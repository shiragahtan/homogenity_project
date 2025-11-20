"""
* **k independent random walks**
* **OPTIMIZED:** Uses pre-computed one-hot columns for masking (Zero overhead).
* **OPTIMIZED:** Defer score calculation until sampling is required.
* **OPTIMIZED:** Iterative Series masking to avoid DataFrame memory allocation.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori  # type: ignore
from numpy.linalg import LinAlgError

#  Config + helpers
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
NUM_WALKS = 300

with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]
ATTRIBUTE_WEIGHTS_RAW: Dict[str, float] = _CFG.get("ATTRIBUTE_WEIGHTS", {})

if ATTRIBUTE_WEIGHTS_RAW:
    lo, hi = min(ATTRIBUTE_WEIGHTS_RAW.values()), max(ATTRIBUTE_WEIGHTS_RAW.values())
    ATTRIBUTE_WEIGHTS: Dict[str, float] = {
        a: 0.0 if math.isclose(hi, lo) else (w - lo) / (hi - lo)
        for a, w in ATTRIBUTE_WEIGHTS_RAW.items()
    }
else:
    ATTRIBUTE_WEIGHTS = {}

sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe

def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    parts: List[pd.DataFrame] = []
    lookup: Dict[str, Tuple[str, str]] = {}
    for col in df.columns:
        dummies = pd.get_dummies(df[col].fillna("⧫NA⧫"), prefix=col, dtype=bool)
        parts.append(dummies)
        lookup.update({c: (col, c.split("_", 1)[1]) for c in dummies.columns})
    return pd.concat(parts, axis=1), lookup

# OPTIMIZATION 1: FASTER MASKING
def _mask(df: pd.DataFrame, filt: Mapping[str, str | int | float]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for a, v in filt.items():
        col = df[a]
        # Fast path for numeric data
        if pd.api.types.is_numeric_dtype(col):
             m &= (col == float(v)) 
        else:
             m &= (col == v)
    return m

def _homog_random_walks_direct(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    k_walks: int = NUM_WALKS,
    size_stop: float = 0.80,
    rng: Optional[random.Random] = None,
) -> bool:
    """Faithful direct-mode random walk with ZERO SETUP COST optimization."""
    rng = rng or random.Random()
    
    try:
        ate_all = calculate_ate_safe(df, treatment_col, outcome_col)
    except LinAlgError:
        return True
        
    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")
    
    # One-Hot Prep
    onehot, lookup = _onehot_lookup(mining_df)
    min_sup = delta / len(df)
    
    # Mining
    freq = apriori(onehot, min_support=min_sup, use_colnames=True)
    freq = freq[freq["itemsets"].apply(lambda s: len({lookup[c][0] for c in s}) == len(s))]
    if freq.empty:
        return True
        
    itemsets = list(freq["itemsets"])

    # OPTIMIZATION 2: DEFERRED SCORING
    if len(itemsets) <= k_walks:
        # DETERMINISTIC MODE (Fastest)
        chosen_idx = list(range(len(itemsets)))
        chosen_idx.sort(key=lambda i: len(itemsets[i]))
        
        chosen_itemsets = [itemsets[i] for i in chosen_idx]
        do_walk = False
    else:
        # RANDOM MODE
        def _item_score(itemset: frozenset[str]) -> float:
            attrs = {lookup[c][0] for c in itemset}
            weight = sum(ATTRIBUTE_WEIGHTS.get(a, 0.0) for a in attrs)
            return len(itemset) + weight

        scores = np.array([_item_score(s) for s in itemsets], dtype=float)
        
        # Level-1 Guarantee logic
        idx_level_1 = [i for i, s in enumerate(itemsets) if len(s) == 1]
        remaining_budget = k_walks - len(idx_level_1)

        if remaining_budget > 0:
            probs = scores / scores.sum()
            # FIX: Variable name matched here
            chosen_idx_random = rng.choices(range(len(itemsets)), weights=probs, k=remaining_budget)
            final_indices = list(set(idx_level_1) | set(chosen_idx_random))
        else:
            final_indices = idx_level_1[:k_walks]
            
        chosen_itemsets = [itemsets[i] for i in final_indices]
        do_walk = True

    cate_cache: Dict[frozenset, float] = {}
    visited: set[frozenset] = set()

    # OPTIMIZATION 3: ITERATIVE SERIES MASKING
    def _eval_fast(itemset: frozenset[str]) -> Optional[bool]:
        if itemset in cate_cache:
            cate = cate_cache[itemset]
        else:
            cols = list(itemset)
            # 1. Grab first column (Cheap View)
            mask = onehot[cols[0]]
            
            # 2. Iteratively AND (No DataFrame allocation)
            for c in cols[1:]:
                mask = mask & onehot[c]
            
            # 3. Filter
            sub_df = df[mask]
            n = len(sub_df)
            
            if n < delta or n / len(df) > size_stop:
                return None
            try:
                cate = calculate_ate_safe(sub_df, treatment_col, outcome_col)
            except LinAlgError:
                return None
            cate_cache[itemset] = cate
        return abs(cate - ate_all) > epsilon

    for root_itemset in chosen_itemsets:
        # Deterministic Mode
        if not do_walk:
            res = _eval_fast(root_itemset)
            if res: return False
            continue 

        # Random Mode
        current = root_itemset
        while current:
            if current in visited:
                break
            visited.add(current)
            res = _eval_fast(current)
            if res:
                return False
            
            # Walk Up Logic
            candidates = []
            for col_name in current:
                parent_attr = lookup[col_name][0]
                w = ATTRIBUTE_WEIGHTS.get(parent_attr, 0.0)
                candidates.append((w, col_name))
            
            candidates.sort()
            
            least_w_col = candidates[0][1]
            if len(candidates) > 1 and rng.random() < 0.15:
                least_w_col = candidates[1][1]
            
            current = frozenset(c for c in current if c != least_w_col)
            
    return True

def _homog_random_walks(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    k_walks: int = NUM_WALKS,
    size_stop: float = 0.80,
    optimization_mode: str = "direct",
    unlearning_threshold: float = 0.1,
    ate_update_obj=None,
    rng: Optional[random.Random] = None,
) -> bool:
    if optimization_mode == "direct":
        return _homog_random_walks_direct(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta,
            epsilon=epsilon,
            k_walks=k_walks,
            size_stop=size_stop,
            rng=rng,
        )
    return True

#  Public API
def calc_utility_for_subgroups(
    mode: int,
    algorithm: Callable[[pd.DataFrame, float], pd.DataFrame],
    df: pd.DataFrame,
    treatment_col: str,
    delta: int,
    epsilon: float,
    *,
    outcome_col: Optional[str] = None,
    tgtO: Optional[str] = None,
    k_walks: int = NUM_WALKS, 
    size_stop: float = 0.8,
    optimization_mode: str = "direct",
    unlearning_threshold: float = 0.1,
    **kwargs: object,
):
    """Drop-in compatible with your driver script."""
    outcome_col = outcome_col or tgtO
    if outcome_col is None:
        raise ValueError("Need outcome_col / tgtO")

    if mode == 0:
        return _homog_random_walks(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta,
            epsilon=epsilon,
            k_walks=k_walks,
            size_stop=size_stop,
            optimization_mode=optimization_mode,
            unlearning_threshold=unlearning_threshold,
        )

    # Exhaustive path (unchanged for reporting mode)
    full_ate = calculate_ate_safe(df, treatment_col, outcome_col)
    exclude = [treatment_col, BINARY_TREATMENT, outcome_col]
    
    mining_df = df.drop(columns=exclude, errors="ignore")
    onehot, lookup = _onehot_lookup(mining_df)
    freq = algorithm(onehot, min_support=delta / len(df), use_colnames=True)
    freq = freq[freq["itemsets"].apply(lambda s: len({lookup[c][0] for c in s}) == len(s))]
    
    records = []
    for it, sz in zip(freq["itemsets"], freq["support"]):
        filt = {lookup[c][0]: lookup[c][1] for c in it}
        # Use fast mask here too
        mask = onehot[list(it)].all(axis=1)
        sub_df = df[mask]
        
        if len(sub_df) < delta: continue
        try:
            cate = calculate_ate_safe(sub_df, treatment_col, outcome_col)
        except LinAlgError: continue
            
        records.append({
            "AttributeValues": str(filt),
            "Size": int(round(sz * len(df))),
            "Utility": cate,
            "UtilityDiff": cate - full_ate
        })
        
    return records, len(records)

# Helper function for exhaustive path (if needed by other scripts)
def _mine_subgroups(
    algorithm: Callable[[pd.DataFrame, float], pd.DataFrame],
    df: pd.DataFrame,
    delta: int,
    *,
    exclude_cols: Sequence[str] = (),
) -> List[Tuple[Dict[str, str], int]]:
    mining_df = df.drop(columns=list(exclude_cols), errors="ignore")
    onehot, lookup = _onehot_lookup(mining_df)
    freq = algorithm(onehot, min_support=delta / len(df), use_colnames=True)
    freq = freq[freq["itemsets"].apply(lambda s: len({lookup[c][0] for c in s}) == len(s))]
    out: List[Tuple[Dict[str, str], int]] = []
    for it, sup in zip(freq["itemsets"], freq["support"]):
        out.append(({lookup[c][0]: lookup[c][1] for c in it}, int(round(sup * len(df)))))
    return out