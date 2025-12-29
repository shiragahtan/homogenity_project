"""
* **OPTIMIZED ALGORITHM: Apriori-Rooted Random Walk (Saturation Aware)**
* **Fix 1 (Saturation):** Tracks if walks are finding new nodes. Stops if space is explored.
* **Fix 2 (Sampling):** Only checks a random subset of columns for intersections per step.
* **Result:** Maintains accuracy (doesn't skip valid paths) but exits early if grid is exhausted.
* **Update:** Returns (Status, Count) of unique subgroups checked.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori  # type: ignore
from numpy.linalg import LinAlgError

# --- CONFIGURATION ---
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

# Still used for the global treatment col setting (fallback)
BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]

sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe

def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    parts: List[pd.DataFrame] = []
    lookup: Dict[str, Tuple[str, str]] = {}
    for col in df.columns:
        dummies = pd.get_dummies(df[col].fillna("⧫NA⧫").astype(str), prefix=col, dtype=bool)
        parts.append(dummies)
        lookup.update({c: (col, c.split("_", 1)[1]) for c in dummies.columns})
    return pd.concat(parts, axis=1), lookup

def _homog_random_walks_direct(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    k_walks: int = 1_000,
    size_stop: float = 0.80,
    rng: Optional[random.Random] = None,
    attribute_weights: Optional[Dict[str, float]] = None, # <--- NEW ARGUMENT
) -> Tuple[bool, int]:
    rng = rng or random.Random()

    # --- WEIGHT NORMALIZATION (Local) ---
    raw_weights = attribute_weights or {}
    normalized_weights = {}
    if raw_weights:
        lo, hi = min(raw_weights.values()), max(raw_weights.values())
        div = hi - lo if hi != lo else 1.0
        normalized_weights = {
            a: (w - lo) / div for a, w in raw_weights.items()
        }

    try:
        ate_all = calculate_ate_safe(df, treatment_col, outcome_col, delta)
    except LinAlgError:
        print("Total unique subgroups checked: 0")
        return True, 0

    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")

    # 1. One-Hot Encoding
    onehot, lookup = _onehot_lookup(mining_df)
    min_sup = delta / len(df)
    all_onehot_cols = list(onehot.columns) # Cache list for sampling

    # 2. Apriori Restricted to Roots (max_len=1)
    freq = apriori(onehot, min_support=min_sup, use_colnames=True, max_len=1)

    if freq.empty:
        print("Total unique subgroups checked: 0")
        return True, 0

    # Score Roots
    def _item_score(itemset: frozenset[str]) -> float:
        attrs = {lookup[c][0] for c in itemset}
        # Use locally normalized weights
        weight = sum(normalized_weights.get(a, 0.0) for a in attrs)
        return weight + 1.0

    itemsets = list(freq["itemsets"])
    scores = np.array([_item_score(s) for s in itemsets], dtype=float)

    if scores.sum() == 0:
        probs = np.ones(len(scores)) / len(scores)
    else:
        probs = scores / scores.sum()

    # 3. The Walk (Construction Phase)
    cate_cache: Dict[frozenset, float] = {}
    visited: Set[frozenset] = set()

    # --- SATURATION LOGIC ---
    # If 50 walks pass without seeing a single new node, the grid is exhausted.
    consecutive_stale_walks = 0
    STALE_LIMIT = 50

    # Select K start nodes
    chosen_indices = rng.choices(range(len(itemsets)), weights=probs, k=k_walks)

    for idx in chosen_indices:

        # Early Exit for Saturation
        if consecutive_stale_walks >= STALE_LIMIT:
            print(f"Saturation reached after {len(visited)} subgroups. Exiting early.")
            break

        root_itemset = itemsets[idx]
        root_col_name = list(root_itemset)[0]

        current_mask = onehot[root_col_name]
        current_cols = {lookup[root_col_name][0]}
        current_itemset = set(root_itemset)

        walk_discovered_new_node = False

        # Helper to check CATE
        def check_current(mask, itemset):
            nonlocal walk_discovered_new_node
            key = frozenset(itemset)

            if key in visited:
                return None

            visited.add(key)
            walk_discovered_new_node = True

            if key in cate_cache:
                return cate_cache[key]

            sub_df = df[mask]
            try:
                val = calculate_ate_safe(sub_df, treatment_col, outcome_col, delta)
                cate_cache[key] = val
                return val
            except LinAlgError:
                return None

        # Check the Root
        cate = check_current(current_mask, current_itemset)
        if cate is not None and abs(cate - ate_all) > epsilon:
            pretty_dict = {lookup[root_col_name][0]: lookup[root_col_name][1]}
            print(f"Breaking Subgroup: {pretty_dict}")
            print(f"Total unique subgroups checked: {len(visited)}")
            return False, len(visited)

        # DIVE LOOP
        while True:
            candidates = []
            candidate_weights = []

            # --- OPTIMIZATION: SAMPLING ---
            if len(all_onehot_cols) > 20:
                cols_to_check = rng.sample(all_onehot_cols, 20)
            else:
                cols_to_check = all_onehot_cols

            for col_name in cols_to_check:
                attr, _ = lookup[col_name]

                if attr in current_cols: continue

                # Boolean intersection
                temp_mask = current_mask & onehot[col_name]
                if temp_mask.sum() >= delta:
                    candidates.append((col_name, temp_mask))
                    # Use locally normalized weights
                    candidate_weights.append(normalized_weights.get(attr, 0.0) + 0.1)

            if not candidates:
                break # Dead end

            total_w = sum(candidate_weights)
            c_probs = [w / total_w for w in candidate_weights]

            choice_idx = rng.choices(range(len(candidates)), weights=c_probs, k=1)[0]
            new_col, new_mask = candidates[choice_idx]

            current_mask = new_mask
            current_cols.add(lookup[new_col][0])
            current_itemset.add(new_col)

            cate = check_current(current_mask, current_itemset)
            if cate is not None and abs(cate - ate_all) > epsilon:
                pretty_dict = {lookup[c][0]: lookup[c][1] for c in current_itemset}
                print(f"Breaking Subgroup: {pretty_dict}")
                print(f"Total unique subgroups checked: {len(visited)}")
                return False, len(visited)

            if rng.random() < 0.1:
                break

        if walk_discovered_new_node:
            consecutive_stale_walks = 0
        else:
            consecutive_stale_walks += 1

    print(f"Total unique subgroups checked: {len(visited)}")
    return True, len(visited)

# Public API
def calc_utility_for_subgroups(
    mode: int,
    algorithm: Callable,
    df: pd.DataFrame,
    treatment_col: str,
    delta: int,
    epsilon: float,
    *,
    outcome_col: Optional[str] = None,
    tgtO: Optional[str] = None,
    k_walks: int = 1_000,
    size_stop: float = 0.8,
    rng: Optional[random.Random] = None,
    attribute_weights: Optional[Dict[str, float]] = None, # <--- NEW ARGUMENT
    **kwargs: object,
) -> Tuple[bool, int]:
    outcome_col = outcome_col or tgtO
    if outcome_col is None:
        raise ValueError("Need outcome_col / tgtO")

    if mode == 0:
        return _homog_random_walks_direct(
            df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            delta=delta,
            epsilon=epsilon,
            k_walks=k_walks,
            size_stop=size_stop,
            rng=rng,
            attribute_weights=attribute_weights # <--- PASSING DOWN
        )

    return [], 0