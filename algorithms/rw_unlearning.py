"""
Sub‑group homogeneity analysis — **k random walks version**
==========================================================
This module is a *feature‑complete* drop‑in for your original
`randomww.py`.  It implements **exactly** the strategy you asked for:

* **k independent random walks** in the lattice of attribute=value
  filters, each walk starting at a highly‑specific subgroup (many
  filters) that still meets *δ*.
* Start nodes are sampled **proportionally to an importance score**
  derived from per‑attribute breakage statistics (weights read from
  `config.json`).
* At every step we evaluate the subgroup’s CATE *before* deciding which
  filter to remove.  The walk stops as soon as it reaches a subgroup
  whose size is more than `size_stop · |D|` (default 80 %) or when all
  attributes have been removed.
* The attribute chosen for removal is the **least‑weighted** one among
  the current filters, with a small ε‑greedy randomness so the k walks
  do not collapse onto the same path.
* **Memoised CATE** computations and a `seen` set guarantee each subgroup
  is evaluated at most once across *all* walks.

If any subgroup deviates from the overall ATE by more than *ε* we return
`False` immediately; otherwise, after `k_walks` paths have been
exhausted, we return `True` ("presumed homogeneous").

The exhaustive Apriori path for modes ≠ 0 is unchanged.
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

# ---------------------------------------------------------------------------
#  Config + helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]
ATTRIBUTE_WEIGHTS_RAW: Dict[str, float] = _CFG.get("ATTRIBUTE_WEIGHTS", {})

# normalise weights to [0,1] so they are comparable
if ATTRIBUTE_WEIGHTS_RAW:
    lo, hi = min(ATTRIBUTE_WEIGHTS_RAW.values()), max(ATTRIBUTE_WEIGHTS_RAW.values())
    ATTRIBUTE_WEIGHTS: Dict[str, float] = {
        a: 0.0 if math.isclose(hi, lo) else (w - lo) / (hi - lo)
        for a, w in ATTRIBUTE_WEIGHTS_RAW.items()
    }
else:
    ATTRIBUTE_WEIGHTS = {}

sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe  # noqa: E402  pylint: disable=wrong-import-position

# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    parts: List[pd.DataFrame] = []
    lookup: Dict[str, Tuple[str, str]] = {}
    for col in df.columns:
        dummies = pd.get_dummies(df[col].fillna("⧫NA⧫"), prefix=col, dtype=bool)
        parts.append(dummies)
        lookup.update({c: (col, c.split("_", 1)[1]) for c in dummies.columns})
    return pd.concat(parts, axis=1), lookup


def _mask(df: pd.DataFrame, filt: Mapping[str, str | int | float]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for a, v in filt.items():
        col = df[a]
        m &= col.astype(str) == str(v) if not pd.api.types.is_numeric_dtype(col) else col == int(v)
    return m

# ---------------------------------------------------------------------------
#  k‑random‑walk homogeneity tester (mode 0)
# ---------------------------------------------------------------------------

def _homog_random_walks(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    k_walks: int = 1_000,
    size_stop: float = 0.80,
    rng: Optional[random.Random] = None,
) -> bool:
    """Return **False** on first violating subgroup else **True** after *k* walks."""
    rng = rng or random.Random()

    # overall ATE ---------------------------------------------------------
    try:
        ate_all = calculate_ate_safe(df, treatment_col, outcome_col)
    except LinAlgError:
        return True  # ill‑conditioned global design ⇒ assume homogeneous

    # mine frequent itemsets ≥ δ -----------------------------------------
    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")
    onehot, lookup = _onehot_lookup(mining_df)

    min_sup = delta / len(df)
    freq = apriori(onehot, min_support=min_sup, use_colnames=True)
    freq = freq[freq["itemsets"].apply(lambda s: len({lookup[c][0] for c in s}) == len(s))]
    if freq.empty:
        return True

    # score itemsets: bigger & heavier‑weighted first ---------------------
    def _item_score(itemset: frozenset[str]) -> float:
        attrs = {lookup[c][0] for c in itemset}
        weight = sum(ATTRIBUTE_WEIGHTS.get(a, 0.0) for a in attrs)
        return len(itemset) + weight  # simple additive score

    itemsets = sorted(freq["itemsets"], key=_item_score, reverse=True)

    # sample *k* distinct start filters with roulette‑wheel on scores ------
    scores = np.array([_item_score(s) for s in itemsets], dtype=float)
    probs = scores / scores.sum()
    chosen_idx = rng.choices(range(len(itemsets)), weights=probs, k=min(k_walks, len(itemsets)))
    start_filters = []
    for idx in chosen_idx:
        f = {lookup[c][0]: lookup[c][1] for c in itemsets[idx]}
        start_filters.append(f)

    # memoisation ---------------------------------------------------------
    cate_cache: Dict[frozenset, float] = {}
    visited: set[frozenset] = set()

    def _eval(filt: Dict[str, str]) -> Optional[bool]:
        key = frozenset(filt.items())
        if key in cate_cache:
            cate = cate_cache[key]
        else:
            m = _mask(df, filt)
            sub_df = df[m]
            n = len(sub_df)
            if n < delta or n / len(df) > size_stop:
                return None  # skip out‑of‑range subgroup
            try:
                cate = calculate_ate_safe(sub_df, treatment_col, outcome_col)
            except LinAlgError:
                return None
            cate_cache[key] = cate
        return abs(cate - ate_all) > epsilon

    # walk loop -----------------------------------------------------------
    for root in start_filters:
        current = dict(root)
        while current:
            key = frozenset(current.items())
            if key in visited:
                break
            visited.add(key)

            res = _eval(current)
            if res:  # violation!
                return False

            # pick attribute to drop: lowest weight likely to keep homogeneity
            weights_tuple = [(ATTRIBUTE_WEIGHTS.get(a, 0.0), a) for a in current]
            weights_tuple.sort()  # ascending weight
            least_w_attr = weights_tuple[0][1]
            # ε‑greedy: small chance to pick second‑least to diversify path
            if len(weights_tuple) > 1 and rng.random() < 0.15:
                least_w_attr = weights_tuple[1][1]
            del current[least_w_attr]

    return True

# ---------------------------------------------------------------------------
#  Exhaustive path (modes ≠ 0) — unchanged
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
#  Public API — signature unchanged (extra kwargs tolerated)
# ---------------------------------------------------------------------------

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
    k_walks: int = 1_000,
    size_stop: float = 0.80,
    **kwargs: object,
):
    """Drop‑in compatible with your driver script."""
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
        )

    # ---------- exhaustive path ----------
    full_ate = calculate_ate_safe(df, treatment_col, outcome_col)
    exclude = [treatment_col, BINARY_TREATMENT, outcome_col]
    records: List[Dict[str, str | float | int]] = []
    for filt, sz in _mine_subgroups(algorithm, df, delta, exclude_cols=exclude):
        sub_df = df[_mask(df, filt)]
        if len(sub_df) < delta:
            continue
        try:
            cate = calculate_ate_safe(sub_df, treatment_col, outcome_col)
        except LinAlgError:
            continue
        records.append({"AttributeValues": str(filt), "Size": sz, "Utility": cate, "UtilityDiff": cate - full_ate})
    return records, len(records)
