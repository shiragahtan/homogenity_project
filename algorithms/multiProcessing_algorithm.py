import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp
from ATE_update import calculate_ate_safe
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori
from typing import Dict, List, Tuple, Any, Callable, Optional
from numpy.linalg import LinAlgError
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

# Load config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

BINARY_TREATMENT = config['TREATMENT_COL']

# ── configuration ───────────────────────────────────────────────────────────
CHUNK_SIZE_BASE = 32       # baseline batch size for Pool.imap_unordered
SUPPORT_SWITCH = 0.07      # support → Apriori, else FP‑Growth
MIN_TASKS_PER_CORE = 4     # if fewer, run serial instead of Pool
MIN_SUBGROUPS_FOR_PARALLEL = 50  # minimum subgroups to use multiprocessing

# ── shared globals (populated by _init_worker) ──────────────────────────────
_DF_GLOBAL: Optional[pd.DataFrame] = None
_TREATMENT_COL_GLOBAL: Optional[str] = None
_TGT_O: Optional[str] = None

# ── helpers ─────────────────────────────────────────────────────────────────

def _choose_algorithm(min_sup: float):
    """Pick Apriori for high support, FP‑Growth otherwise."""
    return apriori if min_sup >= SUPPORT_SWITCH else fpgrowth


# ── optimized frequent‑pattern mining ──────────────────────────────────────

def mine_subgroups_optimized(df: pd.DataFrame, delta: int, exclude_cols: List[str] = None) -> List[Tuple[Dict[str, Any], int]]:
    """Optimized version using OneHotEncoder and sparse matrices."""
    if exclude_cols is None:
        exclude_cols = []
    
    # Filter out columns that should not be used for subgroup mining
    mining_df = df.drop(columns=exclude_cols, errors='ignore')
    
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=bool)
    X = enc.fit_transform(mining_df)
    names = enc.get_feature_names_out()

    n_rows = len(df)
    min_sup = delta / n_rows
    
    # Create sparse DataFrame efficiently
    dense_df = pd.DataFrame(X.toarray(), columns=names, dtype=bool)
    onehot_df = dense_df.astype(pd.SparseDtype(bool, fill_value=False))

    freq = _choose_algorithm(min_sup)(
        onehot_df, min_support=min_sup, use_colnames=True
    )

    # Create lookup dictionary for attribute names
    lookup: Dict[str, Tuple[str, object]] = {}
    for col in mining_df.columns:
        unique_vals = mining_df[col].fillna('⧫NA⧫').unique()
        for val in unique_vals:
            col_name = f"{col}_{val}"
            if col_name in names:
                lookup[col_name] = (col, val)

    # Discard itemsets that mention the same attribute twice (matching apriori_algorithm)
    def valid(itemset):
        attrs = [lookup[col][0] for col in itemset if col in lookup]
        return len(attrs) == len(set(attrs))
    
    freq = freq[freq['itemsets'].apply(valid)]

    results: List[Tuple[Dict[str, Any], int]] = []
    for items, sup in zip(freq["itemsets"], freq["support"]):
        filt: Dict[str, Any] = {}
        ok = True
        for col in items:
            if col in lookup:
                attr, val = lookup[col]
                if attr in filt:                           # duplicate attr → skip
                    ok = False
                    break
                filt[attr] = np.nan if val == "⧫NA⧫" else val
        if ok:
            results.append((filt, int(round(sup * n_rows))))
    return results


# ── vectorised AND‑filter (matching apriori_algorithm) ──────────────────────

def _get_subgroup_mask(filt: Dict[str, Any]) -> pd.Series:
    """Returns a boolean mask for the given filter (matching apriori_algorithm)."""
    if not filt:
        return pd.Series(True, index=_DF_GLOBAL.index)
    mask = pd.Series(True, index=_DF_GLOBAL.index)
    for a, v in filt.items():
        if pd.isna(v):
            mask &= _DF_GLOBAL[a].isna()
        else:
            # Convert value to int to match apriori_algorithm behavior
            mask &= _DF_GLOBAL[a] == int(v)
    return mask


# ── CATE helper – runs inside each worker ───────────────────────────────────

def _compute_cate_optimized(filt: Dict[str, Any]):
    """Optimized CATE computation using calculate_ate_safe."""
    subgroup_mask = _get_subgroup_mask(filt)
    sub_df = _DF_GLOBAL[subgroup_mask]
    
    if sub_df.empty:
        return np.nan

    try:
        cate_value = calculate_ate_safe(sub_df, _TREATMENT_COL_GLOBAL, _TGT_O)
        return cate_value
    except LinAlgError:  # XᵀX still singular
        return np.nan


def _eval_cate_worker(args):
    (filt, sz) = args
    cate = _compute_cate_optimized(filt)
    return {
        "AttributeValues": str(filt),
        "Size": sz,
        "Utility": cate,
    }


# ── worker initialiser ──────────────────────────────────────────────────────

def _init_worker(df, treatment_col, tgtO):
    global _DF_GLOBAL, _TREATMENT_COL_GLOBAL, _TGT_O
    _DF_GLOBAL = df
    _TREATMENT_COL_GLOBAL = treatment_col
    _TGT_O = tgtO


# ── main public entry‑point --------------------------------------------------

def calc_utility_for_subgroups(
    mode: int,                       # 0 = homogeneity check, else collect all
    df: pd.DataFrame,
    treatment_col: str,
    tgtO: str,
    delta: int,
    epsilon: float,
    utility_all: float,
    n_jobs: Optional[int] = None,
):
    """Optimized subgroup utility with multiprocessing and sparse matrices.
    
    Falls back to apriori_algorithm logic if overhead is too high.
    """
    # Exclude treatment columns and target outcome from mining
    exclude_cols = [treatment_col, BINARY_TREATMENT, tgtO]
    
    # Pre-compute features columns once
   
    # Mine subgroups using optimized approach
    subgroups = mine_subgroups_optimized(df, delta, exclude_cols=exclude_cols)
    n_sub = len(subgroups)

    # ─────────────────────────── mode 0: homogeneity check ──────────────────
    if mode == 0:
        # Run serially for fast early-exit (matching apriori_algorithm)
        _init_worker(df, treatment_col, tgtO)
        for filt, _ in subgroups:
            cate = _compute_cate_optimized(filt)
            if pd.isna(cate):
                continue
            if abs(utility_all - cate) > epsilon:
                print(
                    f"\n\033[91msubgroup's cate is: {cate} while utility_all is {utility_all} "
                    f"(Δ={abs(utility_all - cate)}>{epsilon}) → NOT homogeneous\033[0m\n"
                )
                return False
        print("\033[92mHomogenous\033[0m")
        return True

    # ─────────────────────── mode ≠0: collect all subgroups ─────────────────

    # Decide whether to use multiprocessing based on overhead
    cores = n_jobs or mp.cpu_count()
    use_pool = n_sub >= MIN_SUBGROUPS_FOR_PARALLEL and n_sub >= MIN_TASKS_PER_CORE * cores

    records: List[Dict[str, Any]] = []
    args = [(filt, sz) for filt, sz in subgroups]

    if use_pool:
        # Use multiprocessing for large datasets
        chunk = max(1, min(CHUNK_SIZE_BASE, n_sub // (cores * 2)))
        with mp.Pool(
            processes=cores,
            initializer=_init_worker,
            initargs=(df, treatment_col, tgtO),
        ) as pool:
            for rec in pool.imap_unordered(_eval_cate_worker, args, chunksize=chunk):
                records.append(rec)
    else:
        # Fall back to serial processing (matching apriori_algorithm)
        _init_worker(df, treatment_col, tgtO)
        for arg in args:
            records.append(_eval_cate_worker(arg))

    # Filter out records with NaN utility values (matching apriori_algorithm)
    filtered_records = []
    for r in records:
        if r["Utility"] is not None and not np.isnan(r["Utility"]):
            r["UtilityDiff"] = r["Utility"] - utility_all
            filtered_records.append(r)
        else:
            r["UtilityDiff"] = np.nan

    return filtered_records, len(filtered_records)
