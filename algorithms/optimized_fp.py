import sys
from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing as mp
from ATE_update import ATEUpdateLinear
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori
from typing import Dict, List, Tuple, Any, Callable, Optional
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))
# ── configuration ───────────────────────────────────────────────────────────
CHUNK_SIZE_BASE = 32       # baseline batch size for Pool.imap_unordered
SUPPORT_SWITCH = 0.07      # support → Apriori, else FP‑Growth
MIN_TASKS_PER_CORE = 4     # if fewer, run serial instead of Pool

# ── shared globals (populated by _init_worker) ──────────────────────────────
_DF_GLOBAL: Optional[pd.DataFrame] = None


# ── helpers ─────────────────────────────────────────────────────────────────

def _choose_algorithm(min_sup: float):
    """Pick Apriori for high support, FP‑Growth otherwise."""
    return apriori if min_sup >= SUPPORT_SWITCH else fpgrowth


# ── frequent‑pattern mining (sparse + Apriori / FP‑Growth) ──────────────────

def mine_subgroups(df: pd.DataFrame, delta: int, exclude_cols: List[str] = None) -> List[Tuple[Dict[str, Any], int]]:
    """Return [(filter‑dict, size), …] for every subgroup with |S|≥delta."""
    if exclude_cols is None:
        exclude_cols = []
    
    # Filter out columns that should not be used for subgroup mining
    mining_df = df.drop(columns=exclude_cols, errors='ignore')
    
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=bool)
    X = enc.fit_transform(mining_df)
    names = enc.get_feature_names_out()               # e.g. "Country_Germany"

    n_rows = len(df)
    min_sup = delta / n_rows
    onehot_df = pd.DataFrame.sparse.from_spmatrix(X, columns=names)

    freq = _choose_algorithm(min_sup)(
        onehot_df, min_support=min_sup, use_colnames=True
    )

    results: List[Tuple[Dict[str, Any], int]] = []
    for items, sup in zip(freq["itemsets"], freq["support"]):
        filt: Dict[str, Any] = {}
        ok = True
        for col in items:
            attr, val = col.split("_", 1)
            if attr in filt:                           # duplicate attr → skip
                ok = False
                break
            filt[attr] = np.nan if val == "⧫NA⧫" else val
        if ok:
            results.append((filt, int(round(sup * n_rows))))
    return results


# ── vectorised AND‑filter (support already ≥ Δ) ─────────────────────────────

def _get_subgroup_mask(filt: Dict[str, Any]) -> pd.Series:
    """Returns a boolean mask for the given filter."""
    if not filt:
        return pd.Series(True, index=_DF_GLOBAL.index)
    mask = pd.Series(True, index=_DF_GLOBAL.index)
    for a, v in filt.items():
        if pd.isna(v):
            mask &= _DF_GLOBAL[a].isna()
        else:
            mask &= _DF_GLOBAL[a] == v
    return mask


# ── CATE helper – runs inside each worker ───────────────────────────────────

def _compute_cate(treatment: Dict[Any, Any], treatment_col: str, tgtO: str, df: pd.DataFrame, filt: Dict[str, Any]):
    subgroup_mask = _get_subgroup_mask(filt)
    sub_df = df[subgroup_mask]
    
    if sub_df.empty:
        return np.nan
        
    # Create features columns excluding treatment, target, and filter columns
    features_cols = [col for col in df.columns if col not in [*treatment.keys(), treatment_col, *filt.keys(), tgtO]]
    ate_update_obj = ATEUpdateLinear(df[features_cols], df[treatment_col], df[tgtO])
    return ate_update_obj.get_original_ate()


def _eval_cate_worker(args):
    treatment, treatment_col, tgtO, df, (filt, sz) = args
    cate = _compute_cate(treatment, treatment_col, tgtO, df, filt)
    return {
        "AttributeValues": str(filt),
        "Size":            sz,
        "Utility":         cate,
    }


# ── worker initialiser (executed once in every child process) ───────────────

def _init_worker(df):
    global _DF_GLOBAL
    _DF_GLOBAL  = df            # copy‑on‑write: inexpensive after fork


# ── main public entry-point --------------------------------------------------

def calc_utility_for_subgroups(
    mode: int,                       # 0 = homogeneity check, else collect all
    df: pd.DataFrame,
    treatment: Dict[Any, Any],
    treatment_col: str,
    tgtO: str,
    delta: int,
    epsilon: float,
    utility_all: float,
    n_jobs: Optional[int] = None,
):
    """Subgroup utility with self-tuning miner & parallel CATE.

    Returns
    -------
    mode0 → bool
    mode≠0 → (list[dict], int)
    """
    # mine once --------------------------------------------------------------
    # Exclude treatment columns and target outcome from mining
    exclude_cols = [*treatment.keys(), treatment_col, tgtO]
    subgroups = mine_subgroups(df, delta, exclude_cols=exclude_cols)   # [(filt, size), …]
    n_sub = len(subgroups)

    # ─────────────────────────── mode 0: homogeneity check ──────────────────
    if mode == 0:
        # we run serially for the fast early-exit path
        _init_worker(df)
        for filt, _ in subgroups:
            cate = _compute_cate(treatment, treatment_col, tgtO, df, filt)
            if abs(utility_all - cate) > epsilon:
                print(
                    f"\n\033[91msubgroup's cate is: {cate} while utility_all is {utility_all} "
                    f"(Δ={abs(utility_all - cate)}>{epsilon}) → NOT homogeneous\033[0m\n"
                )
                return False
        print("\033[92mHomogenous\033[0m")
        return True

    # ─────────────────────── mode ≠0: collect all subgroups ─────────────────

    cores = n_jobs or mp.cpu_count()
    use_pool = n_sub >= MIN_TASKS_PER_CORE * cores

    records: List[Dict[str, Any]] = []
    args = [(treatment, treatment_col, tgtO, df, sg) for sg in subgroups]

    if use_pool:
        chunk = max(1, min(CHUNK_SIZE_BASE, n_sub // (cores * 2)))
        with mp.Pool(
            processes=cores,
            initializer=_init_worker,
            initargs=(df,),
        ) as pool:
            for rec in pool.imap_unordered(_eval_cate_worker, args, chunksize=chunk):
                records.append(rec)
    else:
        _init_worker(df)
        for arg in args:
            records.append(_eval_cate_worker(arg))

    for r in records:
        if r["Utility"] is not None and not np.isnan(r["Utility"]):
            r["UtilityDiff"] = r["Utility"] - utility_all
        else:
            r["UtilityDiff"] = np.nan


    return records, n_sub
