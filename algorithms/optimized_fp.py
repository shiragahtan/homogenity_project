import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori
from typing import Dict, List, Tuple, Any, Callable, Optional

# ── configuration ───────────────────────────────────────────────────────────
CHUNK_SIZE_BASE = 32       # baseline batch size for Pool.imap_unordered
SUPPORT_SWITCH = 0.07      # ≥ 7 % support → Apriori, else FP‑Growth
MIN_TASKS_PER_CORE = 4     # if fewer, run serial instead of Pool

# ── shared globals (populated by _init_worker) ──────────────────────────────
_DF_GLOBAL: Optional[pd.DataFrame] = None
_DAG_STR = None
_TREATMENT = None
_ATTR_ORD = None
_TARGET_COL = None

# ── helpers ─────────────────────────────────────────────────────────────────

def _choose_algorithm(min_sup: float):
    """Pick Apriori for high support, FP‑Growth otherwise."""
    return apriori if min_sup >= SUPPORT_SWITCH else fpgrowth


# ── frequent‑pattern mining (sparse + Apriori / FP‑Growth) ──────────────────

def mine_subgroups(df: pd.DataFrame, delta: int) -> List[Tuple[Dict[str, Any], int]]:
    """Return [(filter‑dict, size), …] for every subgroup with |S| ≥ delta."""
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=bool)
    X = enc.fit_transform(df)
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


# ── vectorised AND‑filter (support already ≥ Δ) ─────────────────────────────

def _apply_filter(filt: Dict[str, Any]) -> pd.DataFrame:
    if not filt:
        return _DF_GLOBAL
    mask = pd.Series(True, index=_DF_GLOBAL.index)
    for a, v in filt.items():
        mask &= _DF_GLOBAL[a] == v
    return _DF_GLOBAL[mask]


# ── CATE helper – runs inside each worker ───────────────────────────────────

def _compute_cate(cate_func: Callable, filt: Dict[str, Any]):
    sub_df = _apply_filter(filt)
    return cate_func(sub_df, _DAG_STR, _TREATMENT, _ATTR_ORD, _TARGET_COL)


def _eval_cate_worker(args):
    cate_func, (filt, sz) = args
    cate, p = _compute_cate(cate_func, filt)
    return {
        "AttributeValues": str(filt),
        "Size":            sz,
        "Utility":         cate,
        "PValue":          p,
    }


# ── worker initialiser (executed once in every child process) ───────────────

def _init_worker(df, dag_str, treatment, attr_ord, tgt_col):
    global _DF_GLOBAL, _DAG_STR, _TREATMENT, _ATTR_ORD, _TARGET_COL
    _DF_GLOBAL  = df            # copy‑on‑write: inexpensive after fork
    _DAG_STR    = dag_str
    _TREATMENT  = treatment
    _ATTR_ORD   = attr_ord
    _TARGET_COL = tgt_col


# ── main public entry‑point --------------------------------------------------

def calc_utility_for_subgroups(
    mode: int,                       # 0 = homogeneity check, else collect all
    df: pd.DataFrame,
    treatment: Dict[str, Any],
    dag_str: str,
    attr_ordinal,
    target_col: str,
    cate_func: Callable,
    delta: int,
    epsilon: int,
    n_jobs: Optional[int] = None,
):
    """Subgroup utility with self‑tuning miner & parallel CATE.

    Returns
    -------
    mode 0 → bool
    mode ≠0 → (list[dict], int)
    """

    # overall CATE -----------------------------------------------------------
    util_all, _ = cate_func(df, dag_str, treatment, attr_ordinal, target_col)

    # mine once --------------------------------------------------------------
    subgroups = mine_subgroups(df, delta)   # [(filt, size), …]
    n_sub = len(subgroups)

    # ─────────────────────────── mode 0: homogeneity check ──────────────────
    if mode == 0:
        # we run serially for the fast early‑exit path
        _init_worker(df, dag_str, treatment, attr_ordinal, target_col)
        for filt, _ in subgroups:
            cate, _ = _compute_cate(cate_func, filt)
            if cate and abs(util_all - cate) > epsilon:
                print(
                    f"\n\033[91msubgroup's cate is: {cate} while utility_all is {util_all} "
                    f"(Δ={abs(util_all - cate)}>{epsilon}) → NOT homogeneous\033[0m\n"
                )
                return False
        print("\033[92mHomogenous\033[0m")
        return True

    # ─────────────────────── mode ≠0: collect all subgroups ─────────────────

    cores = n_jobs or mp.cpu_count()
    use_pool = n_sub >= MIN_TASKS_PER_CORE * cores

    records: List[Dict[str, Any]] = []
    args = [(cate_func, sg) for sg in subgroups]

    if use_pool:
        chunk = max(1, min(CHUNK_SIZE_BASE, n_sub // (cores * 2)))
        with mp.Pool(
            processes=cores,
            initializer=_init_worker,
            initargs=(df, dag_str, treatment, attr_ordinal, target_col),
        ) as pool:
            for rec in pool.imap_unordered(_eval_cate_worker, args, chunksize=chunk):
                records.append(rec)
    else:
        _init_worker(df, dag_str, treatment, attr_ordinal, target_col)
        for arg in args:
            records.append(_eval_cate_worker(arg))

    for r in records:
        r["UtilityDiff"] = r["Utility"] - util_all

    return records, n_sub
