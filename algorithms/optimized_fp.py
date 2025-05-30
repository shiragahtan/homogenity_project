import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.preprocessing import OneHotEncoder
from mlxtend.frequent_patterns import fpgrowth, apriori

from typing import Dict, List, Tuple, Any, Callable, Optional

# ── configuration ───────────────────────────────────────────────────────────
DELTA = 5000  # Default minimum group size
CHUNK_SIZE = 32  # tasks per batch → good default for 8-32 cores

# ── global refs filled once per worker --------------------------------------
_DF_GLOBAL = None  # full DataFrame slice
_DAG_STR = None
_TREATMENT = None
_ATTR_ORD = None
_TARGET_COL = None


# ── frequent-pattern mining (sparse + FP-Growth) ────────────────────────────
def mine_subgroups(
        df: pd.DataFrame,
        delta: int = DELTA,
) -> List[Tuple[Dict[str, object], int]]:
    # Create the encoder
    enc = OneHotEncoder(handle_unknown='ignore',
                        sparse_output=True,
                        dtype=bool)

    # Fit and transform the data to create one-hot encoded matrix
    X = enc.fit_transform(df)
    names = enc.get_feature_names_out()  # "Country_Germany"

    n_rows = len(df)
    min_sup = delta / n_rows
    onehot_df = pd.DataFrame.sparse.from_spmatrix(X, columns=names)

    if delta > 13000:
        algorithm = apriori
    else:
        algorithm = fpgrowth

    freq = algorithm(onehot_df, min_support=min_sup, use_colnames=True)

    results: List[Tuple[Dict[str, object], int]] = []
    for items, sup in zip(freq['itemsets'], freq['support']):
        filt: Dict[str, object] = {}
        ok = True
        for col in items:
            attr, val = col.split('_', 1)
            if attr in filt:  # duplicate attr → skip
                ok = False
                break
            filt[attr] = val if val != '⧫NA⧫' else np.nan
        if ok:
            results.append((filt, int(round(sup * n_rows))))
    return results


# ── vectorised filter (support already ≥ Δ) ─────────────────────────────────
def _apply_filter(filt: Dict[str, object]) -> pd.DataFrame:
    if not filt:
        return _DF_GLOBAL
    mask = pd.Series(True, index=_DF_GLOBAL.index)
    for a, v in filt.items():
        mask &= (_DF_GLOBAL[a] == v)
    return _DF_GLOBAL[mask]


# ── CATE computation ----------------------------------------------
def _compute_cate(cate_func, filt):
    sub_df = _apply_filter(filt)
    return cate_func(sub_df, _DAG_STR, _TREATMENT, _ATTR_ORD, _TARGET_COL)


# ── worker task -------------------------------------------------------------
def _eval_cate_worker(args):
    cate_func, filt_sz = args
    filt, sz = filt_sz
    cate, p = _compute_cate(cate_func, filt)
    return {
        "AttributeValues": str(filt),
        "Size": sz,
        "Utility": cate,
        "PValue": p,
    }


# ── initialise worker (runs once per fork) ----------------------------------
def _init_worker(df, dag_str, treatment, attr_ord, tgt_col):
    global _DF_GLOBAL, _DAG_STR, _TREATMENT, _ATTR_ORD, _TARGET_COL
    _DF_GLOBAL = df  # each worker gets its own, shared-within-proc copy
    _DAG_STR = dag_str
    _TREATMENT = treatment
    _ATTR_ORD = attr_ord
    _TARGET_COL = tgt_col


# ── main driver -------------------------------------------------------------
def calc_utility_for_subgroups(
        df: pd.DataFrame,
        treatment: Dict[str, object],
        dag_str: str,
        attr_ordinal,
        target_col: str,
        cate_func: Callable,
        delta: int = DELTA,
        n_jobs: Optional[int] = None
):
    util_all, _ = cate_func(df, dag_str, treatment, attr_ordinal, target_col)

    # 1) mine frequent subgroups
    subgroups = mine_subgroups(df, delta)  # [(filt, size), …]

    # 2) parallel CATE (no df pickling, batched)
    with mp.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(df, dag_str, treatment, attr_ordinal, target_col)) as pool:
        records: List[Dict] = []
        # Create argument tuples by pairing cate_func with each subgroup
        args = [(cate_func, sg) for sg in subgroups]
        for rec in pool.imap_unordered(_eval_cate_worker, args, chunksize=CHUNK_SIZE):
            # imap_unordered returns items one-by-one
            records.append(rec)

    for r in records:
        r["UtilityDiff"] = r["Utility"] - util_all

    return records, len(records)
