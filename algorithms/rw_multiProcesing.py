from __future__ import annotations
import json, math, random, sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple
import multiprocessing as mp
import numpy as np, pandas as pd
from mlxtend.frequent_patterns import apriori
from numpy.linalg import LinAlgError
from rw_unlearning import calc_utility_for_subgroups as rw_unlearning_serial

# ── config ───────────────────────────────────────────────────────────
CFG = json.loads(Path(Path(__file__).resolve().parent.parent / "configs" / "config.json").read_text())
BINARY_TREATMENT: str = CFG["TREATMENT_COL"]
_RAW_W = CFG.get("ATTRIBUTE_WEIGHTS", {})
if _RAW_W:
    lo, hi = min(_RAW_W.values()), max(_RAW_W.values())
    ATTR_W = {a: 0.0 if math.isclose(lo, hi) else (w - lo) / (hi - lo) for a, w in _RAW_W.items()}
else:
    ATTR_W = {}

sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe  # heavy import once only

# ── helpers ──────────────────────────────────────────────────────────

def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, str]]]:
    parts, look = [], {}
    for col in df.columns:
        d = pd.get_dummies(df[col].fillna("⧫NA⧫"), prefix=col, dtype=bool)
        parts.append(d)
        look.update({c: (col, c.split("_", 1)[1]) for c in d.columns})
    return pd.concat(parts, axis=1), look


def _mask(df: pd.DataFrame, filt: Mapping[str, str | int | float]) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for a, v in filt.items():
        col = df[a]
        m &= col.astype(str) == str(v) if not pd.api.types.is_numeric_dtype(col) else col == int(v)
    return m

# ── worker globals (shared) ──────────────────────────────────────────
_DF: Optional[pd.DataFrame] = None
_TREAT: Optional[str] = None
_OUTCOME: Optional[str] = None
_ATE_ALL: float = 0.0
_EPS: float = 0.0
_DELTA: int = 0
_STOP: float = 0.8
_VISITED: "mp.Manager().dict" = None  # type: ignore
_CACHE: "mp.Manager().dict" = None    # type: ignore


def _init_worker(df, treat, outcome, ate_all, eps, delta, stop, visited, cache):
    global _DF, _TREAT, _OUTCOME, _ATE_ALL, _EPS, _DELTA, _STOP, _VISITED, _CACHE
    _DF = df; _TREAT = treat; _OUTCOME = outcome
    _ATE_ALL, _EPS, _DELTA, _STOP = ate_all, eps, delta, stop
    _VISITED, _CACHE = visited, cache

# ── single walk ─────────────────────────────────────────────────────

def _run_walk(root: Dict[str, str]) -> bool:  # True → heterogeneity found
    rng = random.Random()
    current = dict(root)
    while current:
        k = frozenset(current.items())
        if k in _VISITED:
            break
        _VISITED[k] = True  # atomic via Manager proxy
        if k in _CACHE:
            cate = _CACHE[k]
        else:
            sub = _DF[_mask(_DF, current)]
            n = len(sub)
            if n < _DELTA or n / len(_DF) > _STOP:
                return False
            try:
                cate = calculate_ate_safe(sub, _TREAT, _OUTCOME)
            except LinAlgError:
                return False
            _CACHE[k] = cate
        if abs(cate - _ATE_ALL) > _EPS:
            return True
        w = sorted((ATTR_W.get(a, 0.0), a) for a in current)
        drop = w[1][1] if len(w) > 1 and rng.random() < 0.15 else w[0][1]
        del current[drop]
    return False

# ── global pool (created once) ───────────────────────────────────────
if sys.platform.startswith("linux"):
    _CTX = mp.get_context("fork")
else:
    _CTX = mp.get_context("spawn")
_POOL: Optional[mp.pool.Pool] = None


def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = _CTX.Pool(processes=_CTX.cpu_count())
    return _POOL

# ── public API ───────────────────────────────────────────────────────

def calc_utility_for_subgroups(
    mode: int,
    algorithm: Callable[[pd.DataFrame, float], pd.DataFrame],  # kept for signature but ignored (always Apriori)
    df: pd.DataFrame,
    treatment_col: str,
    delta: int,
    epsilon: float,
    *,
    outcome_col: Optional[str] = None,
    tgtO: Optional[str] = None,
    k_walks: int = 1_000,
    size_stop: float = 0.8,
    **_: object,
):
    """Exact replica of direct random walks, but auto-parallel."""
    if delta != 5000:
        return rw_unlearning_serial(
            mode=mode,
            algorithm=algorithm,
            df=df,
            treatment_col=treatment_col,
            delta=delta,
            epsilon=epsilon,
            outcome_col=outcome_col,
            tgtO=tgtO,
            k_walks=k_walks,
            size_stop=size_stop,
        )
    if mode != 0:
        raise NotImplementedError
    outcome_col = outcome_col or tgtO
    if outcome_col is None:
        raise ValueError

    try:
        ate_all = calculate_ate_safe(df, treatment_col, outcome_col)
    except LinAlgError:
        return True

    onehot, lookup = _onehot_lookup(df.drop(columns=[c for c in {treatment_col, BINARY_TREATMENT, outcome_col} if c in df], errors="ignore"))
    freq = apriori(onehot, min_support=delta / len(df), use_colnames=True)
    freq = freq[freq["itemsets"].apply(lambda s: len({lookup[c][0] for c in s}) == len(s))]
    if freq.empty:
        return True

    def _score(it):
        return len(it) + sum(ATTR_W.get(lookup[c][0], 0.0) for c in it)

    items = sorted(freq["itemsets"], key=_score, reverse=True)
    if not items:
        return True
    probs = np.array([_score(s) for s in items], float); probs /= probs.sum()
    roots = [{lookup[c][0]: lookup[c][1] for c in s} for s in random.choices(items, probs, k=min(k_walks, len(items)))]

    cores = _CTX.cpu_count()
    if k_walks < cores * 8:  # small job → serial faster
        visited, cache = {}, {}
        _init_worker(df, treatment_col, outcome_col, ate_all, epsilon, delta, size_stop, visited, cache)
        return not any(_run_walk(r) for r in roots)

    manager = _CTX.Manager()
    visited = manager.dict()
    cache = manager.dict()
    pool = _get_pool()
    
    # Initialize all workers with the data
    init_results = []
    for _ in range(cores):
        result = pool.apply_async(_init_worker, args=(df, treatment_col, outcome_col, ate_all, epsilon, delta, size_stop, visited, cache))
        init_results.append(result)
    
    # Wait for all workers to be initialized
    for result in init_results:
        result.wait()
    
    chunk = max(32, len(roots) // (cores * 2))
    for flag in pool.imap_unordered(_run_walk, roots, chunksize=chunk):
        if flag:
            return False
    return True