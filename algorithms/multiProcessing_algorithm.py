from __future__ import annotations
import json
import sys
from pathlib import Path
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Callable, Optional
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori
from numpy.linalg import LinAlgError

# --- Configuration --------------------------------------------------------------
try:
    CONFIG_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'config.json'
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    BINARY_TREATMENT = config.get('TREATMENT_COL', 'T')
except (FileNotFoundError, KeyError) as e:
    print(f"Warning: Could not load config ({e}). Using default values.")
    BINARY_TREATMENT = 'T'

# --- Path for ATE Calculation Utility ---
try:
    sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))
    from ATE_update import calculate_ate_safe
except ImportError:
    print("Warning: 'calculate_ate_safe' not found. Using a placeholder.")
    def calculate_ate_safe(*args, **kwargs) -> float:
        return 0.0

# --- Performance Constants ----------------------------------
OPTIMAL_CORES = min(mp.cpu_count(), os.cpu_count() or mp.cpu_count())
MAX_CHUNK_SIZE = 256
EARLY_EXIT_BATCH_SIZE = 16
SUPPORT_SWITCH = 0.07 
MIN_TASKS_PER_CORE = 8
MIN_SUBGROUPS_FOR_PARALLEL = 32

# --- Shared Memory Globals -----------------------
_DF_GLOBAL: Optional[pd.DataFrame] = None
_TREATMENT_COL_GLOBAL: Optional[str] = None
_TGT_O_GLOBAL: Optional[str] = None
_UTILITY_ALL_GLOBAL: Optional[float] = None
_EPSILON_GLOBAL: Optional[float] = None
_DELTA_GLOBAL: Optional[int] = None


# --- Helper and Worker Functions ------------------------------------------------
def _choose_algorithm(min_sup: float) -> Callable:
    """Dynamically choose the fastest mining algorithm."""
    return apriori if min_sup >= SUPPORT_SWITCH else fpgrowth

def _init_worker(df: pd.DataFrame, treatment_col: str, tgtO: str,
                 utility_all: float = None, epsilon: float = None, delta: int = None):
    """
    Initializes global variables for each worker.
    Crucial: df is NOT copied on Linux/Mac (copy-on-write), saving massive RAM/Time.
    """
    global _DF_GLOBAL, _TREATMENT_COL_GLOBAL, _TGT_O_GLOBAL, _UTILITY_ALL_GLOBAL, _EPSILON_GLOBAL, _DELTA_GLOBAL
    _DF_GLOBAL = df
    _TREATMENT_COL_GLOBAL = treatment_col
    _TGT_O_GLOBAL = tgtO
    _UTILITY_ALL_GLOBAL = utility_all
    _EPSILON_GLOBAL = epsilon
    _DELTA_GLOBAL = delta

def _compute_cate_for_subgroup(filt: Dict[str, Any]) -> float:
    """
    Calculates CATE for a subgroup.
    OPTIMIZED: Uses Numpy masking instead of Pandas Series for raw speed.
    """
    # 1. Fast Numpy Filtering
    # We assume _DF_GLOBAL is static, so direct array access is safe and fast.
    n_rows = len(_DF_GLOBAL)
    mask = np.ones(n_rows, dtype=bool)

    for attr, val in filt.items():
        # Extract column as numpy array (zero-copy if possible)
        col_vals = _DF_GLOBAL[attr].values
        
        if pd.isna(val):
            # Check for NaN/None efficiently
            if col_vals.dtype.kind in 'fc': # Float/Complex
                mask &= np.isnan(col_vals)
            else:
                mask &= pd.isna(col_vals) # Robust object check
        else:
            # Vectorized comparison
            mask &= (col_vals == val)

    # 2. Convert back to Pandas only for the final subset
    # This is much faster than doing it every step of the loop
    sub_df = _DF_GLOBAL[mask]
    
    # 3. Pre-check size (Optimization: Reject before regression)
    if _DELTA_GLOBAL and len(sub_df) < _DELTA_GLOBAL:
        return np.nan

    if sub_df.empty or len(sub_df) < 2:
        return np.nan

    try:
        # 4. Calculate ATE (Pass Delta for strictness)
        return calculate_ate_safe(sub_df, _TREATMENT_COL_GLOBAL, _TGT_O_GLOBAL, delta=_DELTA_GLOBAL)
    except (LinAlgError, ValueError, ZeroDivisionError):
        return np.nan

def _eval_cate_worker(args: Tuple[Dict, int]) -> Dict[str, Any]:
    filt, size = args
    cate = _compute_cate_for_subgroup(filt)
    return {
        "AttributeValues": str(filt),
        "Size": size,
        "Utility": cate,
    }

def _batch_eval_cate_worker(batch_args: List[Tuple[Dict, int]]) -> List[Dict[str, Any]]:
    """Process a batch of subgroups in one go to reduce IPC overhead."""
    results = []
    for args in batch_args:
        result = _eval_cate_worker(args)
        results.append(result)
    return results

def _early_exit_worker(batch_args: List[Tuple[Dict, int]]) -> bool:
    """Mode 0 Worker: Returns True immediately if a violation is found."""
    for filt, _ in batch_args:
        cate = _compute_cate_for_subgroup(filt)
        if pd.notna(cate) and abs(_UTILITY_ALL_GLOBAL - cate) > _EPSILON_GLOBAL:
            return True
    return False

# --- Robust Mining Function --------------------------------
def mine_subgroups_optimized(df: pd.DataFrame, delta: int, exclude_cols: List[str]) -> List[Tuple[Dict, int]]:
    """
    Uses pd.get_dummies (Robust) + Sparse Matrices (Fast).
    Finds ALL subgroups including binary 0/1, unlike OneHotEncoder with drop='if_binary'.
    """
    mining_df = df.drop(columns=exclude_cols, errors='ignore')

    # Safe separator for flattened columns
    sep = "|"
    onehot_parts = []
    lookup: Dict[str, Tuple[str, Any]] = {}

    # 1. Encode with get_dummies (Correctness)
    for col in mining_df.columns:
        # fillna ensures no row loss; dtype=bool saves memory
        d = pd.get_dummies(mining_df[col].fillna('⧫NA⧫'), prefix=col, prefix_sep=sep, dtype=bool)
        onehot_parts.append(d)
        
        # Build Lookup
        for c in d.columns:
            parts = c.split(sep, 1)
            if len(parts) == 2:
                original_col, original_val = parts[0], parts[1]
                if original_val == '⧫NA⧫':
                    lookup[c] = (original_col, np.nan)
                else:
                    # Restore numeric types if possible
                    try:
                        if '.' in original_val:
                            real_val = float(original_val)
                        else:
                            real_val = int(original_val)
                    except ValueError:
                        real_val = original_val
                    lookup[c] = (original_col, real_val)

    # 2. Mine (Speed)
    onehot_df = pd.concat(onehot_parts, axis=1)
    
    # Optional: Convert to sparse if memory is tight (pd.get_dummies returns dense by default)
    # onehot_df = onehot_df.astype(pd.SparseDtype(bool, False)) 

    n_rows = len(df)
    min_sup = delta / n_rows
    algorithm = _choose_algorithm(min_sup)
    
    freq = algorithm(onehot_df, min_support=min_sup, use_colnames=True)

    if freq.empty:
        return []

    # 3. Format
    results: List[Tuple[Dict, int]] = []
    for items, sup in zip(freq['itemsets'], freq['support']):
        filt = {}
        valid_items = True
        attrs_seen = set()
        
        for c in items:
            if c not in lookup:
                valid_items = False
                break
            attr, val = lookup[c]
            
            # Prevent "Age=10 AND Age=20" impossibility
            if attr in attrs_seen:
                valid_items = False
                break
            attrs_seen.add(attr)
            
            filt[attr] = val

        if valid_items:
            results.append((filt, int(round(sup * n_rows))))

    return results

def _calculate_optimal_chunks(n_items: int, n_cores: int) -> Tuple[int, int]:
    target_chunks = n_cores * 3
    chunk_size = max(1, min(MAX_CHUNK_SIZE, n_items // target_chunks))
    if chunk_size < 4: chunk_size = min(4, n_items)
    return chunk_size, (n_items + chunk_size - 1) // chunk_size

def _create_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# --- Main Entry Point ----------------------------------------------------
def calc_utility_for_subgroups(
        mode: int,
        df: pd.DataFrame,
        treatment_col: str,
        tgtO: str,
        delta: int,
        epsilon: float,
        **_: object,
) -> bool | Tuple[List[Dict[str, Any]], int]:
    
    try:
        utility_all = calculate_ate_safe(df, treatment_col, tgtO, delta)
    except LinAlgError:
        return True if mode == 0 else ([], 0)

    exclude_cols = [col for col in {treatment_col, BINARY_TREATMENT, tgtO} if col in df.columns]
    
    # 1. Mining
    subgroups = mine_subgroups_optimized(df, delta, exclude_cols=exclude_cols)
    if not subgroups:
        return True if mode == 0 else ([], 0)

    n_sub = len(subgroups)
    cores = OPTIMAL_CORES
    
    # 2. Parallel Processing
    if mode == 0:
        if n_sub >= MIN_SUBGROUPS_FOR_PARALLEL:
            batch_size = max(EARLY_EXIT_BATCH_SIZE, n_sub // (cores * 4))
            batches = _create_batches(subgroups, batch_size)

            with ProcessPoolExecutor(
                    max_workers=cores,
                    initializer=_init_worker,
                    initargs=(df, treatment_col, tgtO, utility_all, epsilon, delta) 
            ) as executor:
                future_to_batch = {
                    executor.submit(_early_exit_worker, batch): batch
                    for batch in batches
                }
                for future in as_completed(future_to_batch):
                    try:
                        if future.result(): # Found violation
                            for f in future_to_batch: f.cancel()
                            return False
                    except Exception as e:
                        print(f"Warning: {e}")
                        continue
                return True
        else:
            # Serial fallback for tiny tasks
            _init_worker(df, treatment_col, tgtO, utility_all, epsilon, delta)
            for filt, _ in subgroups:
                cate = _compute_cate_for_subgroup(filt)
                if pd.notna(cate) and abs(utility_all - cate) > epsilon:
                    return False
            return True

    elif mode == 1:
        use_pool = n_sub >= MIN_SUBGROUPS_FOR_PARALLEL and n_sub >= MIN_TASKS_PER_CORE * cores
        records = []

        if use_pool:
            chunk_size, _ = _calculate_optimal_chunks(n_sub, cores)
            batches = _create_batches(subgroups, chunk_size)

            with ProcessPoolExecutor(
                    max_workers=cores,
                    initializer=_init_worker,
                    initargs=(df, treatment_col, tgtO, None, None, delta) 
            ) as executor:
                futures = [executor.submit(_batch_eval_cate_worker, batch) for batch in batches]
                for future in as_completed(futures):
                    try:
                        records.extend(future.result())
                    except Exception as e:
                        print(f"Warning: {e}")
                        continue
        else:
            _init_worker(df, treatment_col, tgtO, None, None, delta)
            records = [_eval_cate_worker(arg) for arg in subgroups]

        final_records = []
        for r in records:
            if r:
                # If Utility is valid, calc diff. If NaN, keep it as NaN.
                if pd.notna(r.get("Utility")):
                    r["UtilityDiff"] = r["Utility"] - utility_all
                else:
                    r["UtilityDiff"] = np.nan

                # Append regardless of validity
                final_records.append(r)

        return final_records, len(final_records)
    else:
        raise ValueError("Mode must be 0 or 1.")