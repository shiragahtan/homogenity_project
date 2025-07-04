from __future__ import annotations
import json
import sys
from pathlib import Path
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Callable, Optional
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from queue import Queue

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori
from numpy.linalg import LinAlgError
from sklearn.preprocessing import OneHotEncoder

# --- Configuration --------------------------------------------------------------
try:
    CONFIG_PATH = Path(__file__).resolve().parent.parent / 'configs' / 'config.json'
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    BINARY_TREATMENT = config['TREATMENT_COL']
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

# --- Enhanced Performance Tuning Constants ----------------------------------
OPTIMAL_CORES = min(mp.cpu_count(), os.cpu_count() or mp.cpu_count())
CHUNK_SIZE_BASE = 64         # Increased for better throughput
SUPPORT_SWITCH = 0.07        # Support threshold to switch from FP-Growth to Apriori
MIN_TASKS_PER_CORE = 8       # Increased minimum tasks per core
MIN_SUBGROUPS_FOR_PARALLEL = 32  # Reduced threshold for parallel processing
BATCH_SIZE_MULTIPLIER = 4    # For dynamic batch sizing
MAX_CHUNK_SIZE = 256         # Cap on chunk size
EARLY_EXIT_BATCH_SIZE = 16   # Smaller batches for mode 0 early exit

# --- Shared Memory Globals (Populated by _init_worker) -----------------------
_DF_GLOBAL: Optional[pd.DataFrame] = None
_TREATMENT_COL_GLOBAL: Optional[str] = None
_TGT_O_GLOBAL: Optional[str] = None
_UTILITY_ALL_GLOBAL: Optional[float] = None
_EPSILON_GLOBAL: Optional[float] = None

# --- Helper and Worker Functions ------------------------------------------------
def _choose_algorithm(min_sup: float) -> Callable:
    """Pick Apriori for high support, FP-Growth otherwise."""
    return apriori if min_sup >= SUPPORT_SWITCH else fpgrowth

def _init_worker(df: pd.DataFrame, treatment_col: str, tgtO: str, 
                utility_all: float = None, epsilon: float = None):
    """Initializes global variables for each worker process to avoid data serialization."""
    global _DF_GLOBAL, _TREATMENT_COL_GLOBAL, _TGT_O_GLOBAL, _UTILITY_ALL_GLOBAL, _EPSILON_GLOBAL
    _DF_GLOBAL = df
    _TREATMENT_COL_GLOBAL = treatment_col
    _TGT_O_GLOBAL = tgtO
    _UTILITY_ALL_GLOBAL = utility_all
    _EPSILON_GLOBAL = epsilon

def _compute_cate_for_subgroup(filt: Dict[str, Any]) -> float:
    """Calculates CATE for a single subgroup filter. Optimized version."""
    # Build the filter mask more efficiently
    mask = pd.Series(True, index=_DF_GLOBAL.index)
    
    for attr, val in filt.items():
        col_data = _DF_GLOBAL[attr]
        if pd.isna(val):
            mask &= col_data.isna()
        else:
            # Vectorized comparison with type handling
            col_dtype = col_data.dtype
            if col_dtype.kind in 'biufc':  # numeric types
                try:
                    typed_val = col_dtype.type(val)
                    mask &= (col_data == typed_val)
                except (ValueError, TypeError):
                    mask &= (col_data.astype(str) == str(val))
            else:
                mask &= (col_data == val)
    
    sub_df = _DF_GLOBAL[mask]
    if sub_df.empty or len(sub_df) < 2:  # Need at least 2 rows for meaningful analysis
        return np.nan

    try:
        return calculate_ate_safe(sub_df, _TREATMENT_COL_GLOBAL, _TGT_O_GLOBAL)
    except (LinAlgError, ValueError, ZeroDivisionError):
        return np.nan

def _eval_cate_worker(args: Tuple[Dict, int]) -> Dict[str, Any]:
    """Worker for Mode 1. Calculates utility and returns a full record."""
    filt, size = args
    cate = _compute_cate_for_subgroup(filt)
    return {
        "AttributeValues": str(filt),
        "Size": size,
        "Utility": cate,
    }

def _batch_eval_cate_worker(batch_args: List[Tuple[Dict, int]]) -> List[Dict[str, Any]]:
    """Process a batch of subgroups in a single worker call to reduce overhead."""
    results = []
    for args in batch_args:
        result = _eval_cate_worker(args)
        results.append(result)
    return results

def _early_exit_worker(batch_args: List[Tuple[Dict, int]]) -> bool:
    """Worker for Mode 0 that processes batches and returns True if inhomogeneous found."""
    for filt, _ in batch_args:
        cate = _compute_cate_for_subgroup(filt)
        if pd.notna(cate) and abs(_UTILITY_ALL_GLOBAL - cate) > _EPSILON_GLOBAL:
            return True  # Found inhomogeneous subgroup
    return False  # All subgroups in batch are homogeneous

# --- Optimized Frequent-Pattern Mining ------------------------------------------
def mine_subgroups_optimized(df: pd.DataFrame, delta: int, exclude_cols: List[str]) -> List[Tuple[Dict, int]]:
    """Optimized version using OneHotEncoder, sparse matrices, and algorithm switching."""
    mining_df = df.drop(columns=exclude_cols, errors='ignore')
    
    # Pre-filter columns with too many unique values to avoid memory issues
    max_unique_values = min(1000, len(df) // 10)  # Adaptive threshold
    filtered_cols = []
    for col in mining_df.columns:
        if mining_df[col].nunique() <= max_unique_values:
            filtered_cols.append(col)
    
    if not filtered_cols:
        return []
    
    mining_df = mining_df[filtered_cols]
    
    # Use optimized OneHotEncoder settings
    enc = OneHotEncoder(
        handle_unknown="ignore", 
        sparse_output=True, 
        dtype=bool,
        drop='if_binary'  # Drop one category for binary features
    )
    X_sparse = enc.fit_transform(mining_df.astype(str))
    
    # Use sparse DataFrame for memory efficiency
    onehot_df = pd.DataFrame.sparse.from_spmatrix(X_sparse, columns=enc.get_feature_names_out())

    n_rows = len(df)
    min_sup = delta / n_rows
    
    algorithm = _choose_algorithm(min_sup)
    freq = algorithm(onehot_df, min_support=min_sup, use_colnames=True)

    # Optimized lookup creation
    lookup: Dict[str, Tuple[str, str]] = {}
    for name in enc.get_feature_names_out():
        parts = name.rsplit('_', 1)
        if len(parts) == 2:
            lookup[name] = (parts[0], parts[1])

    # Reconstruct filter dictionaries and sizes with validation
    results: List[Tuple[Dict, int]] = []
    for items, sup in zip(freq['itemsets'], freq['support']):
        # Ensure one attribute per itemset and valid lookup
        attrs = []
        valid_items = []
        for c in items:
            if c in lookup:
                attr = lookup[c][0]
                attrs.append(attr)
                valid_items.append(c)
        
        if len(attrs) == len(set(attrs)) and valid_items:  # One attribute per itemset
            filt = {}
            for c in valid_items:
                attr, val_str = lookup[c]
                # Convert 'nan' string back to np.nan for filtering
                filt[attr] = np.nan if val_str == 'nan' else val_str
            results.append((filt, int(round(sup * n_rows))))
    
    return results

def _calculate_optimal_chunks(n_items: int, n_cores: int) -> Tuple[int, int]:
    """Calculate optimal chunk size and number of chunks for load balancing."""
    # Target: 2-4 chunks per core for good load balancing
    target_chunks = n_cores * 3
    chunk_size = max(1, min(MAX_CHUNK_SIZE, n_items // target_chunks))
    
    # Ensure chunk size is reasonable
    if chunk_size < 4:
        chunk_size = min(4, n_items)
    
    return chunk_size, (n_items + chunk_size - 1) // chunk_size

def _create_batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Create batches of items for processing."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

# --- Main Public Entry-Point ----------------------------------------------------
def calc_utility_for_subgroups(
    mode: int,
    df: pd.DataFrame,
    treatment_col: str,
    tgtO: str,
    delta: int,
    epsilon: float,
    **_: object,
) -> bool | Tuple[List[Dict[str, Any]], int]:
    """
    Performs subgroup analysis using a high-performance, pragmatic strategy.
    Mode 0: Enhanced early-exit parallel processing for homogeneity check
    Mode 1: Optimized batched parallel processing for comprehensive analysis
    """
    try:
        utility_all = calculate_ate_safe(df, treatment_col, tgtO)
    except LinAlgError:
        return True if mode == 0 else ([], 0)

    exclude_cols = [col for col in {treatment_col, BINARY_TREATMENT, tgtO} if col in df.columns]
    subgroups = mine_subgroups_optimized(df, delta, exclude_cols=exclude_cols)

    if not subgroups:
        return True if mode == 0 else ([], 0)

    n_sub = len(subgroups)
    cores = OPTIMAL_CORES

    # --- Mode 0: Enhanced Early-Exit Parallel Processing ---
    if mode == 0:
        # Use parallel processing even for mode 0 with early exit capability
        if n_sub >= MIN_SUBGROUPS_FOR_PARALLEL:
            batch_size = max(EARLY_EXIT_BATCH_SIZE, n_sub // (cores * 4))
            batches = _create_batches(subgroups, batch_size)
            
            # Use ProcessPoolExecutor for better control and early exit
            with ProcessPoolExecutor(
                max_workers=cores,
                initializer=_init_worker,
                initargs=(df, treatment_col, tgtO, utility_all, epsilon)
            ) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(_early_exit_worker, batch): batch 
                    for batch in batches
                }
                
                # Process results as they complete
                for future in as_completed(future_to_batch):
                    try:
                        if future.result():  # Found inhomogeneous subgroup
                            # Cancel remaining futures
                            for f in future_to_batch:
                                f.cancel()
                            return False
                    except Exception as e:
                        print(f"Warning: Error in early exit worker: {e}")
                        continue
                
                return True  # All batches processed, no inhomogeneous subgroups found
        else:
            # Fall back to serial for small jobs
            _init_worker(df, treatment_col, tgtO, utility_all, epsilon)
            for filt, _ in subgroups:
                cate = _compute_cate_for_subgroup(filt)
                if pd.notna(cate) and abs(utility_all - cate) > epsilon:
                    return False
            return True

    # --- Mode 1: Optimized Batched Parallel Processing ---
    elif mode == 1:
        use_pool = n_sub >= MIN_SUBGROUPS_FOR_PARALLEL and n_sub >= MIN_TASKS_PER_CORE * cores

        records: List[Dict[str, Any]] = []
        
        if use_pool:
            # Calculate optimal batch size for better load balancing
            chunk_size, _ = _calculate_optimal_chunks(n_sub, cores)
            batches = _create_batches(subgroups, chunk_size)
            
            # Use ProcessPoolExecutor for better resource management
            with ProcessPoolExecutor(
                max_workers=cores,
                initializer=_init_worker,
                initargs=(df, treatment_col, tgtO)
            ) as executor:
                # Submit all batches
                futures = [executor.submit(_batch_eval_cate_worker, batch) for batch in batches]
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        records.extend(batch_results)
                    except Exception as e:
                        print(f"Warning: Error in batch worker: {e}")
                        continue
        else:
            # Fall back to serial for small jobs
            _init_worker(df, treatment_col, tgtO)
            records = [_eval_cate_worker(arg) for arg in subgroups]
        
        # Post-process results
        final_records = []
        for r in records:
            if r and pd.notna(r.get("Utility")):
                r["UtilityDiff"] = r["Utility"] - utility_all
                final_records.append(r)
        
        return final_records, len(final_records)

    else:
        raise ValueError("Mode must be 0 or 1.")

# --- Additional Performance Monitoring (Optional) -------------------------------
def benchmark_performance(func, *args, **kwargs):
    """Optional utility to benchmark performance of the main function."""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    return result