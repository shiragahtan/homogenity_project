import heapq
from typing import List, Tuple, Set, Dict, Optional, Any
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError

# ... (Keep your imports and CONFIG setup from previous code) ...

def _greedy_best_first_search(
    df: pd.DataFrame,
    *,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    max_depth: int = 3  # Optional: prevent predicates from getting too complex
) -> Tuple[bool, int]:
    
    # 1. Base ATE
    try:
        ate_all = calculate_ate_safe(df, treatment_col, outcome_col)
    except LinAlgError:
        return True, 0

    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")

    # 2. One-Hot Encoding (Reuse your logic)
    onehot, lookup = _onehot_lookup(mining_df)
    
    # We map column names to integer indices for faster processing
    col_names = list(onehot.columns)
    n_cols = len(col_names)
    
    # 3. Priority Queue (Max-Heap)
    # Python's heap is a min-heap, so we store (-size) to simulate max-heap.
    # Structure: (-size, tie_breaker_id, current_mask, set_of_col_indices)
    pq = []
    
    # TRACKING
    visited_signatures: Set[frozenset] = set()
    unique_checks = 0
    
    # 4. Initialize with 1-itemsets (Roots)
    # This matches your intuition: the "biggest" are the roots.
    for i, col in enumerate(col_names):
        mask = onehot[col]
        size = mask.sum()
        
        if size >= delta:
            # We use 'i' as a tiebreaker and part of the signature
            # Store indices as a frozenset to avoid duplicates like {A, B} vs {B, A}
            sig = frozenset([i])
            visited_signatures.add(sig)
            
            # Push to heap
            heapq.heappush(pq, (-size, i, mask, sig))

    # 5. The Greedy Loop
    while pq:
        neg_size, _, current_mask, current_sig_indices = heapq.heappop(pq)
        current_size = -neg_size

        # A. STOPPING CONDITION: Size
        # Since we pop largest first, if this one is too small, ALL remaining are too small.
        if current_size < delta:
            print(f"Stopping: Largest remaining subgroup size ({current_size}) < delta ({delta})")
            break

        # B. CHECK CATE
        unique_checks += 1
        sub_df = df[current_mask]
        
        try:
            val = calculate_ate_safe(sub_df, treatment_col, outcome_col)
            if abs(val - ate_all) > epsilon:
                # Reconstruct readable name for logging
                pretty_name = " AND ".join([lookup[col_names[idx]][0] + "=" + lookup[col_names[idx]][1] for idx in current_sig_indices])
                print(f"Breaking Subgroup Found: {pretty_name} (Size: {current_size})")
                print(f"CATE: {val:.4f} vs ATE: {ate_all:.4f}")
                return False, unique_checks
        except LinAlgError:
            pass # Skip calculation errors

        # C. EXPAND (Generate Children)
        # Only expand if we haven't hit max depth
        if len(current_sig_indices) >= max_depth:
            continue
            
        # Optimization: Only combine with columns having index > max(current_indices)
        # This prevents checking {A, B} and {B, A}. We strictly enforce order A->B.
        start_index = max(current_sig_indices) + 1
        
        for next_idx in range(start_index, n_cols):
            # Check if this combination has been visited (redundancy check)
            new_sig = set(current_sig_indices)
            new_sig.add(next_idx)
            new_sig_frozen = frozenset(new_sig)
            
            if new_sig_frozen in visited_signatures:
                continue
                
            # Create new mask
            new_col_name = col_names[next_idx]
            
            # Optimization: Check if the raw column size is even large enough
            # (If column B has 10 rows, A & B cannot have more than 10)
            if onehot[new_col_name].sum() < delta:
                continue

            new_mask = current_mask & onehot[new_col_name]
            new_size = new_mask.sum()
            
            if new_size >= delta:
                visited_signatures.add(new_sig_frozen)
                heapq.heappush(pq, (-new_size, next_idx, new_mask, new_sig_frozen))

    print(f"Exhausted search. Total unique subgroups checked: {unique_checks}")
    return True, unique_checks