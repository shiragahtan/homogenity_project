from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

# --- CONFIGURATION ---
# Load config to get Treatment Column
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]

# Import ATE calculation helper
sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe


# helper function: One-Hot Encoding & Lookup
def _onehot_lookup(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Preprocesses the data into One-Hot columns.
    Returns:
        1. DataFrame of booleans (The Matrix)
        2. Dictionary mapping {OneHotCol -> OriginalAttribute}
        3. List of column names for indexing
    """
    parts: List[pd.DataFrame] = []
    lookup: Dict[str, str] = {}

    for col in df.columns:
        # Create dummies, treat NAs as a distinct category
        dummies = pd.get_dummies(df[col].fillna("⧫NA⧫").astype(str), prefix=col, dtype=bool)
        parts.append(dummies)

        # Map specific column "Age_18" back to attribute "Age"
        for c in dummies.columns:
            lookup[c] = col

    final_df = pd.concat(parts, axis=1)
    return final_df, lookup, list(final_df.columns)


# CORE ALGORITHM: Greedy Narrowest Path
def _greedy_narrowest_path_fast(
        df: pd.DataFrame,
        *,
        treatment_col: str,
        outcome_col: str,
        delta: int,
        epsilon: float,
        utility_all: float,
) -> bool:
    """
    Greedy Depth-First Search.
    1. Scan ALL possible next filters.
    2. Pick the one producing the MINIMUM subgroup size >= delta.
    3. Check homogeneity.
    4. Repeat.
    """
    # 1. Prepare Data
    # Drop columns we shouldn't filter on (Treatment, Outcome, etc.)
    excl = {treatment_col, BINARY_TREATMENT, outcome_col}
    mining_df = df.drop(columns=[c for c in excl if c in df], errors="ignore")

    # Create the Boolean Matrix (Rows x Features)
    onehot_df, col_to_attr, col_names = _onehot_lookup(mining_df)
    X_matrix = onehot_df.values
    n_features = X_matrix.shape[1]

    # 2. State Initialization
    current_mask = np.ones(len(df), dtype=bool)  # Start with full dataset
    used_attrs: Set[str] = set()  # Attributes already filtered (e.g., 'Age')
    current_path_cols: List[str] = []  # History for debugging/logging

    print(f"--- Starting Greedy Search (Base Size: {len(df)}) ---")

    # 3. The Greedy Loop
    while True:
        # --- A. VECTORIZED COUNTING ---
        # Get the sub-matrix defined by the current mask
        subset_matrix = X_matrix[current_mask]

        # If we ran out of data rows, stop
        if subset_matrix.shape[0] == 0:
            break

        # INSTANTLY count intersection size for ALL columns
        # Summing the boolean columns gives the count of True values
        counts = subset_matrix.sum(axis=0)

        # --- B. FIND BEST CANDIDATE (Min Size >= Delta) ---
        best_idx = -1
        min_size = float('inf')
        found_candidate = False

        # Scan the counts array
        for i in range(n_features):
            size = counts[i]

            # Optimization: Skip if size is clearly not the min or too small
            if size < delta: continue
            if size >= min_size: continue

            # Logic: Check if attribute is already used
            col_name = col_names[i]
            attr_name = col_to_attr[col_name]

            if attr_name in used_attrs:
                continue

            # If we are here, this is the new best candidate
            min_size = size
            best_idx = i
            found_candidate = True

        # --- C. TERMINATION OR UPDATE ---
        if not found_candidate:
            print("Terminating: No further filters satisfy size >= delta.")
            # No valid filters left that satisfy size >= delta
            break

        # Apply the Best Choice
        chosen_col = col_names[best_idx]
        chosen_attr = col_to_attr[chosen_col]

        # Update State
        # Intersect current mask with new column
        current_mask = current_mask & X_matrix[:, best_idx]
        used_attrs.add(chosen_attr)
        # FIX: Lists use .append(), not .add()
        current_path_cols.append(chosen_col)

        # --- D. CHECK HOMOGENEITY & LOG ---
        # Get the actual rows from the original dataframe
        sub_df = df[current_mask]
        current_size = len(sub_df)

        # >>>> DEBUG PRINT <<<<
        print(f"Checking Subgroup: {current_path_cols} | Size: {current_size}")

        if current_size < delta:  # Sanity check
            break

        try:
            # Calculate Utility (ATE) for this subgroup
            ate_sub = calculate_ate_safe(sub_df, treatment_col, outcome_col)

            # Compare with Global Utility
            diff = abs(ate_sub - utility_all)

            if diff > epsilon:
                # Heterogeneity Found!
                print(f"  >>> VIOLATION FOUND! Path={current_path_cols}")
                print(
                    f"  >>> Size: {current_size}, ATE Sub: {ate_sub:.4f}, ATE All: {utility_all:.4f}, Diff: {diff:.4f}")
                return False

        except LinAlgError:
            print(f"  > LinAlgError (Singular Matrix) for path {current_path_cols}. Skipping.")
            pass

    # If we exit the loop, no heterogeneity was found along the narrowest path
    return True


# -------------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------------
def calc_utility_for_subgroups(
        mode: int,
        df: pd.DataFrame,
        treatment_col: str,
        delta: int,
        epsilon: float,
        utility_all: float,
        *,
        outcome_col: Optional[str] = None,
        tgtO: Optional[str] = None,
        **kwargs: object,
):
    """
    Dispatcher.
    Currently configured to run ONLY the Greedy Narrowest Path if mode is appropriate,
    or you can just call the function directly.
    """
    outcome_col = outcome_col or tgtO
    if outcome_col is None:
        raise ValueError("Need outcome_col / tgtO")

    return _greedy_narrowest_path_fast(
        df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        delta=delta,
        epsilon=epsilon,
        utility_all=utility_all
    )