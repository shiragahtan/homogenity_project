"""
CausalForest-based subgroup analysis using EconML's Generalized Random Forest.
This module uses CausalForest to estimate heterogeneous treatment effects
and identify subgroups with significantly different treatment effects.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from econml.grf import CausalForest
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
# Load config to get Treatment Column
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]

# Import ATE calculation helper
sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe


def _discretize_cate_predictions(
    cate_pred: np.ndarray,
    n_bins: int = 5
) -> np.ndarray:
    """
    Discretize continuous CATE predictions into bins.
    
    Args:
        cate_pred: Array of CATE predictions
        n_bins: Number of bins to create
        
    Returns:
        Array of bin assignments for each prediction
    """
    # Use quantile-based binning to ensure roughly equal-sized bins
    try:
        bins = pd.qcut(cate_pred.flatten(), q=n_bins, labels=False, duplicates='drop')
    except ValueError:
        # If qcut fails (e.g., too few unique values), use simple binning
        bins = pd.cut(cate_pred.flatten(), bins=n_bins, labels=False)
    return bins


def _fit_causal_forest(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    exclude_cols: Optional[List[str]] = None,
    n_estimators: int = 100,
    min_samples_leaf: int = 5,
    random_state: int = 42
) -> Tuple[CausalForest, pd.DataFrame, np.ndarray]:
    """
    Fit a CausalForest model and return predictions.
    
    Args:
        df: Input DataFrame
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        exclude_cols: Columns to exclude from features
        n_estimators: Number of trees in the forest
        min_samples_leaf: Minimum samples per leaf
        random_state: Random seed
        
    Returns:
        Tuple of (fitted model, feature DataFrame, CATE predictions)
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Prepare feature columns
    all_exclude = set(exclude_cols + [treatment_col, BINARY_TREATMENT, outcome_col])
    feature_cols = [col for col in df.columns if col not in all_exclude]
    
    # Drop constant columns
    feature_cols = [col for col in feature_cols if df[col].nunique() > 1]
    
    if not feature_cols:
        raise ValueError("No valid feature columns found for CausalForest")
    
    # Prepare data
    X = df[feature_cols].copy()
    T = df[treatment_col].values
    Y = df[outcome_col].values
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Initialize and fit CausalForest
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        forest = CausalForest(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            honest=True,  # Use honest splitting for better inference
            inference=True,  # Enable inference
            n_jobs=-1  # Use all available cores
        )
        
        try:
            forest.fit(X.values, T, Y)
            # Predict CATE for all samples
            cate_pred = forest.predict(X.values)
        except Exception as e:
            print(f"Error fitting CausalForest: {e}")
            raise
    
    return forest, X, cate_pred


def _identify_subgroups_from_cate(
    df: pd.DataFrame,
    cate_pred: np.ndarray,
    feature_df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    n_bins: int = 5
) -> List[Tuple[Dict[str, Any], int, float, pd.DataFrame]]:
    """
    Identify subgroups based on CATE predictions.
    
    Args:
        df: Original DataFrame
        cate_pred: CATE predictions for each sample
        feature_df: Feature DataFrame used for prediction
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        delta: Minimum subgroup size
        n_bins: Number of bins for discretizing CATE
        
    Returns:
        List of tuples (subgroup_description, size, avg_cate, subgroup_df)
    """
    # Discretize CATE predictions into bins
    bins = _discretize_cate_predictions(cate_pred, n_bins=n_bins)
    
    subgroups = []
    
    # For each bin, create a subgroup
    for bin_idx in np.unique(bins[~np.isnan(bins)]):
        mask = (bins == bin_idx)
        subgroup_size = np.sum(mask)
        
        if subgroup_size < delta:
            continue
        
        # Get the actual dataframe rows for this subgroup
        subgroup_df = df[mask].copy()
        
        # Calculate statistics for this bin
        avg_cate = np.mean(cate_pred[mask])
        
        # Create a description based on CATE range
        cate_min = np.min(cate_pred[mask])
        cate_max = np.max(cate_pred[mask])
        
        description = {
            "cate_bin": int(bin_idx),
            "cate_range": f"[{cate_min:.3f}, {cate_max:.3f}]",
            "avg_cate": avg_cate
        }
        
        subgroups.append((description, subgroup_size, avg_cate, subgroup_df))
    
    return subgroups


def calc_utility_for_subgroups(
    mode: int,
    df: pd.DataFrame,
    treatment_col: str,
    delta: int,
    epsilon: float,
    utility_all: float,
    *,
    tgtO: Optional[str] = None,
    n_estimators: int = 100,
    n_bins: int = None,  # Changed default to None for auto-calculation
    **kwargs: object,
):
    """
    Calculate utility for subgroups using CausalForest.
    
    Args:
        mode: 0 for homogeneity check (returns True/False), 
              other modes for all subgroups (returns list of records)
        df: Input DataFrame
        treatment_col: Treatment column name
        delta: Minimum subgroup size threshold
        epsilon: Threshold for homogeneity check
        utility_all: Overall utility (ATE) for comparison
        tgtO: Target outcome column name
        n_estimators: Number of trees in the forest
        n_bins: Number of bins for discretizing CATE predictions (auto-calculated if None)
        **kwargs: Additional parameters
        
    Returns:
        For mode == 0: Tuple of (boolean, count) indicating homogeneity and subgroups checked
        For mode != 0: Tuple of (list of subgroup records, count)
    """
    if tgtO is None:
        raise ValueError("Target outcome column (tgtO) must be specified")
    
    # Check if we have enough data and variation in treatment
    if df.empty or df[treatment_col].nunique() < 2:
        if mode == 0:
            return (True, 0)  # Return tuple for consistency
        return [], 0
    
    if len(df) < delta:
        if mode == 0:
            return (True, 0)  # Return tuple for consistency
        return [], 0
    
    # Auto-calculate n_bins based on dataset size and delta
    # Goal: Create bins that are large enough to meet delta threshold
    # while still providing meaningful heterogeneity detection
    if n_bins is None:
        # Calculate maximum reasonable number of bins
        # Each bin should ideally have at least 1.5*delta samples for safety margin
        max_bins = max(2, int(len(df) / (1.5 * delta)))
        # Limit to reasonable range: 2-7 bins
        n_bins = min(7, max(2, max_bins))
        print(f"Auto-calculated n_bins={n_bins} based on dataset size {len(df)} and delta {delta}")
    
    # Validate n_bins
    if n_bins < 2:
        print(f"Warning: n_bins={n_bins} is too small, setting to 2")
        n_bins = 2
    
    try:
        # Fit CausalForest and get predictions
        print(f"Fitting CausalForest with {n_estimators} trees...")
        forest, feature_df, cate_pred = _fit_causal_forest(
            df=df,
            treatment_col=treatment_col,
            outcome_col=tgtO,
            exclude_cols=[],
            n_estimators=n_estimators,
            min_samples_leaf=max(5, delta // 10),
            random_state=42
        )
        
        # Identify subgroups based on CATE predictions
        subgroups = _identify_subgroups_from_cate(
            df=df,
            cate_pred=cate_pred,
            feature_df=feature_df,
            treatment_col=treatment_col,
            outcome_col=tgtO,
            delta=delta,
            n_bins=n_bins
        )
        
        print(f"Found {len(subgroups)} subgroups (>= delta={delta}) from {n_bins} CATE bins")
        if len(subgroups) == 0:
            print(f"Warning: No subgroups met the minimum size threshold (delta={delta})")
            print(f"Consider: (1) Decreasing delta, or (2) Decreasing n_bins")
        elif len(subgroups) == 1:
            print(f"Warning: Only 1 subgroup found - limited heterogeneity detection possible")
            print(f"Consider increasing dataset size or decreasing delta for better analysis")
        
        # Process subgroups based on mode
        if mode == 0:
            # Homogeneity check: return False if any subgroup violates epsilon
            num_checked = 0
            print(f"\n{'='*70}")
            print(f"HOMOGENEITY CHECK: Comparing subgroups to overall ATE={utility_all:.4f}")
            print(f"Threshold (epsilon): {epsilon:.4f}")
            print(f"{'='*70}")
            
            for idx, (description, size, avg_cate, subgroup_df) in enumerate(subgroups, 1):
                # Calculate the actual ATE for this subgroup using the standard method
                try:
                    subgroup_ate = calculate_ate_safe(subgroup_df, treatment_col, tgtO)
                    
                    if np.isnan(subgroup_ate):
                        print(f"Subgroup {idx}: Skipped (NaN ATE)")
                        continue
                    
                    num_checked += 1
                    diff = abs(subgroup_ate - utility_all)
                    
                    print(f"\nSubgroup {idx}/{len(subgroups)}: {description}")
                    print(f"  Size: {size:,} samples")
                    print(f"  Subgroup ATE: {subgroup_ate:.4f}")
                    print(f"  Overall ATE:  {utility_all:.4f}")
                    print(f"  Difference:   {diff:.4f} (threshold: {epsilon:.4f})")
                    
                    if diff > epsilon:
                        print(f"  ❌ HETEROGENEITY DETECTED! (diff {diff:.4f} > epsilon {epsilon:.4f})")
                        print(f"\n{'='*70}")
                        print(f"RESULT: Dataset is HETEROGENEOUS")
                        print(f"Checked {num_checked}/{len(subgroups)} subgroups before finding violation")
                        print(f"{'='*70}")
                        return (False, num_checked)
                    else:
                        print(f"  ✓ Within threshold")
                        
                except (LinAlgError, ValueError) as e:
                    print(f"Subgroup {idx}: Error calculating ATE: {e}")
                    continue
            
            # No violations found
            print(f"\n{'='*70}")
            print(f"RESULT: Dataset is HOMOGENEOUS")
            print(f"Checked all {num_checked} subgroups - no violations found")
            print(f"{'='*70}")
            return (True, num_checked)
        
        else:
            # Return all subgroups with their statistics
            subgroup_records = []
            
            for description, size, avg_cate, subgroup_df in subgroups:
                try:
                    # Calculate actual ATE for this subgroup
                    subgroup_ate = calculate_ate_safe(subgroup_df, treatment_col, tgtO)
                    
                    if np.isnan(subgroup_ate):
                        continue
                    
                    subgroup_records.append({
                        "AttributeValues": str(description),
                        "Size": size,
                        "Utility": subgroup_ate,
                        "UtilityDiff": subgroup_ate - utility_all,
                        "PredictedCATE": avg_cate
                    })
                    
                except (LinAlgError, ValueError) as e:
                    print(f"Warning: Could not calculate ATE for subgroup {description}: {e}")
                    continue
            
            return subgroup_records, len(subgroup_records)
    
    except Exception as e:
        print(f"Error in CausalForest analysis: {e}")
        if mode == 0:
            return (True, 0)  # Return tuple for consistency
        return [], 0
