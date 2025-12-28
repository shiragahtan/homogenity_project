"""
Binary Search Algorithm for Problem 3: Finding Smallest Epsilon Achieving Homogeneity.

Given a rule r and fixed delta, finds the smallest epsilon threshold
for which the rule becomes homogeneous (no violations).

Monotonicity Property: If a rule is homogeneous at epsilon, it remains 
homogeneous for all epsilon' > epsilon (upward-closure).
"""
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Dict

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

# Add project paths
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from ATE_update import calculate_ate_safe
from apriori_algorithm import calc_utility_for_subgroups as fpgrowth_oracle

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    config = json.load(fp)

TREATMENT_COL = config['TREATMENT_COL']


def oracle_is_homogeneous(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon: float,
    utility_all: float
) -> Tuple[bool, int, Optional[Dict]]:
    """
    Oracle function that checks if rule is homogeneous at given epsilon.
    Uses FPGrowth algorithm for subgroup enumeration.
    
    Args:
        df: Input DataFrame
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        delta: Minimum subgroup size threshold
        epsilon: Homogeneity threshold
        utility_all: Overall ATE
        
    Returns:
        Tuple of (is_homogeneous, num_subgroups_checked, violation_info)
        - is_homogeneous: True if no violations (homogeneous)
        - num_subgroups_checked: Number of subgroups evaluated
        - violation_info: Dict with violating subgroup info (None if homogeneous)
    """
    result = fpgrowth_oracle(
        mode=0,
        algorithm=fpgrowth,
        df=df,
        treatment_col=treatment_col,
        tgtO=outcome_col,
        delta=delta,
        epsilon=epsilon,
        utility_all=utility_all
    )
    
    # Parse result
    if isinstance(result, tuple):
        if len(result) >= 5:
            # (homogeneity_status, count, enum_time, iter_time, violation_info)
            is_homogeneous = result[0]
            num_checked = result[1]
            violation_info = result[4]
            return (is_homogeneous, num_checked, violation_info)
        elif len(result) >= 2:
            # Old format
            is_homogeneous = result[0]
            num_checked = result[1]
            return (is_homogeneous, num_checked, None)
    
    # Fallback
    is_homogeneous = bool(result)
    return (is_homogeneous, 0, None)


def find_smallest_epsilon_achieving_homogeneity(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    epsilon_start: float = 1000.0,
    epsilon_max: float = 1000000.0,
    verbose: bool = True
) -> Tuple[Optional[float], int, Optional[Dict], float]:
    """
    Binary search algorithm to find the smallest epsilon where rule is homogeneous.
    
    Searches the range [0, epsilon_max] using standard binary search.
    
    Args:
        df: Input DataFrame
        treatment_col: Treatment column name
        outcome_col: Outcome column name  
        delta: Fixed minimum subgroup size
        epsilon_start: (Unused - kept for compatibility)
        epsilon_max: Maximum epsilon to consider
        verbose: Print progress messages
        
    Returns:
        Tuple of (smallest_epsilon, total_oracle_calls, largest_epsilon_violation_info, utility_all)
        - smallest_epsilon: Smallest epsilon achieving homogeneity (None if not found)
        - total_oracle_calls: Total oracle invocations
        - largest_epsilon_violation_info: Info about violation at epsilon-1 (None if not applicable)
        - utility_all: Overall population ATE
    """
    if verbose:
        print("="*70)
        print(f"FINDING SMALLEST EPSILON ACHIEVING HOMOGENEITY (Binary Search)")
        print(f"Fixed delta: {delta}")
        print(f"Search range: [0, {epsilon_max:,.0f}]")
        print("="*70)
    
    # Calculate overall ATE once
    utility_all = calculate_ate_safe(df, treatment_col, outcome_col, delta)
    
    total_oracle_calls = 0
    last_violation_info = None  # Track the most recent violation
    
    # ===== BINARY SEARCH =====
    epsilon_low = 0
    epsilon_high = int(epsilon_max)
    
    # First, check if even epsilon_max achieves homogeneity
    total_oracle_calls += 1
    is_homogeneous_at_max, _, _ = oracle_is_homogeneous(
        df, treatment_col, outcome_col, delta, epsilon_high, utility_all
    )
    
    if not is_homogeneous_at_max:
        if verbose:
            print(f"\n⚠ No homogeneity found even at epsilon_max = {epsilon_max:,.0f}")
        return None, total_oracle_calls, last_violation_info, utility_all
    
    if verbose:
        print(f"\n🔍 Binary Search on [0, {epsilon_high:,.0f}]")
        print("-" * 70)
    
    iteration = 0
    
    while epsilon_low < epsilon_high:
        epsilon_mid = (epsilon_low + epsilon_high) // 2
        iteration += 1
        total_oracle_calls += 1
        
        if verbose:
            print(f"\n  Iteration {iteration}:")
            print(f"    Range: [{epsilon_low:,.0f}, {epsilon_high:,.0f}]")
            print(f"    Testing epsilon = {epsilon_mid:,.0f}")
        
        is_homogeneous, num_checked, violation_info = oracle_is_homogeneous(
            df, treatment_col, outcome_col, delta, epsilon_mid, utility_all
        )
        
        if verbose:
            status = "HOMOGENEOUS ✓" if is_homogeneous else "HETEROGENEOUS ✗"
            print(f"    → {status} (checked {num_checked} subgroups)")
        
        if is_homogeneous:
            # Can potentially go lower
            epsilon_high = epsilon_mid
            if verbose:
                print(f"    → Searching lower: new high = {epsilon_high:,.0f}")
        else:
            # Need higher epsilon - track the violation
            last_violation_info = violation_info
            epsilon_low = epsilon_mid + 1
            if verbose:
                print(f"    → Searching higher: new low = {epsilon_low:,.0f}")
    
    smallest_epsilon = epsilon_high
    
    if verbose:
        print("\n" + "="*70)
        print(f"RESULT: Smallest epsilon achieving homogeneity = {smallest_epsilon:,.0f}")
        if last_violation_info:
            print(f"  Largest epsilon with violation = {smallest_epsilon - 1:,.0f}")
            print(f"\n  Violating subgroup at ε={smallest_epsilon-1}:")
            print(f"    Subgroup: {last_violation_info['subgroup']}")
            print(f"    Size: {last_violation_info['size']}")
            print(f"    Utility: {last_violation_info['utility']:.2f}")
            print(f"    Population Utility: {utility_all:.2f}")
            print(f"    |Difference|: {last_violation_info['abs_diff']:.2f}")
        print(f"\nTotal oracle calls: {total_oracle_calls}")
        print("="*70)
    
    return smallest_epsilon, total_oracle_calls, last_violation_info, utility_all


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find smallest epsilon achieving homogeneity')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--outcome', type=str, default='ConvertedSalary', help='Outcome column')
    parser.add_argument('--delta', type=int, required=True, help='Fixed delta threshold')
    parser.add_argument('--epsilon_start', type=float, default=1000.0, help='Starting epsilon')
    parser.add_argument('--epsilon_max', type=float, default=1000000.0, help='Maximum epsilon')
    
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(args.dataset)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Run algorithm
    smallest_epsilon, oracle_calls, violation_info, utility_all = find_smallest_epsilon_achieving_homogeneity(
        df=df,
        treatment_col=TREATMENT_COL,
        outcome_col=args.outcome,
        delta=args.delta,
        epsilon_start=args.epsilon_start,
        epsilon_max=args.epsilon_max,
        verbose=True
    )
    
    if smallest_epsilon is not None:
        print(f"\n✅ Final Answer: ε* = {smallest_epsilon:,.0f}")
        print(f"   Efficiency: Found in {oracle_calls} oracle calls")
        if violation_info:
            print(f"\n   Last Violation (ε={smallest_epsilon-1}):")
            print(f"     Subgroup: {violation_info['subgroup']}")
            print(f"     |Difference|: {violation_info['abs_diff']:.2f}")
    else:
        print(f"\n❌ No homogeneous epsilon found up to {args.epsilon_max:,.0f}")

