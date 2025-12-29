"""
Brute Force Algorithm for Problem 3: Finding Smallest Epsilon Achieving Homogeneity.

This method enumerates ALL subgroups using FPGrowth and finds the maximum
utility difference, which directly gives us the smallest epsilon needed.

Comparison with two-phase search:
- Brute force: Examines all subgroups once, O(number of subgroups)
- Two-phase: Multiple oracle calls, O(log(epsilon_max)) oracle calls
"""
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, Dict
import time

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


def find_smallest_epsilon_bruteforce(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    delta: int,
    verbose: bool = True
) -> Tuple[Optional[float], int, Optional[Dict], float, float]:
    """
    Brute force method to find smallest epsilon by enumerating all subgroups.
    
    Algorithm:
    1. Enumerate ALL subgroups with size >= delta using FPGrowth
    2. Calculate utility for each subgroup
    3. Find the maximum |ATE(subgroup) - ATE(population)|
    4. This maximum IS the smallest epsilon needed for homogeneity
    
    Args:
        df: Input DataFrame
        treatment_col: Treatment column name
        outcome_col: Outcome column name
        delta: Fixed minimum subgroup size
        verbose: Print progress messages
        
    Returns:
        Tuple of (smallest_epsilon, num_subgroups_checked, max_violation_info, utility_all, runtime)
    """
    if verbose:
        print("="*70)
        print(f"BRUTE FORCE: FINDING SMALLEST EPSILON")
        print(f"Fixed delta: {delta}")
        print(f"Method: Enumerate all subgroups and find max utility difference")
        print("="*70)
    
    start_time = time.time()
    
    # Calculate overall ATE
    utility_all = calculate_ate_safe(df, treatment_col, outcome_col, delta)
    
    if verbose:
        print(f"\nPopulation ATE: {utility_all:.2f}")
        print(f"Enumerating all subgroups with size >= {delta}...")
    
    # Run FPGrowth in mode=1 to get all subgroups
    # This will return: (records, count, enum_time, iter_time, max_abs_diff, max_violation_info)
    result = fpgrowth_oracle(
        mode=1,  # Mode 1 = collect all
        algorithm=fpgrowth,
        df=df,
        treatment_col=treatment_col,
        tgtO=outcome_col,
        delta=delta,
        epsilon=float('inf'),  # No early stopping
        utility_all=utility_all
    )
    
    elapsed_time = time.time() - start_time
    
    # Parse result
    if isinstance(result, tuple) and len(result) >= 6:
        subgroup_records, num_checked, enum_time, iter_time, max_abs_diff, max_violation_info = result
    else:
        if verbose:
            print("⚠️  Unexpected return format from FPGrowth")
        return None, 0, None, utility_all, elapsed_time
    
    smallest_epsilon = max_abs_diff if max_abs_diff > 0 else 0
    
    if verbose:
        print(f"\n✓ Enumerated {num_checked} subgroups")
        print(f"✓ Maximum utility difference found: {max_abs_diff:.2f}")
        print(f"✓ Smallest epsilon needed: {smallest_epsilon:.2f}")
        
        if max_violation_info:
            print(f"\nSubgroup with maximum deviation:")
            print(f"  Subgroup: {max_violation_info['subgroup']}")
            print(f"  Size: {max_violation_info['size']}")
            print(f"  Utility: {max_violation_info['utility']:.2f}")
            print(f"  Population Utility: {utility_all:.2f}")
            print(f"  |Difference|: {max_violation_info['abs_diff']:.2f}")
        
        print(f"\nRuntime: {elapsed_time:.2f} seconds")
        print("="*70)
    
    return smallest_epsilon, num_checked, max_violation_info, utility_all, elapsed_time


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find smallest epsilon using brute force')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset CSV')
    parser.add_argument('--outcome', type=str, default='ConvertedSalary', help='Outcome column')
    parser.add_argument('--delta', type=int, required=True, help='Fixed delta threshold')
    
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(args.dataset)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Run algorithm
    smallest_epsilon, num_subgroups, violation_info, utility_all, runtime = find_smallest_epsilon_bruteforce(
        df=df,
        treatment_col=TREATMENT_COL,
        outcome_col=args.outcome,
        delta=args.delta,
        verbose=True
    )
    
    if smallest_epsilon is not None:
        print(f"\n✅ Final Answer: ε* = {smallest_epsilon:,.2f}")
        print(f"   Examined {num_subgroups} subgroups in {runtime:.2f}s")

