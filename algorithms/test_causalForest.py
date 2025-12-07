"""
Simple test script for the CausalForest algorithm.
This demonstrates how to use the CausalForest-based subgroup analysis.
"""
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from causalForest_algorithm import calc_utility_for_subgroups
from ATE_update import calculate_ate_safe

# Load config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

TREATMENT_COL = config['TREATMENT_COL']


def generate_synthetic_data(n_samples=1000, n_features=5, seed=42):
    """
    Generate synthetic data for testing the CausalForest algorithm.
    
    Creates data with heterogeneous treatment effects based on features.
    """
    np.random.seed(seed)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate treatment (binary)
    treatment = np.random.binomial(1, 0.5, n_samples)
    
    # Generate outcome with heterogeneous treatment effect
    # Treatment effect depends on first feature
    base_outcome = 10 + 2 * X[:, 0] + 1.5 * X[:, 1]
    treatment_effect = 3 + 2 * X[:, 0]  # Heterogeneous effect
    noise = np.random.randn(n_samples)
    
    outcome = base_outcome + treatment * treatment_effect + noise
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(n_features)])
    df[TREATMENT_COL] = treatment
    df['Outcome'] = outcome
    
    # Discretize features for subgroup analysis
    for col in [f'Feature_{i}' for i in range(n_features)]:
        df[col] = pd.cut(df[col], bins=3, labels=['Low', 'Medium', 'High'])
    
    return df


def test_homogeneity_check():
    """
    Test CausalForest in homogeneity check mode (mode=0).
    """
    print("=" * 70)
    print("TEST 1: Homogeneity Check (mode=0)")
    print("=" * 70)
    
    # Generate synthetic data with heterogeneous effects
    df = generate_synthetic_data(n_samples=1000, n_features=5)
    
    # Calculate overall ATE
    utility_all = calculate_ate_safe(df, TREATMENT_COL, 'Outcome')
    print(f"Overall ATE: {utility_all:.4f}")
    
    # Run CausalForest homogeneity check
    delta = 100  # Minimum subgroup size
    epsilon = 1.0  # Threshold for heterogeneity
    
    print(f"\nRunning CausalForest with delta={delta}, epsilon={epsilon}")
    
    result = calc_utility_for_subgroups(
        mode=0,  # Homogeneity check
        df=df,
        treatment_col=TREATMENT_COL,
        delta=delta,
        epsilon=epsilon,
        utility_all=utility_all,
        tgtO='Outcome',
        n_estimators=50,  # Use fewer trees for testing
        n_bins=3
    )
    
    print(f"\nResult: {'Homogeneous' if result else 'Heterogeneous (violations found)'}")
    print()


def test_all_subgroups():
    """
    Test CausalForest in all subgroups mode (mode=1).
    """
    print("=" * 70)
    print("TEST 2: All Subgroups Mode (mode=1)")
    print("=" * 70)
    
    # Generate synthetic data
    df = generate_synthetic_data(n_samples=500, n_features=4)
    
    # Calculate overall ATE
    utility_all = calculate_ate_safe(df, TREATMENT_COL, 'Outcome')
    print(f"Overall ATE: {utility_all:.4f}")
    
    # Run CausalForest to get all subgroups
    delta = 50
    epsilon = 0.5
    
    print(f"\nRunning CausalForest with delta={delta}, epsilon={epsilon}")
    
    subgroup_records, num_subgroups = calc_utility_for_subgroups(
        mode=1,  # All subgroups
        df=df,
        treatment_col=TREATMENT_COL,
        delta=delta,
        epsilon=epsilon,
        utility_all=utility_all,
        tgtO='Outcome',
        n_estimators=50,
        n_bins=5
    )
    
    print(f"\nFound {num_subgroups} subgroups:")
    print("-" * 70)
    
    # Display results
    if subgroup_records:
        results_df = pd.DataFrame(subgroup_records)
        print(results_df.to_string(index=False))
    else:
        print("No subgroups found.")
    
    print()


def test_small_dataset():
    """
    Test CausalForest with a small dataset (edge case).
    """
    print("=" * 70)
    print("TEST 3: Small Dataset (Edge Case)")
    print("=" * 70)
    
    # Generate small dataset
    df = generate_synthetic_data(n_samples=50, n_features=3)
    
    utility_all = calculate_ate_safe(df, TREATMENT_COL, 'Outcome')
    print(f"Overall ATE: {utility_all:.4f}")
    
    delta = 20
    epsilon = 1.0
    
    print(f"\nRunning CausalForest with delta={delta} on small dataset (n={len(df)})")
    
    try:
        result = calc_utility_for_subgroups(
            mode=0,
            df=df,
            treatment_col=TREATMENT_COL,
            delta=delta,
            epsilon=epsilon,
            utility_all=utility_all,
            tgtO='Outcome',
            n_estimators=20,
            n_bins=3
        )
        print(f"Result: {'Homogeneous' if result else 'Heterogeneous'}")
    except Exception as e:
        print(f"Error (expected with small dataset): {e}")
    
    print()


def main():
    """
    Run all tests for the CausalForest algorithm.
    """
    print("\n" + "=" * 70)
    print("CAUSALFOREST ALGORITHM TESTS")
    print("=" * 70 + "\n")
    
    try:
        # Test 1: Homogeneity check
        test_homogeneity_check()
        
        # Test 2: All subgroups mode
        test_all_subgroups()
        
        # Test 3: Small dataset edge case
        test_small_dataset()
        
        print("=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nMake sure to install econml:")
        print("  pip install econml==0.16.0")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

