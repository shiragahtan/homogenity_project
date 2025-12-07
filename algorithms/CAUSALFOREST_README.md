# CausalForest Algorithm Implementation

This document describes the CausalForest-based algorithm for subgroup analysis and homogeneity checking.

## Overview

The CausalForest algorithm uses EconML's Generalized Random Forest (GRF) to estimate heterogeneous treatment effects and identify subgroups with significantly different treatment effects. Unlike rule-mining approaches (like Apriori), CausalForest directly estimates conditional average treatment effects (CATEs) for each individual.

## Installation

Install the required package:

```bash
pip install econml==0.16.0
```

Or install all requirements:

```bash
pip install -r ../requirements.txt
```

## How It Works

### 1. **Causal Forest Training**
   - Trains a CausalForest model on the full dataset
   - Uses all features (excluding treatment and outcome) to predict heterogeneous treatment effects
   - Employs honest splitting for better inference properties

### 2. **CATE Prediction**
   - Predicts individual-level conditional average treatment effects (CATEs)
   - Each sample gets a predicted treatment effect based on its features

### 3. **Subgroup Identification**
   - Discretizes continuous CATE predictions into bins (default: 5 bins)
   - Creates subgroups based on similar predicted treatment effects
   - Filters out subgroups smaller than the minimum size threshold (delta)

### 4. **Validation**
   - For each subgroup, calculates the actual ATE using the standard linear regression method
   - Compares subgroup ATEs with the overall ATE
   - Checks for violations of the homogeneity threshold (epsilon)

## Algorithm Parameters

### Core Parameters (from benchmark framework)
- **mode** (int): 
  - `0` = Homogeneity check (returns True/False)
  - `1` = All subgroups mode (returns list of subgroups)
- **df** (pd.DataFrame): Input dataset
- **treatment_col** (str): Name of treatment column
- **delta** (int): Minimum subgroup size threshold
- **epsilon** (float): Threshold for homogeneity violation
- **utility_all** (float): Overall ATE for the full dataset
- **tgtO** (str): Target outcome column name

### CausalForest-Specific Parameters
- **n_estimators** (int, default=100): Number of trees in the forest
  - More trees → More stable predictions, but slower
  - Recommended: 100-500 for production, 20-50 for testing
  
- **n_bins** (int, default=5): Number of bins for discretizing CATE predictions
  - More bins → More granular subgroups, but smaller subgroup sizes
  - Fewer bins → Larger subgroups, but less precision
  - Recommended: 3-7 depending on dataset size

## Usage Examples

### Example 1: Homogeneity Check

```python
from causalForest_algorithm import calc_utility_for_subgroups
from ATE_update import calculate_ate_safe

# Calculate overall ATE
utility_all = calculate_ate_safe(df, treatment_col='Treatment', outcome_col='Outcome')

# Check homogeneity
is_homogeneous = calc_utility_for_subgroups(
    mode=0,
    df=df,
    treatment_col='Treatment',
    delta=100,
    epsilon=1.0,
    utility_all=utility_all,
    tgtO='Outcome',
    n_estimators=100,
    n_bins=5
)

if is_homogeneous:
    print("Dataset is homogeneous")
else:
    print("Heterogeneous effects detected")
```

### Example 2: Identify All Subgroups

```python
# Get all subgroups with their treatment effects
subgroup_records, num_subgroups = calc_utility_for_subgroups(
    mode=1,
    df=df,
    treatment_col='Treatment',
    delta=50,
    epsilon=0.5,
    utility_all=utility_all,
    tgtO='Outcome',
    n_estimators=100,
    n_bins=5
)

# Display results
import pandas as pd
results_df = pd.DataFrame(subgroup_records)
print(results_df)
```

## Output Format

### Mode 0 (Homogeneity Check)
Returns a boolean:
- `True`: Dataset is homogeneous (no violations found)
- `False`: Heterogeneous effects detected (at least one subgroup differs by > epsilon)

### Mode 1 (All Subgroups)
Returns a tuple: `(subgroup_records, num_subgroups)`

Each subgroup record contains:
- **AttributeValues**: Description of the subgroup (CATE bin and range)
- **Size**: Number of samples in the subgroup
- **Utility**: Actual ATE calculated for the subgroup
- **UtilityDiff**: Difference from overall ATE
- **PredictedCATE**: Average predicted CATE from the forest

## Advantages

1. **Direct Effect Estimation**: Estimates treatment effects directly without exhaustive rule mining
2. **Continuous Features**: Naturally handles continuous features without discretization
3. **Statistical Properties**: Provides confidence intervals and inference capabilities
4. **Scalability**: More efficient than exhaustive search methods for high-dimensional data

## Limitations

1. **Interpretability**: Subgroups are defined by CATE ranges, not explicit rules
2. **Sample Size**: Requires sufficient data for stable forest predictions
3. **Computational Cost**: Training the forest can be slower than rule mining for small datasets
4. **Minimum Size Requirement**: May miss small but important subgroups

## Comparison with Other Algorithms

| Algorithm | Approach | Interpretability | Scalability | Direct CATE |
|-----------|----------|------------------|-------------|-------------|
| **CausalForest** | ML-based effect estimation | Medium | High | ✓ |
| **Apriori** | Rule mining + ATE | High | Medium | ✗ |
| **Greedy** | Heuristic search | High | High | ✗ |
| **BruteForce** | Exhaustive search | High | Low | ✗ |

## Testing

Run the test script to verify the implementation:

```bash
cd algorithms
python test_causalForest.py
```

This will run three tests:
1. Homogeneity check with synthetic heterogeneous data
2. All subgroups mode to identify effect heterogeneity
3. Edge case with small dataset

## Integration with Benchmark

The CausalForest algorithm is already integrated into the benchmark framework in `all_subgroups_loop.py`:

```python
# Add to ALGORITHM_NAMES
ALGORITHM_NAMES = ["CausalForest", "Apriori", "Greedy", ...]

# Already in dispatch map (index 8)
ALGORITHM_DISPATCH_MAP = {
    ...
    "CausalForest": 8,
}
```

To run it in the benchmark:
1. Ensure `econml` is installed
2. Run `all_subgroups_loop.py`
3. Select mode (0 for homogeneity, 1 for all subgroups)
4. Select "CausalForest" when prompted

## References

- **EconML Documentation**: https://www.pywhy.org/EconML/
- **CausalForest Paper**: Wager, S., & Athey, S. (2018). "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests"
- **GRF Implementation**: https://www.pywhy.org/EconML/_autosummary/econml.grf.CausalForest.html

## Troubleshooting

### Import Error: econml not found
```bash
pip install econml==0.16.0
```

### "No valid feature columns found"
- Ensure the dataset has features besides treatment and outcome
- Check that features have variation (not all constant)

### Small dataset warnings
- CausalForest requires reasonable sample sizes (recommended: n > 200)
- Reduce `n_estimators` and `n_bins` for small datasets
- Consider using simpler methods (Greedy, Apriori) for very small datasets

### Slow performance
- Reduce `n_estimators` (try 20-50 for quick tests)
- Reduce `n_bins` (try 3-4 instead of 5)
- Use `n_jobs=-1` to parallelize (already set in implementation)

## Future Enhancements

Possible improvements:
1. **Adaptive binning**: Automatically determine optimal number of bins
2. **Feature importance**: Report which features drive heterogeneity
3. **Rule extraction**: Convert CATE-based subgroups to interpretable rules
4. **Confidence intervals**: Use forest inference for uncertainty quantification
5. **Multi-output**: Handle multiple outcomes simultaneously

