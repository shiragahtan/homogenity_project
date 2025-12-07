# CausalForest Quick Start Guide

## Installation (One-time)

```bash
pip install econml==0.16.0
```

## Basic Usage

### 1. Import
```python
from causalForest_algorithm import calc_utility_for_subgroups
from ATE_update import calculate_ate_safe
```

### 2. Calculate Overall ATE
```python
utility_all = calculate_ate_safe(df, treatment_col='Treatment', outcome_col='Outcome')
```

### 3. Run CausalForest

**Homogeneity Check:**
```python
is_homogeneous = calc_utility_for_subgroups(
    mode=0,                    # Homogeneity check
    df=df,                     # Your dataframe
    treatment_col='Treatment', # Treatment column name
    delta=100,                 # Min subgroup size
    epsilon=1.0,              # Homogeneity threshold
    utility_all=utility_all,  # Overall ATE
    tgtO='Outcome',           # Outcome column name
    n_estimators=100,         # Number of trees (optional)
    n_bins=5                  # Number of CATE bins (optional)
)
```

**Find All Subgroups:**
```python
subgroups, count = calc_utility_for_subgroups(
    mode=1,                    # All subgroups mode
    df=df,
    treatment_col='Treatment',
    delta=50,
    epsilon=0.5,
    utility_all=utility_all,
    tgtO='Outcome',
    n_estimators=100,
    n_bins=5
)

# View results
import pandas as pd
print(pd.DataFrame(subgroups))
```

## Parameters Quick Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | int | - | 0=homogeneity check, 1=all subgroups |
| `df` | DataFrame | - | Input data |
| `treatment_col` | str | - | Treatment column name |
| `delta` | int | - | Minimum subgroup size |
| `epsilon` | float | - | Homogeneity threshold |
| `utility_all` | float | - | Overall ATE |
| `tgtO` | str | - | Outcome column name |
| `n_estimators` | int | 100 | Number of trees |
| `n_bins` | int | 5 | Number of CATE bins |

## Testing

```bash
cd algorithms
python test_causalForest.py
```

## Run in Benchmark

```bash
cd algorithms
python all_subgroups_loop.py
# Select mode (0 or 1)
# Choose "CausalForest" when prompted
```

## Tuning Tips

### Fast Testing (Small Dataset)
```python
n_estimators=20-50
n_bins=3
```

### Production (Large Dataset)
```python
n_estimators=100-200
n_bins=5-7
```

### Memory Constrained
```python
n_estimators=50
n_bins=3
```

## Common Issues

**ImportError: econml not found**
```bash
pip install econml==0.16.0
```

**Too slow?**
- Reduce `n_estimators` (try 20-50)
- Reduce `n_bins` (try 3)

**No subgroups found?**
- Increase `n_bins` (try 7)
- Decrease `delta` (min size)
- Check if data has sufficient heterogeneity

**All subgroups violate homogeneity?**
- Increase `epsilon` threshold
- Decrease `n_bins` (larger, less granular subgroups)

## Output Format

### Mode 0
- Returns: `bool` (True = homogeneous, False = heterogeneous)

### Mode 1
- Returns: `(list, int)` 
- List contains dictionaries with:
  - `AttributeValues`: Subgroup description
  - `Size`: Number of samples
  - `Utility`: Actual ATE
  - `UtilityDiff`: Difference from overall
  - `PredictedCATE`: Predicted CATE

## Example Output

```
Fitting CausalForest with 100 trees...
Found 5 subgroups from CausalForest predictions

AttributeValues                               Size  Utility  UtilityDiff  PredictedCATE
{'cate_bin': 0, 'cate_range': '[1.2, 2.5]'}  120   1.85     -0.65        1.92
{'cate_bin': 1, 'cate_range': '[2.5, 3.8]'}  118   3.12      0.62        3.15
{'cate_bin': 2, 'cate_range': '[3.8, 5.1]'}  115   4.67      2.17        4.58
...
```

## For More Details

- **Full Documentation**: See `CAUSALFOREST_README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Code**: See `causalForest_algorithm.py`
- **Tests**: See `test_causalForest.py`

