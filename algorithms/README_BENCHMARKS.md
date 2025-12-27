# Homogeneity Algorithm Benchmarks

## Overview

This directory contains implementations and benchmarks for two optimization problems related to causal homogeneity analysis:

- **Problem 2**: Find the largest δ (minimum subgroup size) that breaks homogeneity for a fixed ε
- **Problem 3**: Find the smallest ε (threshold) that achieves homogeneity for a fixed δ

## Quick Start

### Run All Benchmarks

```bash
cd algorithms
source ../../.venv-py311/bin/activate
python run_all_benchmarks.py --rules 5 --epsilons "10000,20000,30000,40000,50000,60000" --deltas "500,1000,1500,2000,2500,3000"
```

### Output Structure

```
benchmark_results/
├── problem2_largest_delta/
│   ├── benchmark_report.html         # Detailed HTML report
│   ├── find_delta_benchmark_results.xlsx  # Excel results
│   ├── find_delta_benchmark_results.csv   # CSV results
│   ├── benchmark_visualization.png        # 4-panel analysis
│   └── oracle_calls_heatmap.png          # Heatmap visualization
│
├── problem3_smallest_epsilon/
│   ├── benchmark_report.html
│   ├── find_epsilon_benchmark_results.xlsx
│   ├── find_epsilon_benchmark_results.csv
│   ├── benchmark_visualization.png
│   └── oracle_calls_heatmap.png
│
└── summary_report.html               # Combined overview
```

## Command Line Options

### `run_all_benchmarks.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--rules` | 5 | Number of treatment rules to test |
| `--epsilons` | "10000,20000,..." | Comma-separated epsilon values for Problem 2 |
| `--deltas` | "500,1000,..." | Comma-separated delta values for Problem 3 |
| `--delta_min` | 100 | Minimum delta for Problem 2 search range |
| `--delta_max` | 10000 | Maximum delta for Problem 2 search range |
| `--epsilon_start` | 1000.0 | Starting epsilon for Problem 3 exponential search |
| `--epsilon_max` | 500000.0 | Maximum epsilon for Problem 3 |
| `--output` | "../benchmark_results" | Output directory |

### Examples

**Quick test (2 rules, limited values):**
```bash
python run_all_benchmarks.py --rules 2 --epsilons "25000,50000" --deltas "1000,2000" --delta_max 3000
```

**Full benchmark (5 rules, comprehensive):**
```bash
python run_all_benchmarks.py --rules 5 --epsilons "10000,20000,30000,40000,50000,60000" --deltas "500,1000,1500,2000,2500,3000" --delta_max 10000
```

## Individual Algorithm Scripts

You can also run each algorithm independently:

### Problem 2: Find Largest Delta
```bash
python benchmark_find_delta.py --rules 5 --epsilons "10000,20000,30000" --delta_max 10000
```

### Problem 3: Find Smallest Epsilon
```bash
python benchmark_find_epsilon.py --rules 5 --deltas "500,1000,1500" --epsilon_max 500000
```

## Results Interpretation

### Problem 2 Results

Each row shows:
- **Epsilon (ε)**: Fixed homogeneity threshold
- **Largest δ (Heterogeneous)**: Maximum subgroup size where violations exist
- **Smallest δ (Homogeneous)**: Minimum size where rule is homogeneous = largest_δ + 1
- **Violating Subgroup**: The specific subgroup causing the boundary
  - Shows subgroup definition, size, utility differences
  - Explains why algorithm stopped at this δ*
- **Oracle Calls**: Number of FPGrowth invocations (efficiency metric)
- **Runtime**: Time in seconds
- **Efficiency**: Ratio of actual/theoretical oracle calls

### Problem 3 Results

Each row shows:
- **Delta (δ)**: Fixed minimum subgroup size
- **Smallest ε (Homogeneous)**: Minimum threshold for homogeneity
- **Largest ε (Heterogeneous)**: Maximum threshold with violations = smallest_ε - 1
- **Violating Subgroup**: The most extreme violation requiring ε*
  - Shows the subgroup with largest |ATE difference|
  - Explains what violation must be tolerated
- **Oracle Calls**: FPGrowth invocations across both phases
- **Runtime**: Time including exponential + binary search

## Algorithms

### Binary Search (Problem 2)
- **Monotonicity**: Non-homogeneous exhibits downward-closure in δ
- **Complexity**: O(log₂(δ_max - δ_min)) oracle calls
- **Oracle**: FPGrowth with early-stopping

### Two-Phase Search (Problem 3)
- **Phase 1**: Exponential search for upper bound
- **Phase 2**: Binary search refinement
- **Monotonicity**: Homogeneous exhibits upward-closure in ε
- **Complexity**: O(log₂(ε_max)) oracle calls
- **Oracle**: FPGrowth with early-stopping

## Files

### Core Implementations
- `find_largest_delta.py` - Problem 2 algorithm
- `find_smallest_epsilon.py` - Problem 3 algorithm
- `apriori_algorithm.py` - FPGrowth oracle with violation tracking

### Benchmarks
- `run_all_benchmarks.py` - Unified runner (use this!)
- `benchmark_find_delta.py` - Problem 2 benchmark
- `benchmark_find_epsilon.py` - Problem 3 benchmark

### Data
- Treatment rules: `Chosen10Treatments.json`
- Datasets: `../stackoverflow/processed_db/so_countries_treatment_*_encoded.csv`

## Notes

- Warnings from `linear_model_unlearning.py` and `ATE_update.py` are normal (numerical edge cases)
- Full benchmarks can take 10-20 minutes depending on parameters
- Results are cached in Excel/CSV for further analysis
- HTML reports include interactive tables with violation details

