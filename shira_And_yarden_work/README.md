# Brute-Force Rule Mining Benchmark

This directory contains Yarden's brute-force algorithm for finding the **largest homogeneous subgroup with positive ATE**, along with a comprehensive benchmark suite.

## 📁 Files

- **`brute_force_search.py`**: Core algorithm implementation
- **`benchmark_bruteforce_rule_mining.py`**: Benchmark runner for testing algorithm robustness
- **`so.csv`**: Stack Overflow dataset (integer-encoded)
- **`benchmark_bruteforce_outputs/`**: Directory containing all benchmark results

---

## 🎯 Algorithm Overview

The brute-force algorithm finds subgroups that satisfy three conditions:

1. **Homogeneity**: `|ATE(subgroup) - ATE(population)| ≤ epsilon`
2. **Positive utility**: `ATE(subgroup) > 0`
3. **Maximal size**: Returns the largest subgroup satisfying conditions 1 & 2

**How it works:**
- Enumerates all possible subgroups using FP-Growth
- Filters by minimum size (delta threshold)
- Checks homogeneity and positive ATE for each
- Returns the largest qualifying subgroup

---

## 🚀 Running the Benchmark

### Basic Usage

```bash
cd /Users/sgahtan/Desktop/shira/studies/brit_project/project_updated/homogenity_project/shira_And_yarden_work

python benchmark_bruteforce_rule_mining.py \
  --max_rules 10 \
  --epsilons "100,500,1000,5000,10000" \
  --deltas "0.03,0.05,0.08,0.10" \
  --sample_n 5000
```

### Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--max_rules` | Number of treatment rules to test from `Chosen10Treatments.json` | `10` (default) |
| `--epsilons` | Comma-separated epsilon values (homogeneity thresholds) | `"100,1000,5000"` |
| `--deltas` | Comma-separated delta percentages (0-1, min subgroup size as % of dataset) | `"0.03,0.05,0.10"` |
| `--sample_n` | Sample size per run (0 = full dataset) | `5000` |
| `--output_dir` | Output directory name | `benchmark_results` |

### Advanced Options

```bash
python benchmark_bruteforce_rule_mining.py \
  --max_rules 5 \
  --epsilons "500,1000,2500" \
  --deltas "0.05,0.08" \
  --sample_n 3000 \
  --max_candidates 5000 \
  --timeout_seconds 300 \
  --output_dir my_benchmark
```

| Advanced Parameter | Description | Default |
|-------------------|-------------|---------|
| `--max_candidates` | Max candidates to evaluate per case | `10000` |
| `--timeout_seconds` | Timeout per test case (seconds) | `600` |
| `--attrs` | Comma-separated attributes for subgroup mining | Auto-detected |
| `--auto_attrs_k` | If >0, pick K lowest-cardinality attributes | `0` (use all) |

---

## 📊 Understanding the Results

After running, you'll find results in `benchmark_bruteforce_outputs/<timestamp>/`:

### 1. **`results.csv`** - Detailed results per test case

Key columns:
- `rule_id`: Which treatment rule was tested
- `epsilon`, `delta_percent`, `delta_count`: Test parameters
- `found`: `True` if a qualifying subgroup was found
- `found_filters`: The subgroup definition (attribute-value pairs)
- `found_size`: Size of the found subgroup
- `found_ate`: ATE of the found subgroup
- `candidates_enumerated`: Total candidates generated
- `candidates_evaluated`: How many were actually checked
- `total_seconds`: Runtime for this test case
- `skipped_reason`: Why it was skipped (if applicable)

### 2. **`report.html`** - Visual dashboard

Open in a browser to see:
- Success rate by epsilon/delta
- Runtime distributions
- Found subgroup sizes
- Heatmaps of performance metrics

### 3. **`plots/`** - PNG visualizations
- Runtime vs. epsilon
- Evaluated candidates vs. epsilon
- Success rate heatmap

---

## 📈 Example: Comprehensive Benchmark

To generate a robust assessment with 420 test cases:

```bash
python benchmark_bruteforce_rule_mining.py \
  --max_rules 10 \
  --epsilons "100,500,1000,2500,5000,10000,15000" \
  --deltas "0.02,0.03,0.05,0.08,0.10,0.15" \
  --sample_n 5000 \
  --output_dir benchmark_comprehensive
```

**Test matrix:**
- 10 rules × 7 epsilon values × 6 delta values = **420 test cases**
- Expected runtime: ~5-15 minutes
- Results saved to `benchmark_bruteforce_outputs/benchmark_comprehensive/`

---

## 🔍 Quick Analysis

After running, analyze results with Python:

```python
import pandas as pd

df = pd.read_csv('benchmark_bruteforce_outputs/<timestamp>/results.csv')

# Success rate
print(f"Success rate: {df['found'].mean()*100:.1f}%")

# Average performance (successful cases only)
success = df[df['found'] == True]
print(f"Avg runtime: {success['total_seconds'].mean():.2f}s")
print(f"Avg candidates evaluated: {success['candidates_evaluated'].mean():.0f}")
print(f"Avg found size: {success['found_size'].mean():.0f}")

# By epsilon
print("\nSuccess rate by epsilon:")
print(df.groupby('epsilon')['found'].mean())
```

---

## 🐛 Common Issues

### Issue: "insufficient_treatment_split"

**Problem:** Not enough treated/control individuals in the condition.

**Solution:** 
- Use larger `--sample_n`
- Choose rules with better treatment balance
- Check `treated_count` and `control_count` in results

### Issue: Many "found=False" cases

**Problem:** No subgroup exists that satisfies all three conditions.

**Solution:**
- Increase `--epsilons` (relax homogeneity)
- Decrease `--deltas` (allow smaller subgroups)
- This is expected behavior - not all rules have positive homogeneous subgroups

---

## 📝 Citation

This benchmark evaluates the algorithm described in the research paper on treatment effect homogeneity.

---

## 🛠️ Development

To modify the benchmark:
1. Edit `benchmark_bruteforce_rule_mining.py`
2. Test with small parameters first: `--max_rules 2 --epsilons "100,500" --deltas "0.05"`
3. Verify results in `results.csv` and `report.html`

---

**Last Updated:** January 2026


