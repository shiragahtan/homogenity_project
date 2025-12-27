# Unified Benchmark Script

## Quick Start

```bash
cd algorithms
source ../../.venv-py311/bin/activate
python run_all_benchmarks.py --rules 5 --epsilons "10000,20000,30000" --deltas "500,1000,1500"
```

**View results**: `../benchmark_results/summary_report.html`

## What It Does

- **Problem 2**: Finds largest δ breaking homogeneity (fixed ε)
- **Problem 3**: Finds smallest ε achieving homogeneity (fixed δ)
- Generates HTML reports with violation details for both

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--rules N` | Number of rules to test (default: 5) |
| `--epsilons "a,b,c"` | Epsilon values for Problem 2 |
| `--deltas "x,y,z"` | Delta values for Problem 3 |
| `--delta_max N` | Max delta search range (default: 10000) |
| `--epsilon_max N` | Max epsilon search range (default: 500000) |

## Examples

**Quick test (~3 min):**
```bash
python run_all_benchmarks.py --rules 2 --epsilons "25000,50000" --deltas "1000,2000"
```

**Standard (~15 min):**
```bash
python run_all_benchmarks.py --rules 5 --epsilons "10000,20000,30000,40000,50000,60000" --deltas "500,1000,1500,2000,2500,3000"
```

## Output

```
benchmark_results/
├── summary_report.html           ← Open this first
├── problem2_largest_delta/
│   └── benchmark_report.html
└── problem3_smallest_epsilon/
    └── benchmark_report.html
```

## Troubleshooting

- **ModuleNotFoundError**: Activate venv first
- **Problem 3 "Not found"**: Increase `--epsilon_max 1000000`
- **Too slow**: Reduce `--rules` or parameter count
