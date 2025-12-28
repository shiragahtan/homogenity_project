"""Minimal test - 1 rule, 1 delta, generates HTML to verify comparison."""
import sys
import json
import time
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from find_smallest_epsilon import find_smallest_epsilon_achieving_homogeneity
from find_epsilon_bruteforce import find_smallest_epsilon_bruteforce

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r") as fp:
    config = json.load(fp)

TREATMENT_COL = config['TREATMENT_COL']
OUTCOME_COL = 'ConvertedSalary'

print("="*70)
print("MINIMAL TEST: 1 Rule, 1 Delta - Epsilon Comparison")
print("="*70)

# Load first treatment
treatments_file = Path(__file__).parent / "Chosen10Treatments.json"
with open(treatments_file, "r") as f:
    treatment = json.loads(f.readline())

print(f"\n📋 Treatment: {treatment['condition']} -> {treatment['treatment']}")

# Load dataset (first 3000 rows for speed)
dataset_path = Path(__file__).resolve().parent.parent / "stackoverflow" / "so_countries_col_new_encoded.csv"
print(f"📂 Loading dataset (first 3000 rows for speed)...")
df = pd.read_csv(dataset_path, nrows=3000)
print(f"   {len(df)} total rows")

# Apply treatment/condition
condition = treatment['condition']
treatment_val = treatment['treatment']

# Create treatment column
df['TempTreatment'] = 0
for attr, val in treatment_val.items():
    df.loc[df[attr] == val, 'TempTreatment'] = 1

# Filter by condition - keep all rows (no filtering for this minimal test)
df_filtered = df[df[TREATMENT_COL].notna()].copy()

print(f"   {len(df_filtered)} rows after filtering (using all data for speed)")

delta = 500
epsilon_max = 500000

print(f"\n🎯 Parameters:")
print(f"   Delta (fixed): {delta}")
print(f"   Epsilon max: {epsilon_max:,}")

# METHOD 1: Binary Search
print("\n" + "="*70)
print("METHOD 1: Binary Search (Simplified)")
print("="*70)

start = time.time()
epsilon_binary, oracle_calls_binary, violation_binary, ate = find_smallest_epsilon_achieving_homogeneity(
    df=df_filtered,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    delta=delta,
    epsilon_max=epsilon_max,
    verbose=True
)
runtime_binary = time.time() - start

print(f"\n✓ Binary Search Result:")
print(f"   Smallest ε: {epsilon_binary:,}")
print(f"   Oracle calls: {oracle_calls_binary}")
print(f"   Runtime: {runtime_binary:.2f}s")

# METHOD 2: Brute Force
print("\n" + "="*70)
print("METHOD 2: Brute Force (FPGrowth - All Subgroups)")
print("="*70)

epsilon_brute, num_subgroups, violation_brute, ate_brute, runtime_brute = find_smallest_epsilon_bruteforce(
    df=df_filtered,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    delta=delta,
    verbose=True
)

print(f"\n✓ Brute Force Result:")
print(f"   Smallest ε: {epsilon_brute:.2f}")
print(f"   Subgroups examined: {num_subgroups}")
print(f"   Runtime: {runtime_brute:.2f}s")

# Comparison
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

speedup = runtime_binary / runtime_brute
match = abs(epsilon_binary - epsilon_brute) < 2

print(f"Results match: {'✓ YES' if match else '✗ NO'}")
print(f"Binary Search: {epsilon_binary:,} (Oracle calls: {oracle_calls_binary}, Time: {runtime_binary:.2f}s)")
print(f"Brute Force:   {epsilon_brute:.2f} (Subgroups: {num_subgroups}, Time: {runtime_brute:.2f}s)")
print(f"Speedup: {speedup:.2f}x ({'Binary Search' if speedup < 1 else 'Brute Force'} faster)")

# Generate simple HTML
html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Minimal Test Results</title>
    <style>
        body {{ font-family: Arial; padding: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        .winner {{ background: #e8f5e9; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Minimal Test: Epsilon Finding Comparison</h1>
        
        <h2>Test Parameters</h2>
        <p><strong>Rule:</strong> {treatment['condition']} → {treatment['treatment']}</p>
        <p><strong>Delta (fixed):</strong> {delta}</p>
        <p><strong>Dataset rows:</strong> {len(df_filtered):,}</p>
        
        <h2>Methods Compared</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>Description</th>
                <th>Algorithm</th>
            </tr>
            <tr>
                <td><strong>Method 1</strong></td>
                <td>Binary Search</td>
                <td>Standard binary search on [0, {epsilon_max:,}]<br>Uses FPGrowth as oracle</td>
            </tr>
            <tr>
                <td><strong>Method 2</strong></td>
                <td>Brute Force</td>
                <td>FPGrowth enumeration of ALL subgroups<br>Finds maximum utility difference</td>
            </tr>
        </table>
        
        <h2>Results</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Binary Search</th>
                <th>Brute Force</th>
            </tr>
            <tr>
                <td>Smallest ε</td>
                <td>{epsilon_binary:,}</td>
                <td>{epsilon_brute:,.0f}</td>
            </tr>
            <tr>
                <td>Efficiency</td>
                <td>{oracle_calls_binary} oracle calls</td>
                <td>{num_subgroups:,} subgroups examined</td>
            </tr>
            <tr class="{'winner' if runtime_binary < runtime_brute else ''}">
                <td>Runtime</td>
                <td>{runtime_binary:.2f}s</td>
                <td>{runtime_brute:.2f}s</td>
            </tr>
            <tr>
                <td>Winner</td>
                <td colspan="2">{'✓ Binary Search' if runtime_binary < runtime_brute else '✓ Brute Force'} 
                    ({speedup:.2f}x speedup)</td>
            </tr>
        </table>
        
        <h2>Key Points</h2>
        <ul>
            <li><strong>Binary Search:</strong> Uses FPGrowth as an oracle (checks homogeneity at specific epsilon values)</li>
            <li><strong>Brute Force:</strong> Enumerates ALL subgroups once, finds the maximum utility difference</li>
            <li><strong>For Problem 3 (Find Smallest Epsilon):</strong> Both methods are compared</li>
            <li><strong>For Problem 2 (Find Largest Delta):</strong> Only Binary Search with FPGrowth oracle is used</li>
        </ul>
    </div>
</body>
</html>"""

# Save HTML
output_file = Path("minimal_test_results.html")
with open(output_file, "w") as f:
    f.write(html)

print(f"\n✅ HTML report saved: {output_file}")
print(f"\nTo view: open {output_file}")

