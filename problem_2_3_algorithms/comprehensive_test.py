"""Comprehensive test - runs BOTH Problem 2 and Problem 3 with 1 rule."""
import sys
import json
import time
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from find_largest_delta import find_largest_delta_breaking_homogeneity
from find_smallest_epsilon import find_smallest_epsilon_achieving_homogeneity
from find_epsilon_bruteforce import find_smallest_epsilon_bruteforce

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r") as fp:
    config = json.load(fp)

TREATMENT_COL = config['TREATMENT_COL']
OUTCOME_COL = 'ConvertedSalary'

print("="*70)
print("COMPREHENSIVE TEST: Problem 2 & Problem 3")
print("="*70)

# Load first treatment
treatments_file = Path(__file__).parent / "Chosen10Treatments.json"
with open(treatments_file, "r") as f:
    treatment = json.loads(f.readline())

print(f"\n📋 Treatment: {treatment['condition']} -> {treatment['treatment']}")

# Load FULL dataset (for realistic results)
dataset_path = Path(__file__).resolve().parent.parent / "stackoverflow" / "so_countries_col_new_encoded.csv"
print(f"📂 Loading full dataset...")
df = pd.read_csv(dataset_path)
print(f"   {len(df)} total rows")

# Apply treatment/condition properly
condition = treatment['condition']
treatment_val = treatment['treatment']

# Create treatment column
df['TempTreatment'] = 0
for attr, val in treatment_val.items():
    df.loc[df[attr] == val, 'TempTreatment'] = 1

# Filter by condition
df_filtered = df.copy()
for attr, val in condition.items():
    df_filtered = df_filtered[df_filtered[attr] == val]

print(f"   {len(df_filtered)} rows after filtering")

if len(df_filtered) < 1000:
    print("   ⚠️  Using full dataset (condition filtering gave too few rows)")
    df_filtered = df[df[TREATMENT_COL].notna()].copy()
    print(f"   {len(df_filtered)} rows")

# Test parameters
epsilon_fixed = 30000
delta_test = 2000
delta_min = 1000
delta_max = 3000
epsilon_max = 2000000

print("\n" + "="*70)
print("PROBLEM 2: Find Largest Delta (Fixed Epsilon)")
print("="*70)
print(f"Fixed epsilon: {epsilon_fixed:,}")
print(f"Delta range: [{delta_min:,}, {delta_max:,}]")
print("-"*70)

start = time.time()
largest_delta, oracle_calls_delta, violation_delta, ate_delta = find_largest_delta_breaking_homogeneity(
    df=df_filtered,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    epsilon=epsilon_fixed,
    delta_min=delta_min,
    delta_max=delta_max,
    verbose=True
)
runtime_delta = time.time() - start

print(f"\n✓ Problem 2 Result:")
print(f"   Largest δ (breaking homogeneity): {largest_delta if largest_delta else 'N/A'}")
if largest_delta:
    print(f"   → Smallest δ (homogeneous): {largest_delta + 1}")
print(f"   Oracle calls: {oracle_calls_delta}")
print(f"   Runtime: {runtime_delta:.2f}s")

# Problem 3
print("\n" + "="*70)
print("PROBLEM 3: Find Smallest Epsilon (Fixed Delta)")
print("="*70)
print(f"Fixed delta: {delta_test:,}")
print(f"Epsilon range: [0, {epsilon_max:,}]")
print("-"*70)

print("\nMethod 1: Binary Search")
print("-"*40)
start = time.time()
epsilon_binary, oracle_calls_binary, violation_binary, ate_binary = find_smallest_epsilon_achieving_homogeneity(
    df=df_filtered,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    delta=delta_test,
    epsilon_max=epsilon_max,
    verbose=True
)
runtime_binary = time.time() - start

print(f"\n✓ Binary Search Result:")
print(f"   Smallest ε: {epsilon_binary:,}")
print(f"   Oracle calls: {oracle_calls_binary}")
print(f"   Runtime: {runtime_binary:.2f}s")

print("\nMethod 2: Brute Force")
print("-"*40)
epsilon_brute, num_subgroups, violation_brute, ate_brute, runtime_brute = find_smallest_epsilon_bruteforce(
    df=df_filtered,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    delta=delta_test,
    verbose=True
)

print(f"\n✓ Brute Force Result:")
print(f"   Smallest ε: {epsilon_brute:.2f}")
print(f"   Subgroups examined: {num_subgroups}")
print(f"   Runtime: {runtime_brute:.2f}s")

# Generate comprehensive HTML
speedup = runtime_binary / runtime_brute
match = abs(epsilon_binary - epsilon_brute) < 2

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Test Results</title>
    <style>
        body {{ font-family: Arial; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); }}
        h1 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        h2 {{ color: #667eea; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #667eea; color: white; }}
        .winner {{ background: #e8f5e9; font-weight: bold; }}
        .section {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .badge {{ display: inline-block; padding: 5px 10px; border-radius: 5px; font-size: 0.9em; font-weight: bold; }}
        .badge-problem2 {{ background: #ffeb3b; color: #333; }}
        .badge-problem3 {{ background: #4caf50; color: white; }}
        .metric {{ font-size: 1.2em; font-weight: bold; color: #667eea; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Comprehensive Test: Problems 2 & 3</h1>
        
        <div class="section">
            <h3>Test Configuration</h3>
            <p><strong>Rule:</strong> {treatment['condition']} → {treatment['treatment']}</p>
            <p><strong>Dataset rows:</strong> {len(df_filtered):,}</p>
            <p><strong>Population ATE:</strong> {ate_delta:.2f}</p>
        </div>
        
        <h2><span class="badge badge-problem2">PROBLEM 2</span> Find Largest Delta (Fixed Epsilon)</h2>
        <div class="section">
            <h3>Problem Statement</h3>
            <p>Given fixed epsilon (ε = {epsilon_fixed:,}), find the <strong>largest delta</strong> where the rule is still heterogeneous.</p>
            
            <h3>Algorithm Used</h3>
            <p><strong>Binary Search with FPGrowth Oracle</strong></p>
            <ul>
                <li>Search range: δ ∈ [{delta_min:,}, {delta_max:,}]</li>
                <li>Oracle: FPGrowth checks if rule is heterogeneous at each delta</li>
                <li>Monotonicity: If heterogeneous at δ, remains heterogeneous for all δ' < δ</li>
            </ul>
            
            <h3>Results</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Largest δ (Heterogeneous)</td>
                    <td class="metric">{largest_delta if largest_delta else 'N/A'}</td>
                </tr>
                <tr>
                    <td>Smallest δ (Homogeneous)</td>
                    <td class="metric">{largest_delta + 1 if largest_delta else 'N/A'}</td>
                </tr>
                <tr>
                    <td>Oracle Calls (Efficiency)</td>
                    <td>{oracle_calls_delta}</td>
                </tr>
                <tr>
                    <td>Runtime</td>
                    <td>{runtime_delta:.2f}s</td>
                </tr>
            </table>
            
            {f'''<p><strong>Violating Subgroup:</strong></p>
            <ul>
                <li>Subgroup: {violation_delta['subgroup']}</li>
                <li>Size: {violation_delta['size']:,}</li>
                <li>Utility: {violation_delta['utility']:.2f}</li>
                <li>|Difference|: {violation_delta['abs_diff']:.2f}</li>
            </ul>''' if violation_delta else ''}
        </div>
        
        <h2><span class="badge badge-problem3">PROBLEM 3</span> Find Smallest Epsilon (Fixed Delta)</h2>
        <div class="section">
            <h3>Problem Statement</h3>
            <p>Given fixed delta (δ = {delta_test:,}), find the <strong>smallest epsilon</strong> where the rule becomes homogeneous.</p>
            
            <h3>Methods Compared</h3>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Algorithm</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td><strong>Method 1</strong></td>
                    <td>Binary Search</td>
                    <td>Binary search on [0, {epsilon_max:,}] using FPGrowth as oracle</td>
                </tr>
                <tr>
                    <td><strong>Method 2</strong></td>
                    <td>Brute Force</td>
                    <td>FPGrowth enumerates ALL subgroups, finds max utility difference</td>
                </tr>
            </table>
            
            <h3>Results Comparison</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Binary Search</th>
                    <th>Brute Force</th>
                </tr>
                <tr>
                    <td>Smallest ε</td>
                    <td class="metric">{epsilon_binary:,}</td>
                    <td class="metric">{epsilon_brute:,.0f}</td>
                </tr>
                <tr>
                    <td>Results Match</td>
                    <td colspan="2">{'✓ YES' if match else '✗ NO'}</td>
                </tr>
                <tr>
                    <td>Efficiency Metric</td>
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
                    <td colspan="2"><strong>{'Binary Search' if runtime_binary < runtime_brute else 'Brute Force'}</strong> 
                        ({speedup:.2f}x speedup)</td>
                </tr>
            </table>
        </div>
        
        <h2>📝 Summary</h2>
        <div class="section">
            <h3>Key Takeaways</h3>
            <ul>
                <li><strong>Problem 2:</strong> Uses Binary Search with FPGrowth oracle to find largest delta</li>
                <li><strong>Problem 3:</strong> Compares Binary Search vs Brute Force for finding smallest epsilon</li>
                <li><strong>Binary Search:</strong> More efficient when search space is large (O(log n) oracle calls)</li>
                <li><strong>Brute Force:</strong> Can be faster for small datasets (enumerates once)</li>
                <li><strong>FPGrowth:</strong> Used as oracle in Binary Search, and for enumeration in Brute Force</li>
            </ul>
        </div>
    </div>
</body>
</html>"""

# Save HTML
output_file = Path("comprehensive_test_results.html")
with open(output_file, "w") as f:
    f.write(html)

print("\n" + "="*70)
print("✅ COMPREHENSIVE TEST COMPLETE")
print("="*70)
print(f"\nHTML report saved: {output_file}")
print(f"\nSummary:")
print(f"  Problem 2 (Find Largest Delta):")
print(f"    - Largest δ breaking homogeneity: {largest_delta}")
print(f"    - Runtime: {runtime_delta:.2f}s")
print(f"  Problem 3 (Find Smallest Epsilon):")
print(f"    - Binary Search: ε={epsilon_binary:,}, {runtime_binary:.2f}s")
print(f"    - Brute Force: ε={epsilon_brute:.0f}, {runtime_brute:.2f}s")
print(f"    - Winner: {'Binary Search' if runtime_binary < runtime_brute else 'Brute Force'}")












