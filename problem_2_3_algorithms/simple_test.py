"""Simple test using pre-processed treatment data - runs BOTH problems."""
import sys
import json
import time
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from find_largest_delta import find_largest_delta_breaking_homogeneity
from find_smallest_epsilon import find_smallest_epsilon_achieving_homogeneity
from find_epsilon_bruteforce import find_smallest_epsilon_bruteforce
from ATE_update import calculate_ate_safe

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r") as fp:
    config = json.load(fp)

TREATMENT_COL = config['TREATMENT_COL']
OUTCOME_COL = 'ConvertedSalary'

print("="*70)
print("SIMPLE TEST: Problems 2 & 3 with Pre-Processed Data")
print("="*70)

# Use pre-processed treatment file
treatment_file = Path(__file__).resolve().parent.parent / "stackoverflow" / "so_countries_treatment_1_encoded.csv"
print(f"\n📂 Loading pre-processed treatment file...")
df = pd.read_csv(treatment_file)
print(f"   {len(df)} rows")
print(f"   Treatment column: {df[TREATMENT_COL].value_counts().to_dict()}")

# Calculate population ATE
ate_pop = calculate_ate_safe(df, TREATMENT_COL, OUTCOME_COL, 100)
print(f"   Population ATE: {ate_pop:.2f}")

# Test parameters
epsilon_fixed = 30000
delta_test = 1000
delta_min = 500
delta_max = 2000
epsilon_max = 2000000

print("\n" + "="*70)
print("PROBLEM 2: Find Largest Delta (Fixed Epsilon)")
print("="*70)
print(f"Fixed epsilon: {epsilon_fixed:,}")
print(f"Delta range: [{delta_min:,}, {delta_max:,}]")

start = time.time()
largest_delta, oracle_calls_delta, violation_delta, ate_delta = find_largest_delta_breaking_homogeneity(
    df=df,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    epsilon=epsilon_fixed,
    delta_min=delta_min,
    delta_max=delta_max,
    verbose=False
)
runtime_delta = time.time() - start

print(f"\n✓ Problem 2 Result:")
print(f"   Largest δ (breaking homogeneity): {largest_delta if largest_delta else 'N/A'}")
if largest_delta:
    print(f"   → Smallest δ (homogeneous): {largest_delta + 1}")
print(f"   Oracle calls: {oracle_calls_delta}")
print(f"   Runtime: {runtime_delta:.2f}s")
if violation_delta:
    print(f"   Violating subgroup: {violation_delta['subgroup']}")
    print(f"   |Diff|: {violation_delta['abs_diff']:.2f}")

print("\n" + "="*70)
print("PROBLEM 3: Find Smallest Epsilon (Fixed Delta)")
print("="*70)
print(f"Fixed delta: {delta_test:,}")

print("\nMethod 1: Binary Search")
start = time.time()
epsilon_binary, oracle_calls_binary, violation_binary, ate_binary = find_smallest_epsilon_achieving_homogeneity(
    df=df,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    delta=delta_test,
    epsilon_max=epsilon_max,
    verbose=False
)
runtime_binary = time.time() - start

print(f"   Smallest ε: {epsilon_binary:,}")
print(f"   Oracle calls: {oracle_calls_binary}")
print(f"   Runtime: {runtime_binary:.2f}s")

print("\nMethod 2: Brute Force")
epsilon_brute, num_subgroups, violation_brute, ate_brute, runtime_brute = find_smallest_epsilon_bruteforce(
    df=df,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    delta=delta_test,
    verbose=False
)

print(f"   Smallest ε: {epsilon_brute:.2f}")
print(f"   Subgroups examined: {num_subgroups}")
print(f"   Runtime: {runtime_brute:.2f}s")

# Comparison
speedup = runtime_binary / runtime_brute if runtime_brute > 0 else 0
match = abs(epsilon_binary - epsilon_brute) < 2 if epsilon_binary and epsilon_brute else False

# Generate HTML
html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Results: Problems 2 & 3</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
        h1 {{ color: #333; border-bottom: 4px solid #667eea; padding-bottom: 15px; margin-bottom: 30px; }}
        h2 {{ color: #667eea; margin-top: 40px; background: #f0f4ff; padding: 15px; border-left: 5px solid #667eea; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        th, td {{ padding: 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; }}
        tr:hover {{ background: #f5f5f5; }}
        .winner {{ background: #e8f5e9 !important; font-weight: bold; }}
        .badge {{ display: inline-block; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: bold; margin-left: 10px; }}
        .badge-problem2 {{ background: #ffd54f; color: #333; }}
        .badge-problem3 {{ background: #4caf50; color: white; }}
        .metric {{ font-size: 1.3em; font-weight: bold; color: #667eea; }}
        .section {{ background: #f9f9f9; padding: 25px; border-radius: 10px; margin: 20px 0; border: 1px solid #e0e0e0; }}
        .highlight {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        ul {{ line-height: 1.8; }}
        .comparison-box {{ display: flex; gap: 20px; margin: 20px 0; }}
        .method-card {{ flex: 1; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Comprehensive Test Results</h1>
        
        <div class="highlight">
            <strong>Dataset:</strong> {len(df):,} rows | <strong>Population ATE:</strong> {ate_pop:.2f}<br>
            <strong>Treatment Distribution:</strong> Control: {(df[TREATMENT_COL]==0).sum():,}, Treatment: {(df[TREATMENT_COL]==1).sum():,}
        </div>
        
        <h2><span class="badge badge-problem2">PROBLEM 2</span> Find Largest Delta Breaking Homogeneity</h2>
        
        <div class="section">
            <h3>📋 Problem Definition</h3>
            <p>Given a <strong>fixed epsilon (ε = {epsilon_fixed:,})</strong>, find the largest delta (δ) where the rule is still heterogeneous.</p>
            <p><em>Interpretation: The largest minimum subgroup size where we can still find violations.</em></p>
            
            <h3>🔧 Algorithm</h3>
            <p><strong>Binary Search with FPGrowth Oracle</strong></p>
            <ul>
                <li>Search space: δ ∈ [{delta_min:,}, {delta_max:,}]</li>
                <li>Oracle: FPGrowth checks heterogeneity at each delta candidate</li>
                <li>Monotonicity: If heterogeneous at δ, remains heterogeneous for all δ' < δ</li>
                <li>Complexity: O(log(δ_max - δ_min)) oracle calls</li>
            </ul>
            
            <h3>📊 Results</h3>
            <table>
                <tr>
                    <th style="width: 50%">Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Largest δ (Heterogeneous)</td>
                    <td class="metric">{largest_delta if largest_delta else 'None found'}</td>
                </tr>
                <tr>
                    <td>Smallest δ (Homogeneous)</td>
                    <td class="metric">{largest_delta + 1 if largest_delta else delta_min}</td>
                </tr>
                <tr>
                    <td>Oracle Calls</td>
                    <td>{oracle_calls_delta}</td>
                </tr>
                <tr>
                    <td>Runtime</td>
                    <td>{runtime_delta:.2f}s</td>
                </tr>
                <tr>
                    <td>Population ATE</td>
                    <td>{ate_delta:.2f}</td>
                </tr>
            </table>
            
            {f'''<div class="highlight">
                <strong>Violating Subgroup Details:</strong>
                <ul>
                    <li><strong>Subgroup:</strong> {violation_delta['subgroup']}</li>
                    <li><strong>Size:</strong> {violation_delta['size']:,} individuals</li>
                    <li><strong>Subgroup Utility:</strong> {violation_delta['utility']:.2f}</li>
                    <li><strong>Population Utility:</strong> {ate_delta:.2f}</li>
                    <li><strong>|Difference|:</strong> {violation_delta['abs_diff']:.2f} (> ε={epsilon_fixed:,})</li>
                </ul>
            </div>''' if violation_delta else '<p><em>No heterogeneity found in the tested range.</em></p>'}
        </div>
        
        <h2><span class="badge badge-problem3">PROBLEM 3</span> Find Smallest Epsilon Achieving Homogeneity</h2>
        
        <div class="section">
            <h3>📋 Problem Definition</h3>
            <p>Given a <strong>fixed delta (δ = {delta_test:,})</strong>, find the smallest epsilon (ε) where the rule becomes homogeneous.</p>
            <p><em>Interpretation: The minimum threshold needed to consider all subgroups as "similar enough".</em></p>
            
            <h3>🔧 Methods Compared</h3>
            <div class="comparison-box">
                <div class="method-card">
                    <h4>Method 1: Binary Search</h4>
                    <ul>
                        <li>Binary search on [0, {epsilon_max:,}]</li>
                        <li>Uses FPGrowth as oracle</li>
                        <li>Complexity: O(log ε_max)</li>
                    </ul>
                </div>
                <div class="method-card">
                    <h4>Method 2: Brute Force</h4>
                    <ul>
                        <li>FPGrowth enumerates ALL subgroups</li>
                        <li>Finds max utility difference</li>
                        <li>Complexity: O(number of subgroups)</li>
                    </ul>
                </div>
            </div>
            
            <h3>📊 Results Comparison</h3>
            <table>
                <tr>
                    <th style="width: 40%">Metric</th>
                    <th style="width: 30%">Binary Search</th>
                    <th style="width: 30%">Brute Force</th>
                </tr>
                <tr>
                    <td><strong>Smallest ε</strong></td>
                    <td class="metric">{epsilon_binary:,}</td>
                    <td class="metric">{epsilon_brute:,.0f}</td>
                </tr>
                <tr>
                    <td>Results Match?</td>
                    <td colspan="2" style="text-align:center; font-weight:bold;">{'✓ YES' if match else '✗ NO (±1 difference acceptable)'}</td>
                </tr>
                <tr>
                    <td>Efficiency</td>
                    <td>{oracle_calls_binary} oracle calls</td>
                    <td>{num_subgroups:,} subgroups examined</td>
                </tr>
                <tr class="{'winner' if runtime_binary < runtime_brute else ''}">
                    <td>Runtime</td>
                    <td>{runtime_binary:.2f}s</td>
                    <td class="{'winner' if runtime_brute < runtime_binary else ''}">{runtime_brute:.2f}s</td>
                </tr>
                <tr>
                    <td><strong>Winner</strong></td>
                    <td colspan="2" style="text-align:center; font-weight:bold; color:#4caf50;">
                        {'Binary Search' if runtime_binary < runtime_brute else 'Brute Force'} 
                        (speedup: {speedup:.2f}x)
                    </td>
                </tr>
            </table>
        </div>
        
        <h2>📝 Summary & Key Takeaways</h2>
        <div class="section">
            <h3>Algorithm Comparison</h3>
            <table>
                <tr>
                    <th>Problem</th>
                    <th>Algorithms Used</th>
                    <th>Comparison Type</th>
                </tr>
                <tr>
                    <td><strong>Problem 2</strong><br>(Find Largest Delta)</td>
                    <td>Binary Search + FPGrowth oracle</td>
                    <td>❌ No comparison<br>(Single method)</td>
                </tr>
                <tr>
                    <td><strong>Problem 3</strong><br>(Find Smallest Epsilon)</td>
                    <td>1. Binary Search + FPGrowth oracle<br>2. Brute Force (FPGrowth enumeration)</td>
                    <td>✅ Yes<br>(Two methods compared)</td>
                </tr>
            </table>
            
            <h3>Key Insights</h3>
            <ul>
                <li><strong>Binary Search:</strong> More efficient for large search spaces (logarithmic oracle calls)</li>
                <li><strong>Brute Force:</strong> Can be competitive for smaller datasets (single enumeration)</li>
                <li><strong>FPGrowth Role:</strong> Acts as oracle in Binary Search, enumerator in Brute Force</li>
                <li><strong>Trade-off:</strong> Binary Search makes fewer calls but each call has overhead; Brute Force enumerates once but examines all subgroups</li>
            </ul>
        </div>
    </div>
</body>
</html>"""

# Save HTML
output_file = Path("simple_test_results.html")
with open(output_file, "w") as f:
    f.write(html)

print("\n" + "="*70)
print("✅ TEST COMPLETE")
print("="*70)
print(f"\nHTML report: {output_file}")
print(f"\nSummary:")
print(f"  Problem 2: Largest δ = {largest_delta}, Runtime = {runtime_delta:.2f}s")
print(f"  Problem 3: ε = {epsilon_binary:,} (Binary) vs {epsilon_brute:.0f} (Brute)")
print(f"  Winner: {'Binary Search' if runtime_binary < runtime_brute else 'Brute Force'} ({speedup:.2f}x)")
print(f"\nOpen: {output_file}")





