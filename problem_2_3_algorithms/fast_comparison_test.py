"""Fast comparison test - 2 rules with quick search ranges."""
import sys
import json
import time
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from find_largest_delta import find_largest_delta_breaking_homogeneity
from find_smallest_epsilon import find_smallest_epsilon_achieving_homogeneity
from find_epsilon_bruteforce import find_smallest_epsilon_bruteforce

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r") as fp:
    config = json.load(fp)

TREATMENT_COL = config['TREATMENT_COL']
OUTCOME_COL = 'ConvertedSalary'

print("="*80)
print("🚀 FAST COMPARISON TEST: 2 Rules - Row-by-Row Results")
print("="*80)

# Load treatments
treatments_file = Path(__file__).parent / "Chosen10Treatments.json"
with open(treatments_file, "r") as f:
    treatments = [json.loads(line) for line in f]

# Test with first 2 rules only
num_rules = 2
treatments = treatments[:num_rules]

# Parameters
epsilon_fixed = 30000
delta_test = 1000
delta_min = 800
delta_max = 1500  # Smaller range for speed
# Important: Binary search is only correct if epsilon_max is >= the true smallest epsilon.
# In practice (for some rules) the true epsilon can be > 1,000,000, so we keep this high
# to make the comparison meaningful.
epsilon_max = 2_000_000

# Results storage
problem2_results = []
problem3_results = []

print(f"\nTesting {num_rules} rules with FAST parameters...")
print(f"  • Problem 2: δ ∈ [{delta_min}, {delta_max}], fixed ε = {epsilon_fixed:,}")
print(f"  • Problem 3: ε ∈ [0, {epsilon_max:,}], fixed δ = {delta_test:,}")

for i, treatment in enumerate(treatments):
    print(f"\n{'='*80}")
    print(f"Rule {i+1}: {treatment['condition']} -> {treatment['treatment']}")
    print('='*80)
    
    # Load pre-processed treatment file
    treatment_file = Path(__file__).resolve().parent.parent / "stackoverflow" / f"so_countries_treatment_{i+1}_encoded.csv"
    
    if not treatment_file.exists():
        print(f"  ⚠️  Treatment file not found: {treatment_file}")
        continue
        
    df = pd.read_csv(treatment_file)
    print(f"  📊 Dataset: {len(df):,} rows")
    
    # ========================================================================
    # PROBLEM 2: Find Largest Delta
    # ========================================================================
    print(f"\n  🔍 PROBLEM 2: Finding largest delta (fixed ε={epsilon_fixed:,})...")
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
    
    problem2_results.append({
        'Rule_Number': i+1,
        'Condition': str(treatment['condition']),
        'Treatment': str(treatment['treatment']),
        'Fixed_Epsilon': epsilon_fixed,
        'Largest_Delta_Heterogeneous': largest_delta if largest_delta else 'None',
        'Oracle_Calls': oracle_calls_delta,
        'Runtime_Seconds': round(runtime_delta, 2),
        'Population_ATE': round(ate_delta, 2) if ate_delta else 'N/A',
        'Violating_Subgroup_Size': violation_delta['size'] if violation_delta else 'N/A',
        'Abs_Difference': round(violation_delta['abs_diff'], 2) if violation_delta else 'N/A'
    })
    
    print(f"    ✓ Largest δ (heterogeneous) = {largest_delta}")
    print(f"    ✓ Oracle calls = {oracle_calls_delta}")
    print(f"    ✓ Runtime = {runtime_delta:.2f}s")
    
    # ========================================================================
    # PROBLEM 3: Find Smallest Epsilon (Compare 2 methods)
    # ========================================================================
    print(f"\n  🎯 PROBLEM 3: Finding smallest epsilon (fixed δ={delta_test:,})...")
    
    # Method 1: Binary Search
    print(f"    • Running Binary Search method...")
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
    
    # Method 2: Brute Force
    print(f"    • Running Brute Force method...")
    start_brute = time.time()
    epsilon_brute, num_subgroups, violation_brute, ate_brute, runtime_brute = find_smallest_epsilon_bruteforce(
        df=df,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        delta=delta_test,
        verbose=False
    )
    
    # Comparison metrics
    binary_found = epsilon_binary is not None
    if binary_found:
        eps_binary_val = float(epsilon_binary)
        match = abs(eps_binary_val - float(epsilon_brute)) < 2 if epsilon_brute is not None else False
        winner = 'Binary Search' if runtime_binary < runtime_brute else 'Brute Force'
        speedup = (runtime_brute / runtime_binary) if runtime_binary > 0 else 0
        binary_status = "Found"
    else:
        eps_binary_val = None
        match = None
        winner = 'N/A (increase ε_max)'
        speedup = None
        binary_status = f"Not found ≤ {epsilon_max:,}"
    
    problem3_results.append({
        'Rule_Number': i+1,
        'Condition': str(treatment['condition']),
        'Treatment': str(treatment['treatment']),
        'Fixed_Delta': delta_test,
        'Binary_Status': binary_status,
        'Epsilon_Binary_Search': (round(eps_binary_val, 2) if eps_binary_val is not None else 'Not found'),
        'Epsilon_Brute_Force': round(epsilon_brute, 2),
        'Match': ('Yes ✓' if match is True else ('No ✗' if match is False else 'N/A')),
        'Oracle_Calls_Binary': oracle_calls_binary,
        'Oracle_Calls_Brute': 1,
        'Subgroups_Enumerated_Brute': num_subgroups,
        'Runtime_Binary_Seconds': round(runtime_binary, 2),
        'Runtime_Brute_Seconds': round(runtime_brute, 2),
        'Winner': winner,
        'Speedup_Factor': (round(speedup, 2) if speedup is not None else 'N/A'),
        'Population_ATE': round(ate_binary, 2) if ate_binary else 'N/A'
    })
    
    if binary_found:
        print(f"    ✓ Binary Search: ε = {eps_binary_val:,.0f}, Runtime = {runtime_binary:.2f}s")
    else:
        print(f"    ⚠ Binary Search: no homogeneous ε found up to ε_max={epsilon_max:,.0f} (Runtime = {runtime_binary:.2f}s)")
    print(f"    ✓ Brute Force: ε = {epsilon_brute:,.0f}, Runtime = {runtime_brute:.2f}s")
    if binary_found:
        print(f"    ✓ Winner: {winner} (Speedup: {speedup:.2f}x)")
    else:
        print(f"    ✓ Winner: N/A (Binary Search did not solve within ε_max)")

# ============================================================================
# GENERATE CSV FILES
# ============================================================================
df_problem2 = pd.DataFrame(problem2_results)
df_problem3 = pd.DataFrame(problem3_results)

csv_problem2 = "fast_test_problem2_results.csv"
csv_problem3 = "fast_test_problem3_results.csv"

df_problem2.to_csv(csv_problem2, index=False)
df_problem3.to_csv(csv_problem3, index=False)

# ============================================================================
# GENERATE HTML REPORT
# ============================================================================

# Problem 2 HTML table
problem2_table = df_problem2.to_html(index=False, classes='results-table', border=0)

# Problem 3 HTML table with winner highlighting
problem3_html_rows = []
for _, row in df_problem3.iterrows():
    row_class = 'winner-row' if row['Winner'] == 'Binary Search' else 'brute-winner-row'
    cells = ''.join([f'<td>{val}</td>' for val in row.values])
    problem3_html_rows.append(f'<tr class="{row_class}">{cells}</tr>')

problem3_table_body = '\n'.join(problem3_html_rows)
problem3_headers = ''.join([f'<th>{col.replace("_", " ")}</th>' for col in df_problem3.columns])
problem3_table = f'''<table class="results-table">
<thead><tr>{problem3_headers}</tr></thead>
<tbody>{problem3_table_body}</tbody>
</table>'''

html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fast Comparison Test - {num_rules} Rules</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1600px; 
            margin: 0 auto; 
            background: white; 
            padding: 50px; 
            border-radius: 20px; 
            box-shadow: 0 25px 80px rgba(0,0,0,0.35);
        }}
        h1 {{ 
            color: #333; 
            font-size: 2.5em; 
            border-bottom: 5px solid #667eea; 
            padding-bottom: 20px; 
            margin-bottom: 30px;
            text-align: center;
        }}
        h2 {{ 
            color: #667eea; 
            font-size: 1.8em;
            margin: 50px 0 25px 0; 
            background: linear-gradient(to right, #f0f4ff, transparent);
            padding: 20px; 
            border-left: 6px solid #667eea;
            border-radius: 5px;
        }}
        h3 {{ 
            color: #555; 
            margin: 25px 0 15px 0;
            font-size: 1.3em;
        }}
        .info-box {{ 
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            padding: 25px; 
            border-left: 6px solid #ffc107; 
            margin: 25px 0;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(255, 193, 7, 0.2);
        }}
        .info-box strong {{ color: #856404; font-size: 1.1em; }}
        .section {{ 
            background: #f9f9f9; 
            padding: 30px; 
            border-radius: 12px; 
            margin: 30px 0;
            border: 2px solid #e0e0e0;
        }}
        .results-table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin: 25px 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            font-size: 0.95em;
        }}
        .results-table th {{ 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 16px 12px; 
            text-align: left; 
            font-weight: 600;
            position: sticky;
            top: 0;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        .results-table td {{ 
            padding: 14px 12px; 
            border-bottom: 1px solid #e0e0e0;
            color: #333;
        }}
        .results-table tr:hover {{ 
            background: #f5f8ff;
            transition: background 0.2s ease;
        }}
        .winner-row {{ 
            background: linear-gradient(to right, #e8f5e9, #f1f8f4) !important;
            font-weight: 500;
        }}
        .brute-winner-row {{ 
            background: linear-gradient(to right, #fff3e0, #fef5e7) !important;
        }}
        .badge {{ 
            display: inline-block; 
            padding: 8px 16px; 
            border-radius: 25px; 
            font-size: 0.85em; 
            font-weight: bold; 
            margin-left: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .badge-problem2 {{ 
            background: linear-gradient(135deg, #ffd54f 0%, #ffb300 100%);
            color: #333;
            box-shadow: 0 4px 15px rgba(255, 179, 0, 0.3);
        }}
        .badge-problem3 {{ 
            background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }}
        .metric {{ 
            font-size: 1.4em; 
            font-weight: bold; 
            color: #667eea;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 25px;
            border-radius: 12px;
            margin: 30px 0;
            border-left: 6px solid #2196f3;
        }}
        ul {{ 
            line-height: 2; 
            margin-left: 25px;
        }}
        ul li {{ 
            margin: 8px 0;
            color: #555;
        }}
        .icon {{ 
            font-size: 1.3em; 
            margin-right: 8px;
        }}
        .files-list {{
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-family: 'Courier New', monospace;
            border: 1px solid #ddd;
        }}
        .files-list li {{
            color: #333;
            font-size: 0.95em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Fast Comparison Test Results</h1>
        
        <div class="info-box">
            <strong>📋 Test Configuration</strong><br><br>
            • <strong>Number of rules tested:</strong> {num_rules}<br>
            • <strong>Problem 2 fixed epsilon (ε):</strong> {epsilon_fixed:,}<br>
            • <strong>Problem 2 delta range:</strong> [{delta_min:,}, {delta_max:,}]<br>
            • <strong>Problem 3 fixed delta (δ):</strong> {delta_test:,}<br>
            • <strong>Problem 3 epsilon range:</strong> [0, {epsilon_max:,}]<br>
            • <strong>Purpose:</strong> Show row-by-row results for each rule with method comparison
        </div>
        
        <!-- ============================================================ -->
        <!-- PROBLEM 2: Find Largest Delta -->
        <!-- ============================================================ -->
        <h2><span class="icon">🔍</span><span class="badge badge-problem2">PROBLEM 2</span> Find Largest Delta - Results by Rule</h2>
        
        <div class="section">
            <h3>📖 Problem Definition</h3>
            <p>Given a <strong>fixed epsilon (ε = {epsilon_fixed:,})</strong>, find the <strong>largest delta (δ)</strong> where the rule is still heterogeneous (violations exist).</p>
            <p><em>Interpretation: What's the maximum minimum subgroup size where we can still detect treatment effect differences?</em></p>
            
            <h3>⚙️ Algorithm Used</h3>
            <p><strong>Binary Search with FPGrowth Oracle</strong></p>
            <ul>
                <li>Monotonicity property: If heterogeneous at δ, remains heterogeneous for all δ' &lt; δ</li>
                <li>Complexity: O(log(δ_max - δ_min)) oracle calls</li>
                <li>Each oracle call uses FPGrowth to check all subgroups ≥ δ</li>
            </ul>
            
            <h3>📊 Results Table (Each Row = One Rule)</h3>
            {problem2_table}
            
            <div class="summary-box">
                <h3>📌 Column Descriptions</h3>
                <ul>
                    <li><strong>Largest Delta (Heterogeneous):</strong> Maximum δ where violations were still found</li>
                    <li><strong>Oracle Calls:</strong> Number of FPGrowth invocations (efficiency metric)</li>
                    <li><strong>Runtime:</strong> Total execution time in seconds</li>
                    <li><strong>Abs Difference:</strong> |Subgroup ATE - Population ATE| of the violating subgroup found</li>
                </ul>
            </div>
        </div>
        
        <!-- ============================================================ -->
        <!-- PROBLEM 3: Find Smallest Epsilon (Method Comparison) -->
        <!-- ============================================================ -->
        <h2><span class="icon">🎯</span><span class="badge badge-problem3">PROBLEM 3</span> Find Smallest Epsilon - Method Comparison by Rule</h2>
        
        <div class="section">
            <h3>📖 Problem Definition</h3>
            <p>Given a <strong>fixed delta (δ = {delta_test:,})</strong>, find the <strong>smallest epsilon (ε)</strong> where the rule becomes homogeneous.</p>
            <p><em>Interpretation: What's the minimum threshold needed to consider all subgroups "similar enough"?</em></p>
            
            <h3>⚙️ Methods Compared</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                <div style="background: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #4caf50;">
                    <h4 style="color: #2e7d32; margin-bottom: 10px;">🔎 Binary Search</h4>
                    <p>Uses binary search with FPGrowth oracle to efficiently find the smallest ε.</p>
                    <p><strong>Advantage:</strong> Fewer oracle calls (logarithmic complexity)</p>
                </div>
                <div style="background: #fff3e0; padding: 20px; border-radius: 10px; border-left: 5px solid #ff9800;">
                    <h4 style="color: #e65100; margin-bottom: 10px;">📊 Brute Force</h4>
                    <p>Enumerates all subgroups once using FPGrowth and finds maximum difference.</p>
                    <p><strong>Advantage:</strong> Guaranteed exact answer (no approximation)</p>
                </div>
            </div>
            
            <h3>📊 Results Table (Each Row = One Rule)</h3>
            <p><em>Green rows = Binary Search won | Orange rows = Brute Force won</em></p>
            {problem3_table}
            
            <div class="summary-box">
                <h3>📌 Column Descriptions</h3>
                <ul>
                    <li><strong>Epsilon Binary Search:</strong> Smallest ε found by binary search method</li>
                    <li><strong>Epsilon Brute Force:</strong> Smallest ε found by brute force method (exact)</li>
                    <li><strong>Match:</strong> Do both methods find the same result? (within 2 units)</li>
                    <li><strong>Oracle Calls Binary:</strong> Number of FPGrowth invocations in binary search</li>
                    <li><strong>Subgroups Enumerated Brute:</strong> Total subgroups checked in brute force</li>
                    <li><strong>Winner:</strong> Which method had faster runtime for this rule</li>
                    <li><strong>Speedup Factor:</strong> How many times faster the winner was</li>
                </ul>
            </div>
        </div>
        
        <!-- ============================================================ -->
        <!-- SUMMARY -->
        <!-- ============================================================ -->
        <h2><span class="icon">📝</span> Summary</h2>
        
        <div class="section">
            <h3>🏆 Overall Method Comparison (Problem 3)</h3>
            <ul>
                <li><strong>Binary Search wins:</strong> {sum(1 for r in problem3_results if r['Winner'] == 'Binary Search')} / {len(problem3_results)} rules</li>
                <li><strong>Brute Force wins:</strong> {sum(1 for r in problem3_results if r['Winner'] == 'Brute Force')} / {len(problem3_results)} rules</li>
                <li><strong>Average speedup (when Binary Search wins):</strong> {round(sum(r['Speedup_Factor'] for r in problem3_results if r['Winner'] == 'Binary Search') / max(1, sum(1 for r in problem3_results if r['Winner'] == 'Binary Search')), 2)}x</li>
            </ul>
            
            <h3>📂 Generated Files</h3>
            <div class="files-list">
                <ul>
                    <li>📄 CSV (Problem 2): {csv_problem2}</li>
                    <li>📄 CSV (Problem 3): {csv_problem3}</li>
                    <li>🌐 HTML Report: fast_test_comparison_results.html</li>
                </ul>
            </div>
        </div>
    </div>
</body>
</html>"""

# Save HTML
output_file = Path("fast_test_comparison_results.html")
with open(output_file, "w", encoding='utf-8') as f:
    f.write(html)

print("\n" + "="*80)
print("✅ FAST COMPARISON TEST COMPLETE")
print("="*80)
print(f"\n📂 Files generated:")
print(f"  • HTML Report: {output_file}")
print(f"  • CSV (Problem 2): {csv_problem2}")
print(f"  • CSV (Problem 3): {csv_problem3}")
print(f"\n📊 Summary:")
print(f"  • Tested {num_rules} rules")
print(f"  • Problem 2: Binary Search (no comparison)")
print(f"  • Problem 3: Binary Search vs Brute Force")
print(f"  • Binary Search won: {sum(1 for r in problem3_results if r['Winner'] == 'Binary Search')}/{len(problem3_results)} times")
print(f"  • Brute Force won: {sum(1 for r in problem3_results if r['Winner'] == 'Brute Force')}/{len(problem3_results)} times")
print(f"\n🌐 Open the HTML file to see beautiful formatted results!")

