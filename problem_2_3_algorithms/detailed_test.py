"""Detailed test - multiple rules showing row-by-row results in HTML tables."""
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

# Load config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r") as fp:
    config = json.load(fp)

TREATMENT_COL = config['TREATMENT_COL']
OUTCOME_COL = 'ConvertedSalary'

print("="*70)
print("DETAILED TEST: Multiple Rules - Row-by-Row Results")
print("="*70)

# Load treatments
treatments_file = Path(__file__).parent / "Chosen10Treatments.json"
with open(treatments_file, "r") as f:
    treatments = [json.loads(line) for line in f]

# Test with first 3 rules for speed
num_rules = 3
treatments = treatments[:num_rules]

# Parameters
epsilon_fixed = 30000
delta_test = 1000
delta_min = 500
delta_max = 2000
epsilon_max = 2000000

# Results storage
problem2_results = []
problem3_results = []

print(f"\nTesting {num_rules} rules...")

for i, treatment in enumerate(treatments):
    print(f"\n{'='*70}")
    print(f"Rule {i+1}: {treatment['condition']} -> {treatment['treatment']}")
    print('='*70)
    
    # Load pre-processed treatment file
    treatment_file = Path(__file__).resolve().parent.parent / "stackoverflow" / f"so_countries_treatment_{i+1}_encoded.csv"
    
    if not treatment_file.exists():
        print(f"  ⚠️  Treatment file not found: {treatment_file}")
        continue
        
    df = pd.read_csv(treatment_file)
    print(f"  Dataset: {len(df)} rows")
    
    # PROBLEM 2
    print(f"\n  Problem 2: Finding largest delta (ε={epsilon_fixed:,})...")
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
        'Rule': f"Rule {i+1}",
        'Condition': str(treatment['condition']),
        'Treatment': str(treatment['treatment']),
        'Fixed_Epsilon': epsilon_fixed,
        'Largest_Delta_Heterogeneous': largest_delta if largest_delta else 'N/A',
        'Smallest_Delta_Homogeneous': largest_delta + 1 if largest_delta else delta_min,
        'Oracle_Calls': oracle_calls_delta,
        'Runtime': round(runtime_delta, 2),
        'Population_ATE': round(ate_delta, 2),
        'Violating_Subgroup': str(violation_delta['subgroup']) if violation_delta else 'N/A',
        'Subgroup_Size': violation_delta['size'] if violation_delta else 'N/A',
        'Abs_Diff': round(violation_delta['abs_diff'], 2) if violation_delta else 'N/A'
    })
    
    print(f"    ✓ Largest δ = {largest_delta}, Runtime = {runtime_delta:.2f}s")
    
    # PROBLEM 3
    print(f"  Problem 3: Finding smallest epsilon (δ={delta_test:,})...")
    
    # Method 1: Binary Search
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
    epsilon_brute, num_subgroups, violation_brute, ate_brute, runtime_brute = find_smallest_epsilon_bruteforce(
        df=df,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        delta=delta_test,
        verbose=False
    )
    
    speedup = runtime_binary / runtime_brute if runtime_brute > 0 else 0
    match = abs(epsilon_binary - epsilon_brute) < 2 if epsilon_binary and epsilon_brute else False
    
    problem3_results.append({
        'Rule': f"Rule {i+1}",
        'Condition': str(treatment['condition']),
        'Treatment': str(treatment['treatment']),
        'Fixed_Delta': delta_test,
        'Epsilon_Binary': epsilon_binary,
        'Epsilon_Brute': round(epsilon_brute, 2),
        'Match': 'Yes' if match else 'No',
        'Oracle_Calls_Binary': oracle_calls_binary,
        'Subgroups_Brute': num_subgroups,
        'Runtime_Binary': round(runtime_binary, 2),
        'Runtime_Brute': round(runtime_brute, 2),
        'Winner': 'Binary Search' if runtime_binary < runtime_brute else 'Brute Force',
        'Speedup': round(speedup, 2),
        'Population_ATE': round(ate_binary, 2)
    })
    
    print(f"    ✓ Binary Search: ε={epsilon_binary:,}, {runtime_binary:.2f}s")
    print(f"    ✓ Brute Force: ε={epsilon_brute:.0f}, {runtime_brute:.2f}s")
    print(f"    ✓ Winner: {'Binary Search' if runtime_binary < runtime_brute else 'Brute Force'}")

# Generate HTML with tables
df_problem2 = pd.DataFrame(problem2_results)
df_problem3 = pd.DataFrame(problem3_results)

# Problem 2 HTML table
problem2_table = '<table><thead><tr>'
for col in ['Rule', 'Fixed_Epsilon', 'Largest_Delta_Heterogeneous', 'Smallest_Delta_Homogeneous', 
            'Oracle_Calls', 'Runtime', 'Population_ATE', 'Abs_Diff']:
    problem2_table += f'<th>{col.replace("_", " ")}</th>'
problem2_table += '</tr></thead><tbody>'

for _, row in df_problem2.iterrows():
    problem2_table += '<tr>'
    for col in ['Rule', 'Fixed_Epsilon', 'Largest_Delta_Heterogeneous', 'Smallest_Delta_Homogeneous',
                'Oracle_Calls', 'Runtime', 'Population_ATE', 'Abs_Diff']:
        problem2_table += f'<td>{row[col]}</td>'
    problem2_table += '</tr>'
problem2_table += '</tbody></table>'

# Problem 3 HTML table
problem3_table = '<table><thead><tr>'
for col in ['Rule', 'Fixed_Delta', 'Epsilon_Binary', 'Epsilon_Brute', 'Match',
            'Oracle_Calls_Binary', 'Subgroups_Brute', 'Runtime_Binary', 'Runtime_Brute', 'Winner', 'Speedup']:
    problem3_table += f'<th>{col.replace("_", " ")}</th>'
problem3_table += '</tr></thead><tbody>'

for _, row in df_problem3.iterrows():
    winner_class = ' class="winner"' if row['Winner'] == 'Brute Force' else ''
    problem3_table += f'<tr{winner_class}>'
    for col in ['Rule', 'Fixed_Delta', 'Epsilon_Binary', 'Epsilon_Brute', 'Match',
                'Oracle_Calls_Binary', 'Subgroups_Brute', 'Runtime_Binary', 'Runtime_Brute', 'Winner', 'Speedup']:
        problem3_table += f'<td>{row[col]}</td>'
    problem3_table += '</tr>'
problem3_table += '</tbody></table>'

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Detailed Test Results - Row by Row</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); }}
        h1 {{ color: #333; border-bottom: 4px solid #667eea; padding-bottom: 15px; margin-bottom: 30px; }}
        h2 {{ color: #667eea; margin-top: 40px; background: #f0f4ff; padding: 15px; border-left: 5px solid #667eea; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); font-size: 0.9em; }}
        th, td {{ padding: 12px 8px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        th {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: 600; position: sticky; top: 0; }}
        tr:hover {{ background: #f5f5f5; }}
        .winner {{ background: #e8f5e9 !important; }}
        .badge {{ display: inline-block; padding: 6px 12px; border-radius: 20px; font-size: 0.85em; font-weight: bold; margin-left: 10px; }}
        .badge-problem2 {{ background: #ffd54f; color: #333; }}
        .badge-problem3 {{ background: #4caf50; color: white; }}
        .section {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; border: 1px solid #e0e0e0; }}
        .highlight {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Detailed Test Results: {num_rules} Rules</h1>
        
        <div class="highlight">
            <strong>Test Configuration:</strong><br>
            • Number of rules tested: <strong>{num_rules}</strong><br>
            • Problem 2 fixed epsilon: <strong>{epsilon_fixed:,}</strong><br>
            • Problem 3 fixed delta: <strong>{delta_test:,}</strong><br>
            • Each row represents results for one rule
        </div>
        
        <h2><span class="badge badge-problem2">PROBLEM 2</span> Find Largest Delta - Results by Rule</h2>
        
        <div class="section">
            <h3>📋 What This Shows</h3>
            <p>For each rule, given <strong>fixed ε = {epsilon_fixed:,}</strong>, find the largest δ where heterogeneity is still detected.</p>
            
            <h3>📊 Results Table</h3>
            {problem2_table}
            
            <h3>Column Descriptions</h3>
            <ul>
                <li><strong>Largest Delta (Heterogeneous):</strong> Maximum δ where violations were found</li>
                <li><strong>Smallest Delta (Homogeneous):</strong> Minimum δ where rule becomes homogeneous</li>
                <li><strong>Oracle Calls:</strong> Number of FPGrowth invocations (efficiency metric)</li>
                <li><strong>Abs Diff:</strong> |Subgroup ATE - Population ATE| of the violating subgroup</li>
            </ul>
        </div>
        
        <h2><span class="badge badge-problem3">PROBLEM 3</span> Find Smallest Epsilon - Results by Rule</h2>
        
        <div class="section">
            <h3>📋 What This Shows</h3>
            <p>For each rule, given <strong>fixed δ = {delta_test:,}</strong>, find the smallest ε where the rule becomes homogeneous.</p>
            <p><em>Compares Binary Search vs Brute Force methods.</em></p>
            
            <h3>📊 Results Table</h3>
            {problem3_table}
            
            <h3>Column Descriptions</h3>
            <ul>
                <li><strong>Epsilon Binary:</strong> Smallest ε found by Binary Search method</li>
                <li><strong>Epsilon Brute:</strong> Smallest ε found by Brute Force method</li>
                <li><strong>Match:</strong> Do both methods find the same result?</li>
                <li><strong>Oracle Calls Binary:</strong> Number of FPGrowth calls in Binary Search</li>
                <li><strong>Subgroups Brute:</strong> Total subgroups enumerated in Brute Force</li>
                <li><strong>Winner:</strong> Which method was faster for this rule</li>
                <li><strong>Speedup:</strong> Runtime ratio (higher = winner was much faster)</li>
            </ul>
        </div>
        
        <h2>📝 Summary</h2>
        <div class="section">
            <h3>Key Observations</h3>
            <ul>
                <li><strong>Problem 2:</strong> Uses only Binary Search (no comparison)</li>
                <li><strong>Problem 3:</strong> Compares two methods - Binary Search vs Brute Force</li>
                <li><strong>Each row</strong> shows results for a different treatment rule</li>
                <li><strong>Brute Force winner count:</strong> {sum(1 for r in problem3_results if r['Winner'] == 'Brute Force')} / {len(problem3_results)}</li>
                <li><strong>Binary Search winner count:</strong> {sum(1 for r in problem3_results if r['Winner'] == 'Binary Search')} / {len(problem3_results)}</li>
            </ul>
        </div>
    </div>
</body>
</html>"""

# Save HTML
output_file = Path("detailed_test_results.html")
with open(output_file, "w") as f:
    f.write(html)

# Save CSVs
df_problem2.to_csv("problem2_detailed_results.csv", index=False)
df_problem3.to_csv("problem3_detailed_results.csv", index=False)

print("\n" + "="*70)
print("✅ DETAILED TEST COMPLETE")
print("="*70)
print(f"\nFiles generated:")
print(f"  • HTML: {output_file}")
print(f"  • CSV (Problem 2): problem2_detailed_results.csv")
print(f"  • CSV (Problem 3): problem3_detailed_results.csv")
print(f"\nSummary:")
print(f"  • Tested {num_rules} rules")
print(f"  • Problem 2: Binary Search only")
print(f"  • Problem 3: Binary Search vs Brute Force")
print(f"  • Brute Force won {sum(1 for r in problem3_results if r['Winner'] == 'Brute Force')}/{len(problem3_results)} times")


