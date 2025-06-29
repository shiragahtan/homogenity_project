import os
import re
import json
import pandas as pd
from collections import Counter, defaultdict

# Load EPSILONS from config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)
EPSILONS = config['EPSILONS']

# Directory containing the xlsx files
DIRECTORY = '../algorithms_results/Yardens_results/'  # relative to this script

# Regex to match the files and extract delta (ignore rule_num)
file_pattern = re.compile(r'Apriori_subgroups_results_delta_(\d+)_\d+\.xlsx')

# Prepare results: {delta: {epsilon: Counter}}
results = defaultdict(lambda: defaultdict(Counter))

# Summary counter for epsilon 5000 across all rules and deltas
summary_counter = Counter()

for filename in os.listdir(DIRECTORY):
    match = file_pattern.match(filename)
    if not match:
        continue
    delta = int(match.group(1))
    filepath = os.path.join(DIRECTORY, filename)
    try:
        xls = pd.ExcelFile(filepath)
        sheet_name = 'Subgroups' if 'Subgroups' in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue
    # Ensure columns are as expected
    if not all(col in df.columns for col in ['AttributeValues', 'Size', 'Utility', 'UtilityDiff']):
        print(f"Skipping {filename}: missing required columns.")
        continue
    # Sort by abs(UtilityDiff)
    df = df.reindex(df['UtilityDiff'].abs().sort_values(ascending=False).index)
    for epsilon in EPSILONS:
        filtered = df[df['UtilityDiff'].abs() >= epsilon]
        for attrval in filtered['AttributeValues']:
            # Parse the string dict (e.g., "{'Gender': '1', 'HDI': '1'}")
            try:
                keys = list(eval(attrval).keys())
            except Exception:
                continue
            for key in keys:
                results[delta][epsilon][key] += 1
                # Add to summary counter for epsilon 5000
                if epsilon == 5000:
                    summary_counter[key] += 1

# Write results to Excel
output_path = os.path.join(os.path.dirname(__file__), 'attribute_counts_by_delta_epsilon.xlsx')
with pd.ExcelWriter(output_path) as writer:
    # Write individual delta sheets
    for delta in sorted(results):
        # Build a DataFrame: rows=epsilon, columns=attribute keys, values=counts
        all_keys = set()
        for epsilon in results[delta]:
            all_keys.update(results[delta][epsilon].keys())
        all_keys = sorted(all_keys)
        data = []
        for epsilon in sorted(results[delta]):
            row = [results[delta][epsilon].get(key, 0) for key in all_keys]
            data.append(row)
        df_out = pd.DataFrame(data, columns=all_keys, index=sorted(results[delta]))
        df_out.index.name = 'Epsilon'
        df_out.to_excel(writer, sheet_name=f'delta_{delta}')
    
    # Write summary sheet for epsilon 5000
    if summary_counter:
        summary_df = pd.DataFrame(list(summary_counter.items()), columns=['Attribute', 'Count'])
        summary_df = summary_df.sort_values('Count', ascending=False)
        summary_df.to_excel(writer, sheet_name='Summary_Epsilon_5000', index=False)
        print(f"Summary: Found {len(summary_counter)} unique attributes with counts ranging from {min(summary_counter.values())} to {max(summary_counter.values())}")

print(f"Done! Results saved to {output_path}") 