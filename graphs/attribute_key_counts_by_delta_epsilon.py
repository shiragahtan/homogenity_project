import os
import re
import json
import pandas as pd
from collections import Counter, defaultdict

# Load EPSILONS from config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)
# EPSILONS = config['EPSILONS']
EPSILONS = [76000]
DIRECTORY = '../algorithms_results/'
file_pattern = re.compile(r'FPGrowth_subgroups_results_delta_(\d+)_\d+\.xlsx')

results = defaultdict(lambda: defaultdict(Counter))
breaking_groups = defaultdict(lambda: defaultdict(list))
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

    if not all(col in df.columns for col in ['AttributeValues', 'Size', 'Utility', 'UtilityDiff']):
        continue

    df = df.reindex(df['UtilityDiff'].abs().sort_values(ascending=False).index)

    for epsilon in EPSILONS:
        filtered = df[df['UtilityDiff'].abs() >= epsilon]
        for _, row in filtered.iterrows():
            attrval = row['AttributeValues']
            try:
                keys = list(eval(attrval).keys())
                breaking_groups[delta][epsilon].append({
                    'Epsilon': epsilon,
                    'AttributeValues': attrval,
                    'Size': row['Size'],
                    'ATE': row['Utility'],
                    'ATE_diff': row['UtilityDiff'],
                    'Attributes': ', '.join(keys)
                })
                for key in keys:
                    results[delta][epsilon][key] += 1
                    if epsilon == 5000:
                        summary_counter[key] += 1
            except:
                continue

# --- OUTPUT SECTION ---

# 1. Write Counts and Summary to Excel (These are usually small)
output_excel = os.path.join(os.path.dirname(__file__), 'attribute_counts_summary.xlsx')
with pd.ExcelWriter(output_excel) as writer:
    for delta in sorted(results):
        all_keys = sorted({k for eps in results[delta] for k in results[delta][eps].keys()})
        data = [[results[delta][eps].get(k, 0) for k in all_keys] for eps in sorted(results[delta])]
        df_out = pd.DataFrame(data, columns=all_keys, index=sorted(results[delta]))
        df_out.index.name = 'Epsilon'
        df_out.to_excel(writer, sheet_name=f'delta_{delta}')

    if summary_counter:
        summary_df = pd.DataFrame(list(summary_counter.items()), columns=['Attribute', 'Count']).sort_values('Count',
                                                                                                             ascending=False)
        summary_df.to_excel(writer, sheet_name='Summary_Epsilon_5000', index=False)

# 2. Write Breaking Groups to CSV (To handle 1M+ rows)
for delta in breaking_groups:
    all_delta_groups = []
    for epsilon in sorted(breaking_groups[delta]):
        all_delta_groups.extend(breaking_groups[delta][epsilon])

    if all_delta_groups:
        csv_path = os.path.join(os.path.dirname(__file__), f'breaking_groups_delta_{delta}.csv')
        breaking_df = pd.DataFrame(all_delta_groups)
        # Sort by Epsilon then highest difference
        breaking_df = breaking_df.sort_values(['Epsilon', 'ATE_diff'], ascending=[True, False])
        breaking_df.to_csv(csv_path, index=False)
        print(f"Delta {delta}: Saved {len(all_delta_groups)} rows to {csv_path}")

print(f"Done! Summary saved to {output_excel}")