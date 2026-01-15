import os
import re
import json
import pandas as pd
from collections import Counter, defaultdict

# --- CONFIGURATION ---
# Ensure the config path is correct relative to where you run the script
config_path = '../configs/config.json'
if not os.path.exists(config_path):
    # Fallback if running from a different directory depth
    config_path = 'configs/config.json'

try:
    with open(config_path, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Config file not found at {config_path}")
    exit()

# Change this variable to switch datasets: "german_credit" OR "stackoverflow"
CHOSEN_DS = config["CHOSEN_DATASET"]

if CHOSEN_DS not in config['DATASETS']:
    raise ValueError(f"Dataset '{CHOSEN_DS}' not found in config.json")

ds_config = config['DATASETS'][CHOSEN_DS]

# Load dataset-specific values
EPSILONS = ds_config['EPSILONS']
VALID_DELTAS = set(ds_config['DELTAS'])  # Used to filter files belonging to this dataset

# Set the epsilon to use for the "Summary" sheet (defaulting to the first one)
SUMMARY_EPSILON = EPSILONS[0]

print(f"🔹 Analyzing results for: {CHOSEN_DS}")
print(f"   Target Epsilons: {EPSILONS}")
print(f"   Valid Deltas: {VALID_DELTAS}")
print(f"   Summary Epsilon: {SUMMARY_EPSILON}")

DIRECTORY = '../algorithms_results/'

# FIX 1: Updated pattern to match .csv files instead of .xlsx
# Pattern matches filenames like: "so_..._delta_100_0.csv"
file_pattern = re.compile(rf"{re.escape(CHOSEN_DS)}.*_delta_(\d+)_(\d+)\.csv$", re.IGNORECASE)

results = defaultdict(lambda: defaultdict(Counter))
breaking_groups = defaultdict(lambda: defaultdict(list))
summary_counter = Counter()

# --- PROCESSING ---
if not os.path.exists(DIRECTORY):
    print(f"Directory not found: {DIRECTORY}")
    exit()

for filename in os.listdir(DIRECTORY):
    match = file_pattern.search(filename)
    if not match:
        continue

    delta = int(match.group(1))

    # FILTER: Only process files that match the DELTAS of the chosen dataset
    if delta not in VALID_DELTAS:
        continue

    filepath = os.path.join(DIRECTORY, filename)
    try:
        # FIX 2: Read CSV instead of Excel
        # Using engine='python' is safer if you have mixed separators or complex parsing issues
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

    # Ensure required columns exist
    if not all(col in df.columns for col in ['AttributeValues', 'Size', 'Utility', 'UtilityDiff']):
        continue

    # Sort by UtilityDiff for processing priority
    df = df.reindex(df['UtilityDiff'].abs().sort_values(ascending=False).index)

    for epsilon in EPSILONS:
        # Filter rows where violation >= epsilon
        filtered = df[df['UtilityDiff'].abs() >= epsilon]

        for _, row in filtered.iterrows():
            attrval = row['AttributeValues']
            try:
                # Parse dictionary string keys
                keys = list(eval(attrval).keys())

                breaking_groups[delta][epsilon].append({
                    'Epsilon': epsilon,
                    'AttributeValues': attrval,
                    'Size': row['Size'],
                    'ATE': row['Utility'],
                    'ATE_diff': row['UtilityDiff'],
                    'Attributes': ', '.join(keys),
                    'SourceFile': filename
                })

                for key in keys:
                    results[delta][epsilon][key] += 1
                    # Use dynamic summary epsilon instead of hardcoded value
                    if epsilon == SUMMARY_EPSILON:
                        summary_counter[key] += 1
            except Exception:
                continue

# --- OUTPUT SECTION ---

# 1. Write Counts and Summary to Excel (Output remains Excel to support multiple sheets)
output_excel = os.path.join(os.path.dirname(__file__), f'attribute_counts_summary_{CHOSEN_DS}.xlsx')

# Check if we actually have data before opening the writer
if not results and not summary_counter:
    print(f"⚠️ No matching data found for {CHOSEN_DS} (looked for CSVs).")
    print("   Skipping Excel creation to prevent errors.")
else:
    try:
        with pd.ExcelWriter(output_excel) as writer:
            data_written = False
            
            for delta in sorted(results):
                all_keys = sorted({k for eps in results[delta] for k in results[delta][eps].keys()})
                if not all_keys:
                    continue
                
                # Create matrix: Rows=Epsilons, Cols=Attributes
                data = [[results[delta][eps].get(k, 0) for k in all_keys] for eps in sorted(results[delta])]
                df_out = pd.DataFrame(data, columns=all_keys, index=sorted(results[delta]))
                df_out.index.name = 'Epsilon'
                df_out.to_excel(writer, sheet_name=f'delta_{delta}')
                data_written = True

            if summary_counter:
                summary_df = pd.DataFrame(list(summary_counter.items()), columns=['Attribute', 'Count']).sort_values('Count',
                                                                                                                     ascending=False)
                summary_df.to_excel(writer, sheet_name=f'Summary_Eps_{SUMMARY_EPSILON}', index=False)
                data_written = True
            
            if data_written:
                print(f"✅ Summary saved to {output_excel}")
            else:
                print("⚠️ Logic matched data but resulting DataFrames were empty (no output generated).")

    except IndexError as e:
        print(f"❌ Error saving Excel: {e}") 
        print("   (This usually occurs if valid data was found but no sheets could be written.)")

# 2. Write Breaking Groups to CSV
for delta in breaking_groups:
    all_delta_groups = []
    for epsilon in sorted(breaking_groups[delta]):
        all_delta_groups.extend(breaking_groups[delta][epsilon])

    if all_delta_groups:
        csv_path = os.path.join(os.path.dirname(__file__), f'breaking_groups_{CHOSEN_DS}_delta_{delta}.csv')
        breaking_df = pd.DataFrame(all_delta_groups)

        # Sort by Epsilon then highest difference
        breaking_df = breaking_df.sort_values(['Epsilon', 'ATE_diff'], ascending=[True, False])

        breaking_df.to_csv(csv_path, index=False)
        print(f"   Delta {delta}: Saved {len(all_delta_groups)} rows to {csv_path}")

print("Done!")