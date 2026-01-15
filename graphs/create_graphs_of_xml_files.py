import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# --- Configuration Load ---
CONFIG_PATH = '../configs/config.json'

# Load the configuration file
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# Change this variable to switch datasets: "german_credit", "stackoverflow", or "acs"
CHOSEN_DS = config["CHOSEN_DATASET"]

if CHOSEN_DS not in config['DATASETS']:
    raise ValueError(f"Dataset '{CHOSEN_DS}' not found in config.json")

ds_config = config['DATASETS'][CHOSEN_DS]

# Dynamically assign values from config
DATASET_PATH = ds_config['FULL_DATASET_PATH']
TREATMENT_FILE = f"../algorithms/{ds_config['RULES_FILE']}"
DELTAS = ds_config['DELTAS']

print(f"🔹 Loaded Configuration for: {CHOSEN_DS}")
print(f"   Dataset: {DATASET_PATH}")
print(f"   Rules: {TREATMENT_FILE}")
print(f"   Deltas: {DELTAS}")


def plot_subgroups_graph(file_path, size_all, delta, rule, rule_data):
    """
    Plots a scatter graph and returns a dictionary containing summary stats.
    Reads data from CSV and metadata from companion JSON.

    Args:
        file_path: Path to the CSV results file.
        size_all: The total count of rows in the population (filtered by condition).
        delta: The delta parameter used.
        rule: The index of the rule.
        rule_data: The dictionary object of the rule from JSON (for fallback titles).
    """
    # 1. Read the CSV Data
    try:
        subgroups_data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"Warning: CSV file is empty: {file_path}")
        return None

    # 2. Try to load metadata (Condition/Treatment) from companion JSON file
    # Pattern: filename.csv -> filename_metadata.json
    meta_path = str(file_path).replace('.csv', '_metadata.json')
    
    condition_str = "Unknown"
    treatment_str = "Unknown"

    if Path(meta_path).exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                condition_str = str(meta.get('Condition', rule_data.get('condition')))
                treatment_str = str(meta.get('Treatment', rule_data.get('treatment')))
        except Exception as e:
            print(f"Warning: Could not read metadata {meta_path}: {e}")
            # Fallback
            condition_str = str(rule_data.get('condition', 'Unknown'))
            treatment_str = str(rule_data.get('treatment', 'Unknown'))
    else:
        # Fallback to JSON rule data if metadata file doesn't exist
        condition_str = str(rule_data.get('condition', 'Unknown'))
        treatment_str = str(rule_data.get('treatment', 'Unknown'))

    # Clean and coerce numeric columns
    for col in ['Utility', 'UtilityDiff', 'Size']:
        if col in subgroups_data.columns:
            # Check if column is object type (string) before using .str accessor
            if subgroups_data[col].dtype == 'object':
                 subgroups_data[col] = subgroups_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            subgroups_data[col] = pd.to_numeric(subgroups_data[col], errors='coerce')

    # Compute utility_all from the first valid row
    first_valid = subgroups_data.dropna(subset=['Utility', 'UtilityDiff']).head(1)
    if first_valid.empty:
        print(f"Warning: No valid 'Utility'/'UtilityDiff' rows for rule {rule}, delta {delta}. Skipping plot.")
        return None
    
    # Calculate utility_all based on the difference
    utility_all = first_valid.iloc[0]['Utility'] - first_valid.iloc[0]['UtilityDiff']

    # Keep only rows where both Utility and Size are numeric
    valid_subgroups = subgroups_data.dropna(subset=['Utility', 'Size']).copy()

    if valid_subgroups.empty:
        print(f"Warning: No valid subgroups found for rule {rule}, delta {delta}. Skipping plot.")
        return None

    # Ensure Size is integer
    valid_subgroups['Size'] = valid_subgroups['Size'].astype(int)

    # --- CALCULATION FOR SUMMARY ---
    subgroups_above_delta = valid_subgroups[valid_subgroups['Size'] > delta]
    if not subgroups_above_delta.empty:
        utility_diff = (subgroups_above_delta['Utility'] - utility_all).abs().max()
    else:
        utility_diff = 0

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot Subgroups
    plt.scatter(valid_subgroups['Utility'], valid_subgroups['Size'],
                alpha=0.7, edgecolors='w', s=100, label='Subgroups (Valid)')

    # Plot Global Population (Red Dot) using the calculated size_all
    plt.scatter(utility_all, size_all, color='red', s=150, edgecolors='k', label='Utility All (Condition Population)')

    plt.title(
        f"Subgroups: Utility vs Size\nCondition: {condition_str}\nTreatment: {treatment_str}\nMax |Utility_subgroup - Utility_all|: {utility_diff:.2f}")
    plt.xlabel('Utility')
    plt.ylabel('Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    output_png = f'rule{rule}_delta_{delta}.png'
    plt.savefig(output_png, bbox_inches='tight')
    print(f"  Saved graph: {output_png}")

    plt.close()

    return {
        'Rule': rule,
        'Delta': delta,
        'MaxAbsUtilityDiff': utility_diff,
        'Condition': condition_str,
        'Treatment': treatment_str
    }


if __name__ == "__main__":

    # 1. Load Main Dataset
    print(f"Loading main dataset: {DATASET_PATH}")
    if not Path(DATASET_PATH).exists():
        print(f"Error: Dataset {DATASET_PATH} not found.")
        exit(1)

    df_main = pd.read_csv(DATASET_PATH)

    # 2. Load Treatments JSON
    print(f"Loading treatments from {TREATMENT_FILE}...")
    try:
        with open(TREATMENT_FILE, "r") as f:
            good_treatments = []
            for line in f:
                if line.strip():
                    good_treatments.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Treatment file '{TREATMENT_FILE}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: JSON format issue. Checking if file is a standard JSON list...")
        with open(TREATMENT_FILE, "r") as f:
            good_treatments = json.load(f)

    print(f"Found {len(good_treatments)} treatments.")

    # List to hold summary data for all rules
    summary_results = []

    # 3. Iterate over Rules
    for i, rule_data in enumerate(good_treatments):
        rule_idx = i

        # --- A. Calculate Size (Filter Main DF) ---
        condition = rule_data.get("condition", {})

        # Build query mask
        mask = pd.Series([True] * len(df_main))
        for col, val in condition.items():
            if col in df_main.columns:
                mask &= (df_main[col] == val)
            else:
                print(f"Warning: Column '{col}' not found in dataset. Skipping filtering for this col.")

        df_filtered = df_main[mask]
        size_all = len(df_filtered)

        print(f"Processing Rule {rule_idx}: Condition {condition} -> Size: {size_all}")

        if size_all == 0:
            print(f"  Warning: No rows match condition {condition}. Skipping.")
            continue

        # --- B. Process for each Delta ---
        for delta in DELTAS:
            # Changed to look for .csv files
            # Note: Ensure "MultiProcessing" is the correct algorithm name used in your output filenames
            # If you are using RW, change this string to "RW"
            results_path = f'../algorithms_results/{CHOSEN_DS}_MultiProcessing_subgroups_results_delta_{delta}_{rule_idx}.csv'

            if Path(results_path).exists():
                result_data = plot_subgroups_graph(results_path, size_all, delta, rule_idx, rule_data)
                if result_data:
                    summary_results.append(result_data)
            else:
                print(f"  Warning: Results file not found: {results_path}")

    # --- SAVE SUMMARY EXCEL ---
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        cols = ['Rule', 'Delta', 'MaxAbsUtilityDiff', 'Condition', 'Treatment']
        final_cols = [c for c in cols if c in summary_df.columns]
        summary_df = summary_df[final_cols]

        output_summary_path = f'{CHOSEN_DS}_summary_utility_diff.xlsx'
        summary_df.to_excel(output_summary_path, index=False)
        print(f"\nSummary file created successfully: {output_summary_path}")
        print(summary_df)
    else:
        print("\nNo results were generated to summarize.")