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
# Note: The config usually contains just the filename, but your original script
# looked in "../algorithms/". We construct the path to match your folder structure.
TREATMENT_FILE = f"../algorithms/{ds_config['RULES_FILE']}"
DELTAS = ds_config['DELTAS']

print(f"🔹 Loaded Configuration for: {CHOSEN_DS}")
print(f"   Dataset: {DATASET_PATH}")
print(f"   Rules: {TREATMENT_FILE}")
print(f"   Deltas: {DELTAS}")


def plot_subgroups_graph(file_path, size_all, delta, rule, rule_data):
    """
    Plots a scatter graph and returns a dictionary containing summary stats.

    Args:
        file_path: Path to the Excel results file.
        size_all: The total count of rows in the population (filtered by condition).
        delta: The delta parameter used.
        rule: The index of the rule.
        rule_data: The dictionary object of the rule from JSON (for fallback titles).
    """
    excel_data = pd.ExcelFile(file_path)
    subgroups_data = pd.read_excel(excel_data, sheet_name='Subgroups')
    chosen_treatment = pd.read_excel(excel_data, sheet_name='ChosenTreatment')

    # Clean and coerce numeric columns
    for col in ['Utility', 'UtilityDiff', 'Size']:
        if col in subgroups_data.columns:
            subgroups_data[col] = subgroups_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            subgroups_data[col] = pd.to_numeric(subgroups_data[col], errors='coerce')

    # Compute utility_all from the first valid row
    first_valid = subgroups_data.dropna(subset=['Utility', 'UtilityDiff']).head(1)
    if first_valid.empty:
        print(f"Warning: No valid 'Utility'/'UtilityDiff' rows for rule {rule}, delta {delta}. Skipping plot.")
        return None
    utility_all = first_valid.iloc[0]['Utility'] - first_valid.iloc[0]['UtilityDiff']

    # Get Condition/Treatment strings for the Title
    if not chosen_treatment.empty:
        condition_str = str(chosen_treatment.loc[0, 'Condition'])
        treatment_str = str(chosen_treatment.loc[0, 'Treatment'])
    else:
        # Fallback to JSON data if Excel sheet is empty
        condition_str = str(rule_data.get('condition', 'Unknown'))
        treatment_str = str(rule_data.get('treatment', 'Unknown'))

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
            # Read line by line as it seems to be JSON lines based on your description
            # If it's a standard JSON list, use json.load(f).
            # Based on your prompt "{"condition"...} \n {"condition"...}", it implies JSON Lines.
            good_treatments = []
            for line in f:
                if line.strip():
                    good_treatments.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Treatment file '{TREATMENT_FILE}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: JSON format issue. Checking if file is a standard JSON list...")
        # Fallback if the file is actually a list [ {}, {} ]
        with open(TREATMENT_FILE, "r") as f:
            good_treatments = json.load(f)

    print(f"Found {len(good_treatments)} treatments.")

    # List to hold summary data for all rules
    summary_results = []

    # 3. Iterate over Rules
    for i, rule_data in enumerate(good_treatments):
        # Your indices likely start at 0, but your file naming convention seemed to check for `so_countries_treatment_{i+1}` previously.
        # If your Excel results are 0-indexed (rule0), use `i`. If 1-indexed (rule1), use `i+1`.
        # Assuming 0-indexed based on "rule{rule}_delta..." pattern in your code.
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
            # Adjust filename to match your exact output naming convention
            results_path = f'../algorithms_results/{CHOSEN_DS}_MultiProcessing_subgroups_results_delta_{delta}_{rule_idx}.xlsx'

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
        # Reorder if cols exist
        final_cols = [c for c in cols if c in summary_df.columns]
        summary_df = summary_df[final_cols]

        output_summary_path = f'{CHOSEN_DS}_summary_utility_diff.xlsx'
        summary_df.to_excel(output_summary_path, index=False)
        print(f"\nSummary file created successfully: {output_summary_path}")
        print(summary_df)
    else:
        print("\nNo results were generated to summarize.")
