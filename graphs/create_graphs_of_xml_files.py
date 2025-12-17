import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Make sure numpy is imported if you're returning np.nan from the other file,
# although it's not strictly needed in *this* file for .dropna() to work.
# import numpy as np 

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

DELTAS = config['DELTAS']


def plot_subgroups_graph(file_path, data_path, delta, rule):
    """
    Plots a scatter graph and returns a dictionary containing summary stats
    (specifically the MaxAbsUtilityDiff).
    """
    excel_data = pd.ExcelFile(file_path)
    subgroups_data = pd.read_excel(excel_data, sheet_name='Subgroups')
    chosen_treatment = pd.read_excel(excel_data, sheet_name='ChosenTreatment')

    # Clean and coerce numeric columns (strip strings first if present)
    for col in ['Utility', 'UtilityDiff', 'Size']:
        if col in subgroups_data.columns:
            subgroups_data[col] = subgroups_data[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            subgroups_data[col] = pd.to_numeric(subgroups_data[col], errors='coerce')

    # Compute utility_all from the first valid row (both Utility and UtilityDiff must be numeric)
    first_valid = subgroups_data.dropna(subset=['Utility', 'UtilityDiff']).head(1)
    if first_valid.empty:
        print(f"Warning: No valid 'Utility'/'UtilityDiff' rows for rule {rule}, delta {delta}. Skipping plot.")
        return None
    utility_all = first_valid.iloc[0]['Utility'] - first_valid.iloc[0]['UtilityDiff']

    # Evaluate condition/treatment strings to objects (for plot title)
    # We also keep the raw strings for the summary excel
    condition_str = chosen_treatment.loc[0, 'Condition']
    treatment_str = chosen_treatment.loc[0, 'Treatment']

    # Safe eval for display logic
    try:
        condition = eval(condition_str)
        treatment = eval(treatment_str)
    except:
        condition = condition_str
        treatment = treatment_str

    full_dataset = pd.read_csv(data_path)
    size_all = full_dataset.shape[0]

    # Keep only rows where both Utility and Size are numeric
    valid_subgroups = subgroups_data.dropna(subset=['Utility', 'Size']).copy()
    dropped_count = len(subgroups_data) - len(valid_subgroups)
    if dropped_count:
        print(
            f"Info: Dropped {dropped_count} invalid subgroup rows for rule {rule}, delta {delta} (non-numeric Utility/Size).")

    if valid_subgroups.empty:
        print(f"Warning: No valid subgroups found for rule {rule}, delta {delta}. Skipping plot.")
        return None

    # Ensure Size is integer for plotting and calculations
    valid_subgroups['Size'] = valid_subgroups['Size'].astype(int)

    # --- CALCULATION FOR SUMMARY ---
    subgroups_above_delta = valid_subgroups[valid_subgroups['Size'] > delta]
    if not subgroups_above_delta.empty:
        utility_diff = (subgroups_above_delta['Utility'] - utility_all).abs().max()
    else:
        utility_diff = 0

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_subgroups['Utility'], valid_subgroups['Size'],
                alpha=0.7, edgecolors='w', s=100, label='Subgroups (Valid)')

    plt.scatter(utility_all, size_all, color='red', s=150, edgecolors='k', label='Utility All')

    plt.title(
        f"Subgroups: Utility vs Size\nCondition: {condition}\nTreatment: {treatment}\nMax |Utility_subgroup - Utility_all|: {utility_diff:.2f}")
    plt.xlabel('Utility')
    plt.ylabel('Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f'rule{rule}_delta_{delta}.png', bbox_inches='tight')
    print("saved graph on path: ", f'rule{rule}_delta_{delta}.png')

    plt.close()  # Good practice to close figure to free memory

    # Return the summary data for this rule
    return {
        'Rule': rule,
        'Delta': delta,
        'MaxAbsUtilityDiff': utility_diff,
        'Condition': condition_str,
        'Treatment': treatment_str
    }


if __name__ == "__main__":
    TREATMENT_FILE = "../algorithms/Chosen10Treatments.json"
    OUTPUT_DIR_NAME = 'processed_db'

    print(f"Loading treatments from {TREATMENT_FILE}...")
    try:
        with open(TREATMENT_FILE, "r") as f:
            good_treatments = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Treatment file '{TREATMENT_FILE}' not found.")
        good_treatments = []  # Ensure it's defined to avoid crash later

    num_expected_datasets = len(good_treatments)
    print(f"Found {num_expected_datasets} treatments in the JSON file.")
    base_data_dir = Path('../stackoverflow').resolve()
    processed_db_dir = base_data_dir / OUTPUT_DIR_NAME

    treated_rules_datasets = []
    for i in range(1, num_expected_datasets + 1):
        filename = f"so_countries_treatment_{i}_encoded.csv"
        file_path = processed_db_dir / filename

        if file_path.exists():
            treated_rules_datasets.append(str(file_path))
        else:
            print(f"Warning: Dataset for index {i} not found at {file_path}. Skipping.")

    if not treated_rules_datasets:
        print("Error: No processed datasets found. Please run batch_process_treatments.py first.")

    print(f"Identified {len(treated_rules_datasets)} existing datasets for experiments.")

    DELTAS = [1000]

    # List to hold summary data for all rules
    summary_results = []

    for rule in range(len(good_treatments)):
        for delta in DELTAS:
            results_path = f'../algorithms_results/FPGrowth_subgroups_results_delta_{delta}_{rule}.xlsx'

            # Check range to avoid index error if datasets are missing
            if rule < len(treated_rules_datasets):
                data_path = treated_rules_datasets[rule]

                # Check if results file exists before trying to plot
                if Path(results_path).exists():
                    result_data = plot_subgroups_graph(results_path, data_path, delta, rule)
                    if result_data:
                        summary_results.append(result_data)
                else:
                    print(
                        f"Warning: Results file not found at {results_path}. Skipping plot for rule {rule}, delta {delta}.")
            else:
                print(f"Warning: No dataset path available for rule index {rule}.")

    # --- SAVE SUMMARY EXCEL ---
    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        # Reorder columns nicely
        cols = ['Rule', 'Delta', 'MaxAbsUtilityDiff', 'Condition', 'Treatment']
        summary_df = summary_df[cols]

        output_summary_path = 'summary_utility_diff.xlsx'
        summary_df.to_excel(output_summary_path, index=False)
        print(f"\nSummary file created successfully: {output_summary_path}")
        print(summary_df)
    else:
        print("\nNo results were generated to summarize.")
