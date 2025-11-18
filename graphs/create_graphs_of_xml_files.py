import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Load configuration file
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

DELTAS = config['DELTAS']


def plot_subgroups_graph(file_path, data_path, delta, rule):
    """
    Plots a scatter graph from the 'Subgroups' sheet in the given Excel file.
    It filters out invalid subgroups (where Utility is NaN) before plotting.
    """
    # Load the Excel file and the specific sheets
    excel_data = pd.ExcelFile(file_path)
    subgroups_data = pd.read_excel(excel_data, sheet_name='Subgroups')
    chosen_treatment = pd.read_excel(excel_data, sheet_name='ChosenTreatment')

    # 🚀 FIX 1: Convert relevant columns to numeric to avoid TypeError
    numeric_cols = ['Utility', 'UtilityDiff', 'Size']
    for col in numeric_cols:
        # Use errors='coerce' to turn non-numeric values (like NaNs) into actual NaN floats
        subgroups_data[col] = pd.to_numeric(subgroups_data[col], errors='coerce')

    # Step 1: Calculate Utility All (from the first row, which is assumed valid)
    # This calculation now works because the columns are numeric
    utility_all = subgroups_data.loc[0, 'Utility'] - subgroups_data.loc[0, 'UtilityDiff']

    # Step 2: Get the condition, treatment, and full dataset size
    # Assuming 'Condition' and 'Treatment' values in the Excel are valid string representations of objects
    condition = eval(chosen_treatment.loc[0, 'Condition'])
    treatment = eval(chosen_treatment.loc[0, 'Treatment'])
    full_dataset = pd.read_csv(data_path)
    size_all = full_dataset.shape[0]

    # Step 3: Filter out invalid subgroups (where 'Utility' or 'Size' might be NaN after conversion)
    valid_subgroups = subgroups_data.dropna(subset=['Utility', 'Size'])

    if valid_subgroups.empty:
        print(f"Warning: No valid subgroups found for rule {rule}, delta {delta}. Skipping plot.")
        return

    # Step 4: Plot the standard scatter plot (using *valid_subgroups*)
    plt.figure(figsize=(10, 6))
    plt.scatter(valid_subgroups['Utility'], valid_subgroups['Size'],
                alpha=0.7, edgecolors='w', s=100, label='Subgroups (Valid)')

    # Step 5: Plot the special dot for Utility All
    plt.scatter(utility_all, size_all, color='red', s=150, edgecolors='k', label='Utility All')

    # Step 6: Calculate max absolute utility diff (using *valid_subgroups*)
    # Ensure 'Size' is compared against delta after filtering/coercing to numeric
    subgroups_above_delta = valid_subgroups[valid_subgroups['Size'] > delta]

    if not subgroups_above_delta.empty:
        utility_diff = (subgroups_above_delta['Utility'] - utility_all).abs().max()
    else:
        # This can happen if no valid subgroups are above the delta
        utility_diff = 0

    # Step 7: Plot labels and legend
    plt.title(
        f"Subgroups: Utility vs Size\nCondition: {condition}\nTreatment: {treatment}\nMax |Utility_subgroup - Utility_all|: {utility_diff:.2f}")
    plt.xlabel('Utility')
    plt.ylabel('Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.savefig(f'rule{rule + 1}_delta_{delta}.png', bbox_inches='tight')

    # 🚀 FIX 2: Close the figure after saving to prevent the RuntimeWarning (memory leak)
    plt.close()

    print("saved graph on path: ", f'rule{rule + 1}_delta_{delta}.png')


if __name__ == "__main__":
    TREATMENT_FILE = "../algorithms/Chosen10Treatments.json"
    OUTPUT_DIR_NAME = 'processed_db'

    print(f"Loading treatments from {TREATMENT_FILE}...")
    try:
        with open(TREATMENT_FILE, "r") as f:
            # Assuming each line in the JSON file is a separate JSON object
            good_treatments = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Treatment file '{TREATMENT_FILE}' not found.")
        sys.exit(1)  # Add exit for critical failure

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
        # sys.exit(1) # Consider adding an exit here if no data means the script can't proceed

    print(f"Identified {len(treated_rules_datasets)} existing datasets for experiments.")

    for rule in range(len(good_treatments)):
        for delta in DELTAS:
            results_path = f'../algorithms_results/FPGrowth_subgroups_results_delta_{delta}_{rule}.xlsx'

            # Ensure we don't exceed the bounds of treated_rules_datasets
            if rule >= len(treated_rules_datasets):
                print(f"Error: Rule index {rule} is out of range for identified datasets. Breaking loop.")
                break

            data_path = treated_rules_datasets[rule]

            # Check if results file exists before trying to plot
            if Path(results_path).exists():
                plot_subgroups_graph(results_path, data_path, delta, rule)
            else:
                print(
                    f"Warning: Results file not found at {results_path}. Skipping plot for rule {rule}, delta {delta}.")