import json
import pandas as pd
import matplotlib.pyplot as plt

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

DELTAS = config['DELTAS']

def plot_subgroups_graph(file_path, data_path, delta, rule):
    """
    Plots a scatter graph from the 'Subgroups' sheet in the given Excel file.
    Additionally, it calculates and displays the 'utility_all' dot in a different color.
    The plot title includes the condition, treatment, and the difference between max and min utility.
    """
    # Load the Excel file and the specific sheets
    excel_data = pd.ExcelFile(file_path)
    subgroups_data = pd.read_excel(excel_data, sheet_name='Subgroups')
    chosen_treatment = pd.read_excel(excel_data, sheet_name='ChosenTreatment')

    #print("shira Rows with Size > 31051:")
    #print(subgroups_data[subgroups_data['Size'] > 31051])

    # Step 1: Plot the standard scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(subgroups_data['Utility'], subgroups_data['Size'], 
                alpha=0.7, edgecolors='w', s=100, label='Subgroups')

    # Step 2: Calculate Utility All
    utility_all = subgroups_data.loc[0, 'Utility'] - subgroups_data.loc[0, 'UtilityDiff']

    # Step 3: Get the condition and treatment and filter the dataset
    condition = eval(chosen_treatment.loc[0, 'Condition'])
    treatment = eval(chosen_treatment.loc[0, 'Treatment'])
    # key, value = list(condition.items())[0]
    # print("shira", key, value)

    # # Load the full dataset
    full_dataset = pd.read_csv(data_path)

    # # Filter the dataset based on the condition and get the size
    size_all = full_dataset.shape[0]
    # size_all = full_dataset[full_dataset[key] == value].shape[0]
    # print("shira", size_all)
    # Step 4: Plot the special dot for Utility All
    plt.scatter(utility_all, size_all, color='red', s=150, edgecolors='k', label='Utility All')
    
    # Step 5: Calculate max absolute utility diff (homogeneity measure)
    subgroups_above_delta = subgroups_data[subgroups_data['Size'] > delta]
    if not subgroups_above_delta.empty:
        utility_diff = (subgroups_above_delta['Utility'] - utility_all).abs().max()
    else:
        utility_diff = 0

    # Step 6: Plot labels and legend, including condition, treatment, and utility diff in the title
    plt.title(f"Subgroups: Utility vs Size\nCondition: {condition}\nTreatment: {treatment}\nMax |Utility_subgroup - Utility_all|: {utility_diff:.2f}")
    plt.xlabel('Utility')
    plt.ylabel('Size')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(f'rule{rule}_delta_{delta}.png', bbox_inches='tight')
    # plt.show()  # Optionally comment this out if you don't want to display the plot
    print("saved graph on path: ", f'rule{rule}_delta_{delta}.png')

if __name__ == "__main__":
    treatment_file = "../algorithms/Shira_Treatments.json"
    treated_rules_datasets = [
        '../stackoverflow/so_countries_treatment_1_encoded.csv',
        '../stackoverflow/so_countries_treatment_2_encoded.csv',
        '../stackoverflow/so_countries_treatment_3_encoded.csv'
    ]

    with open(treatment_file, "r") as f:
        good_treatments = [json.loads(line) for line in f]

    for rule in range(len(good_treatments)):
        for delta in DELTAS:
            results_path = f'../algorithms_results/Apriori_subgroups_results_delta_{delta}_{rule}.xlsx'
            data_path = treated_rules_datasets[rule]
            plot_subgroups_graph(results_path, data_path, delta, rule)