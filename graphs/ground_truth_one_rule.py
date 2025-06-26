import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load configuration from config file
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

deltas = config['DELTAS']
epsilons = config['EPSILONS']
algorithms = config['ALGORITHM_NAMES']

# Load data from algorithms directory
df_updated = pd.read_excel("homogeneity_results.xlsx")

# Load all treatments from Shira_Treatments.json
treatments_data = []
with open('../algorithms/Shira_Treatments.json', 'r') as f:
    for line in f:
        treatments_data.append(json.loads(line))

# Modified function to show runtime inside heatmap cells and color by homogeneity
def plot_runtime_annotated_heatmap(data, treatment, condition, rule_index=None):
    # Filter the data for the specific treatment and condition
    filtered_df = data[
        (data['treatment'] == str(treatment)) &
        (data['condition'] == str(condition))
        ]

    if filtered_df.empty:
        print(f"No data found for rule {rule_index}: treatment={treatment}, condition={condition}")
        return None

    # Create pivot tables for coloring and annotation
    color_data = filtered_df.pivot_table(
        index='delta',
        columns='epsilon',
        values='homogeneity_status',
        aggfunc=lambda x: x.iloc[0] if not x.empty else False
    ).astype(int)

    runtime_data = filtered_df.pivot_table(
        index='delta',
        columns='epsilon',
        values='run_time_seconds',
        aggfunc=lambda x: round(x.iloc[0], 1) if not x.empty else ""
    )

    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        color_data,
        cmap='RdYlGn',
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': 'Homogeneity Status (1=True, 0=False)'},
        annot=runtime_data,
        fmt='',
        annot_kws={"size": 10, "weight": 'bold'}
    )

    rule_title = f"Rule {rule_index}: " if rule_index is not None else ""
    plt.title(f'{rule_title}Heatmap of Homogeneity (Color) and Runtime (Text)\nTreatment: {treatment}\nCondition: {condition}')
    plt.xlabel('Epsilon')
    plt.ylabel('Delta')
    plt.tight_layout()

    # Save to file instead of showing
    treatment_clean = str(treatment).replace(":", "_").replace(" ", "_").replace("'", "").replace("{", "").replace("}", "")
    condition_clean = str(condition).replace(":", "_").replace(" ", "_").replace("'", "").replace("{", "").replace("}", "")
    filename = f'homogeneity_heatmap_rule{rule_index}_{treatment_clean}__{condition_clean}.png'

    # Create graphs directory if it doesn't exist
    os.makedirs('../graphs', exist_ok=True)

    output_path = f'../graphs/{filename}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")
    return output_path


# Create heatmaps for all rules in Shira_Treatments.json
print(f"Creating heatmaps for {len(treatments_data)} rules from Shira_Treatments.json")

for i, treatment_data in enumerate(treatments_data, 1):
    treatment = treatment_data['treatment']
    condition = treatment_data['condition']

    print(f"\nProcessing Rule {i}:")
    print(f"  Treatment: {treatment}")
    print(f"  Condition: {condition}")

    plot_runtime_annotated_heatmap(df_updated, treatment, condition, rule_index=i)

print("\nAll heatmaps generated successfully!")
