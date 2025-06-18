import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load configuration from config file
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

deltas = config['DELTAS']
epsilons = config['EPSILONS']
algorithms = config['ALGORITHM_NAMES']

df_updated = pd.read_excel("../algorithms/homogeneity_results.xlsx")

# Update rule definitions with actual treatment-condition pairs
rules_explicit = []
with open('../algorithms/Shira_Treatments.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        treatment = str(data['treatment'])
        condition = str(data['condition'])
        rules_explicit.append(f"{treatment}|{condition}")

# Create new row labels
matrix_rows = []
for rule_index, rule in enumerate(rules_explicit, start=1):
    # Iterate through deltas from small to big
    for delta in sorted(deltas):
        matrix_rows.append(f'Rule {rule_index} - Δ={delta}')

# Reset column headers
matrix_cols = []
for eps in epsilons:
    for algo in algorithms:
        matrix_cols.append(f'{algo}\nε={eps}')

# Reinitialize matrix
runtime_matrix = pd.DataFrame(np.nan, index=matrix_rows, columns=matrix_cols)
color_matrix = pd.DataFrame("", index=matrix_rows, columns=matrix_cols)

# Fill in runtimes and colors again with explicit rules
for rule_index, rule in enumerate(rules_explicit, start=1):
    treatment, condition = rule.split('|')
    for delta in deltas:
        row_label = f'Rule {rule_index} - Δ={delta}'
        for eps in epsilons:
            for algo in algorithms:
                match = df_updated[
                    (df_updated['treatment'] == treatment) &
                    (df_updated['condition'] == condition) &
                    (df_updated['delta'] == delta) &
                    (df_updated['epsilon'] == eps) &
                    (df_updated['algorithm'] == algo)
                ]
                if not match.empty:
                    runtime = match['run_time_seconds'].values[0]
                    hom = match['homogeneity_status'].values[0]
                    col_label = f'{algo}\nε={eps}'
                    runtime_matrix.loc[row_label, col_label] = round(runtime, 1)
                    color_matrix.loc[row_label, col_label] = 'green' if hom else 'red'

# Plot heatmap-style table with corrected rules
fig, ax = plt.subplots(figsize=(20, 10))
table = plt.table(cellText=runtime_matrix.fillna("").values,
                  rowLabels=runtime_matrix.index,
                  colLabels=runtime_matrix.columns,
                  loc='center',
                  cellLoc='center',
                  colLoc='center')

# Apply coloring
for i, row in enumerate(runtime_matrix.index):
    for j, col in enumerate(runtime_matrix.columns):
        cell = table[i + 1, j]
        color = color_matrix.loc[row, col]
        if color:
            cell.set_facecolor(color)

# Style adjustments
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
ax.axis('off')
plt.title("Runtime Table: Colored by Homogeneity (Green = Homogeneous, Red = Not)", fontsize=14)
plt.tight_layout()

# Save to file instead of showing
output_path = '../graphs/homogeneity_runtime_table.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Table saved to {output_path}")
