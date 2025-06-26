import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(script_dir, "algorithm_time.xlsx")
df = pd.read_excel(excel_path)

# Pivot the table to create a format suitable for LaTeX table generation
pivot_df = df.pivot_table(index='delta', columns='algorithm', values='run_time_seconds', aggfunc='mean')

# Sort by delta values for better presentation
pivot_df = pivot_df.sort_index()

# Round to 3 decimal places
pivot_df = pivot_df.round(3)

# Add units to the column headers
pivot_df.columns = [f"{col} (Sec)" for col in pivot_df.columns]

# Make the figure size proportional to the table size
fig, ax = plt.subplots(figsize=(2 + 2 * len(pivot_df.columns), 1 + 0.7 * len(pivot_df)))  # Wider and taller for big tables
ax.axis('off')
tbl = ax.table(
    cellText=pivot_df.values,
    colLabels=pivot_df.columns,
    rowLabels=pivot_df.index,
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(14)
tbl.auto_set_column_width(col=list(range(len(pivot_df.columns)+1)))

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove all margins
plt.savefig(os.path.join(script_dir, "runtime_table.png"), bbox_inches='tight', dpi=200, pad_inches=0)
plt.close()
print("Runtime table image saved to runtime_table.png")
