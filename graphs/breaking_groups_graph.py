import pandas as pd
import matplotlib.pyplot as plt

# Load the specific sheet
file_path = 'attribute_counts_by_delta_epsilon.xlsx'
sheet_name = 'breaking_groups_delta_5000'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Create scatter plot of utility difference by group size
plt.figure()
plt.scatter(df['Size'], df['ATE_diff'], marker='x', color='orange')
plt.xlabel('Group Size')
plt.ylabel('Utility Difference (ATE_diff)')
plt.title('Utility Difference by Group Size')
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.savefig('utility_difference_by_group_size.png', dpi=300, bbox_inches='tight')
plt.close()
print("Graph saved as utility_difference_by_group_size.png")
