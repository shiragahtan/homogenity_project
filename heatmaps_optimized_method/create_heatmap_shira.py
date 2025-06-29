import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Ignore FutureWarning about downcasting behavior
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the CSV file
for rule_num in range(1, 4):
  csv_path = f"../optimizations/homogeneity_results_direct_rule_{rule_num}.csv"
  df = pd.read_csv(csv_path)
  treatment_file = "../algorithms/Shira_Treatments.json"
  with open(treatment_file, "r") as f:
    good_treatments = [json.loads(line) for line in f]

  treatment = good_treatments[rule_num - 1]["treatment"]
  condition = good_treatments[rule_num - 1]["condition"]


  # Filter only the relevant mode (optional, based on your file)
  df = df[df["Mode"] == "direct"]

  # Pivot data for heatmap: Runtime as text, Homogeneous as color
  runtime_data = df.pivot(index="Delta", columns="Epsilon", values="Runtime")
  homogeneity_mask = df.pivot(index="Delta", columns="Epsilon", values="Homogeneous")

  # Map True/False to 1/0 for coloring
  color_data = homogeneity_mask.replace({True: 1, False: 0})

  # Create the heatmap
  plt.figure(figsize=(12, 8))
  sns.heatmap(
      color_data,
      annot=runtime_data.round(1),
      fmt=".1f",
      cmap="RdYlGn",
      cbar_kws={'label': 'Homogeneity Status (1=True, 0=False)'},
      linewidths=0.5,
      linecolor='gray'
  )

  # Add titles and labels
  plt.title(f"Optimized method : Random Walks Without Unlearning\n"
            f"Rule {rule_num}: Heatmap of Homogeneity (Color) and Runtime (Text)\n"
            f"Treatment: {treatment}\n"
            f"Condition: {condition}", 
            pad=20)
  plt.xlabel("Epsilon")
  plt.ylabel("Delta")

  # Display the heatmap
  plt.tight_layout()
  plt.savefig(f'homogeneity_heatmap_rule_{rule_num}.png', dpi=300, bbox_inches='tight')
  print(f"Heatmap saved to homogeneity_heatmap_rule_{rule_num}.png")
# plt.show()  # Commented out to save instead of display
