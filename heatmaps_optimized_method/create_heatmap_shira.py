import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
csv_path = "/Users/sgahtan/Desktop/shira/studies/brit_project/project_updated/homogenity_project/optimizations/homogeneity_results_direct_rule_3.csv"
df = pd.read_csv(csv_path)

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
plt.title( "Optimized method : Random Walks Without Unlearning\n"
            "Rule 3: Heatmap of Homogeneity (Color) and Runtime (Text)\n"
          "Treatment: {'DevType': 'Developer, back-end'}\n"
          "Condition: {'SexualOrientation': 'Straight or heterosexual'}", 
          pad=20)
plt.xlabel("Epsilon")
plt.ylabel("Delta")

# Display the heatmap
plt.tight_layout()
plt.show()
