import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Load Config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
CHOSEN_DS = config["CHOSEN_DATASET"]
INPUT_FILE = f"scalability_results_{CHOSEN_DS}.xlsx"


def plot_scalability():
    if not Path(INPUT_FILE).exists():
        print(f"❌ File {INPUT_FILE} not found.")
        return

    df = pd.read_excel(INPUT_FILE)
    sns.set_style("whitegrid")

    # Create 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Scalability Stress Test: {CHOSEN_DS.capitalize()}', fontsize=16)

    # --- ROW SCALABILITY ---
    row_df = df[df['Experiment'] == 'Row_Scalability']

    # Runtime
    sns.lineplot(ax=axes[0, 0], data=row_df, x='X_Value', y='BF_Time', label='FPGrowth', marker='o', color='#d62728')
    sns.lineplot(ax=axes[0, 0], data=row_df, x='X_Value', y='RW_Time', label='Random Walk', marker='o', color='#2ca02c')
    axes[0, 0].set_title('Runtime vs. Data Size')
    axes[0, 0].set_ylabel('Time (s)')
    axes[0, 0].set_xlabel('Number of Rows')

    # Accuracy
    sns.lineplot(ax=axes[0, 1], data=row_df, x='X_Value', y='Accuracy', label='Agreement Rate', marker='s',
                 color='#1f77b4')
    axes[0, 1].set_title('Accuracy vs. Data Size')
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].set_ylabel('Match Rate (0-1)')

    # --- COLUMN SCALABILITY ---
    col_df = df[df['Experiment'] == 'Col_Scalability']

    # Runtime
    sns.lineplot(ax=axes[1, 0], data=col_df, x='X_Value', y='BF_Time', label='FPGrowth', marker='o', color='#d62728')
    sns.lineplot(ax=axes[1, 0], data=col_df, x='X_Value', y='RW_Time', label='Random Walk', marker='o', color='#2ca02c')
    axes[1, 0].set_title('Runtime vs. Attribute Count')
    axes[1, 0].set_xlabel('Number of Attributes')
    axes[1, 0].set_ylabel('Time (s)')

    # Accuracy
    sns.lineplot(ax=axes[1, 1], data=col_df, x='X_Value', y='Accuracy', label='Agreement Rate', marker='s',
                 color='#1f77b4')
    axes[1, 1].set_title('Accuracy vs. Attribute Count')
    axes[1, 1].set_ylim(0, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{CHOSEN_DS}_scalability_plots.png")
    print(f"📈 Plot saved as {CHOSEN_DS}_scalability_plots.png")
    plt.show()


if __name__ == "__main__":
    plot_scalability()