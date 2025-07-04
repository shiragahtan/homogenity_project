import os
import pandas as pd
import matplotlib.pyplot as plt


# --- CONFIGURATION ---
INPUT_FILE = 'homogeneity_results1.xlsx'
OUTPUT_DIR = 'plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD & CLEAN ---
df = pd.read_excel(INPUT_FILE)
# convert to numeric and drop any bad rows
df['epsilon'] = pd.to_numeric(df['epsilon'], errors='coerce')
df['delta']   = pd.to_numeric(df['delta'],   errors='coerce')
df = df.dropna(subset=['epsilon','delta','algorithm','run_time_seconds'])
df = df[df['algorithm'] != 'RWUnlearning']  # Exclude RWUnlearning

# --- AGGREGATE ---
agg = (
    df
    .groupby(['epsilon', 'delta', 'algorithm'])['run_time_seconds']
    .mean()
    .reset_index(name='avg_time_s')
)

# (Optional) export the aggregated table
agg.to_csv(os.path.join(OUTPUT_DIR, 'aggregated_run_times.csv'), index=False)

# --- PLOT 1: Avg run time vs ε for each δ ---
for delta in sorted(agg['delta'].unique()):
    sub = agg[agg['delta'] == delta]
    plt.figure(figsize=(8,5))
    for alg in sub['algorithm'].unique():
        s = sub[sub['algorithm'] == alg].sort_values('epsilon')
        plt.plot(s['epsilon'], s['avg_time_s'], marker='o', label=alg)
    plt.title(f'Avg Run Time vs ε (δ = {delta})')
    plt.xlabel('ε (epsilon)')
    plt.ylabel('Average run time (s)')
    plt.legend(title='Algorithm')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'run_time_vs_epsilon_delta_{int(delta)}.png'), dpi=200)
    plt.close()

# --- PLOT 2: Overall average per algorithm ---
overall = agg.groupby('algorithm')['avg_time_s'].mean().sort_values()
plt.figure(figsize=(6,4))
overall.plot(kind='bar')
plt.title('Overall Average Run Time per Algorithm')
plt.xlabel('Algorithm')
plt.ylabel('Average run time (s)')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'overall_avg_run_time_per_algorithm.png'), dpi=200)
plt.close()

print(f"✅ Done! All plots and the aggregated CSV are in ./{OUTPUT_DIR}/")
