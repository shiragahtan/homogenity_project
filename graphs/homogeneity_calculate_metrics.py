import pandas as pd
import os
import numpy as np
import re

# --- Configuration ---
RESULTS_FILE = "homogeneity_results.xlsx"
OUTPUT_FILE = "homogeneity_metrics_summary.xlsx"


def to_boolean(val):
    """Robust boolean converter."""
    if isinstance(val, bool): return val
    if isinstance(val, (int, float)): return val != 0
    s = str(val).strip().lower()
    return s in ['true', 't', '1', '1.0', 'yes']


def calculate_summary_metrics():
    print("--- Calculating Summarized Metrics (All Algos in One Sheet) ---")

    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    df = pd.read_excel(RESULTS_FILE)
    if 'algorithm' in df.columns:
        df['algorithm'] = df['algorithm'].astype(str).str.strip()

    # --- 1. Identify Ground Truth (GT) ---
    GT_NAME = "Brute-Force Algorithm"
    unique_algos = df['algorithm'].unique()

    # Dynamic GT Detection
    if GT_NAME not in unique_algos:
        matches = [x for x in unique_algos if 'Brute' in x or 'Apriori' in x or 'FPGrowth' in x]
        if matches:
            GT_NAME = matches[0]
            print(f"Ground Truth Algorithm identified as: {GT_NAME}")
        else:
            print("Error: Could not identify Ground Truth (Brute/Apriori).")
            return

    # Identify Test Algorithms (Everyone else)
    test_algorithms = [algo for algo in unique_algos if algo != GT_NAME]

    # Get unique configs
    params = df[['delta', 'epsilon']].drop_duplicates().sort_values(['delta', 'epsilon'])

    final_rows = []

    # --- 2. Iterate Configs and Algorithms ---
    for _, row in params.iterrows():
        delta = int(row['delta'])
        epsilon = float(row['epsilon'])

        # Get Ground Truth for this config
        gt_subset = df[
            (df['algorithm'] == GT_NAME) &
            (df['delta'] == delta) &
            (df['epsilon'] == epsilon)
            ]

        # Build GT Lookup
        gt_lookup = {}
        for _, g_row in gt_subset.iterrows():
            k = (g_row['treatment'], g_row['condition'])
            gt_lookup[k] = to_boolean(g_row['homogeneity_status'])

        if not gt_lookup:
            continue

        # Process each algorithm against this GT
        for algo_name in test_algorithms:
            algo_subset = df[
                (df['algorithm'] == algo_name) &
                (df['delta'] == delta) &
                (df['epsilon'] == epsilon)
                ]

            if algo_subset.empty: continue

            run_precisions = []
            run_specificities = []

            for _, run_row in algo_subset.iterrows():
                key = (run_row['treatment'], run_row['condition'])
                if key not in gt_lookup: continue

                gt_val = gt_lookup[key]
                pred_val = to_boolean(run_row['homogeneity_status'])

                # Metrics Calculation (Scientific Per-Run Method)
                tp = 1 if (pred_val and gt_val) else 0
                fp = 1 if (pred_val and not gt_val) else 0
                tn = 1 if (not pred_val and not gt_val) else 0

                # Precision
                if (tp + fp) > 0:
                    p = tp / (tp + fp)
                else:
                    p = 1.0  # Silence is Golden

                # Specificity
                if (tn + fp) > 0:
                    s = tn / (tn + fp)
                else:
                    s = 1.0  # No Negatives existed or Perfect TN

                run_precisions.append(p)
                run_specificities.append(s)

            # Average
            if run_precisions:
                avg_p = np.mean(run_precisions) * 100
                avg_s = np.mean(run_specificities) * 100
                total_runs = len(run_precisions)
            else:
                avg_p, avg_s, total_runs = 0, 0, 0

            final_rows.append({
                "Algorithm": algo_name,
                "Delta": delta,
                "Epsilon": epsilon,
                "Precision (%)": round(avg_p, 2),
                "Specificity (%)": round(avg_s, 2),
                "Total Runs Checked": total_runs
            })

    # --- 3. Output ---
    if final_rows:
        summary_df = pd.DataFrame(final_rows)
        # Sort for readability
        summary_df = summary_df.sort_values(['Delta', 'Epsilon', 'Algorithm'])

        print(f"Writing summary to {OUTPUT_FILE}...")
        summary_df.to_excel(OUTPUT_FILE, index=False)
        print("Done.")
        print(summary_df)
    else:
        print("No results found.")


if __name__ == "__main__":
    calculate_summary_metrics()
