import os
import json
import pandas as pd

# --- Configuration ---
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

# Change this variable to switch datasets: "german_credit" OR "stackoverflow"
CHOSEN_DS = config["CHOSEN_DATASET"]
INPUT_FILE = f"{CHOSEN_DS}_homogeneity_results.xlsx"
OUTPUT_FILE = f"{CHOSEN_DS}_homogeneity_runtime_summary.xlsx"


def summarize_runtimes(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    print("--- Calculating Runtime Statistics (All Algos in One Sheet) ---")

    # 1. Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found in the current directory.")
        return

    print(f"Reading data from {input_file}...")

    try:
        # Load the data
        df = pd.read_excel(input_file)

        # Ensure necessary standard columns exist
        required_columns = ['algorithm', 'delta', 'epsilon', 'run_time_seconds']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Input file is missing one of the required columns: {required_columns}")
            return

        # Clean algorithm names (strip whitespace)
        if 'algorithm' in df.columns:
            df['algorithm'] = df['algorithm'].astype(str).str.strip()

        # --- RENAME STEP: Change FPGrowth to Brute Force ---
        # This checks if "FPGrowth" is in the name (case-insensitive) and renames it
        mask = df['algorithm'].str.contains('FPGrowth', case=False, na=False)
        if mask.any():
            print(f"  > Renaming {mask.sum()} rows from 'FPGrowth' to 'Brute Force'")
            df.loc[mask, 'algorithm'] = 'Brute Force'

        # Handle Num Subgroups column
        # If it doesn't exist or is NaN, fill with 0
        if 'num_subgroups' not in df.columns:
            df['num_subgroups'] = 0
        else:
            df['num_subgroups'] = pd.to_numeric(df['num_subgroups'], errors='coerce').fillna(0)

        # 2. Group by Algorithm, Delta, and Epsilon
        print("Calculating averages, variance, and standard deviation...")

        summary_df = df.groupby(['algorithm', 'delta', 'epsilon']).agg(
            Runs_Count=('run_time_seconds', 'count'),
            Avg_Runtime_Sec=('run_time_seconds', 'mean'),
            Variance_Runtime=('run_time_seconds', 'var'),
            Std_Dev_Runtime=('run_time_seconds', 'std'),
            Avg_Num_Subgroups=('num_subgroups', 'mean')
        ).reset_index()

        # Fill NaN values (happens if only 1 run exists)
        cols_to_fill = ['Std_Dev_Runtime', 'Variance_Runtime', 'Avg_Num_Subgroups']
        summary_df[cols_to_fill] = summary_df[cols_to_fill].fillna(0)

        # Round for cleaner output
        round_cols = ['Avg_Runtime_Sec', 'Variance_Runtime', 'Std_Dev_Runtime', 'Avg_Num_Subgroups']
        for col in round_cols:
            summary_df[col] = summary_df[col].round(4)

        # Sort for better readability: Delta -> Epsilon -> Avg Runtime (Ascending)
        summary_df = summary_df.sort_values(['delta', 'epsilon', 'Avg_Runtime_Sec'])

        # 3. Write to Excel (Single Sheet)
        print(f"Writing results to {output_file}...")
        summary_df.to_excel(output_file, index=False, sheet_name="Runtime_Summary")

        print("Done! Summary file created successfully.")
        print("\nPreview of Results:")
        print(summary_df.head(10))

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    summarize_runtimes()