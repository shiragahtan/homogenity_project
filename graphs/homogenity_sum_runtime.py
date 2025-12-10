import os
import pandas as pd


def summarize_runtimes(input_file="homogeneity_results.xlsx", output_file="homogeneity_runtime_summary.xlsx"):
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

        # Handle specific columns for Apriori/FPGrowth (enumeration/iteration) & Num Subgroups
        # If they don't exist (older files) or are NaN, fill them with 0 for calculation safety
        extra_cols = ['enumeration_time_sec', 'iteration_time_sec', 'num_subgroups']
        for col in extra_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                # Convert to numeric, forcing errors to NaN, then fill with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 2. Group by Algorithm, Delta, and Epsilon
        print("Calculating averages, variance, and standard deviation...")

        # We use .agg() to calculate mean, variance, std, and the new averages
        summary_df = df.groupby(['algorithm', 'delta', 'epsilon']).agg(
            Runs_Count=('run_time_seconds', 'count'),
            Avg_Runtime_Sec=('run_time_seconds', 'mean'),
            Variance_Runtime=('run_time_seconds', 'var'),
            Std_Dev_Runtime=('run_time_seconds', 'std'),

            # --- New Attributes (Averages) ---
            Avg_Enum_Time=('enumeration_time_sec', 'mean'),
            Avg_Iter_Time=('iteration_time_sec', 'mean'),
            Avg_Num_Subgroups=('num_subgroups', 'mean')  # Added this
        ).reset_index()

        # Fill NaN values for Std/Var (happens if 1 run) and new cols if applicable
        cols_to_fill = ['Std_Dev_Runtime', 'Variance_Runtime',
                        'Avg_Enum_Time', 'Avg_Iter_Time', 'Avg_Num_Subgroups']
        summary_df[cols_to_fill] = summary_df[cols_to_fill].fillna(0)

        # Round for cleaner output
        round_cols = ['Avg_Runtime_Sec', 'Variance_Runtime', 'Std_Dev_Runtime',
                      'Avg_Enum_Time', 'Avg_Iter_Time', 'Avg_Num_Subgroups']

        for col in round_cols:
            summary_df[col] = summary_df[col].round(4)

        # Sort for better readability: Delta -> Epsilon -> Avg Runtime (Ascending)
        summary_df = summary_df.sort_values(['delta', 'epsilon', 'Avg_Runtime_Sec'])

        # 3. Write to Excel (Single Sheet)
        print(f"Writing results to {output_file}...")

        # Using simple to_excel since we want everything in one sheet
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