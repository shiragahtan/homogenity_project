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

        # Ensure necessary columns exist
        required_columns = ['algorithm', 'delta', 'epsilon', 'run_time_seconds']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Input file is missing one of the required columns: {required_columns}")
            return

        # 2. Group by Algorithm, Delta, and Epsilon
        print("Calculating averages, variance, and standard deviation...")

        # We use .agg() to calculate mean, variance, and standard deviation
        summary_df = df.groupby(['algorithm', 'delta', 'epsilon']).agg(
            Runs_Count=('run_time_seconds', 'count'),
            Avg_Runtime_Sec=('run_time_seconds', 'mean'),
            Variance_Runtime=('run_time_seconds', 'var'),
            Std_Dev_Runtime=('run_time_seconds', 'std')
        ).reset_index()

        # Fill NaN values (happens if an algorithm only has 1 run, std dev is undefined)
        summary_df['Std_Dev_Runtime'] = summary_df['Std_Dev_Runtime'].fillna(0)
        summary_df['Variance_Runtime'] = summary_df['Variance_Runtime'].fillna(0)

        # Round for cleaner output
        summary_df['Avg_Runtime_Sec'] = summary_df['Avg_Runtime_Sec'].round(4)
        summary_df['Variance_Runtime'] = summary_df['Variance_Runtime'].round(4)
        summary_df['Std_Dev_Runtime'] = summary_df['Std_Dev_Runtime'].round(4)

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


if __name__ == "__main__":
    summarize_runtimes()