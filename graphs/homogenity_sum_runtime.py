import os
import pandas as pd

def summarize_runtimes(input_file="homogeneity_results.xlsx", output_file="homogeneity_runtime_summary.xlsx"):
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
            average_runtime_seconds=('run_time_seconds', 'mean'),
            variance_runtime_seconds=('run_time_seconds', 'var'),
            std_dev_runtime_seconds=('run_time_seconds', 'std')  # Added Standard Deviation
        ).reset_index()

        # Fill NaN values with 0 (happens if an algorithm only has 1 run, std dev is undefined)
        summary_df['std_dev_runtime_seconds'] = summary_df['std_dev_runtime_seconds'].fillna(0)
        summary_df['variance_runtime_seconds'] = summary_df['variance_runtime_seconds'].fillna(0)

        # 3. Write to Excel with separate sheets
        print(f"Writing results to {output_file}...")

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Get unique algorithms to create sheets
            algorithms = summary_df['algorithm'].unique()

            for alg_name in algorithms:
                # Filter data for this specific algorithm
                alg_data = summary_df[summary_df['algorithm'] == alg_name]

                # Select columns (Now including Standard Deviation)
                sheet_data = alg_data[[
                    'delta', 
                    'epsilon', 
                    'average_runtime_seconds', 
                    'variance_runtime_seconds',
                    'std_dev_runtime_seconds'
                ]]

                # Excel sheet names cannot exceed 31 chars. Truncate if necessary.
                safe_sheet_name = str(alg_name)[:31]

                sheet_data.to_excel(writer, sheet_name=safe_sheet_name, index=False)
                print(f" - Created sheet for: {safe_sheet_name}")

        print("Done! Summary file created successfully.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    summarize_runtimes()