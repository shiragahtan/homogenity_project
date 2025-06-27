import pandas as pd

# Define the input and output file names
input_file = 'algorithms_time.xlsx'  
output_file = 'average_times_per_algorithm.xlsx'

try:
    # Read the Excel file into a pandas DataFrame
    # If your data is not on the first sheet, you might need to specify the sheet name
    # e.g., df = pd.read_excel(input_file, sheet_name='Sheet1')
    df = pd.read_excel(input_file)

    # Ensure 'delta' is treated as a consistent type for grouping.
    # We'll clean it by removing commas (if any) and converting to integer.
    if 'delta' in df.columns:
        df['delta'] = df['delta'].astype(str).str.replace(',', '', regex=False)
        # Convert to numeric, coercing errors will turn non-numeric to NaN
        df['delta'] = pd.to_numeric(df['delta'], errors='coerce')
        # Drop rows where delta could not be converted to a number, if any
        df.dropna(subset=['delta'], inplace=True)
        df['delta'] = df['delta'].astype(int)


    # Group by 'algorithm' and 'delta', then calculate the mean for 'time_seconds' and 'time_minutes'
    # The .reset_index() is used to turn the grouped indices back into columns
    #import ipdb;ipdb.set_trace()
    aggregated_df = df.groupby(['algorithm', 'delta'])[['run_time_seconds', 'run_time_minutes']].mean().reset_index()
    
    # Rename columns to reflect they are averages, if desired (optional)
    aggregated_df.rename(columns={
        'time_seconds': 'average_time_seconds',
        'time_minutes': 'average_time_minutes'
    }, inplace=True)

    # Save the aggregated DataFrame to a new Excel file
    # The index=False argument prevents pandas from writing the DataFrame index as a column in the Excel sheet
    aggregated_df.to_excel(output_file, index=False)

    print(f"Successfully aggregated data and saved to '{output_file}'")

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found. Please make sure the file exists in the same directory as the script, or provide the full path.")
except Exception as e:
    print(f"An error occurred: {e}")