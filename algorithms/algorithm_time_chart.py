import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the input file name (output from the previous script)
input_file = 'average_times_per_algorithm.xlsx'
# Define the output image file name for the chart
output_chart_file = 'algorithm_overall_average_performance_chart.png'

try:
    # Read the Excel file into a pandas DataFrame
    # This file should contain 'algorithm', 'delta', and 'run_time_seconds' columns
    df = pd.read_excel(input_file)

    # Check if the required columns exist
    if 'algorithm' not in df.columns or 'run_time_seconds' not in df.columns:
        print(f"Error: The input file '{input_file}' must contain 'algorithm' and 'run_time_seconds' columns.")
        exit()

    # --- Aggregate data: Group by 'algorithm' and calculate the mean of 'run_time_seconds' ---
    # This will give one average time per algorithm, across all its deltas
    overall_average_df = df.groupby('algorithm')['run_time_seconds'].mean().reset_index()
    # Rename the column to reflect it's an overall average (optional, but good for clarity)
    overall_average_df.rename(columns={'run_time_seconds': 'overall_run_time_seconds'}, inplace=True)


    # --- Create the Bar Chart ---

    # Set the figure size for better readability
    plt.figure(figsize=(12, 8))

    # Create the bar chart
    # x-axis: algorithm names
    # y-axis: overall_run_time_seconds
    plt.bar(overall_average_df['algorithm'], overall_average_df['overall_run_time_seconds'], color='lightcoral') # Changed color for distinction

    # Add titles and labels
    plt.title('Overall Average Run Time per Algorithm (Across All Deltas)', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Overall Average Time (seconds)', fontsize=14)

    # Rotate x-axis labels for better readability if there are many algorithms or long names
    plt.xticks(rotation=45, ha='right') # ha='right' aligns the rotated labels correctly

    # Add a grid for better readability of y-values
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels on top of each bar
    for index, value in enumerate(overall_average_df['overall_run_time_seconds']):
        plt.text(index, value + (overall_average_df['overall_run_time_seconds'].max() * 0.01), # Position text slightly above the bar
                 f"{value:.2f}", # Format to 2 decimal places
                 ha='center', va='bottom', fontsize=9)

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Save the chart to a file
    plt.savefig(output_chart_file)
    print(f"Successfully created bar chart with overall averages and saved to '{output_chart_file}'")

    # Display the chart (optional, comment out if running in a non-GUI environment)
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found. Please make sure the file exists in the same directory as the script, or provide the full path.")
    print("This script expects the output file from your previous aggregation script.")
except Exception as e:
    print(f"An error occurred: {e}")

