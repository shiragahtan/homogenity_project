import pandas as pd
import json

# --- 1. Load your numerical dataset ---
try:
    # Load the encoded DataFrame
    df = pd.read_csv('../stackoverflow/so_countries_col_new.csv')
except FileNotFoundError:
    print("Error: Encoded dataset not found. Please make sure '../stackoverflow/so_countries_col_new.csv' exists.")
    exit(1)

# --- 2. Load treatments from the JSON file ---
treatments_file = 'Shira_Treatments.json'
treatments_list = []
try:
    with open(treatments_file, 'r') as f:
        for line in f:
            # We only need the 'treatment' part of each JSON object
            treatments_list.append(json.loads(line)['treatment'])
except FileNotFoundError:
    print(f"Error: Treatments file '{treatments_file}' not found.")
    exit(1)

# --- 3. Process each treatment and create corresponding files ---
for i, treatment in enumerate(treatments_list):
    # Create a fresh copy of the original DataFrame for each iteration
    df_g = df.copy()

    # Apply the treatment using the specified approach
    keys = list(treatment.keys())
    vals = list(treatment.values())

    # Identify rows that match all treatment conditions
    mask = (df_g[keys] == vals).all(axis=1)

    # For each treatment column, override its values with binary (1 or 0)
    for key in keys:
        # Set values to 1 for rows that match the treatment, 0 for others
        df_g[key] = 0
        df_g.loc[mask, key] = 1

    # Save the result to a CSV file with the same naming convention as before
    output_file_path = f'../stackoverflow/so_countries_encoded_treatment_{i+1}.csv'
    df_g.to_csv(output_file_path, index=False)

    print(f"Treatment {i+1} applied and saved to {output_file_path}")
    print(f"Treatment conditions: {treatment}")
    print(f"Number of rows matching the treatment: {mask.sum()} out of {len(df_g)}")
    print("-" * 50)
