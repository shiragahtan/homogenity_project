import pandas as pd
import json

# --- 1. Load your numerical dataset ---
try:
    # Load the dataset
    df = pd.read_csv('../stackoverflow/so_countries_col_new.csv')
except FileNotFoundError:
    print("Error: Dataset not found. Please make sure '../stackoverflow/so_countries_col_new.csv' exists.")
    exit(1)

# --- 2. Load treatments from the JSON file ---
treatments_file = 'Shira_Treatments.json'
treatments_data = []
try:
    with open(treatments_file, 'r') as f:
        for line in f:
            # Load the entire treatment data object
            treatments_data.append(json.loads(line))
except FileNotFoundError:
    print(f"Error: Treatments file '{treatments_file}' not found.")
    exit(1)

# --- 3. Process each treatment and create corresponding files ---
for i, treatment_data in enumerate(treatments_data):
    # Get the condition and treatment
    condition = treatment_data["condition"]
    attr, val = list(condition.items())[0]
    treatment = treatment_data["treatment"]

    # Create a fresh copy of the original DataFrame for each iteration
    df_g = pd.read_csv('../stackoverflow/so_countries_col_new.csv')
    df_filtered = (pd.read_csv('../stackoverflow/so_countries_col_new.csv')
          .query(f'{attr} == "{val}"')
          .loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
          .drop(columns=[f'{attr}'])  # Remove the filter column since it now contains only one value
          .loc[lambda d: ~d.isin(["UNKNOWN"]).any(axis=1)]  # Remove rows with "UNKNOWN" in any column
          .reset_index(drop=True))

    print(f"Treatment {i+1}: Filtered on condition: {attr} == {val}")
    print(f"Treatment {i+1}: Found {len(df_filtered)} rows matching condition out of {len(df_g)} total rows")

    if len(df_filtered) == 0:
        print(f"Warning: No rows match condition for treatment {i+1}. Using full dataset.")
        df_filtered = df_g

    # Get treatment keys and values
    keys = list(treatment.keys())
    vals = list(treatment.values())

    # Identify rows that match all treatment conditions
    mask = (df_filtered[keys] == vals).all(axis=1)

    # For each treatment column, override its values with binary (1 or 0)
    for key in keys:
        # Set values to 1 for rows that match the treatment, 0 for others
        df_filtered[key] = 0
        df_filtered.loc[mask, key] = 1

    # Save the result to a CSV file with the same naming convention as before
    output_file_path = f'../stackoverflow/so_countries_encoded_treatment_{i+1}.csv'
    df_filtered.to_csv(output_file_path, index=False)

    print(f"Treatment {i+1} applied and saved to {output_file_path}")
    print(f"Treatment conditions: {treatment}")
    print(f"Number of rows matching the treatment: {mask.sum()} out of {len(df_filtered)}")
    print("-" * 50)
