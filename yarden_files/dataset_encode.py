import pandas as pd
import json
import os

# List of treatment files to process
treatment_files = [
    'so_countries_treatment_1.csv',
    'so_countries_treatment_2.csv',
    'so_countries_treatment_3.csv'
]
# treatment_files = [
#     'yarden_reverted_so_updated.csv',
# ]

# Process each treatment file
for treatment_file in treatment_files:
    input_file = os.path.join('..', 'stackoverflow', treatment_file)

    # Create output filename by adding "_encoded" before the extension
    base_name = os.path.splitext(treatment_file)[0]
    output_file = os.path.join('..', 'stackoverflow', f"{base_name}_encoded.csv")

    # Use a consistent mappings file for all encodings
    mappings_file = os.path.join('..', 'stackoverflow', 'categorical_mappings.json')

    print(f"Processing {treatment_file}...")

    try:
        # Load the dataset
        df = pd.read_csv(input_file)

        # Remove the unnamed column if it exists
        unnamed_columns = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_columns:
            print(f"  Removing unnamed columns: {unnamed_columns}")
            df = df.drop(columns=unnamed_columns)

        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        if not categorical_columns:
            print(f"  No categorical columns found in {treatment_file}. The file may already be encoded.")
            continue

        mappings = {}

        for column in categorical_columns:
            unique_values = df[column].unique()

            column_mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}

            mappings[column] = column_mapping

            df[column] = df[column].map(column_mapping)

        # Save the encoded dataset
        df.to_csv(output_file, index=False)

        # Save the mappings to a file
        with open(mappings_file, 'w') as f:
            json.dump(mappings, f, indent=4)

        print(f"  Transformation complete for {treatment_file}!")
        print(f"  Categorical columns transformed: {categorical_columns}")
        print(f"  Mappings saved to {mappings_file}")
        print(f"  Transformed dataset saved to {output_file}")
        print("-" * 60)

    except FileNotFoundError:
        print(f"  Error: File {input_file} not found. Skipping.")
        print("-" * 60)
        continue
