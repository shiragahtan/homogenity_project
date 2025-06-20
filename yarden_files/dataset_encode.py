import pandas as pd
import json
import os

input_file = os.path.join('..', 'stackoverflow', 'so_countries_col_new.csv')
output_file = os.path.join('..', 'stackoverflow', 'so_countries_col_new_encoded.csv')
mappings_file = os.path.join('..', 'stackoverflow', 'categorical_mappings.json')

# Load the dataset
df = pd.read_csv(input_file)

# Remove the unnamed column if it exists
unnamed_columns = [col for col in df.columns if 'Unnamed' in col]
if unnamed_columns:
    print(f"Removing unnamed columns: {unnamed_columns}")
    df = df.drop(columns=unnamed_columns)

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

mappings = {}

for column in categorical_columns:
    unique_values = df[column].unique()

    column_mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}

    mappings[column] = column_mapping

    df[column] = df[column].map(column_mapping)

df.to_csv(output_file, index=False)

with open(mappings_file, 'w') as f:
    json.dump(mappings, f, indent=4)

print("Transformation complete!")
print(f"Categorical columns transformed: {categorical_columns}")
print(f"Mappings saved to {mappings_file}")
print(f"Transformed dataset saved to {output_file}")
