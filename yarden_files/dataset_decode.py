import pandas as pd
import json
import os


def decode_dataset(encoded_file, mappings_file, output_file):
    try:
        # Load encoded dataset
        df = pd.read_csv(encoded_file)

        # Load mappings
        with open(mappings_file, 'r') as f:
            mappings = json.load(f)

        # Invert the mappings
        inverse_mappings = {
            column: {int(v): k for k, v in mapping.items()}
            for column, mapping in mappings.items()
        }

        # Apply inverse mappings to the appropriate columns
        for column, inv_map in inverse_mappings.items():
            if column in df.columns:
                df[column] = df[column].map(inv_map)

        # Save decoded dataset
        df.to_csv(output_file, index=False)
        print(f"Decoded dataset saved to {output_file}")

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    decode_dataset(
        encoded_file='stackoverflow_data_encoded.csv',
        mappings_file='yarden_categorical_mappings.json',
        output_file='../stackoverflow/yarden_reverted_so_decoded_original.csv'
    )
