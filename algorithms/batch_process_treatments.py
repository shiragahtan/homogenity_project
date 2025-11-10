import pandas as pd
from pathlib import Path
import os
import sys
import json
from typing import Dict, Any

# --- Configuration ---
DATASET = '../stackoverflow/so_countries_col_new.csv'
JSON_FILE = 'Chosen10Treatments.json'
TREATMENT_COL = 'TempTreatment'  # Name for the binary treatment column
OUTCOME_COL = 'ConvertedSalary'  # Target outcome column
OUTPUT_DIR_NAME = 'processed_db'

# Add project paths for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

# --- Import the ATE calculation function ---
from ATE_update import calculate_ate_safe


# --- Helper Functions (Loading & Encoding) ---

def load_and_clean_dataframe(dataset_path: str) -> pd.DataFrame:
    """
    Load the main DataFrame, remove 'Unnamed' columns, and filter out 'UNKNOWN' rows.
    """
    print(f"Loading and cleaning dataset: {dataset_path}")
    try:
        df_original = pd.read_csv(dataset_path)
        # Remove columns starting with 'Unnamed'
        df_original = df_original.loc[:, ~df_original.columns.str.startswith('Unnamed')]
        # Filter out rows containing 'UNKNOWN' string
        df_original = df_original[~df_original.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
        return df_original
    except FileNotFoundError:
        print(f"Error: DATASET not found at '{dataset_path}'. Please check the path.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading or cleaning data: {e}")
        sys.exit(1)


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a DataFrame with categorical columns into a fully numerical one
    using a consistent 1-based mapping for each column's unique values.
    """
    df_encoded = df.copy()

    # Identify categorical columns (object dtype)
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()

    for column in categorical_columns:
        # Get unique values (this will include NaN if present)
        unique_values = df_encoded[column].unique()

        # FIX: Changed 'value' to 'val' to match the loop variable (idx, val)
        column_mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}

        # Apply the mapping
        df_encoded[column] = df_encoded[column].map(column_mapping)

    # Ensure boolean columns are 0/1 (int)
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded


# --- Main Processing Function ---

def create_and_save_treated_dataset(
        df_original: pd.DataFrame,
        condition_attr: str,
        condition_val: Any,
        treatment_attr: str,
        treatment_val: Any,
        output_index: int,
        output_dir: Path,
        treatment_col_name: str = TREATMENT_COL,
        outcome_col_name: str = OUTCOME_COL
):
    """
    Filters the base dataset by condition, applies the treatment, DROPS the
    invariant condition column, encodes the DataFrame, calculates the ATE, and saves it.
    """

    total_rows = len(df_original)
    condition_str = f"`{condition_attr}` == {repr(condition_val)}"
    treatment_str = f"`{treatment_attr}` == {repr(treatment_val)}"
    print(f"--- Processing Combo #{output_index}: Condition({condition_str}) + Treatment({treatment_str}) ---")

    # 1. Filter DataFrame based on condition
    try:
        df_filtered = df_original.query(f"`{condition_attr}` == @condition_val").copy()
    except Exception as e:
        print(f"Error querying: {condition_attr} == {condition_val}. Error: {e}. Skipping.")
        return

    count = len(df_filtered)
    if count == 0:
        print(f"No rows match the condition {condition_str}. Skipping save.")
        return

    coverage_pct = (count / total_rows) * 100
    print(f"  ...Coverage: {count}/{total_rows} ({coverage_pct:.2f}%)")

    # 2. Drop the invariant condition column
    if condition_attr in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=[condition_attr])
        print(f"  ...Dropped invariant condition column: '{condition_attr}'")
    else:
        print(f"  Warning: Condition column '{condition_attr}' not found after filtering.")

    # 3. Apply Treatment (Create the binary 'TempTreatment' column)
    if treatment_attr not in df_filtered.columns:
        if treatment_attr == condition_attr:
            print(
                f"  Error: Treatment column '{treatment_attr}' was dropped as the condition column. Cannot apply treatment.")
        else:
            print(f"  Error: Treatment attribute '{treatment_attr}' not found in filtered data. Skipping.")
        return

    df_filtered[treatment_col_name] = (df_filtered[treatment_attr] == treatment_val).astype(int)

    # Check for non-zero treatment applications
    if df_filtered[treatment_col_name].sum() == 0:
        print(f"Warning: Treatment '{treatment_str}' resulted in zero treated individuals. Skipping save.")
        return

    # 4. Encode the filtered DataFrame
    print("  ...Encoding DataFrame...")
    df_encoded = encode_dataframe(df_filtered)

    # Ensure outcome column is numeric
    df_encoded[outcome_col_name] = pd.to_numeric(df_encoded[outcome_col_name], errors='coerce')

    # --- 5. Calculate ATE for the entire filtered subset (ATE_all for this subgroup) ---
    cate_value_for_subgroup = calculate_ate_safe(
        df=df_encoded,
        treatment_col=treatment_col_name,
        outcome_col=outcome_col_name
    )
    print(f"  ...Calculated ATE_all for this subgroup: {cate_value_for_subgroup:.4f}")

    # 6. Save the encoded DataFrame
    output_filename = f"so_countries_treatment_{output_index}_encoded.csv"
    output_path = output_dir / output_filename

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(output_path, index=False)

    print(f"  ✅ Successfully saved: {output_path}")
    print("-" * 70)


# --- Main Execution ---

if __name__ == "__main__":

    # 1. Calculate the absolute path for the output directory
    base_data_dir = Path(DATASET).parent.resolve()
    final_output_dir = base_data_dir / OUTPUT_DIR_NAME
    print(f"All processed databases will be saved to: {final_output_dir}")

    # 2. Load and Clean Main DataFrame ONCE
    df_original = load_and_clean_dataframe(DATASET)
    if df_original.empty:
        print("Initial DataFrame is empty. Exiting.")
        sys.exit(1)

    # 3. Read and Parse JSON Treatments
    print(f"Reading treatment conditions from {JSON_FILE}...")
    try:
        treatments_list = []
        with open(JSON_FILE, 'r') as f:
            for line in f:
                treatments_list.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: JSON file '{JSON_FILE}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON line: {e}")
        sys.exit(1)

    print(f"Found {len(treatments_list)} condition/treatment pairs to process.")

    # 4. Iterate and Process
    for i, item in enumerate(treatments_list):
        output_index = i + 1

        condition_dict: Dict[str, Any] = item["condition"]
        condition_attr, condition_val = list(condition_dict.items())[0]

        treatment_dict: Dict[str, Any] = item["treatment"]
        treatment_attr, treatment_val = list(treatment_dict.items())[0]

        # Call the processing function
        create_and_save_treated_dataset(
            df_original=df_original,
            condition_attr=condition_attr,
            condition_val=condition_val,
            treatment_attr=treatment_attr,
            treatment_val=treatment_val,
            output_index=output_index,
            output_dir=final_output_dir,
            outcome_col_name=OUTCOME_COL
        )

    print("\n--- Batch processing complete. ---")