import os
import sys
import json
import pandas as pd
from pathlib import Path
from itertools import product

# --- ANSI Colors for Console Output ---
GREEN = '\033[92m'
RESET = '\033[0m'

# --- Path Setup ---
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

# Try importing the ATE function
try:
    from ATE_update import calculate_ate_safe
except ImportError:
    print("Error: Could not import 'calculate_ate_safe'. Please check your directory structure and sys.path.")
    sys.exit(1)

# --- Configuration Load ---
CONFIG_PATH = '../configs/config.json'

print(f"Loading configuration from {CONFIG_PATH}...")
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Config file not found at {CONFIG_PATH}")
    sys.exit(1)

# 1. Determine which dataset is active
CHOSEN_DS = config.get("CHOSEN_DATASET")
if not CHOSEN_DS or CHOSEN_DS not in config.get('DATASETS', {}):
    print(f"Error: Dataset '{CHOSEN_DS}' not found in 'DATASETS' block of config.")
    sys.exit(1)

ds_config = config['DATASETS'][CHOSEN_DS]

# 2. Extract Variables
DATASET_PATH = ds_config['FULL_DATASET_PATH']
OUTCOME_COL = ds_config['OUTCOME_COL']
IMMUTABLE_ATTRS = ds_config['IMMUTABLE_ATTRIBUTES']
MUTABLE_ATTRS = ds_config['MUTABLE_ATTRIBUTES']

# 3. Extract Logic Flags & Thresholds
COVERAGE_THRESHOLD = ds_config.get('COVERAGE_THRESHOLD', 10)
USE_ENCODING = ds_config.get('USE_ENCODING', False)
TREATMENT_COL_NAME = config.get('TREATMENT_COL', 'TempTreatment')

print(f"🔹 Running Treatment Generation for: {CHOSEN_DS}")
print(f"   Input File: {DATASET_PATH}")
print(f"   Outcome Column: {OUTCOME_COL}")
print(f"   Coverage Threshold: {COVERAGE_THRESHOLD}%")
print(f"   Encoding Enabled: {USE_ENCODING}")
print("-" * 60)


# --- Helper Functions ---

def get_unique_values(df, attributes):
    """
    Get all unique values for the specified attributes in the DataFrame.
    """
    unique_values = {}
    for attr in attributes:
        if attr in df.columns:
            # Dropna ensures we don't try to query NaNs
            unique_values[attr] = df[attr].dropna().unique().tolist()
        else:
            print(f"Warning: Attribute '{attr}' configured but not found in dataset columns.")
    return unique_values


def encode_dataframe(df):
    """
    Converts a DataFrame with categorical columns into a fully numerical one.
    """
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()

    for column in categorical_columns:
        unique_values = df_encoded[column].unique()
        column_mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}
        df_encoded[column] = df_encoded[column].map(column_mapping)

    bool_columns = df_encoded.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded


# --- Main Execution ---

# 1. Load and Clean Data
print(f"Reading dataset...")
try:
    df_original = pd.read_csv(DATASET_PATH)
    df_original = df_original.loc[:, ~df_original.columns.str.startswith('Unnamed')]

    obj_cols = df_original.select_dtypes(include=['object']).columns
    if not obj_cols.empty:
        df_original = df_original[~df_original[obj_cols].isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)

except FileNotFoundError:
    print(f"Error: Dataset not found at {DATASET_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

total_rows = len(df_original)
if total_rows == 0:
    print("DataFrame is empty after cleaning. Exiting.")
    sys.exit(1)

if OUTCOME_COL not in df_original.columns:
    print(f"Error: Outcome column '{OUTCOME_COL}' not found in DataFrame.")
    sys.exit(1)

print(f"Loaded {total_rows} total clean rows.")

# 2. Prepare Combinations
possible_conditions = get_unique_values(df_original, IMMUTABLE_ATTRS)
possible_treatments = get_unique_values(df_original, MUTABLE_ATTRS)

cond_pairs = [(attr, val) for attr, vals in possible_conditions.items() for val in vals]
treat_pairs = [(attr, val) for attr, vals in possible_treatments.items() for val in vals]

if not cond_pairs or not treat_pairs:
    print("Error: No condition or treatment pairs found.")
    sys.exit(1)

total_combos = len(cond_pairs) * len(treat_pairs)
print(f"Total Combinations to process: {total_combos}")
print("-" * 60)

results = []

# 3. Iterate Cartesian Product
for i, ((c_attr, c_val), (t_attr, t_val)) in enumerate(product(cond_pairs, treat_pairs)):

    combo_str = f"[{c_attr}={c_val}] + [{t_attr}={t_val}]"

    # --- a. Filter DataFrame based on condition ---
    try:
        df_filtered = df_original.query(f"`{c_attr}` == @c_val").copy()
    except Exception as e:
        print(f"ERROR: {combo_str} -> Query Failed: {e}")
        continue

    # --- b. Calculate Coverage ---
    count = len(df_filtered)
    if count == 0:
        print(f"SKIPPED: {combo_str} -> Empty Subgroup (Size: 0)")
        continue

    coverage_pct = (count / total_rows) * 100

    # Strict coverage check based on Config
    if coverage_pct <= COVERAGE_THRESHOLD:
        print(f"SKIPPED: {combo_str} -> Low Coverage ({coverage_pct:.2f}% <= {COVERAGE_THRESHOLD}%)")
        continue

    # --- c. Apply Treatment ---
    df_filtered[TREATMENT_COL_NAME] = (df_filtered[t_attr] == t_val).astype(int)

    # --- d. Preprocessing ---
    if USE_ENCODING:
        df_encoded = encode_dataframe(df_filtered)
    else:
        df_encoded = df_filtered.copy()
        if t_attr in df_encoded.columns:
            df_encoded = df_encoded.drop(columns=[t_attr])

    # Ensure outcome is numeric
    df_encoded[OUTCOME_COL] = pd.to_numeric(df_encoded[OUTCOME_COL], errors='coerce')
    df_encoded = df_encoded.dropna(subset=[OUTCOME_COL])

    if df_encoded.empty:
        print(f"SKIPPED: {combo_str} -> Empty after Outcome cleaning")
        continue

    # --- e. Calculate CATE ---
    try:
        cate_value = calculate_ate_safe(
            df=df_encoded,
            treatment_col=TREATMENT_COL_NAME,
            outcome_col=OUTCOME_COL
        )

        # --- LOGGING ---
        if cate_value > 0:
            print(f"{GREEN}PROCESSED: {combo_str} -> Size: {count}, CATE: {cate_value:.4f}{RESET}")
        else:
            print(f"PROCESSED: {combo_str} -> Size: {count}, CATE: {cate_value:.4f}")

    except Exception as e:
        print(f"ERROR: {combo_str} -> ATE Calc Failed: {e}")
        cate_value = 0

    # --- f. Store Result ---
    # Store everything initially, filter later for file
    results.append({
        "condition_attr": c_attr,
        "condition_val": c_val,
        "treatment_attr": t_attr,
        "treatment_val": t_val,
        "coverage_pct": coverage_pct,
        "subgroup_size": count,
        "cate_value": cate_value
    })

# --- 4. Save Results ---
print("-" * 60)
if results:
    results_df = pd.DataFrame(results)

    # Filter: ONLY Positive Utility for final file
    final_df = results_df[results_df['cate_value'] > 0].sort_values(by='cate_value', ascending=False)

    output_filename = f"{CHOSEN_DS}_high_coverage_positive_utility.csv"
    final_df.to_csv(output_filename, index=False)

    print(f"DONE. Checked {len(results)} combinations.")
    print(f"{GREEN}Saved {len(final_df)} Positive Utility combinations to '{output_filename}'.{RESET}")
else:
    print("No results generated.")