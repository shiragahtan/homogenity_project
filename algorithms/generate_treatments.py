import os
import sys
import json
import pandas as pd
from pathlib import Path
from itertools import product
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))
from ATE_update import calculate_ate_safe

# --- Configuration ---
DATASET = '../german_credit/german_data_encoded.csv'

OUTCOME_COL = 'credit_risk'  # 'tgtO'
TREATMENT_COL = 'TempTreatment'  # This is the 'treatment_col' we will create

# Attribute lists
immutable_attributes = [
    "age",
    "gender",
    "personal_status",
    "foreign_worker",
    "credit_history",
    "employment_duration",
    "property",
    "installment_rate",
    "purpose",
    "people_liable",
    "duration",
    "amount"
]
mutable_attrs = [
    "status",
    "savings",
    "other_debtors",
    "present_residence",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
]

# --- Helper Functions ---

def get_unique_values(df, attributes):
    """
    Get all unique values for the specified attributes in the DataFrame.
    """
    unique_values = {}
    for attr in attributes:
        if attr in df.columns:
            unique_values[attr] = df[attr].dropna().unique().tolist()
    return unique_values


def encode_dataframe(df):
    """
    Converts a DataFrame with categorical columns into a fully numerical one
    using your project's consistent 1-based mapping.
    """
    df_encoded = df.copy()

    # Identify categorical columns
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()

    for column in categorical_columns:
        # Get unique values
        unique_values = df_encoded[column].unique()

        # Create your 1-based mapping: {value: 1, value2: 2, ...}
        column_mapping = {value: idx + 1 for idx, value in enumerate(unique_values)}

        # Apply the mapping
        df_encoded[column] = df_encoded[column].map(column_mapping)

    # Ensure boolean columns are 0/1
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded


# --- 4. Main Execution ---

# Load and Clean Main DataFrame ONCE
print(f"Loading and cleaning dataset: {DATASET}")
try:
    df_original = pd.read_csv(DATASET)
    df_original = df_original.loc[:, ~df_original.columns.str.startswith('Unnamed')]
    df_original = df_original[~df_original.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
except FileNotFoundError:
    print(f"Error: DATASET not found. Please make sure '{DATASET}' exists.")
    exit(1)
except Exception as e:
    print(f"Error loading or cleaning data: {e}")
    exit(1)

total_rows = len(df_original)
if total_rows == 0:
    print("DataFrame is empty after cleaning. Exiting.")
    exit(1)

# Ensure the outcome column exists
if OUTCOME_COL not in df_original.columns:
    print(f"Error: Outcome column '{OUTCOME_COL}' not found in the dataset.")
    exit(1)

print(f"Loaded {total_rows} total clean rows.")

# Get all possible condition and treatment values
possible_conditions = get_unique_values(df_original, immutable_attributes)
possible_treatments = get_unique_values(df_original, mutable_attrs)

# Create flat lists of (attr, val) pairs for the Cartesian product
cond_pairs = [(attr, val) for attr, vals in possible_conditions.items() for val in vals]
treat_pairs = [(attr, val) for attr, vals in possible_treatments.items() for val in vals]

if not cond_pairs or not treat_pairs:
    print("Error: No condition or treatment pairs found. Check attributes and data.")
    exit(1)

print(f"Created {len(cond_pairs)} condition values and {len(treat_pairs)} treatment values.")
print(f"Total combinations to test: {len(cond_pairs) * len(treat_pairs)}")
print("-" * 60)

# Iterate over the Cartesian Product
results = []
for (c_attr, c_val), (t_attr, t_val) in product(cond_pairs, treat_pairs):

    # --- a. Filter DataFrame based on condition ---
    try:
        df_filtered = df_original.query(f"`{c_attr}` == @c_val").copy()
    except Exception as e:
        print(f"Error querying: {c_attr} == {c_val}. Error: {e}. Skipping.")
        continue

    # --- b. Calculate Coverage ---
    count = len(df_filtered)
    if count == 0:
        continue

    coverage_pct = (count / total_rows) * 100

    # [CHANGE] Filter strict coverage > 70% here to avoid unnecessary computation
    if coverage_pct <= 70:
        continue

    # --- c. Apply Treatment ---
    # Create the binary 'TempTreatment' column
    df_filtered[TREATMENT_COL] = (df_filtered[t_attr] == t_val).astype(int)

    # --- d. Encode the filtered DataFrame ---
    df_encoded = encode_dataframe(df_filtered)

    # Ensure outcome column is numeric
    df_encoded[OUTCOME_COL] = pd.to_numeric(df_encoded[OUTCOME_COL], errors='coerce')

    # --- e. Calculate CATE using YOUR imported function ---
    cate_value = calculate_ate_safe(
        df=df_encoded,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL
    )

    # --- f. Print and store results ---
    condition_str = f"{c_attr}:{repr(c_val)}"
    treatment_str = f"{t_attr}:{repr(t_val)}"
    print(f"Combo: {condition_str} + {treatment_str}")
    print(f"  ...Coverage: {count}/{total_rows} ({coverage_pct:.2f}%)")
    print(f"  ...CATE: {cate_value:.4f}")

    results.append({
        "condition_attr": c_attr,
        "condition_val": c_val,
        "treatment_attr": t_attr,
        "treatment_val": t_val,
        "coverage_pct": coverage_pct,
        "cate_value": cate_value
    })

# --- 5. Save all results ---
if results:
    results_df = pd.DataFrame(results)

    # [CHANGE] Filter for positive utility (cate_value > 0)
    results_df = results_df[results_df['cate_value'] > 0]

    # [CHANGE] Sort by utility (cate_value) in decreasing order
    results_df = results_df.sort_values(by='cate_value', ascending=False)

    output_filename = "german_high_coverage_positive_utility.csv"
    results_df.to_csv(output_filename, index=False)
    print("-" * 60)
    print("All combinations processed.")
    print(f"Filtered (Cov > 70%, Utility > 0) and Sorted results saved to '{output_filename}'")
else:
    print("-" * 60)
    print("No combinations met the > 70% coverage criteria.")