import json
import pandas as pd
import sys
import itertools
import math
from pathlib import Path

# --- Configuration ---
CONFIG_PATH = '../configs/config.json'


def calculate_level_metrics(df, feature_cols, level, delta):
    """
    Calculates:
      1. Theoretical Bound (Renamed): Count of ALL unique subgroups (size >= 1) found in data.
      2. Actual Bound (Renamed): Count of subgroups with size >= DELTA.
      3. Max Size: The population of the largest subgroup found at this level.
    """
    count_theoretical = 0  # Size >= 1
    count_actual = 0  # Size >= Delta
    max_subgroup_size = 0

    # Iterate over all combinations of columns for this level
    # e.g., if level=2: (Age, Sex), (Age, Education)...
    for cols in itertools.combinations(feature_cols, level):
        # value_counts() gives us the size of every unique combination found
        counts = df[list(cols)].value_counts()

        # 1. Theoretical: How many unique patterns exist (even if size is 1)?
        count_theoretical += len(counts)

        # 2. Actual: How many patterns meet the Delta threshold?
        count_actual += (counts >= delta).sum()

        # 3. Max Size
        if not counts.empty:
            current_max = counts.max()
            if current_max > max_subgroup_size:
                max_subgroup_size = current_max

    return count_theoretical, count_actual, max_subgroup_size


def main():
    print("--- Subgroup Statistics Calculator (Delta-Constrained) ---")

    # 1. Load Config
    if Path(CONFIG_PATH).exists():
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
    else:
        # Fallback for current directory
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except:
            print(f"Error: Config not found at {CONFIG_PATH}")
            return

    # 2. Setup Paths & Params
    try:
        chosen_ds = config.get("CHOSEN_DATASET")
        ds_config = config['DATASETS'][chosen_ds]
        dataset_path = ds_config['FULL_DATASET_PATH']
        rules_file = ds_config['RULES_FILE']
        target_col = ds_config['TARGET_COLUMN']

        # Get Delta
        deltas = ds_config.get('DELTAS', [1000])  # Default to 1000 if missing
        delta = deltas[0]
        print(f"🔹 Dataset: {chosen_ds}")
        print(f"🔹 Delta Threshold: {delta}")

    except KeyError as e:
        print(f"Configuration error: Missing key {e}")
        return

    # 3. Load Data
    print(f"Loading Dataset: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        # Standard cleaning
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        obj_cols = df.select_dtypes(include=['object']).columns
        if not obj_cols.empty:
            df = df[~df[obj_cols].isin(["UNKNOWN"]).any(axis=1)]
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 4. Load Rules
    rules_path = Path(rules_file)
    if not rules_path.exists():
        rules_path = Path('../algorithms') / rules_file

    if not rules_path.exists():
        rules_path = Path(rules_file)

    if not rules_path.exists():
        print(f"Error: Rules file not found ({rules_file})")
        return

    with open(rules_path, 'r') as f:
        content = f.read().strip()
        if content.startswith('['):
            try:
                rules = json.loads(content)
            except:
                print("JSON Decode Error")
                return
        else:
            rules = [json.loads(line) for line in content.split('\n') if line.strip()]

    print(f"\nProcessing {len(rules)} Rules...\n" + "=" * 90)

    # 5. Process Rules
    for i, rule in enumerate(rules):
        cond_dict = rule['condition']
        treat_dict = rule['treatment']
        cond_col, cond_val = list(cond_dict.items())[0]
        treat_col, _ = list(treat_dict.items())[0]

        print(f"RULE #{i + 1}: IF {cond_col}={cond_val} THEN TREATMENT {treat_col}")

        # A. Filter Population
        if cond_col not in df.columns:
            print(f"  [Skipped] Column {cond_col} missing.")
            continue

        mask = df[cond_col] == cond_val
        if mask.sum() == 0:
            mask = df[cond_col].astype(str) == str(cond_val)

        sub_df = df[mask].copy()

        # B. Define Mining Features
        exclude = {cond_col, treat_col, target_col, 'TempTreatment'}
        feature_cols = [c for c in sub_df.columns if c not in exclude]

        # C. Get Unique Counts
        unique_counts_dict = {}
        feature_details = []
        for col in feature_cols:
            k = sub_df[col].nunique()
            if k > 1:
                unique_counts_dict[col] = k
                feature_details.append(f"{col}({k})")

        active_features = list(unique_counts_dict.keys())

        if not active_features:
            print("  0 Subgroups possible (No variable features).\n")
            continue

        print(f"  Population: {len(sub_df):,}")
        print(f"  Features ({len(active_features)}): {', '.join(feature_details)}")
        print("-" * 90)

        # D. Calculate Stats Per Level
        grand_theo = 0
        grand_actual = 0

        # Header
        print(f"  {'Lvl':<4} | {'Theoretical Bound':<20} | {'Actual Bound':<15} | {'Max Subgroup Size':<18}")
        print(f"  {'':<4} | {'(Size >= 1)':<20} | {'(Size >= ' + str(delta) + ')':<15} | {'':<18}")
        print(f"  {'-' * 4} | {'-' * 20} | {'-' * 15} | {'-' * 18}")

        for level in range(1, len(active_features) + 1):
            # Calculate counts
            count_theo, count_act, max_size = calculate_level_metrics(sub_df, active_features, level, delta)

            grand_theo += count_theo
            grand_actual += count_act

            print(f"  {level:<4} | {count_theo:<20,} | {count_act:<15,} | {max_size:<18,}")

        print(f"  {'-' * 4} | {'-' * 20} | {'-' * 15} | {'-' * 18}")
        print(f"  {'SUM':<4} | \033[92m{grand_theo:<20,}\033[0m | \033[94m{grand_actual:<15,}\033[0m | {'-':<18}")
        print("=" * 90 + "\n")


if __name__ == "__main__":
    main()