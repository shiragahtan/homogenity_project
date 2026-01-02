import pandas as pd
import re
import os
import glob
import ast
import json
from collections import defaultdict, Counter

# --- Configuration ---
# Directory where your source XLSX files are located
TARGET_DIRECTORY = '../algorithms_results'
# Directory where the final summary Excel files will be saved
OUTPUT_DIR = './breaking_subgroups_by_rule'

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

# Change this variable to switch datasets: "german_credit" OR "stackoverflow"
CHOSEN_DS = config["CHOSEN_DATASET"]

if CHOSEN_DS not in config['DATASETS']:
    raise ValueError(f"Dataset '{CHOSEN_DS}' not found in config.json")

ds_config = config['DATASETS'][CHOSEN_DS]

EPSILON_RANGE = ds_config['EPSILONS']

# Regex to match the files and extract delta and index (the rule number)
# Captures: 1: delta value, 2: index value (Rule ID)
FILE_PATTERN = re.compile(r'_delta_(\d+)_(\d+)\.xlsx$', re.IGNORECASE)


# --- Helper Functions ---

def load_rules_metadata(rules_filename):
    """
    Reads the rules file (JSON lines format) and returns a list/dict of metadata.
    Line 0 corresponds to Rule Index 0, Line 1 to Rule Index 1, etc.
    """
    path = os.path.join('../algorithms', rules_filename)
    rules_meta = {}

    print(f"--- Loading Rules Metadata from: {path} ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                data = json.loads(line)
                rules_meta[idx] = {
                    'utility': data.get('utility', 0),
                    'coverage': data.get('coverage', 0)
                }
        print(f"  > Loaded metadata for {len(rules_meta)} rules.")
    except FileNotFoundError:
        print(f"  ! Warning: Rules file not found at {path}. Utility/Coverage will be N/A.")
    except Exception as e:
        print(f"  ! Error reading rules file: {e}")

    return rules_meta


def parse_attribute_values(attrval_str):
    """
    Safely parse the string representation of a dictionary that contains
    attribute-value pairs and return only the attribute keys.
    """
    attrval_str = attrval_str.replace('np.int64', 'int')
    attrval_str = attrval_str.replace('np.float64', 'float')

    try:
        data_dict = ast.literal_eval(attrval_str)
        return list(data_dict.keys())
    except Exception:
        keys = re.findall(r"'([^']+)'\s*:", attrval_str)
        return keys


def group_files_by_rule(target_dir):
    """Scans directory for .xlsx files and groups valid files by the rule index."""
    files_by_rule = defaultdict(list)
    search_pattern = os.path.join(target_dir, '*.xlsx')
    file_paths = glob.glob(search_pattern)

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename.startswith('~$'):
            continue

        match = FILE_PATTERN.search(filename)
        if match:
            delta = match.group(1)
            index = match.group(2)  # Rule Identifier
            files_by_rule[index].append({'path': file_path, 'delta': delta, 'index': index})

    return files_by_rule


def process_rule_group(index, file_list, rules_meta):
    """
    Processes files for a single rule and prepares summary stats.
    """
    rule_attribute_counts = Counter()
    rule_summary_rows = []
    rule_breaking_groups_data = {}

    # Retrieve external metadata for this rule
    rule_idx_int = int(index)
    meta = rules_meta.get(rule_idx_int, {'utility': 0, 'coverage': 0})

    # Format metadata for display
    # Utility with commas (e.g., 21,460.43)
    meta_utility = f"{meta['utility']:,.2f}"
    # Coverage as percentage (e.g., 84%)
    meta_coverage = f"{meta['coverage']:.0f}%"  # Rounded to nearest int based on your image, or use .2f%

    print(f"\n--- Processing Rule Index: {index} (mapped to 'rule {rule_idx_int + 1}') ---")

    for file_info in file_list:
        file_path = file_info['path']
        delta = file_info['delta']
        filename = os.path.basename(file_path)

        try:
            df = pd.read_excel(file_path, sheet_name='Subgroups', engine='openpyxl')
        except Exception as e:
            print(f"  ! Error reading {filename}: {e}")
            continue

        required_cols = ['AttributeValues', 'UtilityDiff']
        if not all(col in df.columns for col in required_cols):
            continue

        df['UtilityDiff'] = pd.to_numeric(df['UtilityDiff'], errors='coerce')
        df.dropna(subset=['UtilityDiff'], inplace=True)
        num_all_subgroups = len(df)

        if num_all_subgroups == 0:
            continue

        for epsilon in EPSILON_RANGE:
            epsilon_float = float(epsilon)
            df_breaking = df[df['UtilityDiff'].abs() > epsilon_float].copy()
            num_breaking_subgroups = len(df_breaking)

            percentage = (num_breaking_subgroups / num_all_subgroups) * 100 if num_all_subgroups > 0 else 0

            # --- Prepare Summary Row (Matches your requested format) ---
            rule_summary_rows.append({
                'sort_index': rule_idx_int,  # Hidden column for sorting later
                'rule': f"rule {rule_idx_int + 1}",  # "rule 1", "rule 2"...
                'Prevelance': f"{percentage:.2f}%",
                'coverage': meta_coverage,
                'utility': meta_utility,
                'num_groups': num_all_subgroups,
                'num_breaking': num_breaking_subgroups,
                # Keep these for internal file generation if needed, but Master will use above
                'Delta': delta,
                'Epsilon': epsilon
            })

            # --- Collect Detail Data ---
            if not df_breaking.empty:
                breaking_list = []
                for _, row in df_breaking.iterrows():
                    attrval = row['AttributeValues']
                    keys = parse_attribute_values(attrval)
                    for key in keys:
                        rule_attribute_counts[key] += 1

                    breaking_list.append({
                        'Epsilon': epsilon,
                        'AttributeValues': attrval,
                        'Size': row.get('Size', 'N/A'),
                        'Utility': row.get('Utility', 'N/A'),
                        'UtilityDiff': row['UtilityDiff'],
                        'Attributes': ', '.join(keys)
                    })

                key = (delta, epsilon)
                if key not in rule_breaking_groups_data:
                    rule_breaking_groups_data[key] = []
                rule_breaking_groups_data[key].extend(breaking_list)

    # --- Write Individual Rule Output ---
    if rule_summary_rows:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f'rule_breaking_summary_index_{index}.xlsx')

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Sheet 1: Attribute Counts
                if rule_attribute_counts:
                    pd.DataFrame(rule_attribute_counts.items(), columns=['Attribute', 'Count']) \
                        .sort_values('Count', ascending=False) \
                        .to_excel(writer, sheet_name='1_Attribute_Counts', index=False)

                # Sheet 2: Summary
                pd.DataFrame(rule_summary_rows).to_excel(writer, sheet_name='2_Summary', index=False)

                # Sheet 3+: Details
                for (delta, epsilon), groups in sorted(rule_breaking_groups_data.items()):
                    pd.DataFrame(groups).sort_values('UtilityDiff', ascending=False) \
                        .to_excel(writer, sheet_name=f'delta_{delta}_eps_{epsilon}', index=False)
        except Exception as e:
            print(f"  ! Error writing individual file: {e}")

    return rule_summary_rows


# --- Main Execution ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Rules Metadata (Utility/Coverage)
    rules_meta = load_rules_metadata(ds_config['RULES_FILE'])

    print("Starting file discovery...")
    files_by_rule = group_files_by_rule(TARGET_DIRECTORY)

    if not files_by_rule:
        print(f"No valid files found in {TARGET_DIRECTORY}.")
        return

    all_rules_summary_data = []

    # 2. Process each rule
    for index, file_list in files_by_rule.items():
        rule_data = process_rule_group(index, file_list, rules_meta)
        if rule_data:
            all_rules_summary_data.extend(rule_data)

    # 3. Generate Master Summary File
    if all_rules_summary_data:
        print("\n--- Generating Master Summary File ---")
        master_output_path = os.path.join(OUTPUT_DIR, 'master_summary_all_rules.xlsx')

        try:
            master_df = pd.DataFrame(all_rules_summary_data)

            # Sort by the hidden integer index first
            master_df = master_df.sort_values(by=['sort_index', 'Delta', 'Epsilon'])

            # Select and Order columns EXACTLY as requested in the image
            # Note: The image didn't show Delta/Epsilon, so they are excluded here to match the look.
            # If you need them to differentiate rows, add them back to this list.
            cols_order = ['rule', 'Prevelance', 'coverage', 'utility', 'num_groups', 'num_breaking']

            master_df = master_df[cols_order]

            master_df.to_excel(master_output_path, index=False)
            print(f"Success! Master summary saved to: {master_output_path}")

        except Exception as e:
            print(f"Error creating master summary file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No data available.")

    print("\nProcessing complete.")


if __name__ == '__main__':
    main()