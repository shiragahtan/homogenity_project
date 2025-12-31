import pandas as pd
import re
import os
import glob
import ast
from collections import defaultdict, Counter

# --- Configuration ---
# Directory where your source XLSX files are located
TARGET_DIRECTORY = '../algorithms_results'
# Directory where the final summary Excel files will be saved
OUTPUT_DIR = './breaking_subgroups_by_rule'
# The range of epsilon values to iterate over
#EPSILON_RANGE = range(5000, 100001, 5000)
#EPSILON_RANGE = [0.1]
EPSILON_RANGE = [10000]
# Regex to match the files and extract delta and index (the rule number)
# Captures: 1: delta value, 2: index value (Rule ID)
FILE_PATTERN = re.compile(r'_delta_(\d+)_(\d+)\.xlsx$', re.IGNORECASE)


# --- Helper Functions ---

def parse_attribute_values(attrval_str):
    """
    Safely parse the string representation of a dictionary that contains
    attribute-value pairs and return only the attribute keys.
    Handles numpy type strings (e.g., np.int64).
    """
    # 1. Standardize string format by replacing numpy types for safer parsing
    attrval_str = attrval_str.replace('np.int64', 'int')
    attrval_str = attrval_str.replace('np.float64', 'float')

    try:
        # 2. Use literal_eval for safe dictionary construction
        data_dict = ast.literal_eval(attrval_str)
        return list(data_dict.keys())
    except Exception:
        # Fallback for unexpected formats
        # Extracts keys that are single words within single quotes: "{'Key1':..., 'Key2':...}"
        keys = re.findall(r"'([^']+)'\s*:", attrval_str)
        return keys


def group_files_by_rule(target_dir):
    """Scans directory for .xlsx files, filters temporary files, and groups valid files by the rule index."""
    files_by_rule = defaultdict(list)
    search_pattern = os.path.join(target_dir, '*.xlsx')
    file_paths = glob.glob(search_pattern)

    for file_path in file_paths:
        filename = os.path.basename(file_path)

        # Skip temporary Excel lock files
        if filename.startswith('~$'):
            continue

        match = FILE_PATTERN.search(filename)
        if match:
            delta = match.group(1)
            index = match.group(2)  # This is the rule identifier
            files_by_rule[index].append({'path': file_path, 'delta': delta, 'index': index})

    return files_by_rule


def process_rule_group(index, file_list):
    """
    Processes all files belonging to a single rule (index) and generates one output file.
    Returns: A list of dictionaries containing the summary stats for this rule (to be aggregated later).
    """

    # Data structures to collect results for this rule
    rule_attribute_counts = Counter()
    rule_percentage_summary = []
    rule_breaking_groups_data = {}  # Key: (delta, epsilon), Value: list of breaking subgroups (dicts)

    print(f"\n--- Starting processing for Rule (Index): {index} ---")

    for file_info in file_list:
        file_path = file_info['path']
        delta = file_info['delta']
        filename = os.path.basename(file_path)

        print(f"  > Reading file: {filename}")

        try:
            # Explicitly specify engine for reading the 'Subgroups' sheet
            df = pd.read_excel(file_path, sheet_name='Subgroups', engine='openpyxl')
        except Exception as e:
            print(f"  ! Error reading 'Subgroups' from {filename}: {e}. Skipping.")
            continue

        # Basic data validation and cleaning
        required_cols = ['AttributeValues', 'UtilityDiff']
        if not all(col in df.columns for col in required_cols):
            print(f"  ! Skipping {filename}: missing required columns ({required_cols}).")
            continue

        df['UtilityDiff'] = pd.to_numeric(df['UtilityDiff'], errors='coerce')
        df.dropna(subset=['UtilityDiff'], inplace=True)
        num_all_subgroups = len(df)

        if num_all_subgroups == 0:
            print(f"  ! Skipping {filename}: 'Subgroups' sheet is empty after cleaning.")
            continue

        # Process for each epsilon
        for epsilon in EPSILON_RANGE:
            epsilon_float = float(epsilon)

            # Filter rows: abs(UtilityDiff) > epsilon
            df_breaking = df[df['UtilityDiff'].abs() > epsilon_float].copy()
            num_breaking_subgroups = len(df_breaking)

            # --- Collect data for Sheet 2 (Percentage Summary) ---
            percentage = (num_breaking_subgroups / num_all_subgroups) * 100 if num_all_subgroups > 0 else 0

            # Store summary data (We add Rule Index here so we can identify it in the master file later)
            rule_percentage_summary.append({
                'Rule Index': int(index),
                'Delta': int(delta),
                'Epsilon': epsilon,
                'Num All Subgroups': num_all_subgroups,
                'Num Breaking Subgroups': num_breaking_subgroups,
                'Percentage Breaking': f"{percentage:.2f}%"
            })

            # --- Collect data for Sheet 1 (Attribute Counts) & Sheet 3+ (Breaking Groups Data) ---
            if not df_breaking.empty:
                breaking_list = []

                for _, row in df_breaking.iterrows():
                    attrval = row['AttributeValues']
                    keys = parse_attribute_values(attrval)

                    # Accumulate counts for Sheet 1
                    for key in keys:
                        rule_attribute_counts[key] += 1

                    # Prepare data for Sheet 3+
                    breaking_list.append({
                        'Epsilon': epsilon,
                        'AttributeValues': attrval,
                        # Use .get() for optional columns
                        'Size': row.get('Size', 'N/A'),
                        'Utility': row.get('Utility', 'N/A'),
                        'UtilityDiff': row['UtilityDiff'],
                        'Attributes': ', '.join(keys)
                    })

                # Store breaking groups data for Sheet 3+
                key = (delta, epsilon)
                # If a key already exists (e.g., from multiple files sharing the same delta/epsilon combo), extend the list
                if key not in rule_breaking_groups_data:
                    rule_breaking_groups_data[key] = []
                rule_breaking_groups_data[key].extend(breaking_list)

    # --- Write Output File for the Rule ---
    if not rule_attribute_counts and not rule_percentage_summary:
        print(f"--- Rule {index} finished with no data to write. ---")
        return []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f'rule_breaking_summary_index_{index}.xlsx')

    print(f"  > Writing individual summary to {output_path}...")

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # 1. Write Sheet 1: Attribute Counts Summary
            if rule_attribute_counts:
                counts_df = pd.DataFrame(rule_attribute_counts.items(), columns=['Attribute', 'Count'])
                counts_df = counts_df.sort_values('Count', ascending=False)
                # Sheet name starting with 1 to ensure it's the first sheet
                counts_df.to_excel(writer, sheet_name='1_Attribute_Counts', index=False)

            # 2. Write Sheet 2: Percentage Summary
            if rule_percentage_summary:
                summary_df = pd.DataFrame(rule_percentage_summary)
                # Remove Rule Index from individual file if desired, or keep it. Keeping it for now.
                summary_df = summary_df.sort_values(['Delta', 'Epsilon'])
                summary_df.to_excel(writer, sheet_name='2_Percentage_Summary', index=False)

            # 3. Write Sheet 3+: Breaking Subgroups Data
            # Sort by delta then epsilon for predictable sheet order
            for (delta, epsilon), groups in sorted(rule_breaking_groups_data.items()):
                sheet_name = f'delta_{delta}_eps_{epsilon}'
                breaking_df = pd.DataFrame(groups)
                # Sort the breaking groups by UtilityDiff
                breaking_df = breaking_df.sort_values('UtilityDiff', ascending=False)
                breaking_df.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"--- Successfully created output for Rule {index} ---")

        # Return the summary data so Main can aggregate it
        return rule_percentage_summary

    except Exception as e:
        print(f"  ! Error writing output file for Rule {index}: {e}")
        return []


# --- Main Execution ---
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting file discovery and grouping...")
    files_by_rule = group_files_by_rule(TARGET_DIRECTORY)

    if not files_by_rule:
        print(f"No valid files found in {TARGET_DIRECTORY}. Check the directory path and file naming convention.")
        return

    print(
        f"Found {sum(len(v) for v in files_by_rule.values())} files grouped into {len(files_by_rule)} rules (indices).")

    # List to hold summary stats from ALL rules
    all_rules_summary_data = []

    # Process each rule group
    for index, file_list in files_by_rule.items():
        # process_rule_group now returns the summary list
        rule_summary_data = process_rule_group(index, file_list)

        # Add to our master list
        if rule_summary_data:
            all_rules_summary_data.extend(rule_summary_data)

    # --- Generate Master Summary File ---
    if all_rules_summary_data:
        print("\n--- Generating Master Summary File ---")
        master_output_path = os.path.join(OUTPUT_DIR, 'master_summary_all_rules.xlsx')

        try:
            master_df = pd.DataFrame(all_rules_summary_data)

            # Reorder columns for better readability
            cols_order = ['Rule Index', 'Delta', 'Epsilon', 'Num All Subgroups', 'Num Breaking Subgroups',
                          'Percentage Breaking']
            # Only select columns that exist (in case something went wrong, though it shouldn't)
            cols_to_use = [c for c in cols_order if c in master_df.columns]
            master_df = master_df[cols_to_use]

            # Sort by Rule Index, then Delta
            master_df = master_df.sort_values(by=['Rule Index', 'Delta', 'Epsilon'])

            master_df.to_excel(master_output_path, index=False)
            print(f"Success! Master summary saved to: {master_output_path}")

        except Exception as e:
            print(f"Error creating master summary file: {e}")
    else:
        print("No data available to generate master summary.")

    print("\n\nAll processing complete. Results are saved in the './breaking_subgroups_by_rule' directory.")


if __name__ == '__main__':
    main()
