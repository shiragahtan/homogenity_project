import pandas as pd
import re
import os
import glob
import ast
import json
from collections import defaultdict, Counter

# --- Configuration ---
TARGET_DIRECTORY = '../algorithms_results'
OUTPUT_DIR = './breaking_subgroups_by_rule'

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

CHOSEN_DS = config["CHOSEN_DATASET"]

if CHOSEN_DS not in config['DATASETS']:
    raise ValueError(f"Dataset '{CHOSEN_DS}' not found in config.json")

ds_config = config['DATASETS'][CHOSEN_DS]
EPSILON_RANGE = ds_config.get('EPSILONS', [])
DELTA_RANGE = ds_config.get('DELTAS', [])

delta_group = "|".join(map(str, DELTA_RANGE))
FILE_PATTERN = re.compile(rf"{re.escape(CHOSEN_DS)}.*_delta_({delta_group})_(\d+)\.xlsx$", re.IGNORECASE)

# --- Helper Functions ---

def load_rules_metadata(rules_filename):
    path = os.path.join('../algorithms', rules_filename)
    rules_meta = {}
    print(f"--- Loading Rules Metadata from: {path} ---")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip(): continue
                data = json.loads(line)
                rules_meta[idx] = {'utility': data.get('utility', 0), 'coverage': data.get('coverage', 0)}
    except Exception as e:
        print(f"  ! Error reading rules file: {e}")
    return rules_meta

def parse_attribute_values(attrval_str):
    if pd.isna(attrval_str): return []
    attrval_str = str(attrval_str).replace('np.int64', 'int').replace('np.float64', 'float')
    try:
        data_dict = ast.literal_eval(attrval_str)
        return list(data_dict.keys())
    except Exception:
        return re.findall(r"'([^']+)'\s*:", attrval_str)

def group_files_by_rule(target_dir):
    files_by_rule = defaultdict(list)
    search_pattern = os.path.join(target_dir, '*.xlsx')
    file_paths = glob.glob(search_pattern)

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if filename.startswith('~$'): continue
        match = FILE_PATTERN.search(filename)
        if match:
            delta = match.group(1)
            index = match.group(2)
            files_by_rule[index].append({'path': file_path, 'delta': delta, 'index': index})
    return files_by_rule

def process_rule_group(index, file_list, rules_meta):
    rule_attribute_counts = Counter()
    rule_summary_rows = []
    rule_breaking_groups_data = {}
    all_subgroups_for_this_rule = [] # To be aggregated in master
    
    rule_idx_int = int(index)
    meta = rules_meta.get(rule_idx_int, {'utility': 0, 'coverage': 0})
    
    print(f"Processing Rule Index: {index}")

    for file_info in file_list:
        file_path = file_info['path']
        delta = file_info['delta']
        
        try:
            df = pd.read_excel(file_path, sheet_name='Subgroups', engine='openpyxl')
        except Exception:
            continue

        if 'UtilityDiff' not in df.columns: continue
        
        df['UtilityDiff'] = pd.to_numeric(df['UtilityDiff'], errors='coerce')
        df.dropna(subset=['UtilityDiff'], inplace=True)
        num_all_subgroups = len(df)
        
        if num_all_subgroups == 0: continue

        for epsilon in EPSILON_RANGE:
            epsilon_float = float(epsilon)
            df_breaking = df[df['UtilityDiff'].abs() > epsilon_float].copy()
            num_breaking_subgroups = len(df_breaking)
            percentage = (num_breaking_subgroups / num_all_subgroups) * 100

            rule_summary_rows.append({
                'sort_index': rule_idx_int,
                'rule': f"rule {rule_idx_int + 1}",
                'Prevelance': f"{percentage:.2f}%",
                'coverage': f"{meta['coverage']:.0f}%",
                'utility': f"{meta['utility']:,.2f}",
                'num_groups': num_all_subgroups,
                'num_breaking': num_breaking_subgroups,
                'Delta': delta,
                'Epsilon': epsilon
            })

            if not df_breaking.empty:
                breaking_list = []
                for _, row in df_breaking.iterrows():
                    keys = parse_attribute_values(row['AttributeValues'])
                    for key in keys: rule_attribute_counts[key] += 1
                    
                    subgroup_item = {
                        'rule': f"rule {rule_idx_int + 1}",
                        'Delta': delta,
                        'Epsilon': epsilon,
                        'Size': row.get('Size', 'N/A'),
                        'UtilityDiff': row['UtilityDiff'],
                        'Conditions': row['AttributeValues'], # The full string of conditions
                        'Attributes_List': ', '.join(keys)
                    }
                    breaking_list.append(subgroup_item)
                    all_subgroups_for_this_rule.append(subgroup_item)
                
                key = (delta, epsilon)
                if key not in rule_breaking_groups_data: rule_breaking_groups_data[key] = []
                rule_breaking_groups_data[key].extend(breaking_list)

    # Save Individual Rule File
    if rule_summary_rows:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f'rule_breaking_summary_index_{index}.xlsx')
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if rule_attribute_counts:
                pd.DataFrame(rule_attribute_counts.items(), columns=['Attribute', 'Count']).sort_values('Count', ascending=False).to_excel(writer, sheet_name='1_Attribute_Counts', index=False)
            pd.DataFrame(rule_summary_rows).to_excel(writer, sheet_name='2_Summary', index=False)
            for (delta, epsilon), groups in sorted(rule_breaking_groups_data.items()):
                pd.DataFrame(groups).sort_values('UtilityDiff', ascending=False).to_excel(writer, sheet_name=f'delta_{delta}_eps_{epsilon}', index=False)

    return rule_summary_rows, all_subgroups_for_this_rule

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rules_meta = load_rules_metadata(ds_config['RULES_FILE'])
    
    print("Starting file discovery...")
    files_by_rule = group_files_by_rule(TARGET_DIRECTORY)

    if not files_by_rule:
        print(f"No files found matching dataset '{CHOSEN_DS}' and deltas {DELTA_RANGE}")
        return

    all_rules_summary_data = []
    all_breaking_details = []

    for index, file_list in files_by_rule.items():
        summary_data, subgroup_data = process_rule_group(index, file_list, rules_meta)
        if summary_data:
            all_rules_summary_data.extend(summary_data)
            all_breaking_details.extend(subgroup_data)

    if all_rules_summary_data:
        print("\n--- Generating Master Summary File ---")
        master_output_path = os.path.join(OUTPUT_DIR, 'master_summary_all_rules.xlsx')
        
        # DataFrame 1: The Summary
        master_df = pd.DataFrame(all_rules_summary_data)
        master_df = master_df.sort_values(by=['sort_index', 'Delta', 'Epsilon'])
        summary_cols = ['rule', 'Prevelance', 'coverage', 'utility', 'num_groups', 'num_breaking', 'Delta', 'Epsilon']
        summary_final = master_df[[c for c in summary_cols if c in master_df.columns]]
        
        # DataFrame 2: The Raw Subgroups
        details_df = pd.DataFrame(all_breaking_details)
        
        with pd.ExcelWriter(master_output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary of all rules
            summary_final.to_excel(writer, sheet_name='Rules_Summary', index=False)
            
            # Sheet 2: The actual breaking subgroups
            if not details_df.empty:
                # Sort by Rule, then by Delta, then by most significant UtilityDiff
                details_df = details_df.sort_values(by=['rule', 'Delta', 'UtilityDiff'], ascending=[True, True, False])
                details_df.to_excel(writer, sheet_name='All_Breaking_Subgroups', index=False)
        
        print(f"Saved: {master_output_path}")

if __name__ == '__main__':
    main()