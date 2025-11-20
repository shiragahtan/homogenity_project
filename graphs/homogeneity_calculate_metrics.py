import pandas as pd
import os
import glob
import re
import numpy as np
from collections import defaultdict

# --- Configuration ---
#RESULTS_FILE = "newer2_old_homogenity_results.xlsx"
RESULTS_FILE = "homogeneity_results.xlsx"
RAW_ALGORITHMS_DIR = "../algorithms_results"
OUTPUT_FILE = "homogeneity_metrics_fixed.xlsx"

FILE_PATTERN = re.compile(r'_delta_(\d+)_(\d+)\.xlsx$', re.IGNORECASE)
EPSILON_RANGE = range(5000, 65001, 5000)

def to_boolean(val):
    """Robust boolean converter."""
    if isinstance(val, bool): return val
    if isinstance(val, (int, float)): return val != 0
    s = str(val).strip().lower()
    return s in ['true', 't', '1', '1.0', 'yes']

def calculate_average_prevalence_from_raw(target_dir):
    print(f"--- Starting Raw Prevalence Calculation from {target_dir} ---")
    prevalence_accumulator = defaultdict(list)
    
    files = glob.glob(os.path.join(target_dir, '*.xlsx'))
    count = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        if filename.startswith('~$'): continue
        match = FILE_PATTERN.search(filename)
        if not match: continue
        
        delta = int(match.group(1))
        
        try:
            df = pd.read_excel(file_path, sheet_name='Subgroups', engine='openpyxl')
            if 'UtilityDiff' not in df.columns: continue
            
            df['UtilityDiff'] = pd.to_numeric(df['UtilityDiff'], errors='coerce')
            df.dropna(subset=['UtilityDiff'], inplace=True)
            
            num_all = len(df)
            if num_all == 0:
                for eps in EPSILON_RANGE:
                    prevalence_accumulator[(delta, float(eps))].append(0.0)
                continue

            for epsilon in EPSILON_RANGE:
                eps_float = float(epsilon)
                num_breaking = len(df[df['UtilityDiff'].abs() > eps_float])
                percentage = (num_breaking / num_all * 100.0) if num_all > 0 else 0.0
                prevalence_accumulator[(delta, eps_float)].append(percentage)
            
            count += 1
            if count % 50 == 0: print(f"Processed {count} raw files...")

        except Exception as e:
            print(f"Skipping {filename}: {e}")

    avg_map = {k: np.mean(v) for k, v in prevalence_accumulator.items()}
    return avg_map

def calculate_final_metrics():
    prevalence_map = calculate_average_prevalence_from_raw(RAW_ALGORITHMS_DIR)

    print("--- Calculating Metrics Per Rule & Averaging (Macro-Average) ---")
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found.")
        return

    df = pd.read_excel(RESULTS_FILE)
    df['algorithm'] = df['algorithm'].str.strip()

    # Identify Algorithms
    RW_NAME = "Weighted RW Algorithm"
    GT_NAME = "Brute-Force Algorithm" 
    
    if RW_NAME not in df['algorithm'].unique():
        matches = [x for x in df['algorithm'].unique() if 'RW' in x]
        if matches: RW_NAME = matches[0]
    if GT_NAME not in df['algorithm'].unique():
        matches = [x for x in df['algorithm'].unique() if 'Brute' in x or 'Apriori' in x]
        if matches: GT_NAME = matches[0]

    final_rows = []
    
    params = df[['delta', 'epsilon']].drop_duplicates().sort_values(['delta', 'epsilon'])
    
    for _, row in params.iterrows():
        delta = int(row['delta'])
        epsilon = float(row['epsilon'])
        
        subset = df[(df['delta'] == delta) & (df['epsilon'] == epsilon)]
        
        # Get unique rules for this delta/epsilon combo
        rules = subset[['treatment', 'condition']].drop_duplicates()
        
        rule_precisions = []
        rule_specificities = []
        
        for _, rule in rules.iterrows():
            t = rule['treatment']
            c = rule['condition']
            
            # 1. Get Ground Truth for this specific rule
            gt_rows = subset[(subset['algorithm'] == GT_NAME) & (subset['treatment'] == t) & (subset['condition'] == c)]
            if gt_rows.empty: continue
            gt_status = to_boolean(gt_rows.iloc[0]['homogeneity_status'])
            
            # 2. Get ALL RW runs for this specific rule
            rw_runs = subset[(subset['algorithm'] == RW_NAME) & (subset['treatment'] == t) & (subset['condition'] == c)]
            if rw_runs.empty: continue

            # 3. Count outcomes across all runs for this single rule
            rule_tp = 0
            rule_fp = 0
            rule_tn = 0
            
            for _, r_run in rw_runs.iterrows():
                rw_status = to_boolean(r_run['homogeneity_status'])
                
                if rw_status is True and gt_status is True:
                    rule_tp += 1
                elif rw_status is True and gt_status is False:
                    rule_fp += 1
                elif rw_status is False and gt_status is False:
                    rule_tn += 1
                # Note: FN ignored as per previous logic (RW assumes homogeneity unless broken)
            
            # 4. Calculate Individual Rule Metrics (Handling 0/0 as 1.0)
            
            # Precision = TP / (TP + FP)
            if (rule_tp + rule_fp) > 0:
                p = rule_tp / (rule_tp + rule_fp)
            else:
                # If no positive predictions were made (TP+FP=0), but we were right 
                # (e.g. correctly predicted all Negatives), score as 1.0 (Perfect)
                p = 1.0 

            # Specificity = TN / (TN + FP)
            if (rule_tn + rule_fp) > 0:
                s = rule_tn / (rule_tn + rule_fp)
            else:
                # If no negative/false predictions were applicable (TN+FP=0), 
                # (e.g. correctly predicted all Positives), score as 1.0 (Perfect)
                s = 1.0

            rule_precisions.append(p)
            rule_specificities.append(s)
        
        # 5. Average across all rules (Macro-Average)
        if rule_precisions:
            avg_prec = np.mean(rule_precisions) * 100.0
            avg_spec = np.mean(rule_specificities) * 100.0
        else:
            avg_prec = 0.0
            avg_spec = 0.0
        
        avg_prev = prevalence_map.get((delta, epsilon), 0.0)
        
        final_rows.append({
            "epsilon": epsilon,
            "delta": delta,
            "Precision (%)": avg_prec,
            "Specificity (%)": avg_spec,
            "Prevalence (%)": avg_prev
        })
        
    out_df = pd.DataFrame(final_rows)
    out_df = out_df.sort_values(['delta', 'epsilon'])
    
    print(f"Writing metrics to {OUTPUT_FILE}...")
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        out_df.to_excel(writer, sheet_name='RW_Metrics_Fixed', index=False)
    print("Done.")

if __name__ == "__main__":
    calculate_final_metrics()