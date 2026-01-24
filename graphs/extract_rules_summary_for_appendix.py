#!/usr/bin/env python3
"""
Extract coverage, utility, and prevalence from existing breaking_per_rule.py output files.
This reads the already-generated Excel files and creates a summary table for the appendix.
"""

import pandas as pd
import json
from pathlib import Path

def load_rules_from_json(json_file: Path):
    """Load rules metadata (coverage, utility) from JSON file."""
    rules = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rules.append(json.loads(line))
    return rules

def extract_prevalence_from_file(file_path: Path):
    """Extract average prevalence across all delta/epsilon combinations."""
    try:
        df = pd.read_excel(file_path, sheet_name='2_Percentage_Summary', engine='openpyxl')
        # Get the 'Percentage Breaking' column and calculate average
        percentages = df['Percentage Breaking'].str.rstrip('%').astype(float)
        avg_prevalence = percentages.mean()
        return avg_prevalence
    except Exception as e:
        print(f"  Warning: Could not read {file_path.name}: {e}")
        return None

def format_condition(condition_dict):
    """Format condition for display."""
    items = []
    for key, value in condition_dict.items():
        items.append(f"{key} = {value}")
    return " AND ".join(items)

def format_treatment(treatment_dict):
    """Format treatment for display."""
    items = []
    for key, value in treatment_dict.items():
        items.append(f"{key} → {value}")
    return " AND ".join(items)

def main():
    print("="*80)
    print("EXTRACTING COVERAGE, UTILITY, AND PREVALENCE FROM EXISTING RESULTS")
    print("="*80)
    
    # Paths
    base_dir = Path(__file__).parent
    results_dir = base_dir / 'breaking_subgroups_by_rule'
    algorithms_dir = base_dir.parent / 'algorithms'
    
    # Load config to get dataset info
    config_path = base_dir.parent / 'configs' / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Determine which dataset was used for these results
    # Based on the file list, this appears to be the StackOverflow dataset
    # Let me check what rules files we have
    
    datasets_to_process = []
    
    for dataset_key, ds_config in config['DATASETS'].items():
        rules_file = ds_config['RULES_FILE']
        rules_path = algorithms_dir / rules_file
        if rules_path.exists():
            datasets_to_process.append((dataset_key, rules_file, rules_path))
    
    all_results = []
    
    for dataset_key, rules_filename, rules_path in datasets_to_process:
        print(f"\n{'='*80}")
        print(f"Processing Dataset: {dataset_key} ({rules_filename})")
        print(f"{'='*80}")
        
        # Load rules metadata
        rules = load_rules_from_json(rules_path)
        print(f"  Found {len(rules)} rules in JSON")
        
        for idx, rule in enumerate(rules):
            # Get coverage and utility from JSON
            coverage = rule.get('coverage', 0)
            utility = rule.get('utility', 0)
            condition = rule.get('condition', {})
            treatment = rule.get('treatment', {})
            
            # Try to find corresponding result file
            result_file = results_dir / f'rule_breaking_summary_index_{idx}.xlsx'
            
            if result_file.exists():
                prevalence = extract_prevalence_from_file(result_file)
                if prevalence is not None:
                    print(f"  ✓ Rule {idx + 1}: Coverage={coverage:.2f}%, Utility={utility:.4f}, Prevalence={prevalence:.2f}%")
                else:
                    prevalence = "N/A"
                    print(f"  ⚠️  Rule {idx + 1}: Could not calculate prevalence")
            else:
                prevalence = "N/A"
                print(f"  ⚠️  Rule {idx + 1}: No result file found")
            
            all_results.append({
                'Dataset': dataset_key,
                'Rule #': idx + 1,
                'Condition': format_condition(condition),
                'Treatment': format_treatment(treatment),
                'Coverage (%)': f"{coverage:.2f}",
                'Utility': f"{utility:.4f}",
                'Prevalence (%)': f"{prevalence:.2f}" if isinstance(prevalence, float) else prevalence
            })
    
    # Create DataFrame and save
    if all_results:
        df = pd.DataFrame(all_results)
        
        output_csv = base_dir / 'rules_summary_from_existing_results.csv'
        df.to_csv(output_csv, index=False)
        
        print(f"\n{'='*80}")
        print("✅ SUCCESS!")
        print(f"{'='*80}")
        print(f"Created: {output_csv}")
        print(f"Total rules processed: {len(df)}")
        
        # Print the table
        print(f"\n{df.to_string(index=False)}")
        
        # Summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            coverage_vals = dataset_df['Coverage (%)'].astype(float)
            utility_vals = dataset_df['Utility'].astype(float)
            
            # Handle N/A values in prevalence
            prevalence_vals = dataset_df['Prevalence (%)'].replace('N/A', float('nan'))
            try:
                prevalence_vals = prevalence_vals.astype(float)
            except:
                prevalence_vals = pd.Series([float('nan')] * len(dataset_df))
            
            print(f"\n{dataset}:")
            print(f"  Number of rules: {len(dataset_df)}")
            print(f"  Avg Coverage: {coverage_vals.mean():.2f}%")
            print(f"  Avg Utility: {utility_vals.mean():.4f}")
            if not prevalence_vals.isna().all():
                print(f"  Avg Prevalence: {prevalence_vals.mean():.2f}%")
            else:
                print(f"  Avg Prevalence: N/A")
    
    else:
        print("\n❌ No data was processed.")

if __name__ == "__main__":
    main()

