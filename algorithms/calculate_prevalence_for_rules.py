#!/usr/bin/env python3
"""
Script to calculate Coverage, Utility, and Prevalence for all rules based on the article definitions.

Definitions:
- Coverage: The number of tuples from D that the rule captures (percentage of data)
- Utility: The CATE of the treatment within the subpopulation
- Prevalence: The fraction of subgroups that VIOLATE homogeneity (w.r.t. δ and ε)
              among all subgroups to which the rule applies
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re
from typing import List, Dict, Tuple

def load_rules_from_json(json_file: Path) -> List[Dict]:
    """Load rules from a JSON file (one rule per line)."""
    rules = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rules.append(json.loads(line))
    return rules

def format_condition(condition_dict: Dict) -> str:
    """Format a condition dictionary as a readable string."""
    items = []
    for key, value in condition_dict.items():
        items.append(f"{key} = {value}")
    return " AND ".join(items)

def format_treatment(treatment_dict: Dict) -> str:
    """Format a treatment dictionary as a readable string."""
    items = []
    for key, value in treatment_dict.items():
        items.append(f"{key} → {value}")
    return " AND ".join(items)

def find_result_files_for_rule(dataset_name: str, rule_index: int, results_dir: Path) -> List[Path]:
    """Find all result files for a specific rule."""
    # Pattern: *_delta_*_{rule_index}.xlsx (files don't include dataset name)
    # We'll use Apriori or FPGrowth or BruteForce results
    pattern = f"*_delta_*_{rule_index}.xlsx"
    all_files = list(results_dir.glob(pattern))
    
    # Filter to get one algorithm's results (prefer FPGrowth > Apriori > BruteForce)
    for preferred_algo in ['FPGrowth', 'Apriori', 'BruteForce', 'MultiProcessing']:
        algo_files = [f for f in all_files if preferred_algo in f.name]
        if algo_files:
            return algo_files
    
    return all_files

def calculate_prevalence_for_rule(result_files: List[Path], epsilon: float) -> Dict[str, float]:
    """
    Calculate prevalence for a rule given its result files.
    
    Prevalence = (# of subgroups that violate homogeneity) / (# of all subgroups)
    A subgroup violates homogeneity if |UtilityDiff| > epsilon
    """
    prevalence_by_delta = {}
    
    for file_path in result_files:
        # Extract delta from filename
        match = re.search(r'delta_(\d+)_', str(file_path))
        if not match:
            continue
        delta = match.group(1)
        
        try:
            # Read the subgroups sheet
            df = pd.read_excel(file_path, sheet_name='Subgroups', engine='openpyxl')
            
            if 'UtilityDiff' not in df.columns:
                continue
            
            # Clean the data
            df['UtilityDiff'] = pd.to_numeric(df['UtilityDiff'], errors='coerce')
            df = df.dropna(subset=['UtilityDiff'])
            
            num_all_subgroups = len(df)
            if num_all_subgroups == 0:
                continue
            
            # Count subgroups that violate homogeneity
            num_breaking = len(df[df['UtilityDiff'].abs() > epsilon])
            
            # Calculate prevalence (as fraction)
            prevalence = num_breaking / num_all_subgroups
            
            prevalence_by_delta[delta] = {
                'prevalence': prevalence,
                'prevalence_pct': prevalence * 100,
                'num_breaking': num_breaking,
                'num_total': num_all_subgroups
            }
            
        except Exception as e:
            print(f"    Warning: Could not process {file_path.name}: {e}")
            continue
    
    return prevalence_by_delta

def load_config() -> Dict:
    """Load the config.json file."""
    config_path = Path(__file__).parent.parent / 'configs' / 'config.json'
    with open(config_path, 'r') as f:
        return json.load(f)

def process_dataset(dataset_key: str, config: Dict, results_dir: Path) -> pd.DataFrame:
    """Process all rules for a specific dataset."""
    
    print(f"\n{'='*80}")
    print(f"Processing Dataset: {dataset_key}")
    print(f"{'='*80}")
    
    ds_config = config['DATASETS'][dataset_key]
    rules_file = ds_config['RULES_FILE']
    epsilons = ds_config.get('EPSILONS', [10000, 15000, 20000])
    deltas = ds_config.get('DELTAS', [5000, 10000, 15000, 20000])
    
    # Use first epsilon as representative
    epsilon = float(epsilons[0]) if epsilons else 10000.0
    
    # Load rules
    rules_path = Path(__file__).parent / rules_file
    if not rules_path.exists():
        print(f"  ⚠️  Rules file not found: {rules_path}")
        return pd.DataFrame()
    
    rules = load_rules_from_json(rules_path)
    print(f"  Found {len(rules)} rules")
    print(f"  Using epsilon = {epsilon} for prevalence calculation")
    
    results = []
    
    for idx, rule in enumerate(rules):
        rule_index = idx  # 0-based index used in filenames
        
        condition = rule.get('condition', {})
        treatment = rule.get('treatment', {})
        utility = rule.get('utility', 0)
        coverage = rule.get('coverage', 0)
        
        # Find result files for this rule
        result_files = find_result_files_for_rule(dataset_key, rule_index, results_dir)
        
        if not result_files:
            print(f"  ⚠️  Rule {idx + 1}: No result files found")
            # Use coverage as fallback if no result files
            results.append({
                'Dataset': dataset_key,
                'Rule #': idx + 1,
                'Condition': format_condition(condition),
                'Treatment': format_treatment(treatment),
                'Coverage (%)': f"{coverage:.2f}",
                'Utility': f"{utility:.4f}",
                'Prevalence (%)': 'N/A',
                'Prevalence_Notes': 'No result files'
            })
            continue
        
        # Calculate prevalence for available deltas
        prevalence_by_delta = calculate_prevalence_for_rule(result_files, epsilon)
        
        if prevalence_by_delta:
            # Average prevalence across all deltas
            avg_prevalence = np.mean([v['prevalence_pct'] for v in prevalence_by_delta.values()])
            
            # Get details for reporting
            delta_details = []
            for delta, stats in sorted(prevalence_by_delta.items()):
                delta_details.append(
                    f"δ={delta}: {stats['prevalence_pct']:.2f}% "
                    f"({stats['num_breaking']}/{stats['num_total']})"
                )
            
            print(f"  ✓ Rule {idx + 1}: Prevalence = {avg_prevalence:.2f}% (avg across deltas)")
            
            results.append({
                'Dataset': dataset_key,
                'Rule #': idx + 1,
                'Condition': format_condition(condition),
                'Treatment': format_treatment(treatment),
                'Coverage (%)': f"{coverage:.2f}",
                'Utility': f"{utility:.4f}",
                'Prevalence (%)': f"{avg_prevalence:.2f}",
                'Prevalence_Details': '; '.join(delta_details)
            })
        else:
            print(f"  ⚠️  Rule {idx + 1}: Could not calculate prevalence")
            results.append({
                'Dataset': dataset_key,
                'Rule #': idx + 1,
                'Condition': format_condition(condition),
                'Treatment': format_treatment(treatment),
                'Coverage (%)': f"{coverage:.2f}",
                'Utility': f"{utility:.4f}",
                'Prevalence (%)': 'N/A',
                'Prevalence_Details': 'Calculation failed'
            })
    
    return pd.DataFrame(results)

def main():
    """Main function."""
    
    print("="*80)
    print("CALCULATING COVERAGE, UTILITY, AND PREVALENCE FOR ALL RULES")
    print("="*80)
    print("\nDefinitions:")
    print("  - Coverage: % of data that matches the rule's condition")
    print("  - Utility: CATE (treatment effect) within the subpopulation")
    print("  - Prevalence: % of subgroups that VIOLATE homogeneity (w.r.t. δ and ε)")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Results directory
    results_dir = Path(__file__).parent.parent / 'algorithms_results' / '10RulesResults'
    
    if not results_dir.exists():
        print(f"\n❌ Error: Results directory not found: {results_dir}")
        print("   Please ensure algorithm results have been generated.")
        return
    
    print(f"\nResults directory: {results_dir}")
    
    # Process each dataset
    all_dataframes = []
    
    for dataset_key in ['stackoverflow', 'german_credit', 'acs']:
        if dataset_key in config['DATASETS']:
            df = process_dataset(dataset_key, config, results_dir)
            if not df.empty:
                all_dataframes.append(df)
    
    # Combine and save
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save main file (without detailed prevalence breakdown)
        output_csv = Path(__file__).parent / 'rules_with_correct_prevalence.csv'
        main_df = combined_df.drop(columns=['Prevalence_Details'], errors='ignore')
        main_df.to_csv(output_csv, index=False)
        
        # Save detailed file (with prevalence breakdown by delta)
        detailed_csv = Path(__file__).parent / 'rules_with_prevalence_detailed.csv'
        combined_df.to_csv(detailed_csv, index=False)
        
        print(f"\n{'='*80}")
        print("✅ SUCCESS!")
        print(f"{'='*80}")
        print(f"Created files:")
        print(f"  1. {output_csv.name} - Summary table for appendix")
        print(f"  2. {detailed_csv.name} - Detailed breakdown by delta")
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        for dataset in combined_df['Dataset'].unique():
            dataset_df = combined_df[combined_df['Dataset'] == dataset]
            
            # Calculate numeric values
            coverage_vals = dataset_df['Coverage (%)'].str.rstrip('%').astype(float)
            utility_vals = dataset_df['Utility'].astype(float)
            
            # Prevalence might have N/A values
            prevalence_vals = dataset_df['Prevalence (%)'].replace('N/A', np.nan)
            prevalence_vals = pd.to_numeric(prevalence_vals, errors='coerce')
            
            print(f"\n{dataset}:")
            print(f"  Number of rules: {len(dataset_df)}")
            print(f"  Avg Coverage: {coverage_vals.mean():.2f}%")
            print(f"  Avg Utility: {utility_vals.mean():.4f}")
            if not prevalence_vals.isna().all():
                print(f"  Avg Prevalence: {prevalence_vals.mean():.2f}%")
            else:
                print(f"  Avg Prevalence: N/A")
        
        print(f"\n{'='*80}\n")
        
    else:
        print("\n❌ No data was processed successfully.")

if __name__ == "__main__":
    main()

