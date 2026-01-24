#!/usr/bin/env python3
"""
Script to generate a summary table of all rules with their Coverage, Utility, and Prevalence
for the appendix (Section 7).

This script reads the treatment JSON files and outputs a formatted table.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

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

def process_rules_file(json_file: Path, dataset_name: str) -> pd.DataFrame:
    """Process a rules JSON file and return a DataFrame with all metrics."""
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name} ({json_file.name})")
    print(f"{'='*80}")
    
    rules = load_rules_from_json(json_file)
    
    results = []
    for idx, rule in enumerate(rules, 1):
        condition = rule.get('condition', {})
        treatment = rule.get('treatment', {})
        utility = rule.get('utility', 0)
        coverage = rule.get('coverage', 0)
        
        # In rule mining, prevalence typically refers to how prevalent/common
        # the condition is in the population, which is the same as coverage
        prevalence = coverage
        
        results.append({
            'Dataset': dataset_name,
            'Rule #': idx,
            'Condition': format_condition(condition),
            'Treatment': format_treatment(treatment),
            'Coverage (%)': f"{coverage:.2f}",
            'Utility': f"{utility:.4f}",
            'Prevalence (%)': f"{prevalence:.2f}"
        })
    
    df = pd.DataFrame(results)
    
    # Print summary to console
    print(f"\nFound {len(rules)} rules")
    print(f"\nRules Summary:")
    print(df.to_string(index=False))
    
    return df

def main():
    """Main function to process all treatment JSON files."""
    
    # Define the paths to the JSON files
    algorithms_dir = Path(__file__).parent
    
    files_to_process = [
        ('algorithms/Chosen10Treatments.json', 'StackOverflow'),
        ('algorithms/GermanChosen10Treatments.json', 'German Credit'),
        ('algorithms/ACSChosen10Treatments.json', 'ACS')
    ]
    
    all_dataframes = []
    
    for json_filename, dataset_name in files_to_process:
        json_path = algorithms_dir / '..' / json_filename.replace('algorithms/', '')
        
        if not json_path.exists():
            # Try alternative path
            json_path = algorithms_dir / json_filename.split('/')[-1]
        
        if json_path.exists():
            df = process_rules_file(json_path, dataset_name)
            all_dataframes.append(df)
        else:
            print(f"Warning: File not found: {json_path}")
    
    # Combine all dataframes
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save as CSV for easy viewing
        csv_file = algorithms_dir / 'rules_summary_for_appendix.csv'
        combined_df.to_csv(csv_file, index=False)
        
        # Try to save as Excel if openpyxl is available
        output_file = algorithms_dir / 'rules_summary_for_appendix.xlsx'
        excel_saved = False
        try:
            print(f"\n{'='*80}")
            print(f"Saving combined results to: {output_file}")
            print(f"{'='*80}")
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Write all rules to one sheet
                combined_df.to_excel(writer, sheet_name='All Rules', index=False)
                
                # Write each dataset to separate sheets
                for df in all_dataframes:
                    dataset_name = df['Dataset'].iloc[0]
                    sheet_name = dataset_name[:31]  # Excel sheet name limit
                    df_without_dataset = df.drop(columns=['Dataset'])
                    df_without_dataset.to_excel(writer, sheet_name=sheet_name, index=False)
            excel_saved = True
        except ImportError:
            print(f"\nNote: openpyxl not installed. Skipping Excel file creation.")
        
        print(f"\n✅ Successfully created:")
        if excel_saved:
            print(f"   - {output_file}")
        print(f"   - {csv_file}")
        print(f"\nTotal rules processed: {len(combined_df)}")
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("Summary Statistics:")
        print(f"{'='*80}")
        for dataset_name in combined_df['Dataset'].unique():
            dataset_df = combined_df[combined_df['Dataset'] == dataset_name]
            print(f"\n{dataset_name}:")
            print(f"  - Number of rules: {len(dataset_df)}")
            print(f"  - Avg Coverage: {dataset_df['Coverage (%)'].str.rstrip('%').astype(float).mean():.2f}%")
            print(f"  - Avg Utility: {dataset_df['Utility'].astype(float).mean():.4f}")
    
    else:
        print("No files were processed successfully.")

if __name__ == "__main__":
    main()

