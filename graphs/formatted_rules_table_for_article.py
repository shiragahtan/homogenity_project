#!/usr/bin/env python3
"""
Create formatted tables for the article from the rules summary CSV.
Generates both LaTeX and Markdown formats.
"""

import pandas as pd
from pathlib import Path

def create_latex_table(df, dataset_name):
    """Create a LaTeX table for a specific dataset."""
    dataset_df = df[df['Dataset'] == dataset_name].copy()
    dataset_df = dataset_df.drop(columns=['Dataset'])
    
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Rules for " + dataset_name.upper() + " Dataset: Coverage, Utility, and Prevalence}\n"
    latex += "\\label{tab:rules_" + dataset_name + "}\n"
    latex += "\\begin{tabular}{|c|p{5cm}|p{5cm}|c|c|c|}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Rule} & \\textbf{Condition} & \\textbf{Treatment} & \\textbf{Coverage} & \\textbf{Utility} & \\textbf{Prevalence} \\\\\n"
    latex += "\\hline\n"
    
    for _, row in dataset_df.iterrows():
        rule_num = row['Rule #']
        condition = row['Condition'].replace('_', '\\_').replace('&', '\\&')
        treatment = row['Treatment'].replace('_', '\\_').replace('&', '\\&').replace('→', '$\\rightarrow$')
        coverage = row['Coverage (%)']
        utility = row['Utility']
        prevalence = row['Prevalence (%)']
        
        latex += f"{rule_num} & {condition} & {treatment} & {coverage}\\% & {utility} & {prevalence}\\% \\\\\n"
        latex += "\\hline\n"
    
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def create_markdown_table(df, dataset_name):
    """Create a Markdown table for a specific dataset."""
    dataset_df = df[df['Dataset'] == dataset_name].copy()
    dataset_df = dataset_df.drop(columns=['Dataset'])
    
    md = f"\n### {dataset_name.upper()} Dataset\n\n"
    md += "| Rule | Condition | Treatment | Coverage (%) | Utility | Prevalence (%) |\n"
    md += "|:----:|-----------|-----------|:------------:|:-------:|:--------------:|\n"
    
    for _, row in dataset_df.iterrows():
        rule_num = row['Rule #']
        condition = row['Condition']
        treatment = row['Treatment']
        coverage = row['Coverage (%)']
        utility = row['Utility']
        prevalence = row['Prevalence (%)']
        
        md += f"| {rule_num} | {condition} | {treatment} | {coverage} | {utility} | {prevalence} |\n"
    
    return md

def create_compact_table(df, dataset_name):
    """Create a compact table suitable for copying to Word/Google Docs."""
    dataset_df = df[df['Dataset'] == dataset_name].copy()
    dataset_df = dataset_df.drop(columns=['Dataset'])
    
    # Rename columns for display
    display_df = dataset_df.copy()
    display_df.columns = ['Rule', 'Condition', 'Treatment', 'Coverage (%)', 'Utility', 'Prevalence (%)']
    
    return display_df.to_string(index=False)

def main():
    # Read the CSV
    csv_path = Path(__file__).parent / 'rules_summary_from_existing_results.csv'
    df = pd.read_csv(csv_path)
    
    output_dir = Path(__file__).parent
    
    # Get unique datasets
    datasets = df['Dataset'].unique()
    
    print("="*80)
    print("GENERATING FORMATTED TABLES FOR ARTICLE")
    print("="*80)
    
    # Create output files
    latex_file = output_dir / 'rules_tables_latex.tex'
    markdown_file = output_dir / 'rules_tables_markdown.md'
    text_file = output_dir / 'rules_tables_plaintext.txt'
    
    latex_content = "% LaTeX Tables for Article Appendix\n\n"
    markdown_content = "# Rules Summary - Coverage, Utility, and Prevalence\n\n"
    text_content = "RULES SUMMARY - COVERAGE, UTILITY, AND PREVALENCE\n"
    text_content += "="*80 + "\n\n"
    
    for dataset in datasets:
        print(f"\nProcessing {dataset.upper()} dataset...")
        
        # LaTeX
        latex_content += create_latex_table(df, dataset)
        latex_content += "\n\\clearpage\n\n"
        
        # Markdown
        markdown_content += create_markdown_table(df, dataset)
        markdown_content += "\n"
        
        # Plain text (compact)
        text_content += f"\n{'='*80}\n"
        text_content += f"{dataset.upper()} DATASET\n"
        text_content += f"{'='*80}\n\n"
        text_content += create_compact_table(df, dataset)
        text_content += "\n\n"
    
    # Save files
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    print("\n" + "="*80)
    print("✅ SUCCESS! Created formatted tables:")
    print("="*80)
    print(f"1. LaTeX format:    {latex_file}")
    print(f"2. Markdown format: {markdown_file}")
    print(f"3. Plain text:      {text_file}")
    
    # Also print to console for easy copy-paste
    print("\n" + "="*80)
    print("PREVIEW - PLAIN TEXT FORMAT (Copy from below)")
    print("="*80)
    print(text_content)

if __name__ == "__main__":
    main()


