import os
import re
import json
import pandas as pd
from collections import defaultdict, Counter

# Load EPSILONS from config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)
EPSILONS = config['EPSILONS']

# Directory containing the xlsx files
DIRECTORY = '../algorithms_results/Yardens_results/'  # relative to this script

# Regex to match the files and extract delta and rule number
file_pattern = re.compile(r'Apriori_subgroups_results_delta_(\d+)_(\d+)\.xlsx')

# Store all breaking subgroups with their details
breaking_subgroups = []

# Track how many times each subgroup breaks rules
subgroup_break_counts = Counter()

for filename in os.listdir(DIRECTORY):
    match = file_pattern.match(filename)
    if not match:
        continue
    
    delta = int(match.group(1))
    rule_num = int(match.group(2))
    
    # Only process delta 5000
    if delta != 5000:
        continue
    
    filepath = os.path.join(DIRECTORY, filename)
    try:
        xls = pd.ExcelFile(filepath)
        sheet_name = 'Subgroups' if 'Subgroups' in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue
    
    # Ensure columns are as expected
    if not all(col in df.columns for col in ['AttributeValues', 'Size', 'Utility', 'UtilityDiff']):
        print(f"Skipping {filename}: missing required columns.")
        continue
    
    # Check each epsilon
    for epsilon in EPSILONS:
        # Find subgroups that break homogeneity for this epsilon
        breaking_mask = df['UtilityDiff'].abs() >= epsilon
        breaking_df = df[breaking_mask]
        
        for _, row in breaking_df.iterrows():
            attrval = row['AttributeValues']
            utility = row['Utility']
            utility_diff = row['UtilityDiff']
            size = row['Size']
            
            # Parse the string dict (e.g., "{'Gender': '1', 'HDI': '1'}")
            try:
                parsed_attrs = eval(attrval)
                # Create a normalized representation for counting
                normalized_attrs = tuple(sorted(parsed_attrs.items()))
                
                # Store the breaking subgroup details
                breaking_subgroups.append({
                    'Rule': rule_num,
                    'Epsilon': epsilon,
                    'AttributeValues': attrval,
                    'Size': size,
                    'Utility': utility,
                    'UtilityDiff': utility_diff,
                    'Attributes': ', '.join(parsed_attrs.keys()),
                    'NormalizedKey': str(normalized_attrs)  # for counting duplicates
                })
                
                # Count how many times this subgroup breaks rules
                subgroup_break_counts[str(normalized_attrs)] += 1
                
            except Exception as e:
                print(f"Error parsing {attrval}: {e}")
                continue

# Create the final DataFrame
if breaking_subgroups:
    df_final = pd.DataFrame(breaking_subgroups)
    
    # Add the break count for each subgroup
    df_final['BreakCount'] = df_final['NormalizedKey'].map(subgroup_break_counts)
    
    # Remove the temporary normalized key column
    df_final = df_final.drop('NormalizedKey', axis=1)
    
    # Sort by size (descending) - largest subgroups first
    df_final = df_final.sort_values('Size', ascending=False)
    
    # Reorder columns for better readability
    column_order = ['Rule', 'Epsilon', 'Size', 'UtilityDiff', 'BreakCount', 'AttributeValues', 'Utility', 'Attributes']
    df_final = df_final[column_order]
    
    # Save to Excel
    output_path = os.path.join(os.path.dirname(__file__), 'breaking_subgroups_delta_5000.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        # Main sheet with all breaking subgroups
        df_final.to_excel(writer, sheet_name='All_Breaking_Subgroups', index=False)
        
        # Summary sheet
        summary_data = []
        for epsilon in EPSILONS:
            epsilon_data = df_final[df_final['Epsilon'] == epsilon]
            summary_data.append({
                'Epsilon': epsilon,
                'Total_Breaking_Subgroups': len(epsilon_data),
                'Unique_Subgroups': len(epsilon_data['AttributeValues'].unique()),
                'Avg_Size': epsilon_data['Size'].mean(),
                'Max_Size': epsilon_data['Size'].max(),
                'Min_Size': epsilon_data['Size'].min(),
                'Avg_UtilityDiff': epsilon_data['UtilityDiff'].abs().mean(),
                'Max_UtilityDiff': epsilon_data['UtilityDiff'].abs().max()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary_by_Epsilon', index=False)
        
        # Most frequently breaking subgroups
        frequent_breakers = df_final.groupby('AttributeValues').agg({
            'Size': 'first',
            'BreakCount': 'first',
            'UtilityDiff': lambda x: x.abs().max()
        }).sort_values('BreakCount', ascending=False).head(20)
        
        frequent_breakers.to_excel(writer, sheet_name='Most_Frequent_Breakers')
    
    print(f"Analysis complete!")
    print(f"Found {len(df_final)} total breaking subgroup instances")
    print(f"Across {len(df_final['AttributeValues'].unique())} unique subgroups")
    print(f"Results saved to: {output_path}")
    
    # Print some statistics
    print(f"\nSummary by epsilon:")
    for epsilon in EPSILONS:
        count = len(df_final[df_final['Epsilon'] == epsilon])
        print(f"  Epsilon {epsilon}: {count} breaking subgroups")
    
    print(f"\nTop 5 largest breaking subgroups:")
    top_5 = df_final.head(5)
    for _, row in top_5.iterrows():
        print(f"  Size {row['Size']}: {row['AttributeValues']} (breaks {row['BreakCount']} times)")

else:
    print("No breaking subgroups found for delta 5000") 