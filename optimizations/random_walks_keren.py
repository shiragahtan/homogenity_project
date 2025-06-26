import time
import pandas as pd
from mlxtend.frequent_patterns import apriori
from dowhy import CausalModel
import random
import warnings
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import json
from pathlib import Path
# import ipdb
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Add the yarden_files directory to the Python path to import ATE_update
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'yarden_files'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nativ_files'))
from ATE_update import ATEUpdateLinear

# Configuration

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

DELTAS = config['DELTAS']
MODES = config['YARDEN_OPTIMIZATION_MODES']
EPSILONS = config['EPSILONS']
TREATMENT_COL = config['TREATMENT_COL']


warnings.filterwarnings("ignore")

def choose_random_key(weights_optimization_method, random_choice, global_key_value_score):
    # Choose random key(node) to walk in the tree for the random walk
    if weights_optimization_method==0:
        return random.choice(list(random_choice.keys()))
    
    if weights_optimization_method==1:
        keys = list(random_choice.keys())
        scores = []
        for key in keys:
            score = global_key_value_score.get((key, random_choice[key]), 0)
            scores.append((key, score))

        sorted_scores = sorted(scores, key=lambda x: x[1])

        weights = {}
        rank = 0
        last_score = None
        for (key, score) in sorted_scores:
            if score != last_score:
                rank += 1
                last_score = score
            weights[key] = rank
        sum_weight = sum(weights.values())
        normalized_weights = [weights[key] / sum_weight for key in keys]
        return random.choices(keys, weights=normalized_weights, k=1)[0]
    
    if weights_optimization_method==2:
        keys = list(random_choice.keys())
        weights = {}
        for key in keys:
            score = global_key_value_score.get((key, random_choice[key]), 0)
            if score < 0:
                weights[key] = 0
            else:
                weights[key] = score + 1

        sum_weight = sum(weights.values())
        if sum_weight == 0:
            scores = []
            for key in keys:
                score = global_key_value_score.get((key, random_choice[key]), 0)
                scores.append((key, score))

            sorted_scores = sorted(scores, key=lambda x: x[1])

            weights = {}
            rank = 0
            last_score = None
            for (key, score) in sorted_scores:
                if score != last_score:
                    rank += 1
                    last_score = score
                weights[key] = rank
            sum_weight = sum(weights.values())
        normalized_weights = [weights[key] / sum_weight for key in keys]
        return random.choices(keys, weights=normalized_weights, k=1)[0]


def k_random_walks(df_ate, k, treatment, outcome, df, desired_ate, size_threshold, weights_optimization_method, 
                   delta, ate_update_obj, mode='hybrid', unlearning_threshold=0.1):
    """
    Perform k random walks with two modes:
    - hybrid: Use unlearning for small removals (≤ unlearning_threshold), direct CATE for large ones
    - direct: Always use direct CATE calculation
    
    Parameters:
    -----------
    mode : str
        'hybrid' or 'direct'
    unlearning_threshold : float
        Threshold for switching between unlearning and direct CATE (as fraction of dataset size)
    """
    total_ate_calculations = 0
    global_used_combinations = set()
    global_key_value_score = dict()
    max_size_of_group = 100000  # max size of the group to be removed

    total_ate_calculations += 1
    print(f"df ATE: {df_ate}")
    df_shape = df.shape[0]

    desired_diff = round(desired_ate-df_ate, 3)
    print(f"dsired ATE: {desired_ate}")
    print(f"dsired diff: {desired_diff}")


    available_itemsets = len(max_size_itemsets) # subgroups to calculate ATE for
    sample_size = min(k, available_itemsets)
    print(f"Available itemsets: {available_itemsets}, Requested k: {k}, Using sample size: {sample_size}")
    
    random_choices = max_size_itemsets.sample(sample_size)['formatted_itemsets'].values

    diff_values = []

    # Random Walk path
    for walk_idx, random_choice in enumerate(random_choices):
        print(f"\nwalk idx: {walk_idx}")
        
        combo_to_remove = []
        key_value = []
        random_choice = {k:int(v) for k,v in random_choice.items()}
        while len(random_choice) > 0:
            already_exist = False
            combo_hash = frozenset(random_choice.items())

            if combo_hash in global_used_combinations:
                already_exist = True 
            else:
                global_used_combinations.add(combo_hash)

            combo_to_remove.append((random_choice.copy(), already_exist))

            random_key = choose_random_key(weights_optimization_method, random_choice, global_key_value_score)
            random_value = random_choice[random_key]
            key_value.append((random_key, random_value))
            del random_choice[random_key]

        dfs_to_remove_data = []
        df_to_remove = None
        # subgroups to remove in reverse order
        for key, value in reversed(key_value):
            if df_to_remove is None:
                df_to_remove = df[df[key] == value]
            else:
                df_to_remove = df_to_remove[df_to_remove[key] == value]
            if (df_to_remove.shape[0] > delta and df_to_remove.shape[0] < max_size_of_group):
                dfs_to_remove_data.append((list(df_to_remove.index), df_to_remove.shape[0]))
            else:
                break
        
        # Create complement data structure - progressive complements matching original structure
        dfs_to_remove_data_shira_keren = []
        all_indices = set(df.index)
        
        for original_indices, original_size in dfs_to_remove_data:
            # Get complement of each progressive filter
            filtered_indices = set(original_indices)
            complement_indices = list(all_indices - filtered_indices)
            complement_size = len(complement_indices)
            dfs_to_remove_data_shira_keren.append((complement_indices, complement_size))

        # Reverse the order of the list
        dfs_to_remove_data_shira_keren = list(reversed(dfs_to_remove_data_shira_keren))

        print(f"Debug: Original progressive filtering had {len(dfs_to_remove_data)} steps")
        print(f"Debug: Complement progressive filtering has {len(dfs_to_remove_data_shira_keren)} steps")
        for i, ((orig_indices, orig_size), (comp_indices, comp_size)) in enumerate(zip(dfs_to_remove_data, dfs_to_remove_data_shira_keren)):
            print(f"  Step {i}: Original {orig_size} rows -> Complement {comp_size} rows")
        print(f"Debug: Total dataset size: {df.shape[0]} rows")

        tuples_removed_num = set()
        calc_idx = 0

        already_removed_indices = []
        for i, (df_to_remove_index, df_to_remove_shape) in enumerate(reversed(dfs_to_remove_data_shira_keren)):

            if df_to_remove_shape / df_shape < size_threshold:
                print(f"the size of the dataset is too big. too close to the original dataset. Breaking")
                break

            if i > 0:
                print(f"key & value to remove now from combo: {key_value[i-1]}")
            if df_to_remove_shape in tuples_removed_num:
                print(f"Combo to remove: {combo_to_remove[i][0]}, ATE already computed, tuples_removed: {df_to_remove_shape}")
            elif combo_to_remove[i][1] == True:
                print(f"Combo to remove: {combo_to_remove[i][0]} already exist in other random walk")
            else:
                unique_indices = [i for i in df_to_remove_index if i not in already_removed_indices]
                
                # Calculate the fraction of data to be removed
                removal_fraction = len(unique_indices) / df_shape
                
                # CATE Calculations
                if mode == 'direct':
                    # Always use direct CATE calculation
                    print(f"Using direct CATE calculation for {len(unique_indices)} indices ({removal_fraction:.1%} of dataset)")
                    subgroup_df = df.iloc[unique_indices]
                    ate = ate_update_obj.calculate_direct_ate(subgroup_df[features_cols], subgroup_df[treatment_key], subgroup_df[outcome])
                    # ate = utility_all
                    
                elif mode == 'hybrid':
                    if removal_fraction <= unlearning_threshold:
                        # Use unlearning for small removals
                        print(f"Using unlearning for {len(unique_indices)} indices ({removal_fraction:.1%} of dataset)")
                        ate = ate_update_obj.calculate_updated_ATE(unique_indices)
                    else:
                        # Use direct CATE calculation for large removals
                        print(f"Using direct CATE calculation for {len(unique_indices)} indices ({removal_fraction:.1%} of dataset)")
                        subgroup_df = df.iloc[unique_indices]
                        # ate = ate_update_obj.calculate_direct_ate(subgroup_df[features_cols], subgroup_df[treatment_key], subgroup_df[outcome])
                        ate = utility_all
                
                # Skip if CATE is 0 (invalid result)
                if ate == 0:
                    print(f"Skipping result: CATE is 0 (invalid calculation)")
                    continue
                
                already_removed_indices += unique_indices
                total_ate_calculations += 1
                total_ate_time += (end_ate_time - start_ate_time)

                diff = round(ate-df_ate, 3)
                diff_values.append(diff)
                impact = round((diff / df_to_remove_shape), 3)*10
                print(f"Combo to remove: {combo_to_remove[i][0]}, ATE: {ate}, tuples_removed: {df_to_remove_shape}, diff: {diff}, impact: {impact}")
                
                if i > 0:
                    if (desired_diff > 0 and ate > prev_ate) or (desired_diff < 0 and ate < prev_ate):
                        global_key_value_score[key_value[i-1]] = global_key_value_score.get(key_value[i-1], 0) + 1
                    else:
                        global_key_value_score[key_value[i-1]] = global_key_value_score.get(key_value[i-1], 0) - 1
                prev_ate = ate

                if (desired_diff > 0 and ate >= desired_ate) or (desired_diff < 0 and ate <= desired_ate):

                    print(f"Desired ATE condition met. Exiting after {walk_idx + 1} random walks.")
                    avg_ate_time = total_ate_time / total_ate_calculations if total_ate_calculations > 0 else 0
                    print(f"Total ATE calculations: {total_ate_calculations}")
                    print(f"Total time for ATE calculations: {(total_ate_time/60):.2f} minutes")
                    print(f"Average time per ATE calculation: {avg_ate_time:.4f} seconds")
                    return True

                calc_idx += 1

            tuples_removed_num.add(df_to_remove_shape)

    print(f"Desired ATE condition wasn't met.")
    avg_ate_time = total_ate_time / total_ate_calculations if total_ate_calculations > 0 else 0
    print(f"Total ATE calculations: {total_ate_calculations}")
    print(f"Total time for ATE calculations: {(total_ate_time/60):.2f} minutes")
    print(f"Average time per ATE calculation: {avg_ate_time:.4f} seconds")

    if len(diff_values) > 0:
        average_diff = np.mean(diff_values)
        variance_diff = np.var(diff_values)
        max_diff = np.max(diff_values)
        min_diff = np.min(diff_values)
        t_statistic, p_value = stats.ttest_1samp(diff_values, 0)

        print(f"Average Diff: {average_diff}")
        print(f"Variance of Diff: {variance_diff}")
        print(f"T-test Statistic: {t_statistic}, P-value: {p_value}")
        print(f"Max Diff: {max_diff}")
        print(f"Min Diff: {min_diff}")
    else:
        print("No diff values calculated.")
    
    return False

def main(csv_name, attributes_for_apriori, treatment, outcome, desired_ate, k, size_threshold, weights_optimization_method, 
         delta, ate_update_obj, mode='hybrid', unlearning_threshold=0.1):
    # Apriori, group mining part
    df = pd.read_csv(csv_name)
    df_shape = df.shape[0]

    # Store original types for ALL columns
    original_types = df.dtypes.to_dict()

    # Only convert the attributes_for_apriori columns to strings for Apriori
    # Keep treatment and outcome columns as numeric
    df_for_apriori = df.copy()
    for col in attributes_for_apriori:
        if col in df_for_apriori.columns:
            df_for_apriori[col] = df_for_apriori[col].astype(str)
    
    df_filtered = df_for_apriori[attributes_for_apriori]
    df_encoded = pd.get_dummies(df_filtered, prefix_sep='::')

    frequent_itemsets = apriori(df_encoded, min_support=3/df_shape, use_colnames=True)
    frequent_itemsets = frequent_itemsets.query('itemsets.str.len() > 0')

    itemset_mappings = {col: col.split('::', 1) for col in df_encoded.columns}
    formatted_itemsets = []
    for itemset in frequent_itemsets['itemsets']:
        formatted_itemsets.append({itemset_mappings[item][0]: itemset_mappings[item][1] for item in itemset})
    frequent_itemsets['formatted_itemsets'] = formatted_itemsets

    global max_size_itemsets
    frequent_itemsets['itemset_size'] = [len(itemset) for itemset in frequent_itemsets['formatted_itemsets']]
    max_size = frequent_itemsets['itemset_size'].max()
    max_size_itemsets = frequent_itemsets[frequent_itemsets['itemset_size'] == max_size]

    # Display results
    print("Frequent Itemsets with Feature:Value Format:")
    print(frequent_itemsets[['formatted_itemsets', 'itemset_size']])

    df = df.astype(original_types)
    ret = k_random_walks(k, treatment, outcome, df, desired_ate, size_threshold, weights_optimization_method, 
                        delta, ate_update_obj, mode, unlearning_threshold)
    return ret


def check_homogenity_with_random_walks(csv_name, desired_ate, treatment, delta, ate_update_obj, mode='hybrid', unlearning_threshold=0.1):
    attributes_for_apriori = ["Continent", "Gender", "RaceEthnicity"]
    outcome = "ConvertedSalary"
    k=1000
    size_threshold=0.2 # we want to look only at sub datasets that are less than 80% in their size from the original dataset
    weights_optimization_method = 1 # 0- no optimization, 1- sorting, 2- real weights
    
    return main(csv_name, attributes_for_apriori, treatment, outcome, desired_ate, k, size_threshold, 
               weights_optimization_method, delta, ate_update_obj, mode, unlearning_threshold)


if __name__ == "__main__":
    # Extract rules and treatments from the json rules file
    # For both modes, run this
    unlearning_threshold = 0.1
    results = []
    # Test positive and negative directions -> returns 1 if not homogenic
    positive_result = check_homogenity_with_random_walks(
        utility_all + epsilon, treatment_key, delta, ate_update_obj, mode, unlearning_threshold)
    negative_result = check_homogenity_with_random_walks(
        utility_all - epsilon, treatment_key, delta, ate_update_obj, mode, unlearning_threshold)    
    is_homogeneous = not(positive_result or negative_result)


