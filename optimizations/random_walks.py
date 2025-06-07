import time
import pandas as pd
from mlxtend.frequent_patterns import apriori
from dowhy import CausalModel
import random
import warnings
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import os
import ipdb
# Add the yarden_files directory to the Python path to import ATE_update
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'yarden_files'))
from ATE_update import ATEUpdateLinear, ATEUpdateLogistic

warnings.filterwarnings("ignore")

def choose_random_key(weights_optimization_method, random_choice, global_key_value_score):
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


def k_random_walks(k, treatment, outcome, df, desired_ate, size_threshold, weights_optimization_method):
    total_ate_calculations = 0
    total_ate_time = 0
    global_used_combinations = set()
    global_key_value_score = dict()
    delta = 8000
    max_size_of_group = 100000

    features_cols = [col for col in df.columns if col not in [treatment, outcome]]

    ate_update_obj = ATEUpdateLinear(df[features_cols], df[treatment], df[outcome])
    start_ate_time = time.time()  # Start timing ATE calculation
    df_ate = ate_update_obj.get_original_ate()
    print(f"shira ate all : df_ate: {df_ate}")
    end_ate_time = time.time()  # End timing ATE calculation
    total_ate_calculations += 1
    total_ate_time += (end_ate_time - start_ate_time)

    print(f"df ATE: {df_ate}")
    df_shape = df.shape[0]

    desired_diff = round(desired_ate-df_ate, 3)
    print(f"dsired ATE: {desired_ate}")
    print(f"dsired diff: {desired_diff}")

    available_itemsets = len(max_size_itemsets)
    sample_size = min(k, available_itemsets)
    print(f"Available itemsets: {available_itemsets}, Requested k: {k}, Using sample size: {sample_size}")
    
    random_choices = max_size_itemsets.sample(sample_size)['formatted_itemsets'].values

    diff_values = []

    for walk_idx, random_choice in enumerate(random_choices):
        print(f"\nwalk idx: {walk_idx}")
        ate_update_obj = ATEUpdateLinear(df[features_cols], df[treatment], df[outcome])

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
        # ipdb.set_trace()

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
                
                # Process large batches in smaller chunks to avoid performance issues
                batch_size = 14000  # Process 1000 indices at a time
                max_indices_to_process = 14000  # Maximum number of indices to process
                
                # If we have too many indices, sample a subset
                if len(unique_indices) > max_indices_to_process:
                    print(f"Too many indices ({len(unique_indices)}). Sampling {max_indices_to_process} random indices.")
                    import random
                    unique_indices = random.sample(unique_indices, max_indices_to_process)
                
                if len(unique_indices) > batch_size:
                    print(f"Processing {len(unique_indices)} indices in batches of {batch_size}")
                    
                    start_ate_time = time.time()
                    for batch_start in range(0, len(unique_indices), batch_size):
                        batch_end = min(batch_start + batch_size, len(unique_indices))
                        batch_indices = unique_indices[batch_start:batch_end]
                        print(f"  Processing batch {batch_start//batch_size + 1}: indices {batch_start} to {batch_end-1}")
                        ipdb.set_trace()
                        ate = ate_update_obj.calculate_updated_ATE(batch_indices)
                    end_ate_time = time.time()
                else:
                    start_ate_time = time.time()  # Start timing ATE calculation
                    ate = ate_update_obj.calculate_updated_ATE(unique_indices)
                    end_ate_time = time.time()  # End timing ATE calculation
                
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

    average_diff = np.mean(diff_values)
    variance_diff = np.var(diff_values)
    max_diff = np.max(diff_values)
    min_diff = np.min(diff_values)
    t_statistic, p_value = stats.ttest_1samp(diff_values, 0)

    # Bin the diff values and plot histogram
    # plt.figure(figsize=(10, 6))
    # plt.hist(diff_values, bins=30, edgecolor="black")
    # plt.xlabel("Diff Values (Binned)")
    # plt.ylabel("Number of Subgroups")
    # plt.title("Distribution of Diff Values Across Subgroups")
    # plt.savefig("diff_histogram.png") 
    # plt.show()

    print(f"Average Diff: {average_diff}")
    print(f"Variance of Diff: {variance_diff}")
    print(f"T-test Statistic: {t_statistic}, P-value: {p_value}")
    print(f"Max Diff: {max_diff}")
    print(f"Min Diff: {min_diff}")
    return False

def main(csv_name, attributes_for_apriori, treatment, outcome, desired_ate, k, size_threshold, weights_optimization_method):
    start_time = time.time()
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

    start_time_apriori = time.time()
    frequent_itemsets = apriori(df_encoded, min_support=3/df_shape, use_colnames=True)
    elapsed_minutes_apriori = (time.time() - start_time_apriori) / 60
    print(f"Apriori time: {elapsed_minutes_apriori:.2f} minutes")
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
    ret = k_random_walks(k, treatment, outcome, df, desired_ate, size_threshold, weights_optimization_method)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time / 60:.2f} minutes")
    return ret




def check_homogenity_with_random_walks(desired_ate):
    csv_name = "../yarden_files/stackoverflow_data_encoded.csv"
    attributes_for_apriori = ["Continent", "Gender", "RaceEthnicity"]
    treatment = "FormalEducation"
    outcome = "ConvertedSalary"
    k=1000
    size_threshold=0.2 # we want to look only at sub datasets that are less than 80% in their size from the original dataset
    weights_optimization_method = 1 # 0- no optimization, 1- sorting, 2- real weights
    
    return main(csv_name, attributes_for_apriori, treatment, outcome, desired_ate, k, size_threshold, weights_optimization_method)


if __name__ == "__main__":
    epsilon = 3000
    csv_name = "../yarden_files/stackoverflow_data_encoded.csv"
    df = pd.read_csv(csv_name)
    treatment = "FormalEducation"
    outcome = "ConvertedSalary"
    features_cols = [col for col in df.columns if col not in [treatment, outcome]]
    ate_update_obj = ATEUpdateLinear(df[features_cols], df[treatment], df[outcome])
    utility_all = ate_update_obj.get_original_ate() # to check it this is the utility_all
    
    if (check_homogenity_with_random_walks(utility_all + epsilon)):
        print("not homogenous (positive side)")
    elif (check_homogenity_with_random_walks(utility_all - epsilon)):
        print("not homogenous (negative side)")
    else:
        print("probably homogenous")

