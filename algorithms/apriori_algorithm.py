"""
Core subgroup analysis algorithms using Apriori.
This module contains functions for finding subgroups and calculating their utility.
"""
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))
from ATE_update import ATEUpdateLinear
from typing import Dict, List, Tuple, Any, Callable, Optional

def mine_subgroups(
    algorithm: Callable,
    df: pd.DataFrame,
    delta: int,
    exclude_cols: List[str] = None
) -> List[Tuple[Dict[str, object], int]]:
    """
    Return [(filter‑dict, size), …] for every subgroup size ≥ delta.

    Args:
        algorithm: Apriori algorithm function to use
        df: Input DataFrame
        delta: Minimum group size threshold
        exclude_cols: List of columns to exclude from mining (e.g., treatment columns)

    Returns:
        List of tuples containing (filter_dict, size) for each subgroup
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Filter out columns that should not be used for subgroup mining
    mining_df = df.drop(columns=exclude_cols, errors='ignore')
    
    # one‑hot encode attribute=value pairs
    onehot_parts = []
    lookup: Dict[str, Tuple[str, object]] = {}    # dummy‑col ➜ (attr, value)
    for col in mining_df.columns:
        d = pd.get_dummies(mining_df[col].fillna('⧫NA⧫'), prefix=col, dtype=bool)
        onehot_parts.append(d)
        lookup.update({c: (col, c.split('_', 1)[1]) for c in d.columns})
    onehot = pd.concat(onehot_parts, axis=1)

    # Apriori algorithm for frequent itemsets
    min_sup = delta / len(df)
    freq = algorithm(onehot, min_support=min_sup, use_colnames=True)

    # discard item‑sets that mention the same attribute twice
    def valid(itemset):
        attrs = [lookup[col][0] for col in itemset]
        return len(attrs) == len(set(attrs))
    freq = freq[freq['itemsets'].apply(valid)]

    # convert back to {attr:value} + absolute size
    results: List[Tuple[Dict[str, object], int]] = []
    n_rows = len(df)
    for items, sup in zip(freq['itemsets'], freq['support']):
        fdict = {lookup[c][0]: lookup[c][1] for c in items}
        results.append((fdict, int(round(sup * n_rows))))
    return results


def filter_by_attribute(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Vectorised AND-filter without early-exit (support already ≥ delta).

    Args:
        df: Input DataFrame
        filters: Dictionary of attribute-value pairs to filter on

    Returns:
        Filtered DataFrame
    """
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for a, v in filters.items():
        mask &= df[a] == int(v)
    return df[mask]


def calc_utility_for_subgroups(
    mode: int,
    algorithm: Callable,
    df: pd.DataFrame,
    treatment: Dict[Any, Any],
    treatment_col: str,
    tgtO: str,
    delta: int,
    epsilon: int,
    utility_all: float
):
    """
    Calculate utility for each subgroup in the DataFrame using Apriori algorithm.

    Args:
        mode: Mode of operation (0 for homogeneity check, other for all subgroups)
        algorithm: Apriori algorithm function to use
        df: Input DataFrame
        treatment: Dictionary mapping treatment variables to their values
        target_col: Target outcome column
        cate_func: Function to calculate CATE values (expects boolean mask)
        delta: Minimum group size threshold
        epsilon: Threshold for homogeneity check
        utility_all: Overall utility value for comparison

    Returns:
        Tuple containing:
        - List of dictionaries with subgroup data (if mode != 0)
        - Number of subgroups (if mode != 0)
        - Boolean indicating homogeneity (if mode == 0)
    """
    # Find all subgroups meeting the minimum size requirement
    # Exclude treatment columns and target outcome from mining
    exclude_cols = [*treatment.keys(), treatment_col, tgtO]
    subgroup_records = []
    for filt, sz in mine_subgroups(algorithm, df, delta, exclude_cols=exclude_cols):
        # Filter the dataframe to the current subgroup
        sub_df = filter_by_attribute(df, filt)
        if sub_df.empty:
            continue
            
        features_cols = [col for col in df.columns if col not in [*treatment.keys(),treatment_col,*filt.keys(), tgtO]]
        ate_update_obj = ATEUpdateLinear(df[features_cols], df[treatment_col], df[tgtO])
        cate = ate_update_obj.get_original_ate()
            
        if mode == 0 and abs(utility_all - cate) > epsilon:
            print(
                f"\n\033[91msubgroup's cate is: {cate} while utility_all is {utility_all} "
                f"(Δ={abs(utility_all - cate)}>{epsilon}) → NOT homogeneous\033[0m\n"
            )
            return False

        # Store subgroup information
        if mode != 0:
            subgroup_records.append({
                "AttributeValues": str(filt),
                "Size": sz,
                "Utility": cate,
                "UtilityDiff": cate - utility_all,
            })

    if mode != 0:
        return subgroup_records, len(subgroup_records)

    print("\033[92mHomogenous\033[0m")
    return True
