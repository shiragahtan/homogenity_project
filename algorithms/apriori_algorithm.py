"""
Core subgroup analysis algorithms using Apriori.
This module contains functions for finding subgroups and calculating their utility.
"""
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional
from mlxtend.frequent_patterns import apriori


def mine_subgroups(
    algorithm: Callable,
    df: pd.DataFrame,
    delta: int,
) -> List[Tuple[Dict[str, object], int]]:
    """
    Return [(filter‑dict, size), …] for every subgroup size ≥ delta.

    Args:
        df: Input DataFrame
        delta: Minimum group size threshold

    Returns:
        List of tuples containing (filter_dict, size) for each subgroup
    """
    # one‑hot encode attribute=value pairs
    onehot_parts = []
    lookup: Dict[str, Tuple[str, object]] = {}    # dummy‑col ➜ (attr, value)
    for col in df.columns:
        d = pd.get_dummies(df[col].fillna('⧫NA⧫'), prefix=col, dtype=bool)
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
    Vectorised AND‑filter without early‑exit (support already ≥ delta).

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
        mask &= df[a] == v
    return df[mask]


def calc_utility_for_subgroups(
    mode: int,
    algorithm: Callable,
    df: pd.DataFrame,
    treatment: Dict[str, object],
    dag_str: str,
    attr_ordinal: Optional[List[str]],
    target_col: str,
    cate_func: Callable,
    delta: int,
    epsilon: int
):
    """
    Calculate utility for each subgroup in the DataFrame using Apriori algorithm.

    Args:
        df: Input DataFrame
        treatment: Dictionary mapping treatment variables to their values
        dag_str: DAG string representation in DOT format
        attr_ordinal: List of ordinal attributes (if any)
        target_col: Target outcome column
        cate_func: Function to calculate CATE values
        delta: Minimum group size threshold

    Returns:
        Tuple containing:
        - List of dictionaries with subgroup data
        - Number of subgroups found
    """
    # Calculate CATE for the entire dataset
    utility_all, _ = cate_func(df, dag_str, treatment, attr_ordinal, target_col)

    # Find all subgroups meeting the minimum size requirement
    subgroup_records = []
    for filt, sz in mine_subgroups(algorithm, df, delta):
        # Filter the dataframe to the current subgroup
        sub_df = filter_by_attribute(df, filt)

        # Calculate CATE for this subgroup
        cate, p = cate_func(sub_df, dag_str, treatment, attr_ordinal, target_col)
        if cate != 0 and mode == 0 and abs(utility_all - cate) > epsilon:
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
                "PValue": p,
            })

    if mode != 0:
        return subgroup_records, len(subgroup_records)

    print("\033[92mHomogenous\033[0m")
    return True
