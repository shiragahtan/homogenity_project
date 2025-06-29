"""
Core subgroup analysis algorithms using Apriori.
This module contains functions for finding subgroups and calculating their utility.
"""
import sys
import json
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))
from ATE_update import calculate_ate_safe
from typing import Dict, List, Tuple, Any, Callable, Optional
from numpy.linalg import LinAlgError
import random
import numpy as np

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

BINARY_TREATMENT = config['TREATMENT_COL']
ATTRIBUTE_WEIGHTS = config.get('ATTRIBUTE_WEIGHTS', {})

# Normalize weights if not already normalized
if ATTRIBUTE_WEIGHTS:
    min_count = min(ATTRIBUTE_WEIGHTS.values())
    max_count = max(ATTRIBUTE_WEIGHTS.values())
    if max_count > min_count:
        ATTRIBUTE_WEIGHTS = {k: (v - min_count) / (max_count - min_count) for k, v in ATTRIBUTE_WEIGHTS.items()}


def mine_subgroups(
    algorithm: Callable,
    df: pd.DataFrame,
    delta: int,
    exclude_cols: List[str] = None,
    preferred_attributes: List[str] = None,
    attribute_weights: Dict[str, float] = None
) -> List[Tuple[Dict[str, object], int]]:
    """
    Return [(filter-dict, size), …] for every subgroup size ≥ delta.
    Optionally prioritize subgroups with preferred attributes.

    Args:
        algorithm: Apriori algorithm function to use
        df: Input DataFrame
        delta: Minimum group size threshold
        exclude_cols: List of columns to exclude from mining (e.g., treatment columns)
        preferred_attributes: List of attributes to prioritize (e.g., ['Gender', 'Age', 'Education'])
        attribute_weights: Dictionary mapping attributes to weights (e.g., {'Gender': 2.0, 'Age': 1.5})

    Returns:
        List of tuples containing (filter_dict, size) for each subgroup, sorted by preference
    """
    if exclude_cols is None:
        exclude_cols = []
    if attribute_weights is None:
        attribute_weights = ATTRIBUTE_WEIGHTS
    
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
        size = int(round(sup * n_rows))
        if size >= delta:  # Only include subgroups meeting size threshold
            results.append((fdict, size))
    
    # Sort by preference if specified
    if preferred_attributes or attribute_weights:
        results = sort_subgroups_by_preference(results, preferred_attributes, attribute_weights)
    
    return results


def sort_subgroups_by_preference(subgroups: List[Tuple[Dict[str, object], int]], 
                                preferred_attributes: List[str] = None, 
                                attribute_weights: Dict[str, float] = None) -> List[Tuple[Dict[str, object], int]]:
    """
    Sort subgroups by preference score (higher score = higher priority).
    """
    if not preferred_attributes and not attribute_weights:
        return subgroups
    
    def preference_score(subgroup_filters: Dict[str, object], _: int) -> float:
        return calculate_preference_score(subgroup_filters, preferred_attributes, attribute_weights)
    
    # Sort by preference score (descending), then by size (descending)
    return sorted(subgroups, key=lambda x: (preference_score(x[0], x[1]), x[1]), reverse=True)


def calculate_preference_score(filters: Dict[str, object], 
                             preferred_attributes: List[str] = None, 
                             attribute_weights: Dict[str, float] = None) -> float:
    """
    Calculate preference score for a set of filters.
    Higher score = higher priority.
    """
    score = 0.0
    
    # Check if filters contain preferred attributes
    if preferred_attributes:
        for attr in preferred_attributes:
            if attr in filters:
                score += 1.0
    
    # Apply attribute weights
    if attribute_weights:
        for attr, weight in attribute_weights.items():
            if attr in filters:
                score += weight
    
    return score


def choose_property_to_remove(filters: Dict[str, object], 
                            preferred_attributes: List[str] = None, 
                            attribute_weights: Dict[str, float] = None) -> str:
    """
    Choose which property to remove when building tree upwards.
    Prefers to remove less important attributes first.
    """
    if not preferred_attributes and not attribute_weights:
        # Default: remove the last property
        return list(filters.keys())[-1]
    
    # Calculate importance score for each attribute
    attribute_scores = {}
    for attr in filters.keys():
        score = 0.0
        
        # Lower score for preferred attributes (we want to keep them)
        if preferred_attributes and attr in preferred_attributes:
            score -= 1.0
        
        # Lower score for weighted attributes (we want to keep them)
        if attribute_weights and attr in attribute_weights:
            score -= attribute_weights[attr]
        
        attribute_scores[attr] = score
    
    # Remove the attribute with the highest score (least important)
    return max(attribute_scores.keys(), key=lambda x: attribute_scores[x])


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
    treatment_col: str,
    tgtO: str,
    delta: int,
    epsilon: int,
    utility_all: float,
    preferred_attributes: List[str] = None,
    attribute_weights: Dict[str, float] = None
):
    """
    Calculate utility for each subgroup in the DataFrame using Apriori algorithm.

    Args:
        mode: Mode of operation (0 for homogeneity check, other for all subgroups)
        algorithm: Apriori algorithm function to use
        df: Input DataFrame
        treatment_col: Treatment column name
        tgtO: Target outcome column
        delta: Minimum group size threshold
        epsilon: Threshold for homogeneity check
        utility_all: Overall utility value for comparison
        preferred_attributes: List of attributes to prioritize (e.g., ['Gender', 'Age', 'Education'])
        attribute_weights: Dictionary mapping attributes to weights (e.g., {'Gender': 2.0, 'Age': 1.5})

    Returns:
        Tuple containing:
        - List of dictionaries with subgroup data (if mode != 0)
        - Number of subgroups (if mode != 0)
        - Boolean indicating homogeneity (if mode == 0)
    """
    exclude_cols = [treatment_col, BINARY_TREATMENT, tgtO]
    if attribute_weights is None:
        attribute_weights = ATTRIBUTE_WEIGHTS
    subgroups = mine_subgroups(algorithm, df, delta, exclude_cols=exclude_cols, 
                              preferred_attributes=preferred_attributes, 
                              attribute_weights=attribute_weights)
    
    seen_subgroups = set()
    subgroup_records = []
    for filt, sz in subgroups:
        subgroup_key = frozenset(filt.items())
        if subgroup_key in seen_subgroups:
            continue  # Skip already processed subgroup
        seen_subgroups.add(subgroup_key)
        # Filter the dataframe to the current subgroup
        sub_df = filter_by_attribute(df, filt)
        if sub_df.empty:
            continue
        try:
            cate = calculate_ate_safe(sub_df, treatment_col, tgtO)
        except LinAlgError:  # XᵀX still singular
            continue
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


TOP_ATTRIBUTES = [
    'Hobbies', 'Country', 'Student', 'Gender', 'SexualOrientation',
    'RaceEthnicity', 'Dependents', 'Continent', 'HDI', 'GDP', 'GINI', 'UndergradMajor', 'FormalEducation'
]

def k_deterministic_walks_unlearning(
    k: int,
    algorithm: Callable,
    df: pd.DataFrame,
    treatment_col: str,
    tgtO: str,
    delta: int,
    size_threshold: float = 0.8,
    attribute_weights: Dict[str, float] = None
):
    """
    Deterministic version: Only uses top 12 attributes, no randomness, always removes lowest-preference property.
    """
    if attribute_weights is None:
        attribute_weights = ATTRIBUTE_WEIGHTS
    n_rows = len(df)
    # Only use top 12 attributes for mining
    mining_df = df[[col for col in TOP_ATTRIBUTES if col in df.columns]]
    # Mine all frequent itemsets
    onehot_parts = []
    lookup = {}
    for col in mining_df.columns:
        d = pd.get_dummies(mining_df[col].fillna('⧫NA⧫'), prefix=col, dtype=bool)
        onehot_parts.append(d)
        lookup.update({c: (col, c.split('_', 1)[1]) for c in d.columns})
    onehot = pd.concat(onehot_parts, axis=1)
    min_sup = delta / n_rows
    freq = algorithm(onehot, min_support=min_sup, use_colnames=True)
    # Discard itemsets that mention the same attribute twice
    def valid(itemset):
        attrs = [lookup[col][0] for col in itemset]
        return len(attrs) == len(set(attrs))
    freq = freq[freq['itemsets'].apply(valid)]
    # Convert to {attr:value}
    results = []
    for items in freq['itemsets']:
        fdict = {lookup[c][0]: lookup[c][1] for c in items}
        results.append(fdict)
    # Find maximal itemset size
    max_size = max(len(f) for f in results) if results else 0
    maximal_subgroups = [f for f in results if len(f) == max_size]
    if not maximal_subgroups:
        print("No maximal subgroups found.")
        return []
    # Deterministically sort maximal subgroups
    maximal_subgroups = sorted(maximal_subgroups, key=lambda d: sorted(d.items()))
    sampled_maximals = maximal_subgroups[:k]
    all_walks = []
    for start_filters in sampled_maximals:
        walk = []
        current_filters = start_filters.copy()
        while current_filters:
            sub_df = filter_by_attribute(df, current_filters)
            size = len(sub_df)
            if size / n_rows >= size_threshold:
                break  # Stop if subgroup is too large
            try:
                cate = calculate_ate_safe(sub_df, treatment_col, tgtO)
            except LinAlgError:
                cate = None
            walk.append({
                'filters': current_filters.copy(),
                'size': size,
                'CATE': cate
            })
            # Remove one property (lowest preference, lexicographically last if tied)
            if len(current_filters) == 1:
                break
            # Find property with lowest weight (if tie, pick lexicographically last)
            min_weight = min(attribute_weights.get(attr, 0) for attr in current_filters)
            candidates = [attr for attr in current_filters if attribute_weights.get(attr, 0) == min_weight]
            key_to_remove = sorted(candidates)[-1]
            del current_filters[key_to_remove]
        all_walks.append(walk)
    return all_walks

def k_mostly_deterministic_walks_unlearning(
    k: int,
    algorithm: Callable,
    df: pd.DataFrame,
    treatment_col: str,
    tgtO: str,
    delta: int,
    size_threshold: float = 0.8,
    attribute_weights: Dict[str, float] = None
):
    if attribute_weights is None:
        attribute_weights = ATTRIBUTE_WEIGHTS
    n_rows = len(df)
    mining_df = df[[col for col in TOP_ATTRIBUTES if col in df.columns]]
    # Mine all frequent itemsets
    onehot_parts = []
    lookup = {}
    for col in mining_df.columns:
        d = pd.get_dummies(mining_df[col].fillna('⧫NA⧫'), prefix=col, dtype=bool)
        onehot_parts.append(d)
        lookup.update({c: (col, c.split('_', 1)[1]) for c in d.columns})
    onehot = pd.concat(onehot_parts, axis=1)
    min_sup = delta / n_rows
    freq = algorithm(onehot, min_support=min_sup, use_colnames=True)
    # Discard itemsets that mention the same attribute twice
    def valid(itemset):
        attrs = [lookup[col][0] for col in itemset]
        return len(attrs) == len(set(attrs))
    freq = freq[freq['itemsets'].apply(valid)]
    # Convert to {attr:value}
    results = []
    for items in freq['itemsets']:
        fdict = {lookup[c][0]: lookup[c][1] for c in items}
        results.append(fdict)
    # Find maximal itemset size
    max_size = max(len(f) for f in results) if results else 0
    maximal_subgroups = [f for f in results if len(f) == max_size]
    if not maximal_subgroups:
        print("No maximal subgroups found.")
        return []
    # Deterministically sort maximal subgroups
    maximal_subgroups = sorted(maximal_subgroups, key=lambda d: sorted(d.items()))
    # Randomly sample k starting points
    sampled_maximals = random.sample(maximal_subgroups, min(k, len(maximal_subgroups)))
    all_walks = []
    for start_filters in sampled_maximals:
        walk = []
        current_filters = start_filters.copy()
        while current_filters:
            sub_df = filter_by_attribute(df, current_filters)
            size = len(sub_df)
            if size / n_rows >= size_threshold:
                break
            try:
                cate = calculate_ate_safe(sub_df, treatment_col, tgtO)
            except LinAlgError:
                cate = None
            walk.append({
                'filters': current_filters.copy(),
                'size': size,
                'CATE': cate
            })
            if len(current_filters) == 1:
                break
            min_weight = min(attribute_weights.get(attr, 0) for attr in current_filters)
            candidates = [attr for attr in current_filters if attribute_weights.get(attr, 0) == min_weight]
            # Randomly pick among tied candidates
            key_to_remove = random.choice(candidates)
            del current_filters[key_to_remove]
        all_walks.append(walk)
    return all_walks

def ultra_fast_k_random_walks_unlearning(
    k: int,
    df: pd.DataFrame,
    treatment_col: str,
    tgtO: str,
    delta: int,
    epsilon: float,
    size_threshold: float = 0.8,
    attribute_weights: Dict[str, float] = None,
    top_attributes: List[str] = None,
    max_steps: int = 5,
    parallel: bool = True
):
    """
    Ultra-fast k random walks: uses your calculate_ate_safe, early stops if CATE is homogeneous, limits steps, uses numpy masks, parallel option.
    """
    if attribute_weights is None:
        attribute_weights = ATTRIBUTE_WEIGHTS
    if top_attributes is None:
        top_attributes = TOP_ATTRIBUTES
    n = len(df)
    arr = df[top_attributes + [treatment_col, tgtO]].values
    attr_idx = {attr: i for i, attr in enumerate(top_attributes)}
    treat_idx = len(top_attributes)
    tgt_idx = len(top_attributes) + 1
    unique_vals = {attr: df[attr].dropna().unique() for attr in top_attributes}

    # Calculate global ATE once
    global_ate = calculate_ate_safe(df, treatment_col, tgtO)

    def single_walk():
        filters = {attr: np.random.choice(unique_vals[attr]) for attr in top_attributes}
        current_filters = filters.copy()
        walk = []
        steps = 0
        while current_filters and steps < max_steps:
            mask = np.ones(n, dtype=bool)
            for a, v in current_filters.items():
                mask &= (arr[:, attr_idx[a]] == v)
            size = mask.sum()
            if size < delta or size / n >= size_threshold:
                break
            sub_treat = arr[mask, treat_idx]
            sub_tgt = arr[mask, tgt_idx]
            sub_df = pd.DataFrame({
                treatment_col: sub_treat,
                tgtO: sub_tgt
            })
            try:
                cate = calculate_ate_safe(sub_df, treatment_col, tgtO)
            except LinAlgError:
                cate = None
            walk.append({
                'filters': current_filters.copy(),
                'size': size,
                'CATE': cate
            })
            # Early stop if homogeneous
            if cate is not None and abs(cate - global_ate) < epsilon:
                break
            if len(current_filters) == 1:
                break
            min_weight = min(attribute_weights.get(attr, 0) for attr in current_filters)
            candidates = [attr for attr in current_filters if attribute_weights.get(attr, 0) == min_weight]
            key_to_remove = np.random.choice(candidates)
            del current_filters[key_to_remove]
            steps += 1
        return walk

    if parallel:
        import multiprocessing as mp
        with mp.Pool() as pool:
            all_walks = pool.map(lambda _: single_walk(), range(k))
    else:
        all_walks = [single_walk() for _ in range(k)]
    return all_walks
