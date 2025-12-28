from typing import Dict, List, Tuple, Optional, Any
import sys
from pathlib import Path
from numpy.linalg import LinAlgError
import pandas as pd

from rw_unlearning import calc_utility_for_subgroups
sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe



def brute_force_find_positive_homogeneous_subgroup(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    epsilon: float,
    delta_percent: float,
) -> Optional[Tuple[Dict[str, Any], float, int]]:
    # Basic checks
    if treatment_col not in df.columns or outcome_col not in df.columns:
        raise ValueError("treatment_col and outcome_col must be columns in df")

    # Convert fractional `delta` (percent of df) to absolute count
    if not (0.0 < float(delta_percent) <= 1.0):
        raise ValueError("delta must be a fraction in (0, 1]")
    delta_count = max(1, int(len(df) * float(delta_percent)))

    # Attributes to consider for subgrouping
    attrs = [c for c in df.columns if c not in (treatment_col, outcome_col)]

    # Precompute unique values per attribute
    attr_values: Dict[str, List[Any]] = {}
    for a in attrs:
        # Use all distinct values (including NaN as one category via .astype(str) is possible,
        # but here we dropna to keep behavior simple)
        attr_values[a] = df[a].dropna().unique().tolist()

    candidates: List[Tuple[Dict[str, Any], int]] = []

    # DFS with pruning
    def _dfs(start_idx: int, current_filters: Dict[str, Any], current_df: pd.DataFrame):
        # current_df already reflects current_filters
        size = len(current_df)
        if size >= delta_count and current_filters:
            candidates.append((current_filters.copy(), size))

        for i in range(start_idx, len(attrs)):
            a = attrs[i]
            # Skip attribute if already in filters (shouldn't happen in this traversal)
            if a in current_filters:
                continue
            for val in attr_values[a]:
                next_df = current_df[current_df[a] == val]
                if len(next_df) >= delta_count:
                    next_filters = dict(current_filters)
                    next_filters[a] = val
                    _dfs(i + 1, next_filters, next_df)
                # else: prune

    _dfs(0, {}, df)

    if not candidates:
        return None

    # Sort by size descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Evaluate candidates in sorted order
    for filt, sz in candidates:
        # Re-filter to be safe
        sub_df = df
        for a, v in filt.items():
            sub_df = sub_df[sub_df[a] == v]
        # Log the subgroup being checked
        print(f"Checking subgroup: {filt} | size: {sz}")
        if len(sub_df) < delta_count:
            continue
        status = calc_utility_for_subgroups(
            mode=0,
            algorithm=None,
            df=sub_df,
            treatment_col=treatment_col,
            delta=delta_count,
            epsilon=epsilon,
            outcome_col=outcome_col,
        )
        # `calc_utility_for_subgroups` returns (bool, int)
        if not isinstance(status, tuple):
            # unexpected return, skip
            continue
        is_homog = bool(status[0])

        try:
            cate = calculate_ate_safe(sub_df, treatment_col, outcome_col, delta_count)
        except LinAlgError:
            continue

        if is_homog and cate > 0:
            return filt, float(cate), int(sz)

    return None




if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "stackoverflow" / "so.csv"
    df = pd.read_csv(data_path)

    treatment_col = "FormalEducation"
    outcome_col = "ConvertedSalary"

    epsilon = 200
    delta_percent = 0.7

    res = brute_force_find_positive_homogeneous_subgroup(df, treatment_col, outcome_col, epsilon, delta_percent)
    if res is None:
        print("No matching subgroup found")
    else:
        filt, cate, size = res
        print("Found subgroup:", filt)
        print("ATE:", cate)
        print("Size:", size)
