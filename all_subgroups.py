import sys
import multiprocessing as mp
import pandas as pd
from typing import Dict, List, Tuple

# Add the project root directory to sys.path
from utility_functions import CATE

DELTA = 20_000
GLOBAL_FILTERS = {'Gender': 'Male'}


def filter_by_attribute(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Filters the DataFrame based on the given attribute-value pairs.
    Stops filtering if the group size falls below DELTA.
    """
    for attribute, value in filters.items():
        df = df[df[attribute] == value]
        if len(df) < DELTA:  # Stop filtering if group size is below DELTA
            return pd.DataFrame()  # Return an empty DataFrame
    return df


def _dfs_serial(df: pd.DataFrame,
                attrs: List[str],
                attr_values: Dict[str, List],
                delta: int,
                parent_filters: Dict[str, object],
                start_idx: int
                ) -> List[Tuple[Dict[str, object], int]]:
    """
    Depth‑first expansion of one branch — NO multiprocessing inside here.
    Returns a flat list of (filters, size); caller takes care of levels.
    """
    out: List[Tuple[Dict[str, object], int]] = []
    parent_df = df
    for a, v in parent_filters.items():
        parent_df = parent_df[parent_df[a] == v]

    # parent itself is already ≥ delta (checked by caller)
    out.append((parent_filters.copy(), len(parent_df)))

    for idx in range(start_idx, len(attrs)):
        attr = attrs[idx]
        for val in attr_values[attr]:
            child_filters = {**parent_filters, attr: val}
            child_df = parent_df[parent_df[attr] == val]
            size = len(child_df)
            if size >= delta:
                out.extend(
                    _dfs_serial(child_df, attrs, attr_values,
                                delta, child_filters, idx + 1)
                )
    return out


# ---------- public entry point  --------------------------------
def generate_pruned_levels_mp(df: pd.DataFrame,
                              attr_values: Dict[str, List],
                              delta: int = 10_000,
                              n_jobs=None
                              ) -> List[List[Tuple[Dict[str, object], int]]]:
    """
    Parallel version:  fan out *only* the first level that survives ≥ delta,
    then recurse serially inside each process.

    Returns the same structure as your original
        List[ level0, level1, level2, … ]
    where each level element is  (filters‑dict, size).
    """
    attrs = list(attr_values.keys())
    root_size = len(df)

    # ----- build first‑level tasks ---------------------------------
    first_level_tasks = []
    for idx, attr in enumerate(attrs):
        for val in attr_values[attr]:
            filt = {attr: val}
            slice_ = df[df[attr] == val]
            size = len(slice_)
            if size >= delta:
                first_level_tasks.append((slice_, attrs, attr_values,
                                          delta, filt, idx + 1))

    # nothing qualifies
    if not first_level_tasks:
        return [[({}, root_size)]]

    # ----- run first level in parallel -----------------------------
    with mp.Pool(processes=n_jobs) as pool:
        branch_results = pool.starmap(_dfs_serial, first_level_tasks)

    # ----- re‑assemble into explicit levels ------------------------
    flat = [({}, root_size)]  # level 0 (root)
    for branch in branch_results:
        flat.extend(branch)  # each branch already includes its own parent

    # bucket into levels by “number of keys in filters”
    max_depth = max(len(f) for f, _ in flat)
    levels: List[List[Tuple[Dict[str, object], int]]] = [[] for _ in range(max_depth + 1)]
    for filt, sz in flat:
        levels[len(filt)].append((filt, sz))

    return levels


def calc_utility_for_subgroups(attr_vals, df, condition, treatment, DAG_str, attrOrdinal, tgtO):
    """
    Calculate utility for each subgroup in the DataFrame and save results to an Excel file.
    """
    # Calculate CATE for the entire dataset (all men)
    utility_all, _ = CATE(df, DAG_str, treatment, attrOrdinal, tgtO)

    # Initialize a list to store subgroup data
    subgroup_data = []

    kept = generate_pruned_levels_mp(df, attr_vals, DELTA, n_jobs=mp.cpu_count())
    print(f"\nGroups with ≥ {DELTA:,} rows (pruned search):")
    num_subgroups = 0
    for lvl, groups in enumerate(kept):
        if lvl == 0:
            continue
        print(f"  level {lvl}: {len(groups):,} groups")
        for filt, sz in groups:
            print(f"\n    {filt}  ->  {sz:,} rows")
            filtered_df = filter_by_attribute(df, filt)
            cate_value, p_value = CATE(filtered_df, DAG_str, treatment, attrOrdinal, tgtO)
            utility_diff = cate_value - utility_all
            print(f"cate_value: {cate_value}, p_value: {p_value}\n")

            num_subgroups += 1
            # Append subgroup data to the list
            subgroup_data.append({
                "AttributeValues": str(filt),
                "Size": sz,
                "Utility": cate_value,
                "UtilityDiff": utility_diff,
                "PValue": p_value
            })

    # Convert the list of subgroup data to a DataFrame
    print(f"\nTotal number of subgroups: {num_subgroups}")
    subgroup_df = pd.DataFrame(subgroup_data)
    summary_df = pd.DataFrame([{"NumSubgroups": num_subgroups}])
    chosen_treatment_df = pd.DataFrame([{"Condition": condition,
                                         "Treatment": treatment}])

    # Save the DataFrame to an Excel file
    output_file = "subgroups_results_delta_20_000.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        chosen_treatment_df.to_excel(writer, sheet_name="ChosenTreatment", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        subgroup_df.to_excel(writer, sheet_name="Subgroups", index=False)

    print(f"Subgroup results saved to {output_file}")

    return subgroup_data


# Output the results
DATA_PATH = sys.argv[1]

df = pd.read_csv(DATA_PATH)

df = df[df['Gender'] == "Male"]
condition = {'Gender': 'Male'}
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
treatment = {'Exercise': '3 - 4 times per week',
             'UndergradMajor': 'Computer science, computer engineering, or software engineering'}
DAG_str = """digraph {
    Continent -> UndergradMajor;
    Continent -> FormalEducation;
    Continent -> Country;
    Continent -> RaceEthnicity;
    Continent -> ConvertedSalary;
    HoursComputer -> ConvertedSalary;
    UndergradMajor -> DevType;
    FormalEducation -> UndergradMajor;
    FormalEducation -> DevType;
    Age -> FormalEducation;
    Age -> Dependents;
    Age -> DevType;
    Age -> ConvertedSalary;
    Gender -> UndergradMajor;
    Gender -> FormalEducation;
    Gender -> DevType;
    Gender -> ConvertedSalary;
    Dependents -> HoursComputer;
    Country -> FormalEducation;
    Country -> RaceEthnicity;
    Country -> ConvertedSalary;
    DevType -> HoursComputer;
    DevType -> ConvertedSalary;
    RaceEthnicity -> ConvertedSalary;
    HDI -> GINI;
    GINI -> ConvertedSalary;
    GINI -> GDP;
    GDP -> ConvertedSalary;
}
"""
attrOrdinal = None  # No ordinal attributes in this example
tgtO = "ConvertedSalary"  # Target outcome column in the dataset
# generate_and_save_subgroups(df, DATA_PATH, treatment, DAG_str, attrOrdinal, tgtO)
non_filter_cols = {"Gender"}  # whatever you want to skip

attr_vals = {
    col: sorted(
        [v for v in df[col].dropna().unique()
         if str(v).upper() not in {"UNKNOWN"}]  # ← keep only “real” values
    )
    for col in df.columns
    if col not in non_filter_cols
}

calc_utility_for_subgroups(attr_vals, df, condition, treatment, DAG_str, attrOrdinal, tgtO)
