import sys
import json
import time
import datetime
import pandas as pd
from pathlib import Path
import multiprocessing as mp

from project_updated.homogenity_project.nativ_files.utility_functions import CATE
from mlxtend.frequent_patterns import fpgrowth, apriori
from naive_DFS_algorithm import calc_utility_for_subgroups as naive_calc_utility_for_subgroups
from apriori_algorithm import calc_utility_for_subgroups as apriori_calc_utility_for_subgroups
from optimized_fp import calc_utility_for_subgroups as optimized_fp_calc_utility_for_subgroups

MAX_DELTA = 20_000
MIN_DELTA = 5_000
ALGORITHM_NAMES = ["NaiveDFS", "Apriori", "FP", "FP_Multi"]


def save_results_to_excel(algorithm_name, subgroup_data, num_subgroups, condition, treatment, delta, index=0):
    """
    Save subgroup analysis results to an Excel file with multiple sheets.

    Args:
        subgroup_data: List of dictionaries containing subgroup information
        num_subgroups: Total number of subgroups
        condition: Dictionary of conditions used to filter the original data
        treatment: Dictionary of treatment interventions
        delta: Minimum group size threshold used in analysis

    Returns:
        str: Path to the output Excel file
    """
    subgroup_df = pd.DataFrame(subgroup_data)
    summary_df = pd.DataFrame([{"NumSubgroups": num_subgroups}])
    chosen_treatment_df = pd.DataFrame([{"Condition": condition,
                                         "Treatment": treatment}])

    # Save the DataFrame to an Excel file
    output_file = f"{algorithm_name}_subgroups_results_delta_{delta}_{index}.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        chosen_treatment_df.to_excel(writer, sheet_name="ChosenTreatment", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        subgroup_df.to_excel(writer, sheet_name="Subgroups", index=False)

    print(f"✔  {len(subgroup_data):,} subgroups saved to {output_file}")
    return output_file


def append_timing_results(algorithm_name, condition, treatment, num_subgroups, delta, runtime_seconds):
    """
    Append algorithm timing results to an Excel file.
    Creates the file if it doesn't exist.
    """
    excel_path = Path("algorithm_time.xlsx")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Data to append
    new_row = {
        "date": current_date,
        "algorithm": algorithm_name,
        "chosen_treatment": str(treatment),
        "chosen_condition": str(condition),
        "num_subgroups": str(num_subgroups),
        "delta": str(delta),
        "run_time_seconds": runtime_seconds,
        "run_time_minutes": runtime_seconds / 60
    }

    # Check if file exists
    if not excel_path.exists():
        # Create new Excel file with headers
        df = pd.DataFrame([new_row])
        df.to_excel(excel_path, index=False)
    else:
        # Append to existing file
        existing_df = pd.read_excel(excel_path)
        updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        updated_df.to_excel(excel_path, index=False)

    print(f"✅ Timing results appended to {excel_path}")


# description:
# Loads a dataset and a list of treatment/condition pairs.
# For each pair, filters the data and finds all subgroups of a minimum size (DELTA) based on attribute combinations.
# Calculates the Conditional Average Treatment Effect (CATE) for each subgroup.
# Saves the results and summary statistics for each experiment to Excel files.

def run_experiments(chosen_algorithm, delta, good_treatments, DATA_PATH, DAG_str, attrOrdinal, tgtO):
    """
    Run experiments for each treatment and save results to an Excel file.
    """
    print(f"Using algorithm: {ALGORITHM_NAMES[chosen_algorithm]}")
    for i, good_treatment in enumerate(good_treatments):
        condition = good_treatment["condition"]
        attr, val = list(condition.items())[0]
        treatment = good_treatment["treatment"]
        # Filter the DataFrame based on the condition
        df = (pd.read_csv(DATA_PATH)
              .query(f'{attr} == "{val}"')
              .loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
              .drop(columns=[f'{attr}'])  # Remove the filter column since it now contains only one value
              .loc[lambda d: ~d.isin(["UNKNOWN"]).any(axis=1)]  # Remove rows with "UNKNOWN" in any column
              .reset_index(drop=True))

        if len(df) < delta:
            continue  # Skip if the filtered DataFrame is too small
        # Calculate utility for subgroups
        print(f"running for condition: {condition} treatment: {treatment}")
        if chosen_algorithm == 0:
            attr_vals = {
            col: sorted(
                    [v for v in df[col].dropna().unique()
                    if str(v).upper() not in {"UNKNOWN"}]  # ← keep only “real” values
                )
                for col in df.columns
                if col not in attr
            }
            start_time = time.time()
            subgroup_data, num_subgroups = naive_calc_utility_for_subgroups(
                attr_vals,
                df,
                treatment,
                DAG_str,
                attrOrdinal,
                tgtO,
                CATE_func=CATE,
                delta=delta
            )
            elapsed_time = time.time() - start_time

        elif chosen_algorithm == 1:
            start_time = time.time()
            subgroup_data, num_subgroups = apriori_calc_utility_for_subgroups(
                apriori,
                df,
                treatment,
                DAG_str,
                attrOrdinal,
                tgtO,
                cate_func=CATE,
                delta=delta
            )
            elapsed_time = time.time() - start_time

        elif chosen_algorithm == 2:
            start_time = time.time()
            subgroup_data, num_subgroups = apriori_calc_utility_for_subgroups(
                fpgrowth,
                df,
                treatment,
                DAG_str,
                attrOrdinal,
                tgtO,
                cate_func=CATE,
                delta=delta
            )
            elapsed_time = time.time() - start_time

        elif chosen_algorithm == 3:
            start_time = time.time()
            subgroup_data, num_subgroups = optimized_fp_calc_utility_for_subgroups(
                df,
                treatment,
                DAG_str,
                attrOrdinal,
                tgtO,
                cate_func=CATE,
                delta=delta,
                n_jobs=mp.cpu_count()
            )
            elapsed_time = time.time() - start_time

        save_results_to_excel(ALGORITHM_NAMES[chosen_algorithm], subgroup_data, num_subgroups, condition, treatment, delta, index=i)
        print(f"⏱  Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        append_timing_results(ALGORITHM_NAMES[chosen_algorithm], condition, treatment, num_subgroups, delta, elapsed_time)


def main():
    # Output the results
    DATA_PATH = sys.argv[1]
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
    treatment_file = "Shira_Treatments.json"

    with open(treatment_file, "r") as f:
        good_treatments = [json.loads(line) for line in f]

    # chosen_algorithm = int(input(f"Choose your algorithm {list(enumerate(ALGORITHM_NAMES))}: \n"))
    for chosen_algorithm in range(1, 4):
        for delta in range(MIN_DELTA, MAX_DELTA + 1, 5000):
            print(f"Running for delta: {delta}")
            run_experiments(chosen_algorithm, delta, good_treatments, DATA_PATH, DAG_str, attrOrdinal, tgtO)


if __name__ == "__main__":
    main()