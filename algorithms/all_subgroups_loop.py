import sys
import json
import datetime
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from time import perf_counter
from functools import partial
from contextlib import contextmanager
# Add project root to sys.path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from ATE_update import ATEUpdateLinear
from mlxtend.frequent_patterns import fpgrowth, apriori
from naive_DFS_algorithm import calc_utility_for_subgroups as naive_calc_utility_for_subgroups
from apriori_algorithm import calc_utility_for_subgroups as apriori_calc_utility_for_subgroups
from optimized_fp import calc_utility_for_subgroups as optimized_fp_calc_utility_for_subgroups

# Load config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

DELTAS = config['DELTAS']
ALGORITHM_NAMES = config['ALGORITHM_NAMES']
MODES = config['MODES']
EPSILONS = config['EPSILONS']
TREATMENT_COL = config['TREATMENT_COL']


""" Timing helper """
@contextmanager
def timer() -> callable:           # yields a function that returns elapsed seconds
    t0 = perf_counter()
    yield lambda: perf_counter() - t0


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


def _append_df_to_excel(excel_path: Path, new_row: dict):
    """
    Append a new row to an Excel file. Creates the file if it doesn't exist.
    """
    if not excel_path.exists():
        # Create new Excel file with headers
        df = pd.DataFrame([new_row])
        df.to_excel(excel_path, index=False)
    else:
        # Append to existing file
        existing_df = pd.read_excel(excel_path)
        updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        updated_df.to_excel(excel_path, index=False)


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

    _append_df_to_excel(excel_path, new_row)
    print(f"✅ Timing results appended to {excel_path}")


def append_homogeneity_results(algorithm_name, treatment, condition, delta, epsilon, homogeneity_status, runtime_seconds):
    """
    Append homogeneity check results to an Excel file.
    Creates the file if it doesn't exist.
    """
    excel_path = Path("homogeneity_results.xlsx")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Data to append
    new_row = {
        "date": current_date,
        "algorithm": algorithm_name,
        "treatment": str(treatment),
        "condition": str(condition),
        "delta": delta,
        "epsilon": epsilon,
        "homogeneity_status": homogeneity_status,
        "run_time_seconds": runtime_seconds,
        "run_time_minutes": runtime_seconds / 60
    }

    _append_df_to_excel(excel_path, new_row)
    print(f"🧬 Homogeneity results appended to {excel_path}")


def run_experiments(chosen_mode, chosen_algorithm, delta, df, tgtO, attr_vals, condition, treatment, i):
    """
    Run experiments for each treatment and save results to an Excel file.
    """
    print(f"Using algorithm: {ALGORITHM_NAMES[chosen_algorithm]}")
    epsilons = EPSILONS
    if chosen_mode != 0:
        epsilons = [epsilons[0]]  # You don't actually use epsilon in AllSubgroups mode, just chose random value.

    # Calculate utility for subgroups
    print(f"\033[94mrunning for condition: {condition} treatment: {treatment}\033[0m")
    features_cols = [col for col in df.columns if col not in [*treatment.keys(),TREATMENT_COL, tgtO]]
    ate_update_obj = ATEUpdateLinear(df[features_cols], df[TREATMENT_COL], df[tgtO])
    utility_all = ate_update_obj.get_original_ate()
    
    for epsilon in epsilons:
        if chosen_mode == 0:
            print(f"Running with epsilon: {epsilon}")

        # Common parameters for all algorithms
        common = dict(
            df=df,
            treatment=treatment,
            tgtO=tgtO,
            treatment_col=TREATMENT_COL,
            delta=delta,
            epsilon=epsilon,
            mode=chosen_mode,
            utility_all=utility_all
        )

        # Parameters for each algorithm
        _naive_kw = dict(common, attr_vals=attr_vals)
        _apriori_kw = dict(common, algorithm=apriori)
        _fpgrowth_kw = dict(common, algorithm=fpgrowth)
        _opt_fp_kw = dict(common, n_jobs=mp.cpu_count())

        algo_dispatch = {
            0: lambda: naive_calc_utility_for_subgroups(**_naive_kw),
            1: lambda: apriori_calc_utility_for_subgroups(**_apriori_kw),
            2: lambda: apriori_calc_utility_for_subgroups(**_fpgrowth_kw),
            3: lambda: optimized_fp_calc_utility_for_subgroups(**_opt_fp_kw),
        }

        try:
            with timer() as elapsed:
                res = algo_dispatch[chosen_algorithm]()
            elapsed_time = elapsed()

            if chosen_mode == 0:  # Homogeneity check
                append_homogeneity_results(
                    algorithm_name=ALGORITHM_NAMES[chosen_algorithm],
                    treatment=treatment,
                    condition=condition,
                    delta=delta,
                    epsilon=epsilon,
                    homogeneity_status=res,
                    runtime_seconds=elapsed_time
                )
            else:  # Only append timing results for AllSubgroups mode
                subgroup_data, num_subgroups = res
                save_results_to_excel(ALGORITHM_NAMES[chosen_algorithm], subgroup_data, num_subgroups, condition,
                                      treatment, delta, index=i)

                append_timing_results(ALGORITHM_NAMES[chosen_algorithm], condition, treatment, num_subgroups, delta,
                                      elapsed_time)

        except KeyError:
            raise ValueError(f"Unknown algorithm id: {chosen_algorithm}")

        print(f"⏱  Total execution time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")


def main():
    # Output the results
    # DATA_PATH = "../yarden_files/yarden_so_decoded.csv"  # Path to the dataset
    tgtO = "ConvertedSalary"  # Target outcome column in the dataset
    treatment_file = "Shira_Treatments.json"
    treated_rules_datasets = [
        '../stackoverflow/so_countries_treatment_1_encoded.csv',
        '../stackoverflow/so_countries_treatment_2_encoded.csv',
        '../stackoverflow/so_countries_treatment_3_encoded.csv'
    ]

    with open(treatment_file, "r") as f:
        good_treatments = [json.loads(line) for line in f]

    # chosen_mode = int(input(f"Choose your algorithm {list(enumerate(MODES))}: \n"))
    chosen_mode = 1
    # chosen_algorithm = int(input(f"Choose your algorithm {list(enumerate(ALGORITHM_NAMES))}: \n"))
    chosen_algorithm = 2  # For example, 1 for Apriori algorithm
    delta = 20000  # Initial delta value
    # run_experiments(chosen_mode, chosen_algorithm, delta, good_treatments, DATA_PATH, tgtO)
    for i, dataset in enumerate(treated_rules_datasets):
        df = pd.read_csv(dataset)
        condition = good_treatments[i]["condition"]
        attr, _ = list(condition.items())[0]
        treatment = good_treatments[i]["treatment"]
        attr_vals = {
            col: sorted(v for v in df[col].dropna().unique()
                        if str(v).upper() != "UNKNOWN")
            for col in df.columns if col not in [attr, TREATMENT_COL, *treatment.keys(), tgtO]
        }
        # for delta in DELTAS:
        if len(df) < delta:
            print(f"Skipping delta {delta} for treatment {i+1}: DataFrame too small ({len(df)} rows).")
            continue  # Skip if the filtered DataFrame is too small

        # for chosen_algorithm in range(0, len(ALGORITHM_NAMES)): # Loop through all algorithms from end to start
        print(f"Running for delta: {delta}")
        run_experiments(chosen_mode, chosen_algorithm, delta, df, tgtO, attr_vals, condition, treatment, i)


if __name__ == "__main__":
    main()