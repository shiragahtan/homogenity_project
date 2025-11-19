import sys
import json
import datetime
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from time import perf_counter
from functools import partial
from contextlib import contextmanager
import os

# Add project root to sys.path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

from ATE_update import calculate_ate_safe
from mlxtend.frequent_patterns import fpgrowth, apriori
from bruteForce_algorithm import calc_utility_for_subgroups as naive_calc_utility_for_subgroups
from apriori_algorithm import calc_utility_for_subgroups as apriori_calc_utility_for_subgroups
from algorithms.multiProcessing_algorithm import \
    calc_utility_for_subgroups as multiProcessing_calc_utility_for_subgroups
from rw_unlearning import calc_utility_for_subgroups as rw_unlearning_calc_utility_for_subgroups
from rw_multiProcesing import calc_utility_for_subgroups as rw_multiProcessing_calc_utility_for_subgroups

# Load config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

# DELTAS = config['DELTAS']
DELTAS = list(range(5000, 20001, 5000))

# ALGORITHM_NAMES = config['ALGORITHM_NAMES']
ALGORITHM_NAMES = ["Apriori", "RW"]  # Your temporary list for iteration

ALGORITHM_DISPATCH_MAP = {
    "BruteForce": 0,  # Naive
    "Apriori": 1,  # Apriori
    "FPGrowth": 2,  # FPGrowth
    "MultiProcessing": 3,  # Optimized FP
    "RW_Direct": 4,  # RW Unlearning Direct
    "RW_Hybrid": 5,  # RW Unlearning Hybrid
}

MODES = config['MODES']
# EPSILONS = config['EPSILONS']
EPSILONS = list(range(5000, 65001, 5000))
TREATMENT_COL = config['TREATMENT_COL']
OPTIMIZATION_MODES = config.get('OPTIMIZATION_MODES', ['direct'])

""" Timing helper """


@contextmanager
def timer() -> callable:  # yields a function that returns elapsed seconds
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

    # Create algotithms_results directory if it doesn't exist (at same level as algorithms)
    results_dir = Path("../algorithms_results")
    results_dir.mkdir(exist_ok=True)

    # Save the DataFrame to an Excel file in the algotithms_results directory
    output_file = results_dir / f"{algorithm_name}_subgroups_results_delta_{delta}_{index}.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        chosen_treatment_df.to_excel(writer, sheet_name="ChosenTreatment", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        subgroup_df.to_excel(writer, sheet_name="Subgroups", index=False)

    print(f"✔  {len(subgroup_data):,} subgroups saved to {output_file}")
    return str(output_file)


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
    # Create algotithms_results directory if it doesn't exist (at same level as algorithms)
    results_dir = Path("../graphs")
    results_dir.mkdir(exist_ok=True)
    
    excel_path = results_dir / "algorithms_time.xlsx"
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


def append_homogeneity_results(algorithm_name, treatment, condition, delta, epsilon, homogeneity_status,
                               runtime_seconds):
    """
    Append homogeneity check results to an Excel file.
    Creates the file if it doesn't exist.
    """
    # Create algotithms_results directory if it doesn't exist (at same level as algorithms)
    results_dir = Path("../graphs")
    results_dir.mkdir(exist_ok=True)

    excel_path = results_dir / "homogeneity_results.xlsx"
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


def run_experiments(chosen_mode, chosen_algorithm_name, delta, df, tgtO, attr_vals, condition, treatment, i,
                    attr_vals_time=0):
    """
    Run experiments for each treatment and save results to an Excel file.
    Args:
        chosen_mode: Mode of operation (0 for homogeneity check, other for all subgroups)
        chosen_algorithm_name: Algorithm to use (e.g., "BruteForce", "Apriori", "FPGrowth", "MultiProcessing", "RW_Direct", "RW_Hybrid")
        delta: Minimum group size threshold
        df: Input DataFrame
        tgtO: Target outcome column name
        attr_vals: Dictionary of attribute values for subgroup generation
        condition: Dictionary of conditions used to filter the original data
        treatment: Dictionary of treatment interventions
        i: Index of the current dataset/treatment being processed
        attr_vals_time: Time taken to compute attribute values (only for naive algorithm)
    """
    # Use the name directly
    algorithm_name = chosen_algorithm_name
    print(f"Using algorithm: {algorithm_name}")
    epsilons = EPSILONS
    if chosen_mode != 0:
        epsilons = [epsilons[0]]  # You don't actually use epsilon in AllSubgroups mode, just chose random value.

    # Calculate utility for subgroups (measure time separately)
    print(f"\033[94mrunning for condition: {condition} treatment: {treatment}\033[0m")
    with timer() as utility_timer:
        utility_all = calculate_ate_safe(df, TREATMENT_COL, tgtO)
    utility_time = utility_timer()

    for epsilon in epsilons:
        if chosen_mode == 0:
            print(f"Running with epsilon: {epsilon}")

        # Common parameters for all algorithms
        common = dict(
            df=df,
            treatment_col=TREATMENT_COL,
            tgtO=tgtO,
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
        _rw_unlearning_kw_direct = dict(common, algorithm=apriori, size_stop=0.8,
                                        optimization_mode=OPTIMIZATION_MODES[0])
        _rw_unlearning_kw_hybrid = dict(common, algorithm=apriori, size_stop=1, optimization_mode=OPTIMIZATION_MODES[1])

        algo_dispatch = {
            "BruteForce": lambda: naive_calc_utility_for_subgroups(**_naive_kw),
            "Apriori": lambda: apriori_calc_utility_for_subgroups(**_apriori_kw),
            "FPGrowth": lambda: apriori_calc_utility_for_subgroups(**_fpgrowth_kw),
            "MultiProcessing": lambda: multiProcessing_calc_utility_for_subgroups(**_opt_fp_kw),
            "RW_Direct": lambda: rw_unlearning_calc_utility_for_subgroups(**_rw_unlearning_kw_direct),
            "RW_Hybrid": lambda: rw_unlearning_calc_utility_for_subgroups(**_rw_unlearning_kw_hybrid),
        }

        dispatch_key = algorithm_name
        if algorithm_name == "Naive":  # Assuming "Naive" is the name for BruteForce
            dispatch_key = "BruteForce"
        elif algorithm_name == "RW":
            dispatch_key = "RW_Direct"
        try:
            with timer() as elapsed:
                res = algo_dispatch[dispatch_key]()
            algorithm_time = elapsed()

            # Add all timing components
            total_time = algorithm_time + utility_time + attr_vals_time

            if chosen_mode == 0: # Homogeneity check
                # 'res' is now a Boolean (True = Homogeneous, False = Not Homogeneous)
                if res:
                    print(f"\033[92mResult: Homogeneous\033[0m")  # Green
                else:
                    print(f"\033[91mResult: NOT Homogeneous (Violation Found)\033[0m")  # Red

                append_homogeneity_results(
                    # Use the algorithm name for logging
                    algorithm_name=algorithm_name,
                    treatment=treatment,
                    condition=condition,
                    delta=delta,
                    epsilon=epsilon,
                    homogeneity_status=res,
                    runtime_seconds=total_time
                )
            else:  # Only append timing results for AllSubgroups mode
                subgroup_data, num_subgroups = res
                save_results_to_excel(algorithm_name, subgroup_data, num_subgroups, condition,
                                      treatment, delta, index=i)

                append_timing_results(algorithm_name, condition, treatment, num_subgroups, delta,
                                      total_time)

        except KeyError:
            raise ValueError(f"Unknown algorithm name: {algorithm_name} (using dispatch key: {dispatch_key})")

        print(f"⏱  Total execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        if attr_vals_time > 0:
            print(f"   - Attribute values calculation: {attr_vals_time:.2f} seconds")
        print(f"   - Utility calculation: {utility_time:.2f} seconds")
        print(f"   - Algorithm execution: {algorithm_time:.2f} seconds")


def clean_results_files(mode):
    """Delete algorithms_time.xlsx and homogeneity_results.xlsx in ../graphs unless -d is passed."""
    skip_delete = '-d' in sys.argv
    results_dir_graphs = Path("../graphs")
    results_dir_graphs.mkdir(exist_ok=True)
    time_xlsx = results_dir_graphs / "algorithms_time.xlsx"
    homog_xlsx = results_dir_graphs / "homogeneity_results.xlsx"
    files_to_delete = [homog_xlsx] if mode == 0 else [time_xlsx]
    if not skip_delete:
        for f in files_to_delete:
            if f.exists():
                f.unlink()
        print("🧹 Results files reset.")
    else:
        print("⚠️  Results files NOT reset (append mode, -d flag given)")


def process_dataset(i, treated_rules_datasets, good_treatments, chosen_mode, chosen_algorithm_name, tgtO):
    """
    Process a single dataset with the given parameters.
    """
    dataset = treated_rules_datasets[i]
    df = pd.read_csv(dataset)
    condition = good_treatments[i]["condition"]
    attr, _ = list(condition.items())[0]
    treatment = good_treatments[i]["treatment"]

    # Measure attr_vals calculation time
    with timer() as attr_timer:
        attr_vals = {
            col: sorted(v for v in df[col].dropna().unique()
                        if str(v).upper() != "UNKNOWN")
            for col in df.columns if col not in [attr, TREATMENT_COL, *treatment.keys(), tgtO]
        }
    attr_vals_time = attr_timer()

    for delta in DELTAS:
        if len(df) < delta:
            print(f"Skipping delta {delta} for treatment {i + 1}: DataFrame too small ({len(df)} rows).")
            continue  # Skip if the filtered DataFrame is too small

        print(f"Running for delta: {delta}")
        # Pass attr_vals_time only for naive DFS (algorithm name "BruteForce"), otherwise pass 0
        attr_time = attr_vals_time if chosen_algorithm_name == "BruteForce" else 0

        # Check for RW algorithm by name
        if chosen_algorithm_name == "RW":
            # For algorithm "RW" (random walks), run multiple times
            num_runs = 5
            for run_num in range(num_runs):
                print(f"--- Run number: {run_num} ---")
                # Pass algorithm name
                run_experiments(chosen_mode, chosen_algorithm_name, delta, df, tgtO, attr_vals, condition, treatment, i,
                                attr_time)
        else:
            # For other algorithms, run once
            # Pass algorithm name
            run_experiments(chosen_mode, chosen_algorithm_name, delta, df, tgtO, attr_vals, condition, treatment, i,
                            attr_time)


def main():
    # Output the results
    tgtO = "ConvertedSalary"  # Target outcome column in the dataset
    TREATMENT_FILE = "Chosen10Treatments.json"  # Use the new file name
    OUTPUT_DIR_NAME = 'processed_db'

    print(f"Loading treatments from {TREATMENT_FILE}...")
    try:
        with open(TREATMENT_FILE, "r") as f:
            good_treatments = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Treatment file '{TREATMENT_FILE}' not found.")
        return

    num_expected_datasets = len(good_treatments)
    print(f"Found {num_expected_datasets} treatments in the JSON file.")
    base_data_dir = Path('../stackoverflow').resolve()
    processed_db_dir = base_data_dir / OUTPUT_DIR_NAME

    treated_rules_datasets = []
    for i in range(1, num_expected_datasets + 1):
        filename = f"so_countries_treatment_{i}_encoded.csv"
        file_path = processed_db_dir / filename

        if file_path.exists():
            treated_rules_datasets.append(str(file_path))
        else:
            print(f"Warning: Dataset for index {i} not found at {file_path}. Skipping.")

    if not treated_rules_datasets:
        print("Error: No processed datasets found. Please run batch_process_treatments.py first.")
        return

    print(f"Identified {len(treated_rules_datasets)} existing datasets for experiments.")

    print(f"Available Algorithms: {ALGORITHM_NAMES}")
    chosen_mode = int(input(f"Choose your experiment mode {list(enumerate(MODES))}: \n"))
    clean_results_files(chosen_mode)

    algorithms_to_run = ALGORITHM_NAMES

    for chosen_algorithm_name in reversed(algorithms_to_run):
        for i in range(len(treated_rules_datasets)):
            process_dataset(i, treated_rules_datasets, good_treatments, chosen_mode, chosen_algorithm_name, tgtO)


if __name__ == "__main__":
    main()