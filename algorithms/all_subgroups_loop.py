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
from greedy_algorithm import calc_utility_for_subgroups as greedy_calc_utility_for_subgroups
from random_algorithm import calc_utility_for_subgroups as random_calc_utility_for_subgroups
from causalForest_algorithm import calc_utility_for_subgroups as causalForest_calc_utility_for_subgroups

# Load config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

# DELTAS = config['DELTAS']
DELTAS = [1000]

# ALGORITHM_NAMES = config['ALGORITHM_NAMES']
ALGORITHM_NAMES = ["Apriori", "RW", "Random", "Greedy"]

RUN_RANDOM = False
RUN_GREEDY = False

if "Random" in ALGORITHM_NAMES:
    RUN_RANDOM = True
    ALGORITHM_NAMES.remove("Random")

if "Greedy" in ALGORITHM_NAMES:
    RUN_GREEDY = True
    ALGORITHM_NAMES.remove("Greedy")

ALGORITHM_DISPATCH_MAP = {
    "BruteForce": 0,
    "Apriori": 1,
    "FPGrowth": 2,
    "MultiProcessing": 3,
    "RW_Direct": 4,
    "RW_Hybrid": 5,
    "Greedy": 6,
    "Random": 7,
    "CausalForest": 8,
}

MODES = config['MODES']
EPSILONS = [30000]
NUM_RW_RUNS = 5
TREATMENT_COL = config['TREATMENT_COL']
OPTIMIZATION_MODES = config.get('OPTIMIZATION_MODES', ['direct'])

""" Timing helper """


@contextmanager
def timer() -> callable:
    t0 = perf_counter()
    yield lambda: perf_counter() - t0


def save_results_to_excel(algorithm_name, subgroup_data, num_subgroups, condition, treatment, delta, index=0):
    """Save subgroup analysis results to an Excel file."""
    subgroup_df = pd.DataFrame(subgroup_data)
    summary_df = pd.DataFrame([{"NumSubgroups": num_subgroups}])
    chosen_treatment_df = pd.DataFrame([{"Condition": condition, "Treatment": treatment}])

    results_dir = Path("../algorithms_results")
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f"{algorithm_name}_subgroups_results_delta_{delta}_{index}.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        chosen_treatment_df.to_excel(writer, sheet_name="ChosenTreatment", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        subgroup_df.to_excel(writer, sheet_name="Subgroups", index=False)

    print(f"✔  {len(subgroup_data):,} subgroups saved to {output_file}")
    return str(output_file)


def _append_df_to_excel(excel_path: Path, new_row: dict):
    """Append a new row to an Excel file."""
    if not excel_path.exists():
        df = pd.DataFrame([new_row])
        df.to_excel(excel_path, index=False)
    else:
        existing_df = pd.read_excel(excel_path)
        updated_df = pd.concat([existing_df, pd.DataFrame([new_row])], ignore_index=True)
        updated_df.to_excel(excel_path, index=False)


def append_timing_results(algorithm_name, condition, treatment, num_subgroups, delta, runtime_seconds):
    """Append algorithm timing results to an Excel file."""
    results_dir = Path("../graphs")
    results_dir.mkdir(exist_ok=True)

    excel_path = results_dir / "algorithms_time.xlsx"
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                               runtime_seconds, num_subgroups=None,
                               enumeration_time=None, iteration_time=None):
    """Append homogeneity check results to an Excel file."""
    results_dir = Path("../graphs")
    results_dir.mkdir(exist_ok=True)

    excel_path = results_dir / "homogeneity_results.xlsx"
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = {
        "date": current_date,
        "algorithm": algorithm_name,
        "treatment": str(treatment),
        "condition": str(condition),
        "delta": delta,
        "epsilon": epsilon,
        "homogeneity_status": homogeneity_status,
        "num_subgroups": num_subgroups,  # Now populated for Apriori/FPGrowth/RW
        "run_time_seconds": runtime_seconds,
        "run_time_minutes": runtime_seconds / 60,
        "enumeration_time_sec": enumeration_time,
        "iteration_time_sec": iteration_time
    }

    _append_df_to_excel(excel_path, new_row)
    print(f"🧬 Homogeneity results appended to {excel_path}")


def run_single_execution(algo_func, algorithm_name, chosen_mode, condition, treatment, delta, epsilon,
                         utility_time, attr_vals_time):
    """
    Helper function to run the actual algorithm logic, timing, and logging.
    Returns: The result of the algorithm (e.g., tuple (bool, count) or list).
    """
    with timer() as elapsed:
        res = algo_func()
    algorithm_time = elapsed()

    # Add all timing components
    total_time = algorithm_time + utility_time + attr_vals_time

    if chosen_mode == 0:  # Homogeneity check
        # Standardize result: Expecting (status, count) or just status
        homogeneity_status = res
        num_checked = None  # Default to None (empty in CSV)
        enum_time = None
        iter_time = None

        # All these algorithms return (Status, Count)
        if isinstance(res, tuple):
            if len(res) == 2:
                # RW / Apriori / FPGrowth / Greedy / Random
                homogeneity_status = res[0]
                num_checked = res[1]
            elif len(res) == 3:
                # Older signatures or specific custom returns
                homogeneity_status, enum_time, iter_time = res

        status_str = "Homogeneous" if homogeneity_status else "NOT Homogeneous (Violation Found)"
        color = "\033[92m" if homogeneity_status else "\033[91m"
        print(f"{color}Result: {status_str}\033[0m")
        if num_checked is not None:
            print(f"Subgroups checked: {num_checked}")

        append_homogeneity_results(
            algorithm_name=algorithm_name,
            treatment=treatment,
            condition=condition,
            delta=delta,
            epsilon=epsilon,
            homogeneity_status=homogeneity_status,
            runtime_seconds=total_time,
            num_subgroups=num_checked,
            enumeration_time=enum_time,
            iteration_time=iter_time
        )
        return res
    else:
        # AllSubgroups mode
        if isinstance(res, tuple) and len(res) == 4:
            subgroup_data, num_subgroups, _, _ = res
        elif isinstance(res, tuple) and len(res) == 2:
            subgroup_data, num_subgroups = res
        else:
            # Fallback for unexpected formats
            subgroup_data, num_subgroups = res, len(res)

        save_results_to_excel(algorithm_name, subgroup_data, num_subgroups, condition,
                              treatment, delta, index=0)

        append_timing_results(algorithm_name, condition, treatment, num_subgroups, delta,
                              total_time)
        return res


def run_experiments(chosen_mode, chosen_algorithm_name, delta, df, tgtO, attr_vals, condition, treatment, i,
                    attr_vals_time=0, force_n_subgroups=None):
    """
    Main experiment runner.
    """

    algorithm_name = chosen_algorithm_name
    print(f"Using algorithm: {algorithm_name}")
    epsilons = EPSILONS
    if chosen_mode != 0:
        epsilons = [epsilons[0]]

    # Calculate utility
    print(f"\033[94mrunning for condition: {condition} treatment: {treatment}\033[0m")
    with timer() as utility_timer:
        utility_all = calculate_ate_safe(df, TREATMENT_COL, tgtO)
    utility_time = utility_timer()

    for epsilon in epsilons:
        if chosen_mode == 0:
            print(f"Running with epsilon: {epsilon}")

        # Common parameters
        common = dict(
            df=df,
            treatment_col=TREATMENT_COL,
            tgtO=tgtO,
            delta=delta,
            epsilon=epsilon,
            mode=chosen_mode,
            utility_all=utility_all
        )

        # Algorithm setup
        _naive_kw = dict(common, attr_vals=attr_vals)
        _apriori_kw = dict(common, algorithm=apriori)
        _fpgrowth_kw = dict(common, algorithm=fpgrowth)
        _opt_fp_kw = dict(common, n_jobs=mp.cpu_count())
        _rw_unlearning_kw_direct = dict(common, algorithm=apriori, size_stop=0.8,
                                        optimization_mode=OPTIMIZATION_MODES[0])

        _random_kw = dict(common, n_subgroups=force_n_subgroups if force_n_subgroups else 1000)

        _greedy_kw = dict(common, n_subgroups=force_n_subgroups if force_n_subgroups else 1000)

        _causalForest_kw = dict(common)

        algo_dispatch = {
            "BruteForce": lambda: naive_calc_utility_for_subgroups(**_naive_kw),
            "Apriori": lambda: apriori_calc_utility_for_subgroups(**_apriori_kw),
            "FPGrowth": lambda: apriori_calc_utility_for_subgroups(**_fpgrowth_kw),
            "MultiProcessing": lambda: multiProcessing_calc_utility_for_subgroups(**_opt_fp_kw),
            "RW_Direct": lambda: rw_unlearning_calc_utility_for_subgroups(**_rw_unlearning_kw_direct),
            "Greedy": lambda: greedy_calc_utility_for_subgroups(**_greedy_kw),
            "Random": lambda: random_calc_utility_for_subgroups(**_random_kw),
            "CausalForest": lambda: causalForest_calc_utility_for_subgroups(**_causalForest_kw),
        }

        # Resolve dispatch key
        dispatch_key = algorithm_name
        if algorithm_name == "Naive":
            dispatch_key = "BruteForce"
        elif algorithm_name == "RW":
            dispatch_key = "RW_Direct"

        try:
            # --- RUN THE MAIN ALGORITHM ---
            result = run_single_execution(
                algo_dispatch[dispatch_key], algorithm_name, chosen_mode,
                condition, treatment, delta, epsilon, utility_time, attr_vals_time
            )

            if algorithm_name == "RW" and chosen_mode == 0:
                # RW returns (status, count)
                rw_count = 0
                if isinstance(result, tuple) and len(result) >= 2:
                    if isinstance(result[1], int):
                        rw_count = result[1]

                if rw_count > 0:
                    # 1. Trigger Random (if enabled)
                    if RUN_RANDOM:
                        print(f"\n\033[95m>>> Triggering Random Baseline with n={rw_count} (matched to RW) <<<\033[0m")
                        run_experiments(
                            chosen_mode, "Random", delta, df, tgtO, attr_vals,
                            condition, treatment, i, attr_vals_time,
                            force_n_subgroups=rw_count
                        )

                    # 2. Trigger Greedy (if enabled)
                    if RUN_GREEDY:
                        print(f"\n\033[96m>>> Triggering Greedy Baseline with n={rw_count} (matched to RW) <<<\033[0m")
                        run_experiments(
                            chosen_mode, "Greedy", delta, df, tgtO, attr_vals,
                            condition, treatment, i, attr_vals_time,
                            force_n_subgroups=rw_count
                        )
                else:
                    print("RW checked 0 subgroups, skipping baselines.")

        except KeyError:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")


def clean_results_files(mode):
    """Delete results files."""
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
    dataset = treated_rules_datasets[i]
    df = pd.read_csv(dataset)
    condition = good_treatments[i]["condition"]
    attr, _ = list(condition.items())[0]
    treatment = good_treatments[i]["treatment"]

    with timer() as attr_timer:
        attr_vals = {
            col: sorted(v for v in df[col].dropna().unique()
                        if str(v).upper() != "UNKNOWN")
            for col in df.columns if col not in [attr, TREATMENT_COL, *treatment.keys(), tgtO]
        }
    attr_vals_time = attr_timer()

    for delta in DELTAS:
        if len(df) < delta:
            print(f"Skipping delta {delta}: DataFrame too small.")
            continue

        print(f"Running for delta: {delta}")
        attr_time = attr_vals_time if chosen_algorithm_name == "BruteForce" else 0

        if chosen_algorithm_name == "RW":
            num_runs = NUM_RW_RUNS
            print(f"Running {num_runs} times")
            for run_num in range(num_runs):
                print(f"--- Run number: {run_num} ---")
                run_experiments(chosen_mode, chosen_algorithm_name, delta, df, tgtO, attr_vals, condition, treatment, i,
                                attr_time)
        else:
            run_experiments(chosen_mode, chosen_algorithm_name, delta, df, tgtO, attr_vals, condition, treatment, i,
                            attr_time)


def main():
    tgtO = "ConvertedSalary"
    TREATMENT_FILE = "Chosen10Treatments.json"
    OUTPUT_DIR_NAME = 'processed_db'

    print(f"Loading treatments from {TREATMENT_FILE}...")
    try:
        with open(TREATMENT_FILE, "r") as f:
            good_treatments = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: {TREATMENT_FILE} not found.")
        return

    num_expected_datasets = len(good_treatments)
    base_data_dir = Path('../stackoverflow').resolve()
    processed_db_dir = base_data_dir / OUTPUT_DIR_NAME

    treated_rules_datasets = []
    for i in range(1, num_expected_datasets + 1):
        filename = f"so_countries_treatment_{i}_encoded.csv"
        file_path = processed_db_dir / filename
        if file_path.exists():
            treated_rules_datasets.append(str(file_path))
        else:
            print(f"Warning: Dataset {i} not found.")

    if not treated_rules_datasets:
        print("Error: No datasets found.")
        return

    print(f"Available Algorithms: {ALGORITHM_NAMES}")
    chosen_mode = int(input(f"Choose mode {list(enumerate(MODES))}: \n"))
    clean_results_files(chosen_mode)

    algorithms_to_run = ALGORITHM_NAMES

    # Don't run RW in AllSubgroups mode usually
    if chosen_mode == 1 and "RW" in algorithms_to_run:
        try:
            algorithms_to_run.remove("RW")
        except:
            pass

    for chosen_algorithm_name in reversed(algorithms_to_run):
        for i in range(len(treated_rules_datasets)):
            process_dataset(i, treated_rules_datasets, good_treatments, chosen_mode, chosen_algorithm_name, tgtO)


if __name__ == "__main__":
    main()