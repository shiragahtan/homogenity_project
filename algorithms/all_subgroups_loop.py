import sys
import json
import datetime
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from time import perf_counter
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
from greedy_algorithm import calc_utility_for_subgroups as greedy_calc_utility_for_subgroups
from random_algorithm import calc_utility_for_subgroups as random_calc_utility_for_subgroups
from causalForest_algorithm import calc_utility_for_subgroups as causalForest_calc_utility_for_subgroups
from algorithms.code.code.main import run_wte_homogeneity_baseline

# --- Configuration ---
# Load config
with open('../configs/config.json', 'r') as f:
    config = json.load(f)

# DATASET PATHS
#FULL_DATASET_PATH = '../stackoverflow/so_countries_col_new_encoded.csv'
FULL_DATASET_PATH = '../german_credit/german_data_encoded.csv'
RULES_FILE = 'GermanChosen10Treatments.json'

# DELTAS = config['DELTAS']
DELTAS = [100]

# --- ALGORITHM SETUP ---
ALGORITHM_NAMES = ["FPGrowth", "RW", "Greedy", "Random", "CausalForest", "WTE"]

# If you want Random to act as a baseline dependent on RW's count, set this True
RUN_RANDOM_BASELINE = True

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
    "WTE": 9,
}

MODES = config['MODES']
EPSILONS = [0.1]
#EPSILONS = [76000]
NUM_RW_RUNS = 5
TREATMENT_COL = config['TREATMENT_COL']  # 'TempTreatment'
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
    # Convert dicts to string for Excel logging
    chosen_treatment_df = pd.DataFrame([{"Condition": str(condition), "Treatment": str(treatment)}])

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
        "num_subgroups": num_subgroups,
        "run_time_seconds": runtime_seconds,
        "run_time_minutes": runtime_seconds / 60,
        "enumeration_time_sec": enumeration_time,
        "iteration_time_sec": iteration_time
    }

    _append_df_to_excel(excel_path, new_row)
    print(f"🧬 Homogeneity results appended to {excel_path}")


def run_single_execution(algo_func, algorithm_name, chosen_mode, condition, treatment, delta, epsilon,
                         utility_time, attr_vals_time, index=0):
    """
    Helper function to run the actual algorithm logic, timing, and logging.
    """
    with timer() as elapsed:
        res = algo_func()
    algorithm_time = elapsed()

    # Add all timing components
    total_time = algorithm_time + utility_time + attr_vals_time

    if chosen_mode == 0:  # Homogeneity check
        homogeneity_status = res
        num_checked = None
        enum_time = None
        iter_time = None

        if isinstance(res, tuple):
            if len(res) == 2:
                homogeneity_status = res[0]
                num_checked = res[1]
            elif len(res) == 3:
                homogeneity_status, enum_time, iter_time = res
            elif len(res) == 4:
                homogeneity_status, num_checked, enum_time, iter_time = res

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
            subgroup_data, num_subgroups = res, len(res) if isinstance(res, list) else 0

        save_results_to_excel(algorithm_name, subgroup_data, num_subgroups, condition,
                              treatment, delta, index=index)

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
    # Note: df here is already filtered and treated
    print(f"\033[94mrunning for condition: {condition} treatment: {treatment}\033[0m")
    with timer() as utility_timer:
        utility_all = calculate_ate_safe(df, TREATMENT_COL, tgtO, delta)
    utility_time = utility_timer()

    for epsilon in epsilons:
        if chosen_mode == 0:
            print(f"Running with epsilon: {epsilon}")

        common = dict(
            df=df,
            treatment_col=TREATMENT_COL,
            tgtO=tgtO,
            delta=delta,
            epsilon=epsilon,
            mode=chosen_mode,
            utility_all=utility_all
        )

        _naive_kw = dict(common, attr_vals=attr_vals)
        _apriori_kw = dict(common, algorithm=apriori)
        _fpgrowth_kw = dict(common, algorithm=fpgrowth)
        _opt_fp_kw = dict(common, n_jobs=mp.cpu_count())
        _rw_unlearning_kw_direct = dict(common, algorithm=apriori, size_stop=0.8,
                                        optimization_mode=OPTIMIZATION_MODES[0])
        _random_kw = dict(common, n_subgroups=force_n_subgroups if force_n_subgroups else 1000)
        _greedy_kw = dict(common)
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
            "WTE": lambda: run_wte_homogeneity_baseline(**common),
        }

        dispatch_key = algorithm_name
        if algorithm_name == "Naive":
            dispatch_key = "BruteForce"
        elif algorithm_name == "RW":
            dispatch_key = "RW_Direct"

        try:
            result = run_single_execution(
                algo_dispatch[dispatch_key], algorithm_name, chosen_mode,
                condition, treatment, delta, epsilon, utility_time, attr_vals_time, index=i
            )

            if algorithm_name == "RW" and chosen_mode == 0:
                rw_count = 0
                if isinstance(result, tuple):
                    if len(result) >= 2 and isinstance(result[1], int):
                        rw_count = result[1]

                should_run_random = (
                        rw_count > 0
                        and RUN_RANDOM_BASELINE
                        and "Random" in ALGORITHM_NAMES
                        and "RW" in ALGORITHM_NAMES
                )

                if should_run_random:
                    print(f"\n\033[95m>>> Triggering Random Baseline with n={rw_count} (matched to RW) <<<\033[0m")
                    run_experiments(
                        chosen_mode, "Random", delta, df, tgtO, attr_vals,
                        condition, treatment, i, attr_vals_time,
                        force_n_subgroups=rw_count
                    )
                elif rw_count == 0:
                    print("RW checked 0 subgroups, skipping random baseline.")

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


def process_dataset_dynamic(i, rule, full_df, chosen_mode, chosen_algorithm_name, tgtO):
    """
    Dynamically applies the rule to the full dataset in-memory.
    Mimics the logic of the previous batch processing script.
    """
    # 1. Parse the rule (Keys and Values are now numeric due to previous step)
    condition_dict = rule["condition"]
    condition_attr, condition_val = list(condition_dict.items())[0]

    treatment_dict = rule["treatment"]
    treatment_attr, treatment_val = list(treatment_dict.items())[0]

    print(f"--- Processing Rule #{i + 1}: {condition_attr}={condition_val} -> {treatment_attr}={treatment_val} ---")

    # 2. Filter DataFrame based on condition
    #    Since data is numeric, we do direct comparison
    try:
        sub_df = full_df[full_df[condition_attr] == condition_val].copy()
    except KeyError as e:
        print(f"Error: Column {e} not found in dataset. Skipping.")
        return

    if sub_df.empty:
        print(f"No rows match condition {condition_attr}={condition_val}. Skipping.")
        return

    # 3. Drop the invariant condition column
    #    (Matches previous logic: remove the column we just filtered on)
    sub_df = sub_df.drop(columns=[condition_attr])

    # 4. Apply Treatment (Create TempTreatment)
    if treatment_attr not in sub_df.columns:
        print(f"Treatment column {treatment_attr} not found (or was dropped). Skipping.")
        return

    # Create binary treatment column: 1 if matches value, 0 otherwise
    sub_df[TREATMENT_COL] = (sub_df[treatment_attr] == treatment_val).astype(int)

    if sub_df[TREATMENT_COL].sum() == 0:
        print(f"Warning: Treatment resulted in 0 treated individuals. Skipping.")
        return

    # 5. Calculate Attribute Values (For BruteForce)
    #    Exclude: The newly created TempTreatment, the Target Outcome, and the keys of the treatment
    #    (Note: condition_attr is already dropped)
    with timer() as attr_timer:
        attr_vals = {
            col: sorted(sub_df[col].dropna().unique())
            for col in sub_df.columns
            if col not in [TREATMENT_COL, tgtO, *treatment_dict.keys()]
        }
    attr_vals_time = attr_timer()

    # 6. Run Experiments for Deltas
    for delta in DELTAS:
        if len(sub_df) < delta:
            print(f"Skipping delta {delta}: DataFrame too small ({len(sub_df)} rows).")
            continue

        print(f"Running for delta: {delta}")
        attr_time = attr_vals_time if chosen_algorithm_name == "BruteForce" else 0

        if chosen_algorithm_name == "RW":
            num_runs = NUM_RW_RUNS
            print(f"Running {num_runs} times")
            for run_num in range(num_runs):
                print(f"--- Run number: {run_num} ---")
                run_experiments(chosen_mode, chosen_algorithm_name, delta, sub_df, tgtO, attr_vals,
                                condition_dict, treatment_dict, i, attr_time)
        else:
            run_experiments(chosen_mode, chosen_algorithm_name, delta, sub_df, tgtO, attr_vals,
                            condition_dict, treatment_dict, i, attr_time)


def main():
    # tgtO = "ConvertedSalary"
    tgtO = "credit_risk"

    # 1. Load the Rules (JSON with numeric values)
    print(f"Loading treatments from {RULES_FILE}...")
    try:
        with open(RULES_FILE, "r") as f:
            # Each line is a JSON object
            rules_list = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: {RULES_FILE} not found.")
        return

    # 2. Load the Full Numeric Dataset ONCE
    print(f"Loading full dataset from {FULL_DATASET_PATH}...")
    if not Path(FULL_DATASET_PATH).exists():
        print(f"Error: Dataset {FULL_DATASET_PATH} not found.")
        return

    full_df = pd.read_csv(FULL_DATASET_PATH)
    print(f"Loaded dataset with {len(full_df)} rows and {len(full_df.columns)} columns.")

    # 3. User Setup
    print(f"Available Algorithms: {ALGORITHM_NAMES}")
    try:
        chosen_mode = int(input(f"Choose mode {list(enumerate(MODES))}: \n"))
    except ValueError:
        chosen_mode = 0  # Default

    clean_results_files(chosen_mode)

    algorithms_to_run = ALGORITHM_NAMES[:]

    # Filter algorithms based on mode
    if chosen_mode == 1 and "RW" in algorithms_to_run:
        if "RW" in algorithms_to_run: algorithms_to_run.remove("RW")
    if RUN_RANDOM_BASELINE and "Random" in algorithms_to_run:
        if "Random" in algorithms_to_run: algorithms_to_run.remove("Random")

    # 4. Main Execution Loop
    for chosen_algorithm_name in reversed(algorithms_to_run):
        for i, rule in enumerate(rules_list):
            # Call the new dynamic processor
            process_dataset_dynamic(i, rule, full_df, chosen_mode, chosen_algorithm_name, tgtO)


if __name__ == "__main__":
    main()