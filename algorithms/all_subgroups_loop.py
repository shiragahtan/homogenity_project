import re
import sys
import json
import datetime
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from time import perf_counter
from contextlib import contextmanager
import os
import queue # Import queue for Empty check

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
TIMEOUT_SECONDS = 3600  # 1 Hour Cutoff
#TIMEOUT_SECONDS = 8000  # 1 Hour Cutoff

with open('../configs/config.json', 'r') as f:
    config = json.load(f)

CHOSEN_DS = config["CHOSEN_DATASET"]

if CHOSEN_DS not in config['DATASETS']:
    raise ValueError(f"Dataset '{CHOSEN_DS}' not found in config.json")

ds_config = config['DATASETS'][CHOSEN_DS]

FULL_DATASET_PATH = ds_config['FULL_DATASET_PATH']
RULES_FILE = ds_config['RULES_FILE']
DELTAS = ds_config['DELTAS']
EPSILONS = ds_config['EPSILONS']
TARGET_COLUMN_NAME = ds_config['TARGET_COLUMN']
ATTRIBUTE_WEIGHTS = ds_config.get('ATTRIBUTE_WEIGHTS', {})

print(f"🔹 Loaded Configuration for: {CHOSEN_DS}")
print(f"   Dataset: {FULL_DATASET_PATH}")
print(f"   Rules: {RULES_FILE}")
print(f"   Target: {TARGET_COLUMN_NAME}")

# --- ENABLED ALL ALGORITHMS AS REQUESTED ---
#ALGORITHM_NAMES = ["FPGrowth", "RW", "Greedy", "CausalForest", "Random", "WTE"]
#ALGORITHM_NAMES = ["FPGrowth", "RW", "CausalForest", "Random", "WTE"]
#ALGORITHM_NAMES = ["FPGrowth", "RW", "CausalForest", "Random"]
#ALGORITHM_NAMES = ["Greedy", "WTE"]
#ALGORITHM_NAMES = ["Greedy"]
#ALGORITHM_NAMES = ["WTE"]
#ALGORITHM_NAMES = ["FPGrowth", "RW"]
#ALGORITHM_NAMES = ["RW"]
#ALGORITHM_NAMES = ["FPGrowth"]
#ALGORITHM_NAMES = ["MultiProcessing"]
ALGORITHM_NAMES = ["RW", "Random"]
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
NUM_RW_RUNS = 3
TREATMENT_COL = config['TREATMENT_COL']
OPTIMIZATION_MODES = config.get('OPTIMIZATION_MODES', ['direct'])

""" Timing helper """
@contextmanager
def timer() -> callable:
    t0 = perf_counter()
    yield lambda: perf_counter() - t0

def worker_wrapper(func, kwargs, result_queue):
    """
    Wrapper function to run the algorithm in a separate process.
    Puts the result into the queue upon completion.
    """
    try:
        res = func(**kwargs)
        result_queue.put(("success", res))
    except Exception as e:
        # Send error string or object back to main process
        result_queue.put(("error", str(e)))

def save_results_to_csv(algorithm_name, subgroup_data, num_subgroups, condition, treatment, delta, index=0):
    """Save subgroup analysis results to a CSV file (Metadata in JSON)."""
    subgroup_df = pd.DataFrame(subgroup_data)
    
    # Metadata dict since CSV has no sheets
    metadata = {
        "NumSubgroups": num_subgroups,
        "Condition": str(condition),
        "Treatment": str(treatment)
    }

    results_dir = Path("../algorithms_results")
    results_dir.mkdir(exist_ok=True)

    # Main Data File
    output_file = results_dir / f"{CHOSEN_DS}_{algorithm_name}_subgroups_results_delta_{delta}_{index}.csv"
    # Metadata File
    meta_file = results_dir / f"{CHOSEN_DS}_{algorithm_name}_subgroups_results_delta_{delta}_{index}_metadata.json"
    
    try:
        subgroup_df.to_csv(output_file, index=False)
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"✔  {len(subgroup_data):,} subgroups saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")
        
    return str(output_file)


def _append_dict_to_csv(csv_path: Path, new_row_dict: dict):
    df_new = pd.DataFrame([new_row_dict])
    if not csv_path.exists():
        df_new.to_csv(csv_path, index=False, mode='w')
    else:
        df_new.to_csv(csv_path, index=False, mode='a', header=False)


def _append_dict_to_excel(excel_path: Path, new_row_dict: dict):
    """Helper to append a dictionary row to an Excel file."""
    df_new = pd.DataFrame([new_row_dict])
    if not excel_path.exists():
        df_new.to_excel(excel_path, index=False)
    else:
        try:
            # Read existing file, concat new data, and save back
            df_existing = pd.read_excel(excel_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_excel(excel_path, index=False)
        except Exception as e:
            print(f"❌ Error appending to Excel: {e}")


def append_timing_results(algorithm_name, condition, treatment, num_subgroups, delta, runtime_seconds):
    results_dir = Path("../graphs")
    results_dir.mkdir(exist_ok=True)
    # Changed extension to .csv
    csv_path = results_dir / f"{CHOSEN_DS}_algorithms_time.csv"
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
    _append_dict_to_csv(csv_path, new_row)
    print(f"✅ Timing results appended to {csv_path}")


def append_homogeneity_results(algorithm_name, treatment, condition, delta, epsilon, homogeneity_status,
                               runtime_seconds, num_subgroups=None,
                               enumeration_time=None, iteration_time=None):
    results_dir = Path("../graphs")
    results_dir.mkdir(exist_ok=True)
    
    # Changed extension to .xlsx for Mode 0
    xlsx_path = results_dir / f"{CHOSEN_DS}_homogeneity_results.xlsx"
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
    
    _append_dict_to_excel(xlsx_path, new_row)
    print(f"🧬 Homogeneity results appended to {xlsx_path}")


def run_single_execution(target_func, target_kwargs, algorithm_name, chosen_mode, condition, treatment, delta, epsilon,
                         utility_time, attr_vals_time, index=0):
    
    result_queue = mp.Queue()
    # Create the process
    p = mp.Process(target=worker_wrapper, args=(target_func, target_kwargs, result_queue))

    print(f"▶️  [{algorithm_name}] Starting execution with {TIMEOUT_SECONDS}s timeout...")
    
    with timer() as elapsed:
        p.start()
        # Wait for the process to finish or timeout
        p.join(timeout=TIMEOUT_SECONDS)
        
        timed_out = False
        if p.is_alive():
            print(f"⏳ [{algorithm_name}] Timed out after {TIMEOUT_SECONDS}s! Killing process...")
            p.terminate()
            p.join() # Ensure it's dead
            timed_out = True
            res = "TIMEOUT"
        else:
            # Process finished, check queue
            if not result_queue.empty():
                status, payload = result_queue.get()
                if status == "success":
                    res = payload
                else:
                    print(f"❌ [{algorithm_name}] Worker raised error: {payload}")
                    res = None # Or raise
            else:
                print(f"❌ [{algorithm_name}] Worker finished but returned no result (Crash?).")
                res = None

    algorithm_time = elapsed()
    total_time = algorithm_time + utility_time + attr_vals_time

    # --- SHOW RUNTIME FOR BOTH MODES HERE ---
    print(f"⏱️  [{algorithm_name}] Finished. Total Time: {total_time:.4f}s (Algo: {algorithm_time:.4f}s + Overhead: {utility_time + attr_vals_time:.4f}s)")
    # ----------------------------------------

    # Handle Timeout Case Specifics
    if timed_out or res == "TIMEOUT":
        if chosen_mode == 0:
            print(f"\033[93mResult: TIMED OUT\033[0m")
            append_homogeneity_results(algorithm_name, treatment, condition, delta, epsilon, 
                                       "Timed Out", total_time, None, None, None)
            return "TIMEOUT", 0
        else:
            print(f"\033[93mResult: TIMED OUT - No subgroups saved.\033[0m")
            append_timing_results(algorithm_name, condition, treatment, "Timed Out", delta, total_time)
            return "TIMEOUT"

    if chosen_mode == 0:  # Homogeneity check
        raw_result = res
        num_checked = None
        enum_time = None
        iter_time = None

        if isinstance(res, tuple):
            if len(res) == 2:
                raw_result = res[0]
                num_checked = res[1]
            elif len(res) == 3:
                raw_result, enum_time, iter_time = res
            elif len(res) == 4:
                raw_result, num_checked, enum_time, iter_time = res
            elif len(res) == 5:
                raw_result, num_checked, enum_time, iter_time, _ = res

        is_homogeneous = bool(raw_result)
        status_str = "Homogeneous" if is_homogeneous else "NOT Homogeneous (Violation Found)"
        color = "\033[92m" if is_homogeneous else "\033[91m"
        print(f"{color}Result: {status_str}\033[0m")
        if num_checked is not None:
            print(f"Subgroups checked: {num_checked}")

        append_homogeneity_results(algorithm_name, treatment, condition, delta, epsilon, is_homogeneous, total_time,
                                   num_checked, enum_time, iter_time)
        
        # Return result with num_checked for potential use in main loop
        return raw_result, num_checked
    else:
        subgroup_data = res
        num_subgroups = 0
        if isinstance(res, tuple):
            subgroup_data = res[0]
            if len(res) >= 2: num_subgroups = res[1]

        save_results_to_csv(algorithm_name, subgroup_data, num_subgroups, condition, treatment, delta, index=index)
        append_timing_results(algorithm_name, condition, treatment, num_subgroups, delta, total_time)
        return res


def run_experiments(chosen_mode, chosen_algorithm_name, delta, df, tgtO, attr_vals, condition, treatment, i,
                    attr_vals_time=0, force_n_subgroups=None, override_epsilons=None):
    algorithm_name = chosen_algorithm_name
    print(f"Using algorithm: {algorithm_name}")
    
    # Allow overriding epsilons for specific random baseline runs
    if override_epsilons is not None:
        epsilons = override_epsilons
    else:
        # Use global EPSILONS loaded from config
        epsilons = EPSILONS
        if chosen_mode != 0:
            epsilons = [epsilons[0]]

    print(f"\033[94mrunning for condition: {condition} treatment: {treatment}\033[0m")
    with timer() as utility_timer:
        utility_all = calculate_ate_safe(df, TREATMENT_COL, tgtO, delta)
    utility_time = utility_timer()

    execution_results = [] # Store (epsilon, num_checked) for each run

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

        # Add weights SPECIFICALLY for RW here
        _rw_unlearning_kw_direct = dict(common, algorithm=apriori, size_stop=0.8,
                                        optimization_mode=OPTIMIZATION_MODES[0],
                                        attribute_weights=ATTRIBUTE_WEIGHTS)

        _random_kw = dict(common, n_subgroups=force_n_subgroups if force_n_subgroups else 1000)
        _greedy_kw = dict(common)
        _causalForest_kw = dict(common)

        # UPDATED: Use Tuples of (Function, Kwargs) instead of Lambdas
        # This is required for mp.Process to pickle the target correctly
        algo_dispatch = {
            "BruteForce": (naive_calc_utility_for_subgroups, _naive_kw),
            "Apriori": (apriori_calc_utility_for_subgroups, _apriori_kw),
            "FPGrowth": (apriori_calc_utility_for_subgroups, _fpgrowth_kw),
            "MultiProcessing": (multiProcessing_calc_utility_for_subgroups, _opt_fp_kw),
            "RW_Direct": (rw_unlearning_calc_utility_for_subgroups, _rw_unlearning_kw_direct),
            "Greedy": (greedy_calc_utility_for_subgroups, _greedy_kw),
            "Random": (random_calc_utility_for_subgroups, _random_kw),
            "CausalForest": (causalForest_calc_utility_for_subgroups, _causalForest_kw),
            "WTE": (run_wte_homogeneity_baseline, common),
        }

        dispatch_key = algorithm_name
        if algorithm_name == "Naive":
            dispatch_key = "BruteForce"
        elif algorithm_name == "RW":
            dispatch_key = "RW_Direct"

        try:
            target_func, target_kw = algo_dispatch[dispatch_key]
            
            result = run_single_execution(
                target_func, target_kw, algorithm_name, chosen_mode,
                condition, treatment, delta, epsilon, utility_time, attr_vals_time, index=i
            )
            
            # Capture num_checked for return
            num_checked = 0
            if result == "TIMEOUT":
                 num_checked = 0
            elif chosen_mode == 0 and isinstance(result, tuple) and len(result) >= 2:
                # run_single_execution returns (raw_res, num_checked)
                num_checked = result[1]
            
            execution_results.append((epsilon, num_checked))

        except KeyError:
            raise ValueError(f"Unknown algorithm name: {algorithm_name}")
            
    return execution_results


def clean_results_files(mode):
    skip_delete = '-d' in sys.argv
    results_dir_graphs = Path("../graphs")
    results_dir_graphs.mkdir(exist_ok=True)
    
    # Changed to .xlsx for Homogeneity Results (Mode 0)
    homog_xlsx = results_dir_graphs / f"{CHOSEN_DS}_homogeneity_results.xlsx"
    # Kept as .csv for Mode 1 Results
    time_csv = results_dir_graphs / f"{CHOSEN_DS}_algorithms_time.csv"
    
    files_to_delete = [homog_xlsx] if mode == 0 else [time_csv]
    if not skip_delete:
        for f in files_to_delete:
            if f.exists():
                f.unlink()
        print("🧹 Results files reset.")
    else:
        print("⚠️  Results files NOT reset (append mode, -d flag given)")


def encode_dataframe_local(df):
    """
    Replicates the exact logic of the old batch script:
    Maps unique values in THIS SUBSET to 1..N based on appearance order.
    """
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()

    for column in categorical_columns:
        # Get unique values in this filtered subset
        unique_values = df_encoded[column].unique()
        # Map them to 1, 2, 3...
        column_mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}
        df_encoded[column] = df_encoded[column].map(column_mapping)

    # Handle booleans
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded


def process_dataset_dynamic(i, rule, full_df, chosen_mode, chosen_algorithm_name, tgtO):
    # 1. Parse string rule
    condition_dict = rule["condition"]
    condition_attr, condition_val = list(condition_dict.items())[0]
    treatment_dict = rule["treatment"]
    treatment_attr, treatment_val = list(treatment_dict.items())[0]

    print(f"--- Processing Rule #{i + 1}: {condition_attr}={condition_val} -> {treatment_attr}={treatment_val} ---")

    # 2. Filter (String comparison)
    try:
        sub_df = full_df[full_df[condition_attr] == condition_val].copy()
    except KeyError as e:
        print(f"Error: Column {e} not found in dataset. Skipping.")
        return

    if sub_df.empty:
        print(f"No rows match condition {condition_attr}={condition_val}. Skipping.")
        return

    # 3. Drop invariant
    sub_df = sub_df.drop(columns=[condition_attr])

    # 4. Apply Treatment
    if treatment_attr not in sub_df.columns:
        print(f"Treatment column {treatment_attr} not found. Skipping.")
        return

    sub_df[TREATMENT_COL] = (sub_df[treatment_attr] == treatment_val).astype(int)

    # Drop the original source column to prevent multicollinearity
    # This prevents the algorithms from seeing two identical columns (Source & Treatment)
    if treatment_attr in sub_df.columns:
        sub_df = sub_df.drop(columns=[treatment_attr])

    if sub_df[TREATMENT_COL].sum() == 0:
        print(f"Warning: Treatment resulted in 0 treated individuals. Skipping.")
        return

    # 5. LOCAL ENCODING (Matches old batch script)
    if CHOSEN_DS != "acs":
        sub_df_encoded = encode_dataframe_local(sub_df)
        sub_df_encoded = sub_df_encoded.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))
        # Ensure outcome is numeric
        sub_df_encoded[tgtO] = pd.to_numeric(sub_df_encoded[tgtO], errors='coerce')

    else:
        sub_df_encoded = sub_df.copy()
        sub_df_encoded = sub_df_encoded.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))

    # 6. Calculate Attribute Values using local encoding
    with timer() as attr_timer:
        attr_vals = {
            col: sorted(sub_df_encoded[col].dropna().unique())
            for col in sub_df_encoded.columns
            if col not in [TREATMENT_COL, tgtO, *treatment_dict.keys()]
        }
    attr_vals_time = attr_timer()

    # 7. Run Experiments
    for delta in DELTAS:
        if len(sub_df_encoded) < delta:
            print(f"Skipping delta {delta}: DataFrame too small.")
            continue
        print(f"Running for delta: {delta}")
        attr_time = attr_vals_time if chosen_algorithm_name == "BruteForce" else 0

        if chosen_algorithm_name == "RW":
            # --- MODIFIED FLOW: RUN ALL RW FIRST, THEN ALL RANDOM ---
            rw_runs_data = [] # To store list of [(eps, count), ...] per run

            # Step 1: Run RW 3 times (or NUM_RW_RUNS)
            for run_num in range(NUM_RW_RUNS):
                print(f"--- Run number: {run_num} ---")
                
                # run_experiments now returns list of (eps, count)
                run_results = run_experiments(
                    chosen_mode, chosen_algorithm_name, delta, sub_df_encoded, tgtO, attr_vals,
                    condition_dict, treatment_dict, i, attr_time
                )
                rw_runs_data.append(run_results)

            # Step 2: Run Random Baseline if needed
            if RUN_RANDOM_BASELINE and "Random" in ALGORITHM_NAMES:
                print(f"\n\033[95m>>> Starting Random Baseline Sequence (Matching {NUM_RW_RUNS} RW runs) <<<\033[0m")
                
                # Iterate exactly in the order RW ran
                for run_idx, run_results in enumerate(rw_runs_data):
                    print(f"--- Matching Random Run #{run_idx} ---")
                    
                    # For each epsilon that was checked in this run
                    for (eps, count) in run_results:
                        if count == 0:
                            print(f"Skipping Random for epsilon {eps} (RW checked 0 subgroups).")
                            continue
                            
                        print(f"Triggering Random for epsilon {eps} with n={count}")
                        # We force specific epsilon to keep strict alignment
                        run_experiments(
                            chosen_mode, "Random", delta, sub_df_encoded, tgtO, attr_vals, 
                            condition_dict, treatment_dict, i, attr_time, 
                            force_n_subgroups=count, override_epsilons=[eps]
                        )

        else:
            # Standard single run for other algorithms
            run_experiments(chosen_mode, chosen_algorithm_name, delta, sub_df_encoded, tgtO, attr_vals, condition_dict,
                            treatment_dict, i, attr_time)


def main():
    # Use global TARGET_COLUMN_NAME loaded from config
    tgtO = TARGET_COLUMN_NAME

    # 1. Load Rules
    print(f"Loading treatments from {RULES_FILE}...")
    try:
        with open(RULES_FILE, "r") as f:
            rules_list = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: {RULES_FILE} not found.")
        return

    # 2. Load & Clean Dataset (The exact steps from batch file)
    print(f"Loading full dataset from {FULL_DATASET_PATH}...")
    if not Path(FULL_DATASET_PATH).exists():
        print(f"Error: Dataset {FULL_DATASET_PATH} not found.")
        return

    full_df = pd.read_csv(FULL_DATASET_PATH)

    # --- EXACT CLEANING LOGIC FROM OLD BATCH FILE ---
    full_df = full_df.loc[:, ~full_df.columns.str.startswith('Unnamed')]
    full_df = full_df[~full_df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
    # -----------------------------------------------

    print(f"Loaded and cleaned dataset with {len(full_df)} rows and {len(full_df.columns)} columns.")

    # 3. Setup
    print(f"Available Algorithms: {ALGORITHM_NAMES}")
    try:
        chosen_mode = int(input(f"Choose mode {list(enumerate(MODES))}: \n"))
    except ValueError:
        chosen_mode = 0

    clean_results_files(chosen_mode)

    algorithms_to_run = ALGORITHM_NAMES[:]
    if chosen_mode == 1:
        for algo in ["RW", "Greedy", "Random", "CausalForest", "WTE"]:
            if algo in algorithms_to_run: algorithms_to_run.remove(algo)
    if RUN_RANDOM_BASELINE and "Random" in algorithms_to_run:
        if "Random" in algorithms_to_run: algorithms_to_run.remove("Random")

    # 4. Execution
    for chosen_algorithm_name in reversed(algorithms_to_run):
        for i, rule in enumerate(rules_list):
            process_dataset_dynamic(i, rule, full_df, chosen_mode, chosen_algorithm_name, tgtO)


if __name__ == "__main__":
    main()
