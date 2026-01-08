import pandas as pd
import numpy as np
import time
import random
import sys
import json
import logging
import re
from pathlib import Path
from mlxtend.frequent_patterns import fpgrowth

# --- USER IMPORTS ---
# Ensure these are in the same folder or python path
import rw_unlearning as rw_algo
import apriori_algorithm as bf_algo
from yarden_files.ATE_update import calculate_ate_safe

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
        config = json.load(fp)
except FileNotFoundError:
    logger.error(f"❌ Error: Config file not found at {CONFIG_PATH}")
    sys.exit(1)

CHOSEN_DS = config["CHOSEN_DATASET"]
ds_config = config['DATASETS'][CHOSEN_DS]

# Paths
raw_path = ds_config['FULL_DATASET_PATH']
FULL_DATASET_PATH = (CONFIG_PATH.parent / raw_path).resolve()

# Columns
TREATMENT_COL = config['TREATMENT_COL']
OUTCOME_COL = ds_config.get('OUTCOME_COL', ds_config['TARGET_COLUMN'])

# Parameters
BASE_DELTA = ds_config['DELTAS'][0]
EPSILON = ds_config['EPSILONS'][0]
ATTRIBUTE_WEIGHTS = ds_config.get('ATTRIBUTE_WEIGHTS', {})
MAX_ATTRS_FROM_CONFIG = ds_config.get("MAX_SCALABILITY_ATTRIBUTES", 20)

# --- EXPERIMENT CONSTANTS ---
ROW_STEP_PERCENT = 10  # Jumps of 10% (10, 20, 30...)
ATTR_START = 3  # Start with 3 attributes
ATTR_STEP = 2  # Add 2 attributes per step
REPEATS = 3  # Runs per configuration


# ==========================================
# 2. DATA PRE-PROCESSING (From your main.py)
# ==========================================
def load_and_clean_data():
    logger.info(f"Loading dataset: {FULL_DATASET_PATH}")
    if not FULL_DATASET_PATH.exists():
        logger.error("Dataset file not found!")
        sys.exit(1)

    df = pd.read_csv(FULL_DATASET_PATH)

    # 1. Remove Unnamed columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    # 2. Remove rows with "UNKNOWN"
    df = df[~df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)

    # 3. Rename columns to avoid regex issues (standardize)
    df = df.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))

    # 4. Ensure Outcome is numeric
    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors='coerce')
    df = df.dropna(subset=[OUTCOME_COL])

    # 5. Ensure Treatment is binary (if not already)
    # (Assuming the dataset usually comes with a pre-set treatment col,
    # otherwise we might need to create it like in your main.py.
    # For scalability, we assume T exists.)
    if TREATMENT_COL not in df.columns:
        logger.warning(f"⚠️ Treatment col '{TREATMENT_COL}' missing. Creating dummy random treatment for testing.")
        df[TREATMENT_COL] = np.random.randint(0, 2, df.shape[0])

    logger.info(f"✅ Data Loaded & Cleaned: {len(df)} rows, {len(df.columns)} cols")
    return df


# ==========================================
# 3. EXPERIMENT LOOP
# ==========================================
def get_base_ate(df):
    try:
        return calculate_ate_safe(df, TREATMENT_COL, OUTCOME_COL, 0)
    except Exception:
        return 0.0


def run_experiment_loop(df, current_delta, weights, exp_type, x_val):
    results = []

    # Calculate ATE for *this specific sample*
    utility_all = get_base_ate(df)

    logger.info(f"   [Context] Rows: {len(df)} | Delta: {current_delta} | Global ATE: {utility_all:.4f}")

    for i in range(REPEATS):
        logger.info(f"   ► Iteration {i + 1}/{REPEATS}...")

        # --- 1. RUN FPGrowth (Brute Force / Ground Truth) ---
        bf_start = time.time()
        try:
            # Note: Using mode=0 (Homogeneity Check)
            bf_verdict, bf_count, _, _, _ = bf_algo.calc_utility_for_subgroups(
                0, fpgrowth, df, TREATMENT_COL, OUTCOME_COL, current_delta, EPSILON, utility_all
            )
        except Exception as e:
            logger.error(f"      ❌ BF Failed: {e}")
            bf_verdict, bf_count = True, 0

        bf_time = time.time() - bf_start

        # --- 2. RUN Random Walk (Your Algo) ---
        rw_start = time.time()
        try:
            rw_verdict, rw_count = rw_algo.calc_utility_for_subgroups(
                0, None, df, TREATMENT_COL, current_delta, EPSILON,
                outcome_col=OUTCOME_COL,
                utility_all=utility_all,
                k_walks=1500,
                attribute_weights=weights
            )
        except Exception as e:
            logger.error(f"      ❌ RW Failed: {e}")
            rw_verdict, rw_count = True, 0

        rw_time = time.time() - rw_start

        # Log Summary for this run
        logger.info(f"      [BF] {bf_time:.2f}s, {bf_count} checked | [RW] {rw_time:.2f}s, {rw_count} checked")

        # --- Metrics ---
        match = (bf_verdict == rw_verdict)
        results.append({
            "Experiment": exp_type,
            "X_Value": x_val,
            "Iteration": i + 1,
            "BF_Time": bf_time,
            "RW_Time": rw_time,
            "BF_Checked": bf_count,
            "RW_Checked": rw_count,
            "Accuracy": 1.0 if match else 0.0
        })

    return results


# ==========================================
# 4. SCALABILITY TESTS
# ==========================================
def run_row_scalability(full_df, all_results):
    logger.info("\n" + "=" * 60)
    logger.info("🚀 STARTING ROW SCALABILITY (Data Size)")
    logger.info("=" * 60)

    fractions = [x / 100.0 for x in range(ROW_STEP_PERCENT, 101, ROW_STEP_PERCENT)]

    for frac in fractions:
        # 1. Sample Rows
        if frac == 1.0:
            df_sample = full_df.copy()
        else:
            df_sample = full_df.sample(frac=frac, random_state=42)

        # 2. Scale Delta (Proportional to data size)
        # If full data (100%) uses Delta=500, then 10% data uses Delta=50
        scaled_delta = int(BASE_DELTA * frac)
        scaled_delta = max(5, scaled_delta)  # Safety floor

        logger.info(f"\n🔹 Testing {frac * 100:.0f}% Data ({len(df_sample)} rows)")

        res = run_experiment_loop(
            df_sample, scaled_delta, ATTRIBUTE_WEIGHTS, "Row_Scalability", len(df_sample)
        )
        all_results.extend(res)


def run_col_scalability(full_df, all_results):
    logger.info("\n" + "=" * 60)
    logger.info("🚀 STARTING COLUMN SCALABILITY (Attributes)")
    logger.info("=" * 60)

    feature_pool = [c for c in full_df.columns if c not in [TREATMENT_COL, OUTCOME_COL]]

    # Limit attributes based on Config Max
    limit = min(len(feature_pool), MAX_ATTRS_FROM_CONFIG)
    attr_counts = list(range(ATTR_START, limit + 1, ATTR_STEP))

    # Use full dataset (or fixed sample)
    df_fixed = full_df.copy()

    for count in attr_counts:
        logger.info(f"\n🔹 Testing {count} Attributes")

        # 1. Sample Attributes
        random.seed(42)
        selected_feats = random.sample(feature_pool, count)
        cols_to_use = selected_feats + [TREATMENT_COL, OUTCOME_COL]

        df_subset = df_fixed[cols_to_use]

        # 2. Filter Weights
        relevant_weights = {k: v for k, v in ATTRIBUTE_WEIGHTS.items() if k in selected_feats}

        # Use Base Delta (fixed rows -> fixed delta)
        res = run_experiment_loop(
            df_subset, BASE_DELTA, relevant_weights, "Col_Scalability", count
        )
        all_results.extend(res)


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_and_clean_data()

    all_data = []
    run_row_scalability(df, all_data)
    run_col_scalability(df, all_data)

    # Save Results
    out_df = pd.DataFrame(all_data)
    filename = f"scalability_results_{CHOSEN_DS}.xlsx"
    out_df.to_excel(filename, index=False)
    logger.info(f"\n🎉 Experiments Completed! Results saved to {filename}")