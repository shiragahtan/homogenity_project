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

# --- 1. USER-DEFINED IMPORTS (Fixed Path) ---
# Add project root to sys.path for module resolution
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / 'yarden_files'))

# Import Algorithms
import rw_unlearning as rw_algo
import brute_force_algorithm as bf_algo
from ATE_update import calculate_ate_safe

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# ==========================================
# 2. CONFIGURATION
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
RULES_FILE = ds_config['RULES_FILE']

# Columns & Params
TREATMENT_COL = config['TREATMENT_COL']
OUTCOME_COL = ds_config.get('OUTCOME_COL', ds_config['TARGET_COLUMN'])
BASE_DELTA = ds_config['DELTAS'][0]
EPSILON = ds_config['EPSILONS'][0]
ATTRIBUTE_WEIGHTS = ds_config.get('ATTRIBUTE_WEIGHTS', {})
MAX_ATTRS = ds_config.get("MAX_SCALABILITY_ATTRIBUTES", 20)
USE_ENCODING = ds_config.get("USE_ENCODING", True)

# --- EXPERIMENT CONSTANTS ---
ROW_STEP_PERCENT = 10  # 10, 20... 100
ATTR_START = 3  # Start with 3 attributes
ATTR_STEP = 2  # Step by 2
REPEATS = 3  # Average over 3 runs


# ==========================================
# 3. PRE-PROCESSING
# ==========================================
def encode_dataframe_local(df):
    """Maps unique values to integers (1..N)."""
    df_encoded = df.copy()

    # Map Categoricals
    categorical_columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
    for column in categorical_columns:
        if column == OUTCOME_COL: continue
        unique_values = df_encoded[column].unique()
        column_mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}
        df_encoded[column] = df_encoded[column].map(column_mapping)

    # Handle Booleans
    bool_columns = df_encoded.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df_encoded[col] = df_encoded[col].astype(int)

    return df_encoded


def load_and_prep_data():
    logger.info(f"Loading dataset: {FULL_DATASET_PATH}")
    df = pd.read_csv(FULL_DATASET_PATH)

    # Standard Cleaning
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df[~df.isin(["UNKNOWN"]).any(axis=1)].reset_index(drop=True)
    df = df.rename(columns=lambda x: re.sub(r'[,:\[\]\{\}"]', '_', x))

    # Filter by First Rule (to match your study context)
    try:
        with open(RULES_FILE, 'r') as f:
            first_rule = json.loads(f.readline())
            cond_dict = first_rule.get("condition", {})
            if cond_dict:
                attr, val = list(cond_dict.items())[0]
                if attr in df.columns:
                    logger.info(f"🔹 Filtering by Rule: {attr} == {val}")
                    df = df[df[attr] == val].copy()
                    df = df.drop(columns=[attr])
    except Exception as e:
        logger.warning(f"⚠️ Rule filter skipped: {e}")

    # Ensure Treatment/Outcome
    if TREATMENT_COL not in df.columns:
        df[TREATMENT_COL] = np.random.randint(0, 2, df.shape[0])
    df[OUTCOME_COL] = pd.to_numeric(df[OUTCOME_COL], errors='coerce')
    df = df.dropna(subset=[OUTCOME_COL])

    return df


# ==========================================
# 4. EXPERIMENT ENGINE
# ==========================================
def run_experiment_batch(df, current_delta, weights, exp_type, x_value_for_plot):
    results = []

    # 1. Encode Locally if needed
    if USE_ENCODING and CHOSEN_DS != "acs":
        df_ready = encode_dataframe_local(df)
    else:
        df_ready = df.copy()

    # 2. Calculate ATE
    try:
        utility_all = calculate_ate_safe(df_ready, TREATMENT_COL, OUTCOME_COL, 0)
    except:
        utility_all = 0.0

    logger.info(f"   ► Rows: {len(df_ready)} | Delta: {current_delta} | ATE: {utility_all:.2f}")

    for i in range(REPEATS):
        # --- A. FPGrowth ---
        t0 = time.time()
        try:
            bf_verdict, bf_count, _, _, _ = bf_algo.calc_utility_for_subgroups(
                0, fpgrowth, df_ready, TREATMENT_COL, OUTCOME_COL, current_delta, EPSILON, utility_all
            )
        except Exception:
            bf_verdict, bf_count = True, 0
        bf_time = time.time() - t0

        # --- B. Random Walk ---
        t0 = time.time()
        try:
            rw_verdict, rw_count = rw_algo.calc_utility_for_subgroups(
                0, None, df_ready, TREATMENT_COL, current_delta, EPSILON,
                outcome_col=OUTCOME_COL, utility_all=utility_all,
                k_walks=1500, attribute_weights=weights
            )
        except Exception:
            rw_verdict, rw_count = True, 0
        rw_time = time.time() - t0

        # --- C. Store Results ---
        # Accuracy: 1 if verdicts match, 0 if not
        match = 1.0 if (bf_verdict == rw_verdict) else 0.0

        results.append({
            "Experiment": exp_type,
            "X_Value": x_value_for_plot,  # <--- This will now be 0.1, 0.2, etc.
            "Iteration": i + 1,
            "BF_Time": bf_time, "RW_Time": rw_time,
            "BF_Checked": bf_count, "RW_Checked": rw_count,
            "Accuracy": match
        })

    return results


# ==========================================
# 5. SCALABILITY TESTS
# ==========================================
def run_row_scalability(full_df, all_results):
    logger.info("\n🚀 STARTING ROW SCALABILITY")

    # 10, 20, ... 100
    fractions = [x / 100.0 for x in range(ROW_STEP_PERCENT, 101, ROW_STEP_PERCENT)]

    for frac in fractions:
        # Sample
        if frac == 1.0:
            df_sample = full_df.copy()
        else:
            df_sample = full_df.sample(frac=frac, random_state=42)

        # Scale Delta
        scaled_delta = int(BASE_DELTA * frac)
        scaled_delta = max(5, scaled_delta)

        # Pass 'frac' as the X_Value so the graph shows 0.1, 0.2...
        logger.info(f"\n🔹 Testing {frac * 100:.0f}% Data")
        res = run_experiment_batch(
            df_sample, scaled_delta, ATTRIBUTE_WEIGHTS, "Row_Scalability", frac
        )
        all_results.extend(res)


def run_col_scalability(full_df, all_results):
    logger.info("\n🚀 STARTING COLUMN SCALABILITY")

    feature_pool = [c for c in full_df.columns if c not in [TREATMENT_COL, OUTCOME_COL]]
    limit = min(len(feature_pool), MAX_ATTRS)
    attr_counts = list(range(ATTR_START, limit + 1, ATTR_STEP))

    df_fixed = full_df.copy()

    for count in attr_counts:
        random.seed(42)
        selected = random.sample(feature_pool, count)
        cols = selected + [TREATMENT_COL, OUTCOME_COL]

        df_subset = df_fixed[cols]
        relevant_weights = {k: v for k, v in ATTRIBUTE_WEIGHTS.items() if k in selected}

        logger.info(f"\n🔹 Testing {count} Attributes")
        # Pass 'count' as X_Value (3, 5, 7...)
        res = run_experiment_batch(
            df_subset, BASE_DELTA, relevant_weights, "Col_Scalability", count
        )
        all_results.extend(res)


if __name__ == "__main__":
    df = load_and_prep_data()
    results = []
    run_row_scalability(df, results)
    run_col_scalability(df, results)

    out = pd.DataFrame(results)
    out.to_excel(f"scalability_results_{CHOSEN_DS}.xlsx", index=False)
    logger.info(f"🎉 Done. Saved to scalability_results_{CHOSEN_DS}.xlsx")
