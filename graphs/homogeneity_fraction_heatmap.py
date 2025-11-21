#!/usr/bin/env python3
"""
Generate per‑rule heat‑maps that show the **number** of homogeneous executions
(colour‑coded 0–6 red → 7–15 green) and annotate each cell with
"homogeneous/total" and the average run‑time (seconds).

The script reads *homogeneity_results.xlsx* and writes PNGs to the
*homogeneity_rule_heatmaps* folder, creating separate heatmaps for each algorithm.
"""

import os
import json
from pathlib import Path
import ast  # Added for safely evaluating string representations of dictionaries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

#RESULTS_FILE = Path("newer7_old_homogeneity_results.xlsx")  # input Excel
RESULTS_FILE = Path("homogeneity_results.xlsx")
# UPDATED PATH: The rule mapping file is located in the '../algorithms/' folder
RULE_MAP_FILE = Path("../algorithms/Chosen10Treatments.json")
OUTPUT_DIR = Path("homogeneity_rule_heatmaps")  # output folder

# The point at which the diverging colour‑map switches from red → green.
THRESHOLD_HOMOGENEOUS = 7



def load_rule_indices(json_file_path):
    """
    Loads the (condition, treatment) pairs from the JSONL file and creates
    two maps for index and filename construction.
    """
    value_to_index_map = {}
    index_to_full_string_map = {}

    # Read the JSONL file line by line
    with open(json_file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            # Extract full key-value pairs
            cond_key = list(data["condition"].keys())[0]
            cond_val = list(data["condition"].values())[0]
            treat_key = list(data["treatment"].keys())[0]
            treat_val = list(data["treatment"].values())[0]

            # Create the full strings (Attr__Value)
            full_cond_str = f"{cond_key}__{cond_val}"
            full_treat_str = f"{treat_key}__{treat_val}"

            # Key for the map is (condition_value, treatment_value)
            rule_key_value = (str(cond_val), str(treat_val))
            rule_number = idx + 1

            # Only map the first occurrence (idx is the 'real index')
            if rule_key_value not in value_to_index_map:
                value_to_index_map[rule_key_value] = rule_number
                index_to_full_string_map[rule_number] = (full_cond_str, full_treat_str)

    return value_to_index_map, index_to_full_string_map


print(f"Reading rule indices from {RULE_MAP_FILE.resolve()}")
# Load the two required maps
value_to_index_map, index_to_full_string_map = load_rule_indices(RULE_MAP_FILE)

print(f"Reading results from {RESULTS_FILE.resolve()}")
df = pd.read_excel(RESULTS_FILE)

# Convert possible string booleans ("TRUE"/"FALSE") → bool --------------------
if df["homogeneity_status"].dtype == object:
    df["homogeneity_status"] = (
        df["homogeneity_status"].astype(str)
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
    )

# Ensure output directory exists ---------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Distinct treatment/condition/algorithm rules -----------------------------------------
rules = df[["treatment", "condition", "algorithm"]].drop_duplicates().reset_index(drop=True)
print(f"Found {len(rules)} unique treatment/condition/algorithm rule(s)")

# SORT RULES BY ALGORITHM NAME -----------------------------------------------
rules = rules.sort_values(by="algorithm").reset_index(drop=True)
print("Rules sorted by algorithm name for processing order.")

# Diverging colour‑map: dark‑red → white → dark‑green -------------------------
cmap = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.60, 0.00, 0.00), (1.00, 1.00, 1.00), (0.00, 0.45, 0.00),
    ],
    N=256,
)

for rule_idx, rule in rules.iterrows():
    # These hold the raw string from the Excel file (either value or dict string)
    treatment_input = rule["treatment"]
    condition_input = rule["condition"]
    algorithm = rule["algorithm"]

    # Safely parse the string to get the actual value for the lookup key
    try:
        condition_dict = ast.literal_eval(str(condition_input))
        condition_val = str(list(condition_dict.values())[0])
    except (ValueError, TypeError, SyntaxError, IndexError):
        # Assume it's already the raw value if parsing fails
        condition_val = str(condition_input)

    try:
        treatment_dict = ast.literal_eval(str(treatment_input))
        treatment_val = str(list(treatment_dict.values())[0])
    except (ValueError, TypeError, SyntaxError, IndexError):
        # Assume it's already the raw value if parsing fails
        treatment_val = str(treatment_input)

    # DETERMINE CORRECT RULE NUMBER AND FILENAME STRINGS FROM MAPPING
    rule_key = (condition_val, treatment_val)
    rule_number = value_to_index_map.get(rule_key, None)

    full_cond_str = None
    full_treat_str = None

    if rule_number is None:
        # Print the original input strings for debugging
        print(f"\nProcessing rule (No Index): {treatment_input!s} | {condition_input!s} | {algorithm!s}")

        rule_number_str = f"NO_IDX_{rule_idx + 1}"
        # Fallback for full strings uses the raw extracted values if no index is found
        full_cond_str = condition_val
        full_treat_str = treatment_val
    else:
        print(f"\nProcessing rule {rule_number}: {treatment_input!s} | {condition_input!s} | {algorithm!s}")
        rule_number_str = str(rule_number)
        full_cond_str, full_treat_str = index_to_full_string_map[rule_number]

    # Sub‑set for the current rule -------------------------------------------
    # Subsetting MUST use the original input strings if the Excel file contains the dict strings
    rule_df = df[
        (df["treatment"] == treatment_input) & (df["condition"] == condition_input) & (df["algorithm"] == algorithm)]
    if rule_df.empty:
        print("    → no data; skipping …")
        continue

    # Aggregation: number homogeneous + total --------------------------------
    agg = (
        rule_df
        .groupby(["delta", "epsilon"])["homogeneity_status"]
        .agg(num_hom="sum", total="count")
        .reset_index()
    )

    # Compute fraction homogeneous for coloring
    agg["fraction_hom"] = agg["num_hom"] / agg["total"]
    heatmap_data = agg.pivot(index="delta", columns="epsilon", values="fraction_hom")

    # Average run‑time for annotation ----------------------------------------
    runtimes = (
        rule_df
        .groupby(["delta", "epsilon"])["run_time_seconds"]
        .mean()
        .reset_index()
        .pivot(index="delta", columns="epsilon", values="run_time_seconds")
    )

    # Compose annotation text -------------------------------------------------
    annot = heatmap_data.copy().astype(str)
    for row in agg.itertuples(index=False):
        annot.loc[row.delta, row.epsilon] = (
            f"{int(row.num_hom)}/{int(row.total)}\n{runtimes.loc[row.delta, row.epsilon]:.1f}s"
        )

    # Use a diverging colormap centered at 0.5 for the fraction
    cmap_fraction = LinearSegmentedColormap.from_list(
        "red_white_green", [(0.60, 0.00, 0.00), (1.0, 1.0, 1.0), (0.00, 0.45, 0.00)], N=256
    )
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    cmap_to_use = cmap_fraction

    # ---------------------------- PLOT ---------------------------------------
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap=cmap_to_use,
        norm=norm,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="grey",
        cbar_kws={"label": "Fraction Homogeneous"},
    )

    plt.title(
        f"Rule Heatmap: Treatment={treatment_val}, Condition={condition_val}, Algorithm={algorithm}\n"
        "(Annotation: Homogeneous/Total and Runtime)"
    )
    plt.xlabel("Epsilon")
    plt.ylabel("Delta")
    plt.tight_layout()

    # Clean up filename components (remove spaces, colons, braces, quotes, parens, commas)
    trans = str.maketrans({":": "_", " ": "_", "'": "", "{": "", "}": "", "(": "", ")": "", ",": ""})
    safe_a = str(algorithm).translate(trans)

    # Apply translation to the *full* strings derived from the JSON mapping
    safe_t = full_treat_str.translate(trans)
    safe_c = full_cond_str.translate(trans)

    # Construct filename: heatmap_rule_<idx>_<algo_name>_<treatment>__<condition>.png
    filename = OUTPUT_DIR / f"heatmap_rule_{rule_number_str}_{safe_a}_{safe_t}__{safe_c}.png"

    plt.savefig(filename, dpi=300)
    plt.close()

    # Print location; fall back to absolute path if not a sub‑path ------------
    try:
        rel = filename.resolve().relative_to(Path.cwd().resolve())
        print(f"    → saved to {rel}")
    except ValueError:
        print(f"    → saved to {filename.resolve()}")

print("\n✅  All heat‑maps generated.")