#!/usr/bin/env python3
"""
Generate per‑rule heat‑maps that show the **number** of homogeneous executions
(colour‑coded 0–6 red → 7–15 green) and annotate each cell with
"homogeneous/total" and the average run‑time (seconds).

The script reads *homogeneity_results.xlsx* and writes PNGs to the
*homogeneity_rule_heatmaps* folder, creating separate heatmaps for each algorithm.
"""

# ============================== IMPORTS ======================================
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ============================ CONFIGURATION ==================================
RESULTS_FILE = Path("homogeneity_results1.xlsx")     # input Excel
OUTPUT_DIR   = Path("homogeneity_rule_heatmaps")    # output folder

# The point at which the diverging colour‑map switches from red → green.
# All counts < THRESHOLD_HOMOGENEOUS are rendered with red hues, all counts
# >= THRESHOLD_HOMOGENEOUS with green hues.  The total number of executions is
# 15, so a value of 7 cleanly splits the scale 0‑6 vs 7‑15.
THRESHOLD_HOMOGENEOUS = 7

# ============================ DATA LOADING ===================================
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

# Diverging colour‑map: dark‑red → white → dark‑green -------------------------
cmap = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.60, 0.00, 0.00),   # dark red   ~ #990000
        (1.00, 1.00, 1.00),   # white (placed at the threshold)
        (0.00, 0.45, 0.00),   # dark green ~ #007300
    ],
    N=256,
)

# ============================== MAIN LOOP ====================================
for rule_idx, rule in rules.iterrows():
    treatment = rule["treatment"]
    condition = rule["condition"]
    algorithm = rule["algorithm"]
    print(f"\nProcessing rule {rule_idx + 1}: {treatment!s} | {condition!s} | {algorithm!s}")

    # Sub‑set for the current rule -------------------------------------------
    rule_df = df[(df["treatment"] == treatment) & (df["condition"] == condition) & (df["algorithm"] == algorithm)]
    if rule_df.empty:
        print("   → no data; skipping …")
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
        f"Rule Heatmap: Treatment={treatment}, Condition={condition}, Algorithm={algorithm}\n"
        "(Annotation: Homogeneous/Total and Runtime)"
    )
    plt.xlabel("Epsilon")
    plt.ylabel("Delta")
    plt.tight_layout()

    # ---------------------------- SAVE ---------------------------------------
    # Clean up filename components (remove spaces, colons, braces, quotes)
    trans = str.maketrans({":": "_", " ": "_", "'": "", "{" : "", "}" : ""})
    safe_t = str(treatment).translate(trans)
    safe_c = str(condition).translate(trans)
    safe_a = str(algorithm).translate(trans)
    filename = OUTPUT_DIR / f"heatmap_rule_{rule_idx + 1}_{safe_a}_{safe_t}__{safe_c}.png"

    plt.savefig(filename, dpi=300)
    plt.close()

    # Print location; fall back to absolute path if not a sub‑path ------------
    try:
        rel = filename.resolve().relative_to(Path.cwd().resolve())
        print(f"   → saved to {rel}")
    except ValueError:
        print(f"   → saved to {filename.resolve()}")

print("\n✅  All heat‑maps generated.")
