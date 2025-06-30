#!/usr/bin/env python3
"""
Generate per‑rule heat‑maps that show the **number** of homogeneous executions
(colour‑coded 0–4 red → 5 white → 6–max green) and annotate each cell with
"homogeneous/total" and the average run‑time (seconds).

The script reads *homogeneity_results.xlsx* and writes PNGs to the
*homogeneity_rule_heatmaps* folder.
"""

# ============================== IMPORTS ======================================
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ============================ CONFIGURATION ==================================
RESULTS_FILE = Path("homogeneity_results.xlsx")     # input Excel
OUTPUT_DIR   = Path("homogeneity_rule_heatmaps")    # output folder
MIDPOINT     = 5                                     # white centre of palette

# ============================ DATA LOADING ===================================
print(f"Reading results from {RESULTS_FILE.resolve()}")
df = pd.read_excel(RESULTS_FILE)

# Convert possible string booleans ("TRUE"/"FALSE") → bool ---------------------
if df["homogeneity_status"].dtype == object:
    df["homogeneity_status"] = (
        df["homogeneity_status"].astype(str)
        .str.upper()
        .map({"TRUE": True, "FALSE": False})
    )

# Ensure output directory exists ---------------------------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Distinct treatment/condition rules -----------------------------------------
rules = df[["treatment", "condition"]].drop_duplicates().reset_index(drop=True)
print(f"Found {len(rules)} unique treatment/condition rule(s)")

# Diverging colour‑map: dark‑red → white → dark‑green -------------------------
cmap = LinearSegmentedColormap.from_list(
    "red_white_green",
    [
        (0.60, 0.00, 0.00),   # dark red   ~ #990000
        (1.00, 1.00, 1.00),   # white
        (0.00, 0.45, 0.00),   # dark green ~ #007300
    ],
    N=256,
)

# ============================== MAIN LOOP ====================================
for rule_idx, rule in rules.iterrows():
    treatment = rule["treatment"]
    condition = rule["condition"]
    print(f"\nProcessing rule {rule_idx + 1}: {treatment!s} | {condition!s}")

    # Sub‑set for the current rule -------------------------------------------
    rule_df = df[(df["treatment"] == treatment) & (df["condition"] == condition)]
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

    heatmap_data = agg.pivot(index="delta", columns="epsilon", values="num_hom")

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

    # Normalisation: centre palette at MIDPOINT ------------------------------
    vmax = heatmap_data.max().max()
    norm = TwoSlopeNorm(vmin=0, vcenter=MIDPOINT, vmax=vmax)

    # ---------------------------- PLOT ---------------------------------------
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        norm=norm,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="grey",
        cbar_kws={"label": "Number Homogeneous"},
    )

    plt.title(
        f"Rule Heatmap: Treatment={treatment}, Condition={condition}\n"
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
    filename = OUTPUT_DIR / f"heatmap_rule_{rule_idx + 1}_{safe_t}__{safe_c}.png"

    plt.savefig(filename, dpi=300)
    plt.close()

    # Print location; fall back to absolute path if not a sub‑path ------------
    try:
        rel = filename.resolve().relative_to(Path.cwd().resolve())
        print(f"   → saved to {rel}")
    except ValueError:
        print(f"   → saved to {filename.resolve()}")

print("\n✅  All heat‑maps generated.")
