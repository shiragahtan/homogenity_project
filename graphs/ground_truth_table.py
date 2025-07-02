import json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

"""
Homogeneity‑runtime gradient table — corrected column alignment
-----------------------------------------------------------
This version fixes the cell-indexing logic so:
1. The row‑label column (c == 0) is skipped when applying colours.
2. Fraction lookups use c-1 so that data columns align with the 'frac' DataFrame.
"""

# -------------------------------------------------------------------------
# 1) CONFIG
# -------------------------------------------------------------------------
with open("../configs/config.json") as f:
    cfg = json.load(f)

DELTAS   = sorted(cfg["DELTAS"])
EPSILONS = sorted(cfg["EPSILONS"])
ALGOS    = cfg["ALGORITHM_NAMES"][:-2]

# -------------------------------------------------------------------------
# 2) LOAD DATA
# -------------------------------------------------------------------------
df = pd.read_excel("homogeneity_results_no_gaps_bool.xlsx")

# ensure proper dtypes
df["delta"]   = df["delta"].astype(int)
df["epsilon"] = df["epsilon"].astype(int)
if df["homogeneity_status"].dtype == object:
    df["homogeneity_status"] = (
        df["homogeneity_status"].astype(str)
                                 .str.strip().str.upper()
                                 .map({"TRUE": True, "FALSE": False})
    )
df["algorithm"] = df["algorithm"].astype(str).str.strip()

# -------------------------------------------------------------------------
# 3) TREATMENTS
# -------------------------------------------------------------------------
rules = []
with open("../algorithms/Shira_Treatments.json") as f:
    for line in f:
        rec = json.loads(line)
        rules.append((str(rec["treatment"]), str(rec["condition"])))

# -------------------------------------------------------------------------
# 4) AGGREGATE
# -------------------------------------------------------------------------
agg = (
    df.groupby(["treatment","condition","delta","epsilon","algorithm"],
               as_index=False)
      .agg(run_time=("run_time_seconds","mean"),
           hom=("homogeneity_status","sum"),
           tot=("homogeneity_status","count"))
)
agg["frac"] = agg["hom"] / agg["tot"]

# -------------------------------------------------------------------------
# 5) BUILD MATRICES
# -------------------------------------------------------------------------
rows = [f"Rule {i+1} - Δ={d}" for i in range(len(rules)) for d in DELTAS]
cols = [f"{algo}\nε={eps}" for eps in EPSILONS for algo in ALGOS]

runtime = pd.DataFrame("", index=rows, columns=cols)
frac    = pd.DataFrame(np.nan, index=rows, columns=cols)

for i,(t,c) in enumerate(rules,1):
    for d in DELTAS:
        rkey = f"Rule {i} - Δ={d}"
        for eps in EPSILONS:
            for algo in ALGOS:
                mask = (
                    (agg["treatment"]==t)&(agg["condition"]==c)&
                    (agg["delta"]==d)&(agg["epsilon"]==eps)&
                    (agg["algorithm"]==algo)
                )
                if mask.any():
                    runtime.at[rkey,f"{algo}\nε={eps}"] = f"{agg.loc[mask,'run_time'].iat[0]:.1f}"
                    frac.at[rkey,   f"{algo}\nε={eps}"] = agg.loc[mask,'frac'].iat[0]

# -------------------------------------------------------------------------
# 6) COLOUR MAP
# -------------------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list("rwg", [(0.60,0,0),(1,1,1),(0,0.45,0)], N=256)
norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

# -------------------------------------------------------------------------
# 7) TABLE PLOT
# -------------------------------------------------------------------------
CELL_W, CELL_H = 0.70, 0.65
fig_w = CELL_W * len(cols) + 3
fig_h = CELL_H * len(rows) + 2

plt.close('all')
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

tbl = ax.table(
    cellText  = runtime.values,
    rowLabels = runtime.index,
    colLabels = runtime.columns,
    cellLoc   = 'center', loc='center', colLoc='center'
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(6)
tbl.scale(1.0, 1.6)

# -------------------------------------------------------------------------
# 8) COLOURING LOOP (fixed skip+indexing)
# -------------------------------------------------------------------------
for (r, c), cell in tbl.get_celld().items():
    if r == 0 or c == -1:
        cell.get_text().set_weight('bold')
        cell.set_facecolor('white')
        continue

    fr = frac.iat[r-1, c]  # <-- Corrected indexing

    if np.isnan(fr):
        color = '#d8d8d8'
    elif fr == 1.0:
        color = '#228B22'
    elif fr == 0.0:
        color = '#B22222'
    else:
        color = cmap(norm(fr))

    # print(f"DEBUG: cell(r={r-1}, c={c}), label=({runtime.index[r-1]}, {runtime.columns[c]}), frac={fr:.2f}, color={color}")

    cell.set_facecolor(color)
    if color in ['#228B22', '#B22222']:
        cell.get_text().set_color('white')
    else:
        cell.get_text().set_color('black')

plt.title("Runtime Table — Gradient by Homogeneity (red 0 ↔ 1 green)", pad=20)
plt.tight_layout()
plt.savefig("../graphs/homogeneity_runtime_gradient_table.png", dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved full table to ../graphs/homogeneity_runtime_gradient_table.png")
