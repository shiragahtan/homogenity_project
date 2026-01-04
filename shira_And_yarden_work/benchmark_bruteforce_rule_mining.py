"""
Benchmark for brute_force_find_positive_homogeneous_subgroup

Tests the algorithm across different combinations of:
- Treatments (from Chosen10Treatments.json)
- Epsilon values (homogeneity thresholds)
- Delta values (minimum subgroup size percentages)

Outputs:
- results.csv: Full results table
- report.html: Interactive report with visualizations
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

# Setup imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "algorithms"))

from brute_force_search import brute_force_find_positive_homogeneous_subgroup


def load_treatments(json_path: Path) -> List[str]:
    """Load treatment column names from JSONL file (ignore conditions)"""
    treatment_cols = []
    
    with open(json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if "treatment" in entry:
                # Extract column name from treatment dict
                # e.g., {"Exercise": "Daily..."} -> "Exercise"
                for col_name in entry["treatment"].keys():
                    if col_name not in treatment_cols:
                        treatment_cols.append(col_name)
    
    return treatment_cols


def run_benchmark(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_cols: List[str],
    epsilons: List[float],
    deltas: List[float],
) -> pd.DataFrame:
    """
    Run the benchmark across all combinations.
    
    Returns DataFrame with columns:
    - treatment_col, epsilon, delta_percent
    - found, found_filters, found_size, found_ate
    - candidates_enumerated, candidates_evaluated, runtime_seconds
    """
    results = []
    total = len(treatment_cols) * len(epsilons) * len(deltas)
    count = 0
    
    print(f"\n🚀 Starting benchmark...")
    print(f"   Treatments: {len(treatment_cols)}")
    print(f"   Epsilons: {len(epsilons)}")
    print(f"   Deltas: {len(deltas)}")
    print(f"   Total cases: {total}\n")
    
    for treatment_col in treatment_cols:
        if treatment_col not in df.columns:
            print(f"⚠️  Skipping {treatment_col} (not in dataset)")
            continue
            
        for epsilon in epsilons:
            for delta_percent in deltas:
                count += 1
                print(f"[{count}/{total}] Treatment={treatment_col}, ε={epsilon}, δ={delta_percent*100:.1f}%")
                
                try:
                    result = brute_force_find_positive_homogeneous_subgroup(
                        df=df,
                        treatment_col=treatment_col,
                        outcome_col=outcome_col,
                        epsilon=epsilon,
                        delta_percent=delta_percent,
                    )
                    
                    results.append({
                        "treatment_col": treatment_col,
                        "epsilon": epsilon,
                        "delta_percent": delta_percent,
                        "found": result.found,
                        "found_filters": str(result.found_filters) if result.found_filters else None,
                        "found_size": result.found_size,
                        "found_ate": result.found_ate,
                        "candidates_enumerated": result.candidates_enumerated,
                        "candidates_evaluated": result.candidates_evaluated,
                        "runtime_seconds": result.runtime_seconds,
                    })
                    
                    if result.found:
                        print(f"   ✅ Found subgroup (size={result.found_size}, ATE={result.found_ate:.1f}) in {result.runtime_seconds:.1f}s")
                    else:
                        print(f"   ❌ No subgroup found ({result.candidates_evaluated}/{result.candidates_enumerated} evaluated) in {result.runtime_seconds:.1f}s")
                
                except Exception as e:
                    print(f"   ⚠️  ERROR: {e}")
                    results.append({
                        "treatment_col": treatment_col,
                        "epsilon": epsilon,
                        "delta_percent": delta_percent,
                        "found": False,
                        "found_filters": None,
                        "found_size": None,
                        "found_ate": None,
                        "candidates_enumerated": 0,
                        "candidates_evaluated": 0,
                        "runtime_seconds": 0.0,
                    })
    
    return pd.DataFrame(results)


def create_visualizations(df: pd.DataFrame, plots_dir: Path):
    """Generate all visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        import matplotlib.pyplot as plt
        print("⚠️  seaborn not installed, using matplotlib only")
    
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Filter successful cases
    success = df[df["found"] == True].copy()
    
    if len(success) == 0:
        print("⚠️  No successful cases to plot")
        return
    
    # Aggregate by epsilon and delta
    agg_eps = success.groupby("epsilon").agg(
        mean_runtime=("runtime_seconds", "mean"),
        mean_evaluated=("candidates_evaluated", "mean"),
        count=("found", "count"),
    ).reset_index()
    
    agg_delta = success.groupby("delta_percent").agg(
        mean_runtime=("runtime_seconds", "mean"),
        mean_evaluated=("candidates_evaluated", "mean"),
        count=("found", "count"),
    ).reset_index()
    
    # Plot 1: Mean runtime vs epsilon
    plt.figure(figsize=(9, 5))
    plt.plot(agg_eps["epsilon"], agg_eps["mean_runtime"], marker="o", linewidth=2)
    plt.xlabel("Epsilon (homogeneity threshold)")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("Mean Runtime vs Epsilon")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_runtime_vs_epsilon.png", dpi=160)
    plt.close()
    
    # Plot 2: Mean runtime vs delta
    plt.figure(figsize=(9, 5))
    plt.plot(agg_delta["delta_percent"], agg_delta["mean_runtime"], marker="o", linewidth=2)
    plt.xlabel("Delta (minimum subgroup size %)")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("Mean Runtime vs Delta")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_runtime_vs_delta.png", dpi=160)
    plt.close()
    
    # Plot 3: Mean evaluated vs epsilon
    plt.figure(figsize=(9, 5))
    plt.plot(agg_eps["epsilon"], agg_eps["mean_evaluated"], marker="o", linewidth=2, color="green")
    plt.xlabel("Epsilon (homogeneity threshold)")
    plt.ylabel("Mean candidates evaluated")
    plt.title("Mean Candidates Evaluated vs Epsilon")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mean_evaluated_vs_epsilon.png", dpi=160)
    plt.close()
    
    # Plot 4: Success rate heatmap
    if len(df["epsilon"].unique()) > 1 and len(df["delta_percent"].unique()) > 1:
        pivot = df.pivot_table(
            index="delta_percent",
            columns="epsilon",
            values="found",
            aggfunc="mean",
        )
        
        plt.figure(figsize=(9, 5))
        plt.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(label="Success rate")
        plt.yticks(range(len(pivot.index)), [f"{d:.2f}" for d in pivot.index])
        plt.xticks(range(len(pivot.columns)), [f"{int(e)}" for e in pivot.columns], rotation=45)
        plt.xlabel("Epsilon")
        plt.ylabel("Delta (%)")
        plt.title("Success Rate Heatmap")
        plt.tight_layout()
        plt.savefig(plots_dir / "success_rate_heatmap.png", dpi=160)
        plt.close()
    
    print(f"✅ Generated {len(list(plots_dir.glob('*.png')))} plots")


def save_results(df: pd.DataFrame, out_dir: Path):
    """Save results as CSV and HTML report"""
    out_dir.mkdir(exist_ok=True, parents=True)
    plots_dir = out_dir / "plots"
    
    # Save CSV
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved results to {csv_path}")
    
    # Generate plots
    create_visualizations(df, plots_dir)
    
    # Generate HTML report
    total = len(df)
    found_count = int(df["found"].sum())
    mean_runtime = df[df["found"] == True]["runtime_seconds"].mean() if found_count > 0 else 0
    
    def img_tag(name: str) -> str:
        p = plots_dir / name
        if not p.exists():
            return "<div style='color: #999;'>Plot not available</div>"
        return f'<img src="plots/{name}" style="width:100%; border-radius:8px; border:1px solid #ddd;" />'
    
    table_html = df.to_html(index=False, classes="", border=0, float_format=lambda x: f"{x:.2f}")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Brute Force Rule Mining Benchmark</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 24px; background: #f9f9f9; }}
        h1 {{ margin: 0 0 8px 0; }}
        .meta {{ color: #666; margin-bottom: 20px; font-size: 14px; }}
        .stat {{ display: inline-block; margin-right: 20px; }}
        .good {{ color: #0a7; font-weight: 600; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }}
        .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .card h2 {{ margin: 0 0 12px 0; font-size: 16px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: white; }}
        th, td {{ border-bottom: 1px solid #eee; padding: 10px; text-align: left; }}
        th {{ background: #fafafa; position: sticky; top: 0; font-weight: 600; }}
        .table-container {{ background: white; border-radius: 12px; padding: 16px; max-height: 600px; overflow: auto; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>🔬 Brute Force Rule Mining Benchmark</h1>
    <div class="meta">
        <span class="stat">Total cases: <strong>{total}</strong></span>
        <span class="stat">Found: <span class="good">{found_count}</span> ({found_count/total*100:.1f}%)</span>
        <span class="stat">Mean runtime: <strong>{mean_runtime:.2f}s</strong></span>
    </div>
    
    <div class="grid">
        <div class="card">
            <h2>Mean Runtime vs Epsilon</h2>
            {img_tag("mean_runtime_vs_epsilon.png")}
        </div>
        <div class="card">
            <h2>Mean Runtime vs Delta</h2>
            {img_tag("mean_runtime_vs_delta.png")}
        </div>
        <div class="card">
            <h2>Mean Candidates Evaluated vs Epsilon</h2>
            {img_tag("mean_evaluated_vs_epsilon.png")}
        </div>
        <div class="card">
            <h2>Success Rate Heatmap</h2>
            {img_tag("success_rate_heatmap.png")}
        </div>
    </div>
    
    <h2>Results Table</h2>
    <div class="table-container">
        {table_html}
    </div>
</body>
</html>"""
    
    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"✅ Saved HTML report to {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark brute force rule mining algorithm")
    parser.add_argument("--dataset", default="so.csv", help="Path to dataset CSV")
    parser.add_argument("--outcome", default="ConvertedSalary", help="Outcome column name")
    parser.add_argument("--treatments_file", default="../algorithms/Chosen10Treatments.json", 
                        help="Path to treatments JSON file")
    parser.add_argument("--epsilons", default="100,500,1000,5000,10000", 
                        help="Comma-separated epsilon values")
    parser.add_argument("--deltas", default="0.02,0.05,0.10,0.15", 
                        help="Comma-separated delta percentages")
    parser.add_argument("--output_dir", default="benchmark_outputs", 
                        help="Output directory name")
    
    args = parser.parse_args()
    
    # Load data
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / args.dataset)
    print(f"📊 Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    
    # Load treatments
    treatments_path = script_dir / args.treatments_file
    treatment_cols = load_treatments(treatments_path)
    print(f"🎯 Loaded {len(treatment_cols)} treatments from {treatments_path.name}")
    
    # Parse parameters
    epsilons = [float(x.strip()) for x in args.epsilons.split(",")]
    deltas = [float(x.strip()) for x in args.deltas.split(",")]
    
    # Run benchmark
    start_time = time.time()
    results_df = run_benchmark(df, args.outcome, treatment_cols, epsilons, deltas)
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total benchmark time: {total_time:.1f} seconds")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = script_dir / args.output_dir / timestamp
    save_results(results_df, out_dir)
    
    print(f"\n✅ Benchmark complete! Results saved to:")
    print(f"   {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
