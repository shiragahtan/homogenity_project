"""
Unified Benchmark Runner for Problems 2 and 3.

Runs both algorithms and generates organized results:
- Problem 2: Find Largest Delta Breaking Homogeneity (Fixed Epsilon)
- Problem 3: Find Smallest Epsilon Achieving Homogeneity (Fixed Delta)

Results are organized in separate subdirectories with HTML reports.
"""
import sys
import os
import argparse
import time
import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import benchmark modules
from benchmark_find_delta import run_benchmark as run_delta_benchmark
from benchmark_find_delta import generate_summary_statistics, create_visualizations
from benchmark_find_epsilon import run_benchmark as run_epsilon_benchmark
from benchmark_find_epsilon import generate_summary_statistics as gen_epsilon_summary
from benchmark_find_epsilon import create_visualizations as create_epsilon_viz

import pandas as pd
import base64
from typing import Optional, List
import json
import re


def _encode_png_as_data_uri(png_path: Path) -> Optional[str]:
    if not png_path.exists():
        return None
    b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _safe_read_csv(csv_path: Path) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def _df_to_html_table(df: pd.DataFrame) -> str:
    # Wide tables: wrap in a scroll container and keep a reasonable font size.
    return "<div class='table-scroll'>" + df.to_html(index=False, escape=False) + "</div>"


def generate_delta_html_report(results_df, summary_df, output_dir):
    """Generate HTML report for Problem 2 (largest delta)."""
    from benchmark_find_delta import save_results
    # Use the integrated save_results which includes HTML generation
    save_results(results_df, summary_df, output_dir)


def generate_epsilon_html_report(results_df, summary_df, output_dir):
    """Generate HTML report for Problem 3 (smallest epsilon)."""
    from benchmark_find_epsilon import save_results
    # Use the integrated save_results which includes HTML generation
    save_results(results_df, summary_df, output_dir)


def generate_combined_summary_single_html(output_dir: str) -> Path:
    """
    Generate a single self-contained HTML report (no links/iframes to other HTML files).
    It embeds:
      - Full CSV tables for Problem 2 and Problem 3
      - Summary stats for each problem
      - Key PNG plots (base64-embedded)
      - Epsilon comparison table (if present)
    """
    output_path = Path(output_dir)
    html_path = output_path / "summary_report.html"

    # Inputs
    p2_dir = output_path / "problem2_largest_delta"
    p3_dir = output_path / "problem3_smallest_epsilon"
    p2_csv = p2_dir / "find_delta_benchmark_results.csv"
    p3_csv = p3_dir / "find_epsilon_benchmark_results.csv"

    df_p2 = _safe_read_csv(p2_csv)
    df_p3 = _safe_read_csv(p3_csv)

    # Optional: epsilon comparison (lives next to this script, not inside output_dir)
    comparison_dir = Path(__file__).resolve().parent / "benchmark_results_epsilon_comparison"
    comp_csv = comparison_dir / "epsilon_comparison_results.csv"
    df_comp = _safe_read_csv(comp_csv)

    # Embedded images (if present)
    p2_heat = _encode_png_as_data_uri(p2_dir / "oracle_calls_heatmap.png")
    p3_heat = _encode_png_as_data_uri(p3_dir / "oracle_calls_heatmap.png")

    # Summaries (compute from CSV if available)
    delta_summary = generate_summary_statistics(df_p2) if df_p2 is not None else None
    epsilon_summary = gen_epsilon_summary(df_p3) if df_p3 is not None else None

    def _load_rule_metrics_from_existing_results(dataset_key: str = "so") -> dict[int, dict]:
        """
        Load per-rule Coverage/Utility/Prevalence from the existing repo output:
        `graphs/rules_summary_from_existing_results.csv`.
        """
        metrics: dict[int, dict] = {}
        csv_path = Path(__file__).resolve().parent.parent / "graphs" / "rules_summary_from_existing_results.csv"
        if not csv_path.exists():
            return metrics

        try:
            df_rules = pd.read_csv(csv_path)
        except Exception:
            return metrics

        # Normalize expected columns
        needed = {"Dataset", "Rule #", "Coverage (%)", "Utility", "Prevalence (%)"}
        if not needed.issubset(set(df_rules.columns)):
            return metrics

        df_rules = df_rules[df_rules["Dataset"].astype(str).str.lower() == dataset_key.lower()].copy()
        if df_rules.empty:
            return metrics

        for _, r in df_rules.iterrows():
            try:
                rule_id = int(r["Rule #"])
            except Exception:
                continue
            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None
            metrics[rule_id] = {
                "coverage_pct": _to_float(r["Coverage (%)"]),
                "utility": _to_float(r["Utility"]),
                "prevalence_pct": _to_float(r["Prevalence (%)"]),
            }

        return metrics

    # Use the canonical "existing results" summary as requested by the user
    rule_metrics = _load_rule_metrics_from_existing_results(dataset_key="so")

    # Glossary (high-signal columns; supports both benchmark tables)
    glossary_rows = [
        ("Rule_ID", "Index of the tested rule (1..N)."),
        ("Condition", "Antecedent attributes describing a subgroup definition for the rule."),
        ("Treatment", "Treatment attribute/value used by the rule."),
        ("Dataset_Size", "Number of rows in the dataset for that rule."),
        ("Oracle_Calls", "Number of oracle evaluations performed by binary search."),
        ("Runtime_Seconds", "Total wall-clock runtime for that experiment (seconds)."),
        ("Runtime_Minutes", "Runtime_Seconds / 60."),
        ("Theoretical_Max_Iterations", "⌈log2(search_range_size)⌉ for the binary search."),
        ("Efficiency_Ratio", "Oracle_Calls / Theoretical_Max_Iterations (closer to 1.0 is better)."),
        ("Population_Utility", "Utility/ATE over the full population (utility_all)."),
        ("Violating_Subgroup", "Subgroup (attributes) that caused boundary decision (the violation witness)."),
        ("Subgroup_Size", "Size of that subgroup."),
        ("Subgroup_Utility", "Utility/ATE within that subgroup."),
        ("Utility_Difference", "Subgroup_Utility - Population_Utility."),
        ("Abs_Utility_Difference", "|Utility_Difference|."),
        ("Epsilon", "Homogeneity threshold ε (fixed input for Problem 2)."),
        ("Largest_Delta_Heterogeneous", "Problem 2 output: largest δ such that a violation still exists."),
        ("Delta", "Minimum subgroup size δ (fixed input for Problem 3)."),
        ("Smallest_Epsilon_Homogeneous", "Problem 3 output: smallest ε such that no violation exists (or 'None' if not found within ε_max)."),
        ("Coverage (%)", "From `graphs/rules_summary_from_existing_results.csv`: % of data matching the rule’s condition."),
        ("Utility", "From `graphs/rules_summary_from_existing_results.csv`: rule utility (CATE-like metric) for the subpopulation."),
        ("Prevalence (%)", "From `graphs/rules_summary_from_existing_results.csv`: % of subgroups violating homogeneity (|UtilityDiff| > ε)."),
    ]
    glossary_df = pd.DataFrame(glossary_rows, columns=["Column", "Meaning / How it's computed"])

    def _summary_block(summary_df: Optional[pd.DataFrame]) -> str:
        if summary_df is None or summary_df.empty:
            return "<p><em>No summary available (missing CSV).</em></p>"
        return _df_to_html_table(summary_df)

    def _table_block(df: Optional[pd.DataFrame]) -> str:
        if df is None or df.empty:
            return "<p><em>No results available (missing CSV).</em></p>"
        return _df_to_html_table(df)

    def _img_block(title: str, data_uri: Optional[str]) -> str:
        if not data_uri:
            return ""
        return (
            f"<h3 style='margin-top:18px;margin-bottom:10px;'>{title}</h3>"
            f"<img class='img' src='{data_uri}' alt='{title}'/>"
        )

    def _build_plotly_runtime_log_charts() -> tuple[str, str]:
        """
        Return (problem2_plot_html, problem3_plot_html).
        Each is a Plotly chart rendered to an HTML snippet, with log-scale runtime (y-axis).
        Legend shows full rule description: Condition → Treatment.
        """
        if df_p2 is None and df_p3 is None:
            return "", ""

        try:
            import plotly.graph_objects as go  # type: ignore
            import plotly.io as pio  # type: ignore
        except ModuleNotFoundError:
            note = (
                "<p><em>Plotly is not installed, so interactive runtime charts are unavailable. "
                "Install with <code>pip install plotly</code> and regenerate with "
                "<code>python run_all_benchmarks.py --output benchmark_results_full --only_summary --prune_html</code>.</em></p>"
            )
            return note, note

        colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8",
            "#F7DC6F", "#BB8FCE", "#85C1E2", "#F8B739", "#52B788",
        ]

        p2_html = ""
        p3_html = ""

        if df_p2 is not None and not df_p2.empty:
            fig_p2 = go.Figure()
            for i, (rule_id, g) in enumerate(df_p2.groupby("Rule_ID"), start=0):
                g = g.sort_values("Epsilon")
                condition = str(g["Condition"].iloc[0])
                treatment = str(g["Treatment"].iloc[0])
                rid = int(rule_id)
                m = rule_metrics.get(rid, {})
                cov = m.get("coverage_pct")
                util = m.get("utility")
                prev = m.get("prevalence_pct")
                cov_s = f"{cov:.2f}%" if isinstance(cov, (int, float)) else "N/A"
                util_s = f"{util:,.2f}" if isinstance(util, (int, float)) else "N/A"
                prev_s = f"{prev:.2f}%" if isinstance(prev, (int, float)) else "N/A"
                rule_label = f"Rule {rid} (Cov {cov_s}, Util {util_s}, Prev {prev_s}): {condition} → {treatment}"
                color = colors[i % len(colors)]

                fig_p2.add_trace(
                    go.Scatter(
                        x=g["Epsilon"],
                        y=g["Runtime_Seconds"],
                        mode="lines+markers",
                        name=rule_label,
                        line=dict(width=2, color=color),
                        marker=dict(size=8, color=color),
                        customdata=g[["Largest_Delta_Heterogeneous", "Oracle_Calls"]].to_numpy(),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "ε: %{x:,.0f}<br>"
                            "Runtime: %{y:.3f}s<br>"
                            "δ*: %{customdata[0]}<br>"
                            "Oracle calls: %{customdata[1]}<br>"
                            "<extra></extra>"
                        ),
                    )
                )

            fig_p2.update_layout(
                title="Runtime vs ε (Problem 2) — log scale",
                xaxis_title="ε (fixed for each experiment)",
                yaxis_title="Runtime (seconds, log scale)",
                yaxis_type="log",
                template="plotly_white",
                height=620,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                margin=dict(l=60, r=360, t=60, b=60),
            )
            p2_html = pio.to_html(
                fig_p2,
                include_plotlyjs="cdn",
                full_html=False,
                config={"displayModeBar": False, "responsive": True},
            )

        if df_p3 is not None and not df_p3.empty:
            fig_p3 = go.Figure()
            for i, (rule_id, g) in enumerate(df_p3.groupby("Rule_ID"), start=0):
                g = g.sort_values("Delta")
                condition = str(g["Condition"].iloc[0])
                treatment = str(g["Treatment"].iloc[0])
                rid = int(rule_id)
                m = rule_metrics.get(rid, {})
                cov = m.get("coverage_pct")
                util = m.get("utility")
                prev = m.get("prevalence_pct")
                cov_s = f"{cov:.2f}%" if isinstance(cov, (int, float)) else "N/A"
                util_s = f"{util:,.2f}" if isinstance(util, (int, float)) else "N/A"
                prev_s = f"{prev:.2f}%" if isinstance(prev, (int, float)) else "N/A"
                rule_label = f"Rule {rid} (Cov {cov_s}, Util {util_s}, Prev {prev_s}): {condition} → {treatment}"
                color = colors[i % len(colors)]

                fig_p3.add_trace(
                    go.Scatter(
                        x=g["Delta"],
                        y=g["Runtime_Seconds"],
                        mode="lines+markers",
                        name=rule_label,
                        line=dict(width=2, color=color),
                        marker=dict(size=8, color=color),
                        customdata=g[["Smallest_Epsilon_Homogeneous", "Oracle_Calls"]].to_numpy(),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "δ: %{x}<br>"
                            "Runtime: %{y:.3f}s<br>"
                            "ε*: %{customdata[0]}<br>"
                            "Oracle calls: %{customdata[1]}<br>"
                            "<extra></extra>"
                        ),
                    )
                )

            fig_p3.update_layout(
                title="Runtime vs δ (Problem 3) — log scale",
                xaxis_title="δ (fixed for each experiment)",
                yaxis_title="Runtime (seconds, log scale)",
                yaxis_type="log",
                template="plotly_white",
                height=620,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                margin=dict(l=60, r=360, t=60, b=60),
            )
            # plotlyjs already included above (from p2); if p2 missing, include CDN here.
            include_js = False if p2_html else "cdn"
            p3_html = pio.to_html(
                fig_p3,
                include_plotlyjs=include_js,
                full_html=False,
                config={"displayModeBar": False, "responsive": True},
            )

        return p2_html, p3_html

    p2_runtime_plot_html, p3_runtime_plot_html = _build_plotly_runtime_log_charts()

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Benchmark Results (Single Report) - Problems 2 & 3</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      padding: 20px;
      min-height: 100vh;
    }}
    .container {{
      max-width: 1500px;
      margin: 0 auto;
      background: white;
      border-radius: 15px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      overflow: hidden;
    }}
    header {{
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 36px;
      text-align: center;
    }}
    h1 {{ font-size: 2.2em; margin-bottom: 8px; }}
    .subtitle {{ font-size: 1.05em; opacity: 0.95; }}
    .content {{ padding: 28px; }}
    .section {{
      margin: 18px 0 28px 0;
      border: 2px solid #e9ecef;
      border-radius: 10px;
      overflow: hidden;
    }}
    .section-title {{
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 14px 18px;
      font-size: 1.15em;
      font-weight: 700;
    }}
    .section-body {{ padding: 18px; }}
    h2 {{ margin: 6px 0 12px 0; color: #2d2d2d; }}
    h3 {{ margin: 14px 0 10px 0; color: #2d2d2d; }}
    p {{ margin: 8px 0 10px 0; color: #333; line-height: 1.45; }}
    details {{ margin: 10px 0 4px 0; }}
    summary {{ cursor: pointer; font-weight: 700; color: #3949ab; }}

    .table-scroll {{
      width: 100%;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
      border: 1px solid #e9ecef;
      border-radius: 10px;
      margin: 12px 0 14px 0;
    }}
    table {{
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    thead tr {{ background: #f5f6fa; }}
    th, td {{
      padding: 10px 10px;
      border-bottom: 1px solid #e9ecef;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    tbody tr:nth-child(even) {{ background: #fafbff; }}
    tbody tr:hover {{ background: #eef2ff; }}
    .img {{
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid #e9ecef;
    }}
    footer {{
      background: #f8f9fa;
      padding: 18px;
      text-align: center;
      color: #6c757d;
      font-size: 0.9em;
    }}
    .note {{
      background: #fff3cd;
      border-left: 4px solid #ffc107;
      padding: 12px 14px;
      border-radius: 6px;
      margin: 10px 0 14px 0;
    }}
    .plot-wrap {{
      width: 100%;
      margin: 12px 0 16px 0;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>🎯 Homogeneity Algorithm Benchmarks (Single Report)</h1>
      <div class="subtitle">Problems 2 & 3 — generated on {now_str}</div>
    </header>

    <div class="content">
      <div class="section">
        <div class="section-title">📚 Glossary (how columns are computed)</div>
        <div class="section-body">
          <details open>
            <summary>Show / hide</summary>
            {_df_to_html_table(glossary_df)}
          </details>
        </div>
      </div>

      <div class="section">
        <div class="section-title">📊 Problem 2 — Find Largest Delta Breaking Homogeneity (fixed ε)</div>
        <div class="section-body">
          <div class="note"><strong>Interpretation:</strong> δ* is the <em>largest</em> minimum subgroup size such that a violation still exists (heterogeneous). For δ &gt; δ*, the rule becomes homogeneous.</div>
          <h2>Summary</h2>
          {_summary_block(delta_summary)}
          <h2>Runtime vs ε (log scale) — by rule</h2>
          <div class="plot-wrap">{p2_runtime_plot_html}</div>
          <h2>All experiments (full table)</h2>
          {_table_block(df_p2)}
          <details>
            <summary>Other plots (optional)</summary>
            {_img_block("Oracle-calls heatmap (Problem 2)", p2_heat)}
          </details>
        </div>
      </div>

      <div class="section">
        <div class="section-title">🎯 Problem 3 — Find Smallest Epsilon Achieving Homogeneity (fixed δ)</div>
        <div class="section-body">
          <div class="note"><strong>Interpretation:</strong> ε* is the <em>smallest</em> threshold such that no subgroup violates homogeneity. For ε ≥ ε*, the rule stays homogeneous.</div>
          <h2>Summary</h2>
          {_summary_block(epsilon_summary)}
          <h2>Runtime vs δ (log scale) — by rule</h2>
          <div class="plot-wrap">{p3_runtime_plot_html}</div>
          <h2>All experiments (full table)</h2>
          {_table_block(df_p3)}
          <details>
            <summary>Other plots (optional)</summary>
            {_img_block("Oracle-calls heatmap (Problem 3)", p3_heat)}
          </details>
        </div>
      </div>

      <div class="section">
        <div class="section-title">⚖️ Optional — Smallest ε method comparison (Binary Search vs Brute Force)</div>
        <div class="section-body">
          <p>This section is populated if <code>benchmark_results_epsilon_comparison/epsilon_comparison_results.csv</code> exists.</p>
          {_table_block(df_comp)}
        </div>
      </div>
    </div>

    <footer>
      <div>Single-file report: no links/iframes to other HTML needed.</div>
      <div>CSV outputs are still saved in the output folders for programmatic use.</div>
    </footer>
  </div>
</body>
</html>
"""

    html_path.write_text(html, encoding="utf-8")
    print(f"✅ Single-file summary report saved: {html_path}")
    return html_path


def prune_html_files(keep_html: Path, additional_dirs: Optional[List[Path]] = None) -> None:
    """
    Delete all .html files under keep_html.parent (and optionally additional dirs),
    except keep_html itself.
    """
    keep_html = keep_html.resolve()
    dirs = [keep_html.parent]
    if additional_dirs:
        dirs.extend(additional_dirs)

    removed = 0
    for d in dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.html"):
            if p.resolve() == keep_html:
                continue
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    print(f"🧹 Pruned HTML files (kept {keep_html.name}): removed {removed}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run all benchmark experiments')
    parser.add_argument('--rules', type=int, default=5, help='Number of rules to test (default: 5)')
    parser.add_argument('--epsilons', type=str, default='10000,20000,30000,40000,50000,60000',
                       help='Comma-separated epsilon values for Problem 2')
    parser.add_argument('--deltas', type=str, default='500,1000,1500,2000,2500,3000',
                       help='Comma-separated delta values for Problem 3')
    parser.add_argument('--delta_min', type=int, default=100, help='Min delta for Problem 2 search')
    parser.add_argument('--delta_max', type=int, default=10000, help='Max delta for Problem 2 search')
    parser.add_argument('--epsilon_start', type=float, default=1000.0, help='Starting epsilon for Problem 3')
    parser.add_argument('--epsilon_max', type=float, default=1_000_000_000.0, help='Max epsilon for Problem 3')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory (default: benchmark_results)')
    parser.add_argument('--only_summary', action='store_true',
                       help='Only (re)generate the single summary_report.html from existing CSV/PNG outputs (no benchmarks).')
    parser.add_argument('--prune_html', action='store_true',
                       help='After generating summary_report.html, delete other .html files so only one remains.')
    
    args = parser.parse_args()
    
    # Resolve output directory relative to this script (so it works from any cwd)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    args.output = str(output_path)
    
    # Create output directory structure
    base_output = Path(args.output)
    base_output.mkdir(exist_ok=True, parents=True)
    
    problem2_dir = base_output / 'problem2_largest_delta'
    problem3_dir = base_output / 'problem3_smallest_epsilon'
    
    problem2_dir.mkdir(exist_ok=True)
    problem3_dir.mkdir(exist_ok=True)
    
    # Parse lists
    epsilon_values = [float(x.strip()) for x in args.epsilons.split(',')]
    delta_values = [int(x.strip()) for x in args.deltas.split(',')]
    
    print("="*80)
    print("UNIFIED BENCHMARK RUNNER")
    print("="*80)
    print(f"Testing {args.rules} rules")
    print(f"Problem 2 - Epsilons: {epsilon_values}")
    print(f"Problem 3 - Deltas: {delta_values}")
    print(f"Results will be saved to: {base_output}")
    print("="*80)
    
    total_start = time.time()

    if args.only_summary:
        keep = generate_combined_summary_single_html(str(base_output))
        if args.prune_html:
            comparison_dir = Path(__file__).resolve().parent / "benchmark_results_epsilon_comparison"
            prune_html_files(keep_html=keep, additional_dirs=[comparison_dir])
        return
    
    # ===== PROBLEM 2: FIND LARGEST DELTA =====
    print("\n" + "="*80)
    print("PROBLEM 2: FINDING LARGEST DELTA BREAKING HOMOGENEITY")
    print("="*80)
    
    problem2_start = time.time()
    
    delta_results = run_delta_benchmark(
        num_rules=args.rules,
        epsilon_values=epsilon_values,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        output_dir=str(problem2_dir)
    )
    
    problem2_time = time.time() - problem2_start
    
    print("\n📊 Generating Problem 2 summary statistics...")
    delta_summary = generate_summary_statistics(delta_results)
    
    print("📈 Creating Problem 2 visualizations...")
    create_visualizations(delta_results, str(problem2_dir))
    
    print("💾 Saving Problem 2 results...")
    generate_delta_html_report(delta_results, delta_summary, str(problem2_dir))
    
    print(f"\n✅ Problem 2 completed in {problem2_time/60:.2f} minutes")
    
    # ===== PROBLEM 3: FIND SMALLEST EPSILON =====
    print("\n" + "="*80)
    print("PROBLEM 3: FINDING SMALLEST EPSILON ACHIEVING HOMOGENEITY")
    print("="*80)
    
    problem3_start = time.time()
    
    epsilon_results = run_epsilon_benchmark(
        num_rules=args.rules,
        delta_values=delta_values,
        epsilon_start=args.epsilon_start,
        epsilon_max=args.epsilon_max,
        output_dir=str(problem3_dir)
    )
    
    problem3_time = time.time() - problem3_start
    
    print("\n📊 Generating Problem 3 summary statistics...")
    epsilon_summary = gen_epsilon_summary(epsilon_results)
    
    print("📈 Creating Problem 3 visualizations...")
    create_epsilon_viz(epsilon_results, str(problem3_dir))
    
    print("💾 Saving Problem 3 results...")
    generate_epsilon_html_report(epsilon_results, epsilon_summary, str(problem3_dir))
    
    print(f"\n✅ Problem 3 completed in {problem3_time/60:.2f} minutes")
    
    # ===== GENERATE SINGLE SUMMARY =====
    print("\n" + "="*80)
    print("GENERATING SINGLE SUMMARY REPORT")
    print("="*80)
    
    keep = generate_combined_summary_single_html(str(base_output))
    if args.prune_html:
        comparison_dir = Path(__file__).resolve().parent / "benchmark_results_epsilon_comparison"
        prune_html_files(keep_html=keep, additional_dirs=[comparison_dir])
    
    total_time = time.time() - total_start
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*80)
    print("🎉 ALL BENCHMARKS COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Timing Summary:")
    print(f"   Problem 2 (Largest Delta):     {problem2_time/60:6.2f} minutes")
    print(f"   Problem 3 (Smallest Epsilon):  {problem3_time/60:6.2f} minutes")
    print(f"   Total Runtime:                  {total_time/60:6.2f} minutes")
    
    print(f"\n📂 Results saved to: {base_output}/")
    print(f"\n📊 Problem 2 Files:")
    print(f"   - HTML Report:   {problem2_dir}/benchmark_report.html")
    print(f"   - Excel Results: {problem2_dir}/find_delta_benchmark_results.xlsx")
    print(f"   - CSV Results:   {problem2_dir}/find_delta_benchmark_results.csv")
    print(f"   - Visualizations: {problem2_dir}/")
    
    print(f"\n🎯 Problem 3 Files:")
    print(f"   - HTML Report:   {problem3_dir}/benchmark_report.html")
    print(f"   - Excel Results: {problem3_dir}/find_epsilon_benchmark_results.xlsx")
    print(f"   - CSV Results:   {problem3_dir}/find_epsilon_benchmark_results.csv")
    print(f"   - Visualizations: {problem3_dir}/")
    
    print(f"\n🌐 Combined Summary:")
    print(f"   - {base_output}/summary_report.html")
    
    print("\n" + "="*80)
    print("✨ Open summary_report.html in your browser to view all results!")
    print("="*80)


if __name__ == "__main__":
    main()

