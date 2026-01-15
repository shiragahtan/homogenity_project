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
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_runtime_graphs(output_dir):
    """Generate runtime analysis graphs for both problems."""
    base_path = Path(output_dir)
    problem2_csv = base_path / 'problem2_largest_delta' / 'find_delta_benchmark_results.csv'
    problem3_csv = base_path / 'problem3_smallest_epsilon' / 'find_epsilon_benchmark_results.csv'
    
    if not problem2_csv.exists() or not problem3_csv.exists():
        print("⚠️  Warning: CSV files not found, skipping graph generation")
        return
    
    print("\n📊 Generating runtime analysis graphs...")
    
    df_delta = pd.read_csv(problem2_csv)
    df_epsilon = pd.read_csv(problem3_csv)
    
    # Prepare rule labels
    df_delta['Rule_Label'] = df_delta.apply(
        lambda x: f"Rule {x['Rule_ID']}: {x['Condition']} → {x['Treatment']}", axis=1
    )
    df_epsilon['Rule_Label'] = df_epsilon.apply(
        lambda x: f"Rule {x['Rule_ID']}: {x['Condition']} → {x['Treatment']}", axis=1
    )
    
    # Define color palette for rules
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788']
    
    # ========== GRAPH 1: Runtime vs Epsilon ==========
    fig1 = go.Figure()
    
    for rule_id, rule_group in df_delta.groupby('Rule_ID'):
        color_idx = (rule_id - 1) % len(colors)
        rule_label = rule_group['Rule_Label'].iloc[0]
        
        fig1.add_trace(go.Scatter(
            x=rule_group['Epsilon'],
            y=rule_group['Runtime_Seconds'],
            mode='lines+markers',
            name=rule_label,
            line=dict(width=3, color=colors[color_idx]),
            marker=dict(size=10),
            customdata=rule_group['Largest_Delta_Heterogeneous'],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Epsilon: %{x:,.0f}<br>' +
                          'Found Delta: %{customdata}<br>' +
                          'Runtime: %{y:.2f}s<br>' +
                          '<extra></extra>'
        ))
    
    # Get the search range info
    delta_min = df_delta['Largest_Delta_Heterogeneous'].min()
    delta_max = df_delta['Largest_Delta_Heterogeneous'].max()
    
    fig1.update_layout(
        title=f"Problem 2: Runtime vs Epsilon (Finding Largest Delta)<br><sub>Searching for delta in range [{delta_min:,.0f}, {delta_max:,.0f}] for each epsilon</sub>",
        xaxis_title="Epsilon (Homogeneity Threshold) - Fixed for each test",
        yaxis_title="Runtime (seconds)",
        template="plotly_white",
        font=dict(size=14),
        height=600,
        hovermode='closest',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    # ========== GRAPH 2: Runtime vs Delta ==========
    fig2 = go.Figure()
    
    for rule_id, rule_group in df_epsilon.groupby('Rule_ID'):
        color_idx = (rule_id - 1) % len(colors)
        rule_label = rule_group['Rule_Label'].iloc[0]
        
        fig2.add_trace(go.Scatter(
            x=rule_group['Delta'],
            y=rule_group['Runtime_Seconds'],
            mode='lines+markers',
            name=rule_label,
            line=dict(width=3, color=colors[color_idx]),
            marker=dict(size=10),
            customdata=rule_group['Smallest_Epsilon_Homogeneous'],
            hovertemplate='<b>%{fullData.name}</b><br>' +
                          'Delta (Fixed): %{x}<br>' +
                          'Found Epsilon: %{customdata:,.2f}<br>' +
                          'Runtime: %{y:.2f}s<br>' +
                          '<extra></extra>'
        ))
    
    # Get the epsilon search range info
    epsilon_min = df_epsilon['Smallest_Epsilon_Homogeneous'].min()
    epsilon_max = df_epsilon['Smallest_Epsilon_Homogeneous'].max()
    
    fig2.update_layout(
        title=f"Problem 3: Runtime vs Delta (Finding Smallest Epsilon)<br><sub>Searching for epsilon (found range: [{epsilon_min:,.0f}, {epsilon_max:,.0f}]) for each fixed delta</sub>",
        xaxis_title="Delta (Minimum Subgroup Size) - Fixed for each test",
        yaxis_title="Runtime (seconds)",
        template="plotly_white",
        font=dict(size=14),
        height=600,
        hovermode='closest',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    
    # ========== GRAPH 3: Combined View ==========
    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Runtime vs Epsilon (Problem 2)', 'Runtime vs Delta (Problem 3)'),
        horizontal_spacing=0.12
    )
    
    for rule_id, rule_group in df_delta.groupby('Rule_ID'):
        color_idx = (rule_id - 1) % len(colors)
        fig3.add_trace(go.Scatter(
            x=rule_group['Epsilon'], y=rule_group['Runtime_Seconds'],
            mode='lines+markers', name=f"Rule {rule_id}",
            line=dict(width=2, color=colors[color_idx]),
            marker=dict(size=8), legendgroup=f"rule{rule_id}", showlegend=True
        ), row=1, col=1)
    
    for rule_id, rule_group in df_epsilon.groupby('Rule_ID'):
        color_idx = (rule_id - 1) % len(colors)
        fig3.add_trace(go.Scatter(
            x=rule_group['Delta'], y=rule_group['Runtime_Seconds'],
            mode='lines+markers', name=f"Rule {rule_id}",
            line=dict(width=2, color=colors[color_idx]),
            marker=dict(size=8), legendgroup=f"rule{rule_id}", showlegend=False
        ), row=1, col=2)
    
    fig3.update_xaxes(title_text="Epsilon", row=1, col=1)
    fig3.update_xaxes(title_text="Delta", row=1, col=2)
    fig3.update_yaxes(title_text="Runtime (seconds)", type="log", row=1, col=1)
    fig3.update_yaxes(title_text="Runtime (seconds)", type="log", row=1, col=2)
    
    fig3.update_layout(
        title_text="Combined Runtime Analysis (Log Scale)<br><sub>Comparing both problems side by side</sub>",
        template="plotly_white", font=dict(size=13), height=550
    )
    
    # Save graphs
    fig1.write_html(str(base_path / "graph_runtime_vs_epsilon.html"))
    fig2.write_html(str(base_path / "graph_runtime_vs_delta.html"))
    fig3.write_html(str(base_path / "graph_combined_analysis.html"))
    
    print(f"✅ Runtime graphs saved to {base_path}/")


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


def generate_combined_summary(delta_summary, epsilon_summary, output_dir):
    """Generate a combined summary HTML report."""
    output_path = Path(output_dir)
    html_path = output_path / 'summary_report.html'
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results Summary - Problems 2 & 3</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        h1 {{ font-size: 2.5em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }}
        .subtitle {{ font-size: 1.2em; opacity: 0.9; }}
        .content {{ padding: 40px; }}
        
        .problem-section {{
            margin: 30px 0;
            border: 2px solid #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }}
        .problem-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            font-size: 1.3em;
            font-weight: bold;
        }}
        .problem-content {{
            padding: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        thead {{
            background: #f8f9fa;
        }}
        th {{
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e9ecef;
        }}
        tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        .btn {{
            display: inline-block;
            padding: 12px 24px;
            margin: 10px 5px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: transform 0.2s;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0,0,0,0.3);
        }}
        
        .btn-epsilon {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .emoji {{ font-size: 1.2em; }}
        
        footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span class="emoji">🎯</span> Homogeneity Algorithm Benchmarks</h1>
            <p class="subtitle">Problems 2 & 3: Binary Search Performance Analysis</p>
            <p class="subtitle" style="font-size: 0.9em; margin-top: 10px;">
                Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </header>
        
        <div class="content">
            <div class="problem-section">
                <div class="problem-header">
                    <span class="emoji">📊</span> Problem 2: Find Largest Delta Breaking Homogeneity
                </div>
                <div class="problem-content">
                    <p style="margin-bottom: 15px;">
                        <strong>Objective:</strong> Given a fixed epsilon (ε), find the largest minimum subgroup size (δ*) 
                        where the rule remains heterogeneous.
                    </p>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
"""
    
    for _, row in delta_summary.iterrows():
        html += f"""
                            <tr>
                                <td><strong>{row['Metric']}</strong></td>
                                <td>{row['Value']}</td>
                            </tr>
"""
    
    html += """
                        </tbody>
                    </table>
                    <div style="margin-top: 20px; text-align: center;">
                        <a href="problem2_largest_delta/benchmark_report.html" class="btn">
                            <span class="emoji">📈</span> View Detailed Report
                        </a>
                        <a href="problem2_largest_delta/find_delta_benchmark_results.xlsx" class="btn">
                            <span class="emoji">📊</span> Download Excel
                        </a>
                    </div>
                </div>
            </div>
            
            <div class="problem-section">
                <div class="problem-header btn-epsilon">
                    <span class="emoji">🎯</span> Problem 3: Find Smallest Epsilon Achieving Homogeneity
                </div>
                <div class="problem-content">
                    <p style="margin-bottom: 15px;">
                        <strong>Objective:</strong> Given a fixed delta (δ), find the smallest epsilon (ε*) 
                        where the rule becomes homogeneous.
                    </p>
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
"""
    
    for _, row in epsilon_summary.iterrows():
        html += f"""
                            <tr>
                                <td><strong>{row['Metric']}</strong></td>
                                <td>{row['Value']}</td>
                            </tr>
"""
    
    html += """
                        </tbody>
                    </table>
                    <div style="margin-top: 20px; text-align: center;">
                        <a href="problem3_smallest_epsilon/benchmark_report.html" class="btn btn-epsilon">
                            <span class="emoji">📈</span> View Detailed Report
                        </a>
                        <a href="problem3_smallest_epsilon/find_epsilon_benchmark_results.xlsx" class="btn btn-epsilon">
                            <span class="emoji">📊</span> Download Excel
                        </a>
                    </div>
                </div>
            </div>
            
            <div class="problem-section">
                <div class="problem-header">
                    <span class="emoji">📊</span> Runtime Analysis Graphs
                </div>
                <div class="problem-content">
                    <p style="margin-bottom: 15px;">
                        <strong>Interactive visualizations</strong> showing how runtime varies with epsilon and delta values.
                        Each rule is shown in a different color for easy comparison.
                    </p>
                    
                    <div style="margin-top: 20px; text-align: center;">
                        <a href="graph_runtime_vs_epsilon.html" class="btn" target="_blank">
                            <span class="emoji">📈</span> Runtime vs Epsilon (Problem 2)
                        </a>
                        <a href="graph_runtime_vs_delta.html" class="btn btn-epsilon" target="_blank">
                            <span class="emoji">📉</span> Runtime vs Delta (Problem 3)
                        </a>
                        <a href="graph_combined_analysis.html" class="btn" target="_blank">
                            <span class="emoji">📊</span> Combined Analysis
                        </a>
                    </div>
                    
                    <div style="margin-top: 30px;">
                        <h3 style="text-align: center; margin-bottom: 20px;">Runtime vs Epsilon (Problem 2)</h3>
                        <iframe src="graph_runtime_vs_epsilon.html" width="100%" height="650" frameborder="0"></iframe>
                    </div>
                    
                    <div style="margin-top: 30px;">
                        <h3 style="text-align: center; margin-bottom: 20px;">Runtime vs Delta (Problem 3)</h3>
                        <iframe src="graph_runtime_vs_delta.html" width="100%" height="650" frameborder="0"></iframe>
                    </div>
                    
                    <div style="margin-top: 30px;">
                        <h3 style="text-align: center; margin-bottom: 20px;">Combined Analysis (Log Scale)</h3>
                        <iframe src="graph_combined_analysis.html" width="100%" height="600" frameborder="0"></iframe>
                    </div>
                </div>
            </div>
        </div>
        
        <footer>
            <p><span class="emoji">⚡</span> Both algorithms use binary search with FPGrowth oracle</p>
            <p>Dataset: Stack Overflow Developer Survey</p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Combined summary report saved: {html_path}")


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
    parser.add_argument('--epsilon_max', type=float, default=3000000.0, help='Max epsilon for Problem 3')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory (default: benchmark_results)')
    
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
    
    # ===== GENERATE COMBINED SUMMARY =====
    print("\n" + "="*80)
    print("GENERATING COMBINED SUMMARY REPORT")
    print("="*80)
    
    generate_combined_summary(delta_summary, epsilon_summary, str(base_output))
    
    # ===== GENERATE RUNTIME GRAPHS =====
    generate_runtime_graphs(str(base_output))
    
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
    
    print(f"\n📊 Runtime Analysis Graphs:")
    print(f"   - {base_output}/graph_runtime_vs_epsilon.html")
    print(f"   - {base_output}/graph_runtime_vs_delta.html")
    print(f"   - {base_output}/graph_combined_analysis.html")
    
    print("\n" + "="*80)
    print("✨ Open summary_report.html in your browser to view all results with interactive graphs!")
    print("="*80)


if __name__ == "__main__":
    main()

