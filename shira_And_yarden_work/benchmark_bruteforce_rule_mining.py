"""
Benchmark: brute_force_find_positive_homogeneous_subgroup (rule mining)

Goal
----
Given a fixed (epsilon, delta, treatment), measure how the brute-force rule miner behaves:
- runtime (total + enumeration + evaluation)
- number of candidate subgroups enumerated
- number of candidates evaluated until the best subgroup is found
- found subgroup size (coverage) and utility (ATE)
- success / failure rates across a grid of inputs

Outputs
-------
Creates a timestamped directory under:
  shira_And_yarden_work/benchmark_bruteforce_outputs/<timestamp>/
containing:
- results.csv
- report.html
- plots/*.png

Notes
-----
- This benchmark intentionally restricts the subgroup attributes to a small list of low-cardinality
  columns by default, otherwise brute-force enumeration can explode.
- Some underlying utilities read configs using paths relative to the CWD. We set CWD appropriately.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class RunResult:
    treatment_col: str
    outcome_col: str
    epsilon: float
    delta_percent: float
    delta_count: int
    sample_n: int
    attrs: str
    found: bool
    found_filters: Optional[Dict[str, Any]]
    found_size: Optional[int]
    found_ate: Optional[float]
    candidates_enumerated: int
    candidates_evaluated: int
    # timing
    enum_seconds: float
    eval_seconds: float
    total_seconds: float


def _ensure_import_environment(project_root: Path) -> None:
    """
    Some modules (ATE_update) load configs using relative paths like '../configs/config.json'.
    To keep behavior consistent without rewriting those modules, we set CWD to <root>/algorithms.
    """
    os.chdir(project_root / "algorithms")
    # Allow importing project packages from anywhere
    import sys

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    alg_dir = project_root / "algorithms"
    if alg_dir.exists() and str(alg_dir) not in sys.path:
        sys.path.insert(0, str(alg_dir))
    yarden_files = project_root / "yarden_files"
    if yarden_files.exists() and str(yarden_files) not in sys.path:
        sys.path.insert(0, str(yarden_files))


def _dfs_candidates(
    df: pd.DataFrame,
    attrs: List[str],
    delta_count: int,
) -> List[Tuple[Dict[str, Any], int]]:
    """Enumerate candidate filters with pruning by size >= delta_count (same structure as the script)."""
    attr_values: Dict[str, List[Any]] = {a: df[a].dropna().unique().tolist() for a in attrs}
    candidates: List[Tuple[Dict[str, Any], int]] = []

    def _dfs(start_idx: int, current_filters: Dict[str, Any], current_df: pd.DataFrame) -> None:
        size = len(current_df)
        if size >= delta_count and current_filters:
            candidates.append((current_filters.copy(), size))

        for i in range(start_idx, len(attrs)):
            a = attrs[i]
            if a in current_filters:
                continue
            for val in attr_values[a]:
                next_df = current_df[current_df[a] == val]
                if len(next_df) >= delta_count:
                    nf = dict(current_filters)
                    nf[a] = val
                    _dfs(i + 1, nf, next_df)

    _dfs(0, {}, df)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


@contextlib.contextmanager
def _silence_prints():
    """Mute noisy prints from underlying oracles while benchmarking."""
    _print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        yield
    finally:
        builtins.print = _print


def run_one(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    attrs: List[str],
    epsilon: float,
    delta_percent: float,
    sample_n: int,
) -> RunResult:
    from ATE_update import calculate_ate_safe
    from rw_unlearning import calc_utility_for_subgroups

    # sample for deterministic runtime
    if sample_n > 0 and len(df) > sample_n:
        df_use = df[[treatment_col, outcome_col, *attrs]].sample(n=sample_n, random_state=0).reset_index(drop=True)
    else:
        df_use = df[[treatment_col, outcome_col, *attrs]].copy().reset_index(drop=True)

    delta_count = max(1, int(len(df_use) * float(delta_percent)))

    t0 = time.perf_counter()
    t_enum0 = time.perf_counter()
    candidates = _dfs_candidates(df_use, attrs=attrs, delta_count=delta_count)
    enum_seconds = time.perf_counter() - t_enum0

    found_filters: Optional[Dict[str, Any]] = None
    found_size: Optional[int] = None
    found_ate: Optional[float] = None
    candidates_evaluated = 0

    t_eval0 = time.perf_counter()
    for filt, _sz in candidates:
        candidates_evaluated += 1
        sub_df = df_use
        for a, v in filt.items():
            sub_df = sub_df[sub_df[a] == v]

        # condition (1): homogeneity
        with _silence_prints():
            status = calc_utility_for_subgroups(
                mode=0,
                algorithm=None,
                df=sub_df,
                treatment_col=treatment_col,
                delta=delta_count,
                epsilon=epsilon,
                outcome_col=outcome_col,
            )
        is_homog = bool(status[0]) if isinstance(status, tuple) else bool(status)
        if not is_homog:
            continue

        # condition (3): utility > 0 (ATE)
        try:
            cate = calculate_ate_safe(sub_df, treatment_col, outcome_col, delta_count)
        except Exception:
            continue
        if cate is None or not pd.notna(cate) or float(cate) <= 0:
            continue

        found_filters = filt
        found_size = int(len(sub_df))
        found_ate = float(cate)
        break

    eval_seconds = time.perf_counter() - t_eval0
    total_seconds = time.perf_counter() - t0

    return RunResult(
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        epsilon=float(epsilon),
        delta_percent=float(delta_percent),
        delta_count=int(delta_count),
        sample_n=int(len(df_use)),
        attrs=",".join(attrs),
        found=found_filters is not None,
        found_filters=found_filters,
        found_size=found_size,
        found_ate=found_ate,
        candidates_enumerated=int(len(candidates)),
        candidates_evaluated=int(candidates_evaluated),
        enum_seconds=float(enum_seconds),
        eval_seconds=float(eval_seconds),
        total_seconds=float(total_seconds),
    )


def _save_report(results_df: pd.DataFrame, out_dir: Path, plots_dir: Path) -> None:
    # Always write an HTML report. Plots are optional (seaborn/matplotlib).
    have_plots = True
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        have_plots = False

    plots_dir.mkdir(parents=True, exist_ok=True)

    df = results_df.copy()
    df["found_int"] = df["found"].astype(int)

    if have_plots:
        # Runtime vs delta_percent, colored by epsilon
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="delta_percent", y="total_seconds", hue="epsilon", marker="o")
        plt.title("Runtime vs delta_percent")
        plt.tight_layout()
        plt.savefig(plots_dir / "runtime_vs_delta.png", dpi=160)
        plt.close()

        # Candidates evaluated until found
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="delta_percent", y="candidates_evaluated", hue="epsilon", marker="o")
        plt.title("Candidates evaluated until found (or exhausted)")
        plt.tight_layout()
        plt.savefig(plots_dir / "evaluated_vs_delta.png", dpi=160)
        plt.close()

        # Found size
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df[df["found"]], x="delta_percent", y="found_size", hue="epsilon", marker="o")
        plt.title("Found subgroup size vs delta_percent (only successes)")
        plt.tight_layout()
        plt.savefig(plots_dir / "found_size_vs_delta.png", dpi=160)
        plt.close()

        # Success rate
        agg = df.groupby(["treatment_col", "epsilon", "delta_percent"], as_index=False)["found_int"].mean()
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=agg, x="delta_percent", y="found_int", hue="epsilon", marker="o")
        plt.ylim(-0.05, 1.05)
        plt.title("Success rate vs delta_percent")
        plt.tight_layout()
        plt.savefig(plots_dir / "success_rate_vs_delta.png", dpi=160)
        plt.close()

    # HTML report with embedded images + table
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
    h1, h2 { margin: 0 0 12px 0; }
    .meta { color: #555; margin-bottom: 16px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .card { border: 1px solid #e6e6e6; border-radius: 12px; padding: 12px; background: #fff; }
    img { width: 100%; height: auto; border-radius: 10px; border: 1px solid #eee; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #eee; padding: 8px; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #fafafa; z-index: 1; }
    .ok { color: #0a7; font-weight: 600; }
    .bad { color: #b00; font-weight: 600; }
    """

    def img_tag(name: str) -> str:
        p = plots_dir / name
        if not p.exists():
            return "<div class='meta'>Plot not available (missing plotting dependencies).</div>"
        rel = os.path.relpath(p, out_dir)
        return f"<img src='{rel}' alt='{name}' />"

    df_for_html = results_df.copy()
    df_for_html["found_filters"] = df_for_html["found_filters"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else "")
    df_for_html["found"] = df_for_html["found"].apply(lambda x: "YES" if x else "NO")
    df_for_html["found"] = df_for_html["found"].apply(lambda x: f"<span class='ok'>{x}</span>" if x == "YES" else f"<span class='bad'>{x}</span>")

    table_html = df_for_html.to_html(index=False, escape=False)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Brute Force Rule Mining Benchmark</title>
  <style>{css}</style>
</head>
<body>
  <h1>Brute Force Rule Mining Benchmark</h1>
  <div class="meta">Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>

  <div class="grid">
    <div class="card"><h2>Runtime</h2>{img_tag("runtime_vs_delta.png")}</div>
    <div class="card"><h2>Evaluated candidates</h2>{img_tag("evaluated_vs_delta.png")}</div>
    <div class="card"><h2>Found size</h2>{img_tag("found_size_vs_delta.png")}</div>
    <div class="card"><h2>Success rate</h2>{img_tag("success_rate_vs_delta.png")}</div>
  </div>

  <h2 style="margin-top: 20px;">Results table</h2>
  <div class="card" style="overflow:auto; max-height: 70vh;">{table_html}</div>
</body>
</html>"""

    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark brute-force rule mining robustness.")
    parser.add_argument("--dataset", type=str, default="so.csv", help="Dataset CSV filename (relative to this folder)")
    parser.add_argument("--outcome", type=str, default="ConvertedSalary", help="Outcome column")
    parser.add_argument("--treatments", type=str, default="FormalEducation", help="Comma-separated treatment columns")
    parser.add_argument("--epsilons", type=str, default="50,100,200,500", help="Comma-separated epsilons")
    parser.add_argument("--deltas", type=str, default="0.05,0.1,0.2", help="Comma-separated delta percents (0..1)")
    parser.add_argument(
        "--attrs",
        type=str,
        default="Hobby,Student,Continent,HDI",
        help="Comma-separated subgroup attribute columns to enumerate",
    )
    parser.add_argument("--sample_n", type=int, default=3000, help="Sample size per run (0 = full dataset)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_bruteforce_outputs",
        help="Output directory (relative to this folder)",
    )
    args = parser.parse_args()

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    _ensure_import_environment(project_root)

    dataset_path = this_dir / args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    treatments = [t.strip() for t in args.treatments.split(",") if t.strip()]
    epsilons = [float(x.strip()) for x in args.epsilons.split(",") if x.strip()]
    deltas = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    attrs = [a.strip() for a in args.attrs.split(",") if a.strip()]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = this_dir / args.output_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    results: List[RunResult] = []
    total = len(treatments) * len(epsilons) * len(deltas)
    i = 0
    print(f"Running {total} experiments...")

    for tcol in treatments:
        if tcol not in df.columns:
            print(f"⚠️  treatment '{tcol}' not in dataset. Skipping.")
            continue
        for eps in epsilons:
            for dp in deltas:
                i += 1
                print(f"[{i}/{total}] treatment={tcol} epsilon={eps} delta_percent={dp}")
                rr = run_one(
                    df=df,
                    treatment_col=tcol,
                    outcome_col=args.outcome,
                    attrs=attrs,
                    epsilon=eps,
                    delta_percent=dp,
                    sample_n=args.sample_n,
                )
                results.append(rr)

    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_df.to_csv(out_dir / "results.csv", index=False)
    # Excel is optional (requires openpyxl)
    try:
        results_df.to_excel(out_dir / "results.xlsx", index=False)
    except ModuleNotFoundError:
        print("ℹ️  openpyxl not installed; skipping Excel output (results.xlsx). CSV/HTML will still be generated.")

    _save_report(results_df, out_dir=out_dir, plots_dir=plots_dir)

    print("\n✅ Done.")
    print(f"- CSV:  {out_dir / 'results.csv'}")
    print(f"- XLSX: {out_dir / 'results.xlsx'}")
    print(f"- HTML: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()


