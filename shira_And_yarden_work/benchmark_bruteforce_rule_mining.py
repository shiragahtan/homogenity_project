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
    rule_id: Optional[int]
    condition: Optional[Dict[str, Any]]
    treatment_def: Optional[Dict[str, Any]]
    treatment_col: str
    outcome_col: str
    epsilon: float
    delta_percent: float
    delta_count: int
    sample_n: int
    attrs: str
    n_condition_rows: int
    treated_count: int
    control_count: int
    skipped_reason: str
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


def _normalize_label(x: str) -> str:
    # normalize common typography differences (e.g., Bachelor’s vs Bachelor's)
    return (
        str(x)
        .strip()
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("\u00a0", " ")
    )


def _build_value_maps(
    *,
    project_root: Path,
    encoded_csv_path: Path,
    decoded_csv_path: Optional[Path],
    columns: List[str],
) -> Dict[str, Dict[str, int]]:
    """
    Build mapping {col: {label_string: code_int}} for the given columns.
    Prefer learning mapping from (decoded_csv, encoded_csv) if provided and aligned;
    fall back to yarden_categorical_mappings.json for whatever is available there.
    """
    maps: Dict[str, Dict[str, int]] = {}

    # 1) Try to learn mapping from decoded+encoded pair (best coverage, includes FormalEducation)
    if decoded_csv_path and decoded_csv_path.exists():
        usecols = [c for c in columns if c]  # keep order
        try:
            enc = pd.read_csv(encoded_csv_path, usecols=usecols)
            dec = pd.read_csv(decoded_csv_path, usecols=usecols)
            if len(enc) == len(dec):
                for c in usecols:
                    if c not in enc.columns or c not in dec.columns:
                        continue
                    # Build label->code from observed pairs
                    m: Dict[str, int] = {}
                    s_dec = dec[c].astype(str).map(_normalize_label)
                    s_enc = enc[c]
                    for lab, code in zip(s_dec, s_enc):
                        try:
                            icode = int(code)
                        except Exception:
                            continue
                        if lab not in m:
                            m[lab] = icode
                    if m:
                        maps[c] = m
        except Exception:
            # ignore; we'll still try json-based mapping below
            pass

    # 2) Supplement with json mapping file (if present)
    mapping_path = project_root / "yarden_files" / "yarden_categorical_mappings.json"
    if mapping_path.exists():
        try:
            raw = json.loads(mapping_path.read_text(encoding="utf-8"))
            for c in columns:
                if c in raw and isinstance(raw[c], dict):
                    maps.setdefault(c, {})
                    for k, v in raw[c].items():
                        try:
                            maps[c][_normalize_label(k)] = int(v)
                        except Exception:
                            continue
        except Exception:
            pass

    # Small manual aliasing to improve match rates for the provided Chosen10Treatments.json
    # (these are semantically equivalent but use different wording)
    if "RaceEthnicity" in maps:
        aliases = {
            "White or of European descent": "European Descent",
            "White or of European descent ": "European Descent",
        }
        for src, dst in aliases.items():
            if dst in maps["RaceEthnicity"] and src not in maps["RaceEthnicity"]:
                maps["RaceEthnicity"][src] = maps["RaceEthnicity"][dst]

    return maps


def _encode_value(col: str, v: Any, value_maps: Dict[str, Dict[str, int]]) -> Optional[int]:
    """Convert a label or code into an int code if possible; returns None if unknown."""
    if v is None:
        return None
    # already numeric-like
    try:
        return int(v)
    except Exception:
        pass
    s = _normalize_label(str(v))
    m = value_maps.get(col) or {}
    # exact
    if s in m:
        return int(m[s])
    # case-insensitive fallback
    s_low = s.lower()
    for k, code in m.items():
        if str(k).lower() == s_low:
            return int(code)
    return None


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
    *,
    rule_id: Optional[int] = None,
    condition: Optional[Dict[str, Any]] = None,
    treatment_def: Optional[Dict[str, Any]] = None,
    # speed / robustness
    min_samples_per_group: int = 30,
    max_candidates_evaluated: int = 200,
    timeout_seconds: float = 5.0,
) -> RunResult:
    from ATE_update import calculate_ate_safe
    from rw_unlearning import calc_utility_for_subgroups

    # sample for deterministic runtime
    cols = [treatment_col, outcome_col, *attrs]
    cols = [c for c in cols if c in df.columns]
    if sample_n > 0 and len(df) > sample_n:
        df_use = df[cols].sample(n=sample_n, random_state=0).reset_index(drop=True)
    else:
        df_use = df[cols].copy().reset_index(drop=True)

    delta_count = max(1, int(len(df_use) * float(delta_percent)))
    n_condition_rows = int(len(df_use))
    treated_count = int((df_use[treatment_col] == 1).sum()) if treatment_col in df_use.columns else 0
    control_count = int((df_use[treatment_col] == 0).sum()) if treatment_col in df_use.columns else 0

    t0 = time.perf_counter()
    t_enum0 = time.perf_counter()
    candidates = _dfs_candidates(df_use, attrs=attrs, delta_count=delta_count)
    enum_seconds = time.perf_counter() - t_enum0

    found_filters: Optional[Dict[str, Any]] = None
    found_size: Optional[int] = None
    found_ate: Optional[float] = None
    candidates_evaluated = 0

    t_eval0 = time.perf_counter()
    # If treatment split is too small, don't waste time; ATE will be NaN anyway.
    if treated_count < min_samples_per_group or control_count < min_samples_per_group:
        eval_seconds = time.perf_counter() - t_eval0
        total_seconds = time.perf_counter() - t0
        return RunResult(
            rule_id=rule_id,
            condition=condition,
            treatment_def=treatment_def,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            epsilon=float(epsilon),
            delta_percent=float(delta_percent),
            delta_count=int(delta_count),
            sample_n=int(len(df_use)),
            attrs=",".join(attrs),
            n_condition_rows=n_condition_rows,
            treated_count=treated_count,
            control_count=control_count,
            skipped_reason="insufficient_treatment_split",
            found=False,
            found_filters=None,
            found_size=None,
            found_ate=None,
            candidates_enumerated=int(len(candidates)),
            candidates_evaluated=0,
            enum_seconds=float(enum_seconds),
            eval_seconds=float(eval_seconds),
            total_seconds=float(total_seconds),
        )

    deadline = time.perf_counter() + float(timeout_seconds)
    for filt, _sz in candidates:
        if candidates_evaluated >= int(max_candidates_evaluated):
            break
        if time.perf_counter() > deadline:
            break
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
        rule_id=rule_id,
        condition=condition,
        treatment_def=treatment_def,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        epsilon=float(epsilon),
        delta_percent=float(delta_percent),
        delta_count=int(delta_count),
        sample_n=int(len(df_use)),
        attrs=",".join(attrs),
        n_condition_rows=n_condition_rows,
        treated_count=treated_count,
        control_count=control_count,
        skipped_reason="",
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
    except Exception:
        have_plots = False

    plots_dir.mkdir(parents=True, exist_ok=True)

    if results_df.empty or "found" not in results_df.columns:
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Brute Force Rule Mining Benchmark</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
.meta {{ color: #555; margin-bottom: 16px; }}
.card {{ border: 1px solid #e6e6e6; border-radius: 12px; padding: 12px; background: #fff; }}
</style></head>
<body>
<h1>Brute Force Rule Mining Benchmark</h1>
<div class="meta">Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
<div class="card">No rows were produced (likely all rules had empty condition matches). Try different rules or disable condition filtering.</div>
</body></html>"""
        (out_dir / "report.html").write_text(html, encoding="utf-8")
        return

    df = results_df.copy()
    df["found_int"] = df["found"].astype(int)

    if have_plots:
        import numpy as np
        import matplotlib.pyplot as plt

        # Aggregate across rules if present
        group_cols = [c for c in ["epsilon", "delta_percent"] if c in df.columns]
        agg = (
            df.groupby(group_cols, as_index=False)
            .agg(
                mean_runtime=("total_seconds", "mean"),
                mean_evaluated=("candidates_evaluated", "mean"),
                success_rate=("found_int", "mean"),
                mean_found_size=("found_size", "mean"),
            )
            .sort_values(group_cols)
        )

        eps_vals = sorted(agg["epsilon"].unique())
        del_vals = sorted(agg["delta_percent"].unique())

        # Mean runtime vs epsilon (lines by delta)
        plt.figure(figsize=(9, 5))
        for dp in del_vals:
            sub = agg[agg["delta_percent"] == dp]
            plt.plot(sub["epsilon"], sub["mean_runtime"], marker="o", label=f"delta={dp}")
        plt.xscale("log")
        plt.xlabel("epsilon (log)")
        plt.ylabel("mean runtime (s)")
        plt.title("Mean runtime vs epsilon")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "mean_runtime_vs_epsilon.png", dpi=160)
        plt.close()

        # Mean candidates evaluated vs epsilon
        plt.figure(figsize=(9, 5))
        for dp in del_vals:
            sub = agg[agg["delta_percent"] == dp]
            plt.plot(sub["epsilon"], sub["mean_evaluated"], marker="o", label=f"delta={dp}")
        plt.xscale("log")
        plt.xlabel("epsilon (log)")
        plt.ylabel("mean candidates evaluated")
        plt.title("Mean evaluated candidates vs epsilon")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "mean_evaluated_vs_epsilon.png", dpi=160)
        plt.close()

        # Success rate heatmap (delta x epsilon)
        mat = np.full((len(del_vals), len(eps_vals)), np.nan, dtype=float)
        for i, dp in enumerate(del_vals):
            for j, ep in enumerate(eps_vals):
                v = agg[(agg["delta_percent"] == dp) & (agg["epsilon"] == ep)]["success_rate"]
                if not v.empty:
                    mat[i, j] = float(v.iloc[0])
        plt.figure(figsize=(9, 4.5))
        im = plt.imshow(mat, aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, label="success rate")
        plt.yticks(range(len(del_vals)), [str(d) for d in del_vals])
        plt.xticks(range(len(eps_vals)), [str(int(e)) if float(e).is_integer() else str(e) for e in eps_vals], rotation=45)
        plt.xlabel("epsilon")
        plt.ylabel("delta_percent")
        plt.title("Success rate heatmap")
        plt.tight_layout()
        plt.savefig(plots_dir / "success_rate_heatmap.png", dpi=160)
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
    <div class="card"><h2>Mean runtime vs epsilon</h2>{img_tag("mean_runtime_vs_epsilon.png")}</div>
    <div class="card"><h2>Mean evaluated vs epsilon</h2>{img_tag("mean_evaluated_vs_epsilon.png")}</div>
    <div class="card"><h2>Success rate heatmap</h2>{img_tag("success_rate_heatmap.png")}</div>
  </div>

  <h2 style="margin-top: 20px;">Results table</h2>
  <div class="card" style="overflow:auto; max-height: 70vh;">{table_html}</div>
</body>
</html>"""

    (out_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark brute-force rule mining robustness.")
    parser.add_argument("--dataset", type=str, default="so.csv", help="Dataset CSV filename (relative to this folder)")
    parser.add_argument(
        "--decoded_dataset",
        type=str,
        default="yarden_files/yarden_so_decoded.csv",
        help="Optional decoded CSV (relative to repo root) used to infer label->code mappings for so.csv.",
    )
    parser.add_argument("--outcome", type=str, default="ConvertedSalary", help="Outcome column")
    parser.add_argument("--treatments", type=str, default="FormalEducation", help="Comma-separated treatment columns")
    parser.add_argument(
        "--treatments_file",
        type=str,
        default="",
        help="Optional JSON-lines file with {'condition':{...}, 'treatment':{...}} entries (relative to repo root or absolute).",
    )
    parser.add_argument("--epsilons", type=str, default="50,100,200,500", help="Comma-separated epsilons")
    parser.add_argument("--deltas", type=str, default="0.05,0.1,0.2", help="Comma-separated delta percents (0..1)")
    parser.add_argument(
        "--attrs",
        type=str,
        default="Hobby,Student,Continent,HDI",
        help="Comma-separated subgroup attribute columns to enumerate",
    )
    parser.add_argument("--sample_n", type=int, default=3000, help="Sample size per run (0 = full dataset)")
    parser.add_argument("--min_cases", type=int, default=50, help="Warn if fewer than this many result rows are produced")
    parser.add_argument("--max_rules", type=int, default=10, help="Max rules to read from treatments_file (for runtime)")
    parser.add_argument("--max_candidates", type=int, default=200, help="Max candidates evaluated per case")
    parser.add_argument("--timeout_seconds", type=float, default=5.0, help="Timeout per case (seconds)")
    parser.add_argument(
        "--auto_attrs_k",
        type=int,
        default=6,
        help="If >0, pick K lowest-cardinality attributes from the provided --attrs list (per rule) for speed.",
    )
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
    base_attrs = [a.strip() for a in args.attrs.split(",") if a.strip()]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = this_dir / args.output_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    # Build label->code mappings for columns we might see in rules
    decoded_path = Path(args.decoded_dataset)
    if not decoded_path.is_absolute():
        decoded_path = project_root / decoded_path

    def _collect_rule_columns(rules: List[Dict[str, Any]]) -> List[str]:
        cols: List[str] = []
        for r in rules:
            cond = (r.get("condition") or {}) if isinstance(r, dict) else {}
            tr = (r.get("treatment") or {}) if isinstance(r, dict) else {}
            cols.extend(list(cond.keys()))
            cols.extend(list(tr.keys()))
        # also include subgroup attrs list (may need mapping too)
        cols.extend(base_attrs)
        # dedupe while preserving order
        seen = set()
        out = []
        for c in cols:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    value_maps: Dict[str, Dict[str, int]] = {}

    def _apply_filters(d: pd.DataFrame, filt: Dict[str, Any]) -> pd.DataFrame:
        out = d
        for k, v in filt.items():
            if k not in out.columns:
                return out.iloc[0:0]
            v2 = _encode_value(k, v, value_maps)
            if v2 is None:
                return out.iloc[0:0]
            out = out[out[k] == v2]
        return out

    results: List[RunResult] = []

    # Mode A: Use condition+treatment rules from a JSON-lines file (Chosen10Treatments.json)
    if args.treatments_file:
        tf_path = Path(args.treatments_file)
        if not tf_path.is_absolute():
            tf_path = project_root / tf_path
        if not tf_path.exists():
            raise FileNotFoundError(f"treatments_file not found: {tf_path}")

        rules: List[Dict[str, Any]] = []
        for line in tf_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rules.append(json.loads(line))

        # Now that we have rules, build value maps for relevant columns
        value_maps = _build_value_maps(
            project_root=project_root,
            encoded_csv_path=dataset_path,
            decoded_csv_path=decoded_path if decoded_path.exists() else None,
            columns=_collect_rule_columns(rules),
        )

        rules = rules[: max(0, int(args.max_rules))]
        total = len(rules) * len(epsilons) * len(deltas)
        i = 0
        print(f"Running {total} experiments (from treatments_file)...")

        for ridx, rule in enumerate(rules, start=1):
            cond = rule.get("condition") or {}
            treat_def = rule.get("treatment") or {}

            df_cond = _apply_filters(df, cond)
            if df_cond.empty:
                # Still emit rows (so the CSV has the requested number of cases),
                # but mark as skipped.
                for eps in epsilons:
                    for dp in deltas:
                        results.append(
                            RunResult(
                                rule_id=ridx,
                                condition=cond,
                                treatment_def=treat_def,
                                treatment_col="__TREATMENT__",
                                outcome_col=args.outcome,
                                epsilon=float(eps),
                                delta_percent=float(dp),
                                delta_count=0,
                                sample_n=0,
                                attrs="",
                                n_condition_rows=0,
                                treated_count=0,
                                control_count=0,
                                skipped_reason="empty_condition",
                                found=False,
                                found_filters=None,
                                found_size=None,
                                found_ate=None,
                                candidates_enumerated=0,
                                candidates_evaluated=0,
                                enum_seconds=0.0,
                                eval_seconds=0.0,
                                total_seconds=0.0,
                            )
                        )
                continue

            # Binary treatment indicator: 1 iff all treatment_def key/values match
            treat_col_name = "__TREATMENT__"
            df_rule = df_cond.copy()
            treated_mask = pd.Series(True, index=df_rule.index)
            for k, v in treat_def.items():
                if k not in df_rule.columns:
                    treated_mask &= False
                    continue
                v2 = _encode_value(k, v, value_maps)
                if v2 is None:
                    treated_mask &= False
                    continue
                treated_mask &= df_rule[k] == v2
            df_rule[treat_col_name] = treated_mask.astype(int)

            # Subgroup attrs: avoid using condition/treatment keys; optionally pick lowest-cardinality K
            attrs = [a for a in base_attrs if a not in set(cond.keys()) | set(treat_def.keys())]
            if int(args.auto_attrs_k) > 0 and attrs:
                k = int(args.auto_attrs_k)
                card = {a: int(df_rule[a].nunique(dropna=True)) if a in df_rule.columns else 10**9 for a in attrs}
                attrs = [a for a, _ in sorted(card.items(), key=lambda kv: kv[1])[:k]]

            for eps in epsilons:
                for dp in deltas:
                    i += 1
                    print(f"[{i}/{total}] rule={ridx} eps={eps} delta={dp} cond={cond} treat={treat_def}")
                    rr = run_one(
                        df=df_rule,
                        treatment_col=treat_col_name,
                        outcome_col=args.outcome,
                        attrs=attrs,
                        epsilon=eps,
                        delta_percent=dp,
                        sample_n=args.sample_n,
                        rule_id=ridx,
                        condition=cond,
                        treatment_def=treat_def,
                        max_candidates_evaluated=int(args.max_candidates),
                        timeout_seconds=float(args.timeout_seconds),
                    )
                    results.append(rr)
    else:
        # Mode B: treat a column as the treatment directly.
        total = len(treatments) * len(epsilons) * len(deltas)
        i = 0
        print(f"Running {total} experiments...")

        for tcol in treatments:
            if tcol not in df.columns:
                print(f"⚠️  treatment '{tcol}' not in dataset. Skipping.")
                continue
            attrs = [a for a in base_attrs if a != tcol]
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
                        max_candidates_evaluated=int(args.max_candidates),
                        timeout_seconds=float(args.timeout_seconds),
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

    if len(results_df) < int(args.min_cases):
        print(f"⚠️  Only produced {len(results_df)} rows (< min_cases={args.min_cases}). Consider expanding epsilons/deltas or max_rules.")

    print("\n✅ Done.")
    print(f"- CSV:  {out_dir / 'results.csv'}")
    print(f"- XLSX: {out_dir / 'results.xlsx'}")
    print(f"- HTML: {out_dir / 'report.html'}")


if __name__ == "__main__":
    main()


