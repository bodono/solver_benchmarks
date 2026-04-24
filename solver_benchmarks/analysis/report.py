"""Write full benchmark analysis reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.plots import write_analysis_plots
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.reports import (
    completion_summary,
    failure_rates,
    failures_with_successful_alternatives,
    kkt_certificate_summary,
    kkt_summary,
    missing_results,
    objective_spreads,
    pairwise_speedups,
    performance_ratio_matrix,
    problem_solver_comparison,
    safe_filename,
    slowest_solves,
    solver_metrics,
    solver_problem_tables,
    status_matrix,
)


def write_run_report(
    run_dir: str | Path,
    *,
    metric: str = "run_time_seconds",
    output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> list[Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(run_dir)
    if results.empty:
        return []

    outputs: list[Path] = []
    tables = {
        "solver_metrics.csv": solver_metrics(results),
        "status_counts.csv": solver_summary(run_dir),
        "completion.csv": completion_summary(run_dir, results, repo_root=repo_root),
        "failure_rates.csv": failure_rates(results),
        "missing_results.csv": missing_results(run_dir, results, repo_root=repo_root),
        f"performance_profile_{metric}.csv": performance_profile(results, metric=metric),
        f"shifted_geomean_{metric}.csv": shifted_geomean(results, metric=metric),
        f"shifted_geomean_{metric}_success_only.csv": shifted_geomean(
            results,
            metric=metric,
            penalize_failures=False,
        ),
        f"pairwise_speedups_{metric}.csv": pairwise_speedups(results, metric=metric),
        f"performance_ratios_{metric}.csv": performance_ratio_matrix(
            results,
            metric=metric,
        ),
        "problem_solver_comparison.csv": problem_solver_comparison(results),
        "objective_spreads.csv": objective_spreads(results),
        f"slowest_solves_{metric}.csv": slowest_solves(results, metric=metric),
        f"failures_with_successful_alternatives_{metric}.csv": (
            failures_with_successful_alternatives(results, metric=metric)
        ),
        "status_matrix.csv": status_matrix(results),
        "kkt_summary.csv": kkt_summary(results),
        "kkt_certificate_summary.csv": kkt_certificate_summary(results),
    }
    for name, table in tables.items():
        path = _write_table(output_dir / name, table)
        if path is not None:
            outputs.append(path)
    solver_tables_dir = output_dir / "solver_problem_tables"
    for solver_id, table in solver_problem_tables(results).items():
        path = _write_table(solver_tables_dir / f"{safe_filename(solver_id)}.csv", table)
        if path is not None:
            outputs.append(path)
    outputs.extend(write_analysis_plots(run_dir, metric=metric, output_dir=output_dir))
    _write_index(output_dir / "README.md", run_dir, metric, outputs)
    outputs.append(output_dir / "README.md")
    return outputs


def _write_table(path: Path, table: pd.DataFrame) -> Path | None:
    if table.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    include_index = not isinstance(table.index, pd.RangeIndex)
    table.to_csv(path, index=include_index)
    return path


def _write_index(path: Path, run_dir: Path, metric: str, outputs: list[Path]) -> None:
    lines = [
        "# Benchmark Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Primary metric: `{metric}`",
        "",
        "Generated artifacts:",
        "",
    ]
    for output in sorted(outputs):
        lines.append(f"- `{output.relative_to(path.parent)}`")
    lines.append("")
    path.write_text("\n".join(lines))
