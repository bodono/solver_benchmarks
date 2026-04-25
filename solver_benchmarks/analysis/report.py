"""Write full benchmark analysis reports."""

from __future__ import annotations

from pathlib import Path
import json
import math

import pandas as pd

from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.plots import write_analysis_plots
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.reports import (
    claimed_optimal_kkt_thresholds,
    completion_summary,
    difficulty_scaling,
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
    setup_solve_breakdown,
    slowest_solves,
    solver_metrics,
    solver_problem_tables,
    status_matrix,
)
from solver_benchmarks.core.config import manifest_dataset_entries


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
        "claimed_optimal_kkt_thresholds.csv": claimed_optimal_kkt_thresholds(results),
        "kkt_certificate_summary.csv": kkt_certificate_summary(results),
        f"difficulty_scaling_{metric}.csv": difficulty_scaling(results, metric=metric),
        "setup_solve_breakdown.csv": setup_solve_breakdown(results),
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
    plot_outputs = write_analysis_plots(run_dir, metric=metric, output_dir=output_dir)
    outputs.extend(plot_outputs)
    markdown = _render_markdown_report(
        run_dir=run_dir,
        output_dir=output_dir,
        metric=metric,
        results=results,
        tables=tables,
        plot_outputs=plot_outputs,
        artifact_outputs=outputs,
    )
    index_path = output_dir / "index.md"
    readme_path = output_dir / "README.md"
    index_path.write_text(markdown)
    readme_path.write_text(markdown)
    outputs.extend([index_path, readme_path])
    return outputs


def _write_table(path: Path, table: pd.DataFrame) -> Path | None:
    if table.empty:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    include_index = not isinstance(table.index, pd.RangeIndex)
    table.to_csv(path, index=include_index)
    return path


def _render_markdown_report(
    *,
    run_dir: Path,
    output_dir: Path,
    metric: str,
    results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    plot_outputs: list[Path],
    artifact_outputs: list[Path],
) -> str:
    manifest = _load_manifest(run_dir)
    config = manifest.get("config", {})
    dataset_entries = manifest_dataset_entries(config)
    dataset_names = [entry["name"] for entry in dataset_entries] or [_unknown()]
    dataset_label = (
        dataset_names[0]
        if len(dataset_names) == 1
        else ", ".join(dataset_names)
    )
    lines = [
        "# Benchmark Report",
        "",
        "This report is generated by `bench report` and collects the main run",
        "metadata, summary statistics, diagnostics, plots, and links to full CSV",
        "tables in one place.",
        "",
        "## Run Overview",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Datasets: `{dataset_label}`",
        f"- Primary metric: `{metric}`",
        f"- Result rows: `{len(results)}`",
        f"- Problems with results: `{results['problem'].nunique() if 'problem' in results else 0}`",
        f"- Solver variants: `{results['solver_id'].nunique() if 'solver_id' in results else 0}`",
        f"- Config hash: `{config.get('config_hash', _unknown())}`",
        "",
    ]

    solvers = pd.DataFrame(config.get("solvers", []))
    if not solvers.empty:
        display = solvers[[column for column in ("id", "solver", "timeout_seconds") if column in solvers]]
        lines.extend(_section_table("Configured Solvers", display, max_rows=50))

    lines.extend(
        _section_table(
            "Completion",
            tables.get("completion.csv", pd.DataFrame()),
            intro=(
                "Use this first to confirm the run is complete. `missing = 0`, "
                "`unexpected = 0`, and `duplicate_rows = 0` are the expected clean state."
            ),
        )
    )
    lines.extend(
        _section_table(
            "Solver Metrics",
            tables.get("solver_metrics.csv", pd.DataFrame()),
            intro=(
                "Only accurate `optimal` solves count as successes. Inaccurate, "
                "timeout, skipped, and error statuses count as failures."
            ),
        )
    )
    lines.extend(_section_table("Status Counts", tables.get("status_counts.csv", pd.DataFrame())))
    lines.extend(_section_table("Failure Rates", tables.get("failure_rates.csv", pd.DataFrame())))

    lines.extend(
        _section_table(
            "Setup vs Solve Time",
            tables.get("setup_solve_breakdown.csv", pd.DataFrame()),
            intro=(
                "Many adapters split `run_time_seconds` into a setup phase "
                "(KKT factorization, scaling) and a solve phase (the iterative "
                "loop). Solvers that do not report a split show "
                "`with_breakdown = 0`."
            ),
        )
    )
    lines.extend(
        _plot_block(
            output_dir,
            plot_outputs,
            [("setup_solve_breakdown.png", "Setup vs Solve Time")],
        )
    )

    lines.extend(["## Performance Plots", ""])
    lines.extend(
        _plot_block(
            output_dir,
            plot_outputs,
            [
                (f"performance_profile_{metric}.png", "Dolan-More Performance Profile"),
                (f"cactus_{metric}.png", "Cactus Plot"),
                (f"shifted_geomean_{metric}.png", "Penalized Shifted Geomean"),
                (f"performance_ratio_heatmap_{metric}.png", "Per-Problem Performance Ratio Heatmap"),
            ],
        )
    )
    lines.extend(
        _section_table(
            "Shifted Geomean",
            tables.get(f"shifted_geomean_{metric}.csv", pd.DataFrame()),
            intro="Penalized shifted geomeans assign non-successful solves the configured failure penalty.",
        )
    )
    lines.extend(
        _section_table(
            "Shifted Geomean, Successful Solves Only",
            tables.get(f"shifted_geomean_{metric}_success_only.csv", pd.DataFrame()),
        )
    )

    lines.extend(["## Pairwise Comparisons", ""])
    lines.extend(
        _plot_block(
            output_dir,
            plot_outputs,
            [(f"pairwise_scatter_{metric}.png", "Pairwise Solver Scatter")],
        )
    )
    lines.extend(_section_table("Pairwise Speedups", tables.get(f"pairwise_speedups_{metric}.csv", pd.DataFrame())))

    lines.extend(["## Difficulty Scaling", ""])
    lines.extend(
        _plot_block(
            output_dir,
            plot_outputs,
            [(f"difficulty_scaling_{metric}.png", "Difficulty Scaling")],
        )
    )
    lines.extend(
        _section_table(
            f"Median {metric} by Problem Size",
            tables.get(f"difficulty_scaling_{metric}.csv", pd.DataFrame()),
            intro=(
                "Problems are bucketed into equal-population quantile bins of "
                "`metadata.n`. Each row reports a solver's median runtime on "
                "successful solves within that bucket; failures are not "
                "averaged in."
            ),
        )
    )

    lines.extend(["## Problem-Level Views", ""])
    lines.extend(
        _plot_block(
            output_dir,
            plot_outputs,
            [
                ("status_heatmap.png", "Status Heatmap"),
                ("failure_rates.png", "Failure Rates"),
            ],
        )
    )
    lines.extend(
        _section_table(
            "Slowest Solves",
            tables.get(f"slowest_solves_{metric}.csv", pd.DataFrame()),
        )
    )
    lines.extend(
        _section_table(
            "Failures With Successful Alternatives",
            tables.get(f"failures_with_successful_alternatives_{metric}.csv", pd.DataFrame()),
        )
    )
    lines.extend(_section_table("Objective Spreads", tables.get("objective_spreads.csv", pd.DataFrame())))
    lines.extend(
        [
            "Full problem-level tables:",
            "",
            f"- [Cross-solver comparison]({_relative(output_dir / 'problem_solver_comparison.csv', output_dir)})",
            f"- [Status matrix]({_relative(output_dir / 'status_matrix.csv', output_dir)})",
            f"- [Per-solver problem tables]({_relative(output_dir / 'solver_problem_tables', output_dir)})",
            "",
        ]
    )

    lines.extend(["## KKT Diagnostics", ""])
    lines.extend(
        _plot_block(
            output_dir,
            plot_outputs,
            [
                ("kkt_residual_boxplot.png", "KKT Residual Boxplot"),
                ("kkt_residual_heatmap.png", "KKT Residual Heatmap"),
                ("kkt_accuracy_profile.png", "KKT Accuracy Profile"),
            ],
        )
    )
    lines.extend(_section_table("KKT Summary", tables.get("kkt_summary.csv", pd.DataFrame())))
    lines.extend(
        _section_table(
            "Claimed-Optimal KKT Thresholds",
            tables.get("claimed_optimal_kkt_thresholds.csv", pd.DataFrame()),
            intro=(
                "Counts of claimed-optimal solves whose worst relative KKT "
                "residual is at or below each threshold. `count_above_max` "
                "flags claims of optimality with residuals above the loosest "
                "threshold — solutions that should not have been labelled "
                "optimal."
            ),
        )
    )
    lines.extend(_section_table("KKT Certificate Summary", tables.get("kkt_certificate_summary.csv", pd.DataFrame())))

    if len(dataset_names) > 1 and "dataset" in results.columns:
        lines.extend(_per_dataset_breakdown(results, dataset_names, metric=metric))

    lines.extend(["## Provenance", ""])
    environment_columns = [
        column
        for column in [
            "solver_id",
            "metadata.environment_id",
            "metadata.runtime.python_executable",
            "metadata.runtime.python_version",
            "metadata.runtime.platform",
        ]
        if column in results
    ]
    if environment_columns:
        environment = (
            results[environment_columns]
            .drop_duplicates()
            .sort_values(environment_columns[:1])
            .reset_index(drop=True)
        )
        lines.extend(_section_table("Runtime Environments", environment, max_rows=50))

    lines.extend(
        [
            "Manifest excerpt:",
            "",
            "```json",
            json.dumps(_manifest_excerpt(manifest), indent=2, sort_keys=True, default=str),
            "```",
            "",
            "## Artifact Index",
            "",
        ]
    )
    for output in sorted(set(artifact_outputs)):
        lines.append(f"- [{_relative(output, output_dir)}]({_relative(output, output_dir)})")
    lines.append("")
    return "\n".join(lines)


def _per_dataset_breakdown(
    results: pd.DataFrame,
    dataset_names: list[str],
    *,
    metric: str,
) -> list[str]:
    """Emit headline tables (solver_metrics, failure_rates, geomean, KKT)
    once per dataset so a multi-dataset run can be read both as the
    aggregated tables above and as per-dataset slices.
    """
    lines = [
        "## By Dataset",
        "",
        "Headline tables sliced per dataset. The sections above are the",
        "cross-dataset aggregates over the same rows.",
        "",
    ]
    for dataset_name in dataset_names:
        subset = results[results["dataset"] == dataset_name]
        if subset.empty:
            lines.extend([f"### {dataset_name}", "", "No rows for this dataset.", ""])
            continue
        lines.extend([f"### {dataset_name}", ""])
        lines.extend(
            _section_table(
                "Solver Metrics",
                solver_metrics(subset),
                level=4,
            )
        )
        lines.extend(
            _section_table(
                "Failure Rates",
                failure_rates(subset),
                level=4,
            )
        )
        lines.extend(
            _section_table(
                f"Shifted Geomean ({metric})",
                shifted_geomean(subset, metric=metric),
                level=4,
            )
        )
        lines.extend(
            _section_table(
                "KKT Summary",
                kkt_summary(subset),
                level=4,
            )
        )
    return lines


def _section_table(
    title: str,
    table: pd.DataFrame,
    *,
    intro: str | None = None,
    max_rows: int = 20,
    max_cols: int = 12,
    level: int = 2,
) -> list[str]:
    heading = "#" * max(1, min(level, 6))
    lines = [f"{heading} {title}", ""]
    if intro:
        lines.extend([intro, ""])
    if table.empty:
        lines.extend(["No rows for this section.", ""])
        return lines
    lines.extend(_dataframe_to_markdown(table, max_rows=max_rows, max_cols=max_cols))
    lines.append("")
    return lines


def _dataframe_to_markdown(
    table: pd.DataFrame,
    *,
    max_rows: int,
    max_cols: int,
) -> list[str]:
    frame = table.copy()
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    total_rows = len(frame)
    total_cols = len(frame.columns)
    if total_rows > max_rows:
        frame = frame.head(max_rows)
    if total_cols > max_cols:
        frame = frame.iloc[:, :max_cols]
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(_escape_cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join(_escape_cell(_format_cell(row[column])) for column in frame.columns)
            + " |"
        )
    if total_rows > max_rows or total_cols > max_cols:
        rows.extend(
            [
                "",
                f"Showing first {len(frame)} rows and {len(frame.columns)} columns "
                f"of {total_rows} rows and {total_cols} columns. See the linked CSV for the full table.",
            ]
        )
    return rows


def _plot_block(
    output_dir: Path,
    plot_outputs: list[Path],
    plots: list[tuple[str, str]],
) -> list[str]:
    available = {path.name: path for path in plot_outputs}
    lines = []
    for filename, title in plots:
        path = available.get(filename)
        if path is None:
            continue
        relative = _relative(path, output_dir)
        lines.extend([f"### {title}", "", f"![{title}]({relative})", ""])
    if not lines:
        return ["No plots were generated for this section.", ""]
    return lines


def _format_cell(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except ValueError:
        pass
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6g}"
        return ""
    return str(value)


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _manifest_excerpt(manifest: dict) -> dict:
    config = manifest.get("config", {})
    return {
        "run_id": manifest.get("run_id"),
        "created_at_utc": manifest.get("created_at_utc"),
        "datasets": manifest_dataset_entries(config),
        "include": config.get("include", []),
        "exclude": config.get("exclude", []),
        "config_hash": config.get("config_hash"),
        "solver_ids": [solver.get("id") for solver in config.get("solvers", [])],
    }


def _unknown() -> str:
    return "unknown"
