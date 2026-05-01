"""Write full benchmark analysis reports."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import pandas as pd

from solver_benchmarks import __version__ as BENCHMARK_VERSION
from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.plots import write_analysis_plots
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.tables import (
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

PLOT_IMAGE_WIDTH = 680
KKT_PLOT_IMAGE_WIDTH = 920


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
    tables = {
        name: _sort_report_table(name, table, metric=metric)
        for name, table in tables.items()
    }
    for name, table in tables.items():
        path = _write_table(output_dir / name, table)
        if path is not None:
            outputs.append(path)
    solver_tables_dir = output_dir / "solver_problem_tables"
    for solver_id, table in solver_problem_tables(results).items():
        table = _sort_solver_problem_table(table)
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
    # Write the rendered markdown once as both index.md (default
    # GitHub directory landing page) and README.md (rendered on web
    # views). The previous code wrote the same bytes twice; sharing
    # avoids drift if anyone hand-edits one of them.
    index_path = output_dir / "index.md"
    readme_path = output_dir / "README.md"
    index_path.write_text(markdown)
    # Use a relative symlink when the platform supports it so the two
    # paths cannot drift, falling back to a copy on platforms (or
    # filesystems) that disallow symlinks.
    try:
        if readme_path.exists() or readme_path.is_symlink():
            readme_path.unlink()
        readme_path.symlink_to(index_path.name)
    except (OSError, NotImplementedError):
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
    """Compose the per-section render helpers.

    Each helper returns a list[str] of markdown lines for that section
    so the orchestration here stays linear and readable. Helpers accept
    the data they need explicitly — there's no shared mutable state.
    """
    manifest = _load_manifest(run_dir)
    config = manifest.get("config", {})
    dataset_entries = manifest_dataset_entries(config)
    dataset_labels = [_dataset_display_label(entry) for entry in dataset_entries] or [_unknown()]
    dataset_label = (
        dataset_labels[0]
        if len(dataset_labels) == 1
        else ", ".join(dataset_labels)
    )

    lines: list[str] = []
    lines.extend(_render_summary_block(dataset_entries, results, tables, metric))
    lines.extend(
        _render_scope_block(
            run_dir=run_dir,
            manifest=manifest,
            config=config,
            dataset_label=dataset_label,
            metric=metric,
            results=results,
            tables=tables,
        )
    )
    lines.extend(_render_completion_block(tables))
    lines.extend(_render_setup_solve_block(tables, output_dir, plot_outputs))
    lines.extend(_render_performance_plots_block(tables, output_dir, plot_outputs, metric))
    lines.extend(_render_pairwise_block(tables, output_dir, plot_outputs, metric))
    lines.extend(_render_difficulty_scaling_block(tables, output_dir, plot_outputs, metric))
    lines.extend(_render_problem_level_block(tables, output_dir, plot_outputs, metric))
    lines.extend(_render_kkt_diagnostics_block(tables, output_dir, plot_outputs))
    if len(dataset_entries) > 1 and "dataset" in results.columns:
        lines.extend(_per_dataset_breakdown(results, dataset_entries, metric=metric))
    lines.extend(_render_provenance_block(run_dir, manifest, results))
    lines.extend(_render_artifact_index_block(output_dir, artifact_outputs))
    return "\n".join(lines)


def _render_summary_block(
    dataset_entries: list[dict],
    results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    metric: str,
) -> list[str]:
    return [
        "# Benchmark Report",
        "",
        "Generated by `bench report`. The report starts with run scope,",
        "software versions, and headline solver performance; detailed",
        "diagnostics, provenance, and raw artifact links follow.",
        "",
        "## Executive Summary",
        "",
        _executive_summary_text(
            dataset_entries=dataset_entries,
            results=results,
            tables=tables,
            metric=metric,
        ),
        "",
    ]


def _render_scope_block(
    *,
    run_dir: Path,
    manifest: dict,
    config: dict,
    dataset_label: str,
    metric: str,
    results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> list[str]:
    lines: list[str] = []
    lines.extend(
        _section_table(
            "Run Scope",
            _run_scope_table(
                run_dir=run_dir,
                manifest=manifest,
                config=config,
                dataset_label=dataset_label,
                metric=metric,
                results=results,
                tables=tables,
            ),
            level=3,
            max_rows=50,
            max_cols=2,
        )
    )
    lines.extend(
        _section_table(
            "Headline Solver Performance",
            _headline_solver_performance(
                results=results,
                tables=tables,
                config=config,
                metric=metric,
            ),
            intro=(
                "Sorted by penalized shifted geomean when available. Lower "
                "geomean and median values are better; only accurate "
                "`optimal` solves count as successes."
            ),
            max_rows=50,
            max_cols=12,
        )
    )
    lines.extend(
        _section_table(
            "Software and Runtime",
            _software_versions_table(results, config),
            intro=(
                "Benchmark package and solver package versions captured in "
                "result metadata."
            ),
            max_rows=50,
            max_cols=8,
        )
    )
    configured_solvers = _configured_solvers_table(config)
    if not configured_solvers.empty:
        lines.extend(
            _section_table(
                "Configured Solver Variants",
                configured_solvers,
                max_rows=50,
            )
        )
    return lines


def _render_completion_block(tables: dict[str, pd.DataFrame]) -> list[str]:
    lines: list[str] = []
    lines.extend(
        _section_table(
            "Completion",
            tables.get("completion.csv", pd.DataFrame()),
            source_link="completion.csv",
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
            source_link="solver_metrics.csv",
            intro=(
                "Only accurate `optimal` solves count as successes. Inaccurate, "
                "timeout, skipped, and error statuses count as failures."
            ),
        )
    )
    lines.extend(
        _section_table(
            "Status Counts",
            tables.get("status_counts.csv", pd.DataFrame()),
            source_link="status_counts.csv",
        )
    )
    lines.extend(
        _section_table(
            "Failure Rates",
            tables.get("failure_rates.csv", pd.DataFrame()),
            source_link="failure_rates.csv",
        )
    )
    return lines


def _render_setup_solve_block(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    plot_outputs: list[Path],
) -> list[str]:
    lines: list[str] = []
    lines.extend(
        _section_table(
            "Setup vs Solve Time",
            tables.get("setup_solve_breakdown.csv", pd.DataFrame()),
            source_link="setup_solve_breakdown.csv",
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
    return lines


def _render_performance_plots_block(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    plot_outputs: list[Path],
    metric: str,
) -> list[str]:
    lines: list[str] = ["## Performance Plots", ""]
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
            source_link=f"shifted_geomean_{metric}.csv",
            intro="Penalized shifted geomeans assign non-successful solves the configured failure penalty.",
        )
    )
    lines.extend(
        _section_table(
            "Shifted Geomean, Successful Solves Only",
            tables.get(f"shifted_geomean_{metric}_success_only.csv", pd.DataFrame()),
            source_link=f"shifted_geomean_{metric}_success_only.csv",
        )
    )
    return lines


def _render_pairwise_block(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    plot_outputs: list[Path],
    metric: str,
) -> list[str]:
    lines: list[str] = ["## Pairwise Comparisons", ""]
    lines.extend(
        _plot_block(
            output_dir,
            plot_outputs,
            [(f"pairwise_scatter_{metric}.png", "Pairwise Solver Scatter")],
        )
    )
    lines.extend(
        _section_table(
            "Pairwise Speedups",
            tables.get(f"pairwise_speedups_{metric}.csv", pd.DataFrame()),
            source_link=f"pairwise_speedups_{metric}.csv",
        )
    )
    return lines


def _render_difficulty_scaling_block(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    plot_outputs: list[Path],
    metric: str,
) -> list[str]:
    lines: list[str] = ["## Difficulty Scaling", ""]
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
            source_link=f"difficulty_scaling_{metric}.csv",
            intro=(
                "Problems are bucketed into equal-population quantile bins of "
                "`metadata.n`. Each row reports a solver's median runtime on "
                "successful solves within that bucket; failures are not "
                "averaged in."
            ),
        )
    )
    return lines


def _render_problem_level_block(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    plot_outputs: list[Path],
    metric: str,
) -> list[str]:
    lines: list[str] = ["## Problem-Level Views", ""]
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
            source_link=f"slowest_solves_{metric}.csv",
        )
    )
    lines.extend(
        _section_table(
            "Failures With Successful Alternatives",
            tables.get(f"failures_with_successful_alternatives_{metric}.csv", pd.DataFrame()),
            source_link=f"failures_with_successful_alternatives_{metric}.csv",
        )
    )
    lines.extend(
        _section_table(
            "Objective Spreads",
            tables.get("objective_spreads.csv", pd.DataFrame()),
            source_link="objective_spreads.csv",
        )
    )
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
    return lines


def _render_kkt_diagnostics_block(
    tables: dict[str, pd.DataFrame],
    output_dir: Path,
    plot_outputs: list[Path],
) -> list[str]:
    lines: list[str] = ["## KKT Diagnostics", ""]
    lines.extend(
        _plot_block(
            output_dir,
            plot_outputs,
            [
                ("kkt_residual_boxplot.png", "KKT Residual Boxplot"),
                ("kkt_residual_heatmap.png", "KKT Residual Heatmap"),
                ("kkt_accuracy_profile.png", "KKT Accuracy Profile"),
            ],
            width=KKT_PLOT_IMAGE_WIDTH,
        )
    )
    lines.extend(
        _section_table(
            "KKT Summary",
            tables.get("kkt_summary.csv", pd.DataFrame()),
            source_link="kkt_summary.csv",
        )
    )
    lines.extend(
        _section_table(
            "Claimed-Optimal KKT Thresholds",
            tables.get("claimed_optimal_kkt_thresholds.csv", pd.DataFrame()),
            source_link="claimed_optimal_kkt_thresholds.csv",
            intro=(
                "Counts of claimed-optimal solves whose worst relative KKT "
                "residual is at or below each threshold. `count_above_max` "
                "flags claims of optimality with residuals above the loosest "
                "threshold — solutions that should not have been labelled "
                "optimal."
            ),
        )
    )
    lines.extend(
        _section_table(
            "KKT Certificate Summary",
            tables.get("kkt_certificate_summary.csv", pd.DataFrame()),
            source_link="kkt_certificate_summary.csv",
        )
    )
    return lines


def _render_provenance_block(
    run_dir: Path,
    manifest: dict,
    results: pd.DataFrame,
) -> list[str]:
    lines: list[str] = ["## Provenance", ""]
    lines.extend(_system_summary_lines(manifest))
    environment_columns = [
        column
        for column in [
            "solver_id",
            "metadata.environment_id",
            "metadata.runtime.python_executable",
            "metadata.runtime.python_version",
            "metadata.runtime.platform",
            "metadata.runtime.cpu_model",
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
        lines.extend(_section_table("Runtime Environments", environment, max_rows=50, level=3))
    lines.extend(_source_config_section(run_dir, manifest))
    lines.extend(
        [
            "### Manifest Excerpt",
            "",
            "```json",
            json.dumps(_manifest_excerpt(manifest), indent=2, sort_keys=True, default=str),
            "```",
            "",
        ]
    )
    return lines


def _render_artifact_index_block(
    output_dir: Path, artifact_outputs: list[Path]
) -> list[str]:
    lines: list[str] = ["## Artifact Index", ""]
    for output in sorted(set(artifact_outputs)):
        lines.append(f"- [{_relative(output, output_dir)}]({_relative(output, output_dir)})")
    lines.append("")
    return lines


def _per_dataset_breakdown(
    results: pd.DataFrame,
    dataset_entries: list[dict],
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
    for entry in dataset_entries:
        label = _dataset_display_label(entry)
        subset = results[results["dataset"] == entry["id"]]
        if subset.empty:
            lines.extend([f"### {label}", "", "No rows for this dataset.", ""])
            continue
        lines.extend([f"### {label}", ""])
        lines.extend(
            _section_table(
                "Solver Metrics",
                _sort_report_table(
                    "solver_metrics.csv",
                    solver_metrics(subset),
                    metric=metric,
                ),
                level=4,
            )
        )
        lines.extend(
            _section_table(
                "Failure Rates",
                _sort_report_table(
                    "failure_rates.csv",
                    failure_rates(subset),
                    metric=metric,
                ),
                level=4,
            )
        )
        lines.extend(
            _section_table(
                f"Shifted Geomean ({metric})",
                _sort_report_table(
                    f"shifted_geomean_{metric}.csv",
                    shifted_geomean(subset, metric=metric),
                    metric=metric,
                ),
                level=4,
            )
        )
        lines.extend(
            _section_table(
                "KKT Summary",
                _sort_report_table(
                    "kkt_summary.csv",
                    kkt_summary(subset),
                    metric=metric,
                ),
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
    source_link: str | None = None,
) -> list[str]:
    heading = "#" * max(1, min(level, 6))
    lines = [f"{heading} {title}", ""]
    if intro:
        lines.extend([intro, ""])
    if table.empty:
        lines.extend(["No rows for this section.", ""])
        return lines
    lines.extend(
        _dataframe_to_markdown(
            table,
            max_rows=max_rows,
            max_cols=max_cols,
            source_link=source_link,
        )
    )
    lines.append("")
    return lines


# Sort specs for fixed-name report tables. Each entry maps a CSV
# filename to a list of (column, ascending) tuples consumed by
# _sort_by_columns. Tables whose sort depends on the metric (so the
# filename includes the metric) live in _METRIC_SCOPED_SORTS below;
# tables with bespoke sort logic live in _SPECIAL_SORT_DISPATCH.
_FIXED_SORTS: dict[str, list[tuple[str, bool]]] = {
    "solver_metrics.csv": [
        ("success_rate", False),
        ("failure_rate", True),
        ("run_time_median_seconds", True),
        ("run_time_total_seconds", True),
        ("solver_id", True),
    ],
    "completion.csv": [
        ("missing", False),
        ("unexpected", False),
        ("duplicate_rows", False),
        ("complete", True),
        ("solver_id", True),
        ("dataset", True),
    ],
    "status_counts.csv": [
        ("count", False),
        ("solver_id", True),
        ("status", True),
    ],
    "failure_rates.csv": [
        ("failure_rate", False),
        ("failure_count", False),
        ("success_rate", True),
        ("solver_id", True),
    ],
    "objective_spreads.csv": [
        ("relative_spread", False),
        ("absolute_spread", False),
        ("solver_count", False),
    ],
    "kkt_summary.csv": [
        ("kkt_missing", False),
        ("primal_res_rel_max", False),
        ("dual_res_rel_max", False),
        ("duality_gap_rel_max", False),
        ("comp_slack_max", False),
        ("kkt_count", True),
        ("solver_id", True),
    ],
    "claimed_optimal_kkt_thresholds.csv": [
        ("count_above_max", False),
        ("worst_max", False),
        ("worst_p95", False),
        ("missing_residuals", False),
        ("solver_id", True),
    ],
    "kkt_certificate_summary.csv": [
        ("cert_invalid", False),
        ("Aty_rel_max", False),
        ("Px_rel_max", False),
        ("cert_valid", True),
        ("solver_id", True),
    ],
    "setup_solve_breakdown.csv": [
        ("total_median", True),
        ("solve_median", True),
        ("setup_median", True),
        ("with_breakdown", False),
        ("solver_id", True),
    ],
}


def _metric_scoped_sort(name: str, metric: str) -> list[tuple[str, bool]] | None:
    """Sort spec for tables whose filename embeds the metric.

    These can't live in the static _FIXED_SORTS dict because the keys
    depend on the runtime ``metric`` argument.
    """
    if name == f"slowest_solves_{metric}.csv":
        return [(metric, False), *_problem_sort_columns()]
    if name == f"failures_with_successful_alternatives_{metric}.csv":
        return [
            ("status", True),
            (f"best_success_{metric}", True),
            ("solver_id", True),
            *_problem_sort_columns(),
        ]
    if name == f"difficulty_scaling_{metric}.csv":
        return [
            ("size_bin", True),
            ("median_time", True),
            ("p95_time", True),
            ("success_count", False),
            ("solver_id", True),
        ]
    if name.startswith(f"shifted_geomean_{metric}"):
        return [
            (metric, True),
            ("failure_count", True),
            ("success_count", False),
            ("solver_id", True),
        ]
    return None


def _sort_report_table(
    name: str,
    table: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Dispatch ``name`` to the right sort strategy.

    Lookup order: fixed dict, metric-scoped dict, named specials,
    fallback to per-problem sort. Adding a new table type is now a
    one-line entry in ``_FIXED_SORTS`` (or ``_metric_scoped_sort`` if
    the filename embeds the metric).
    """
    if table.empty:
        return table

    fixed = _FIXED_SORTS.get(name)
    if fixed is not None:
        if name == "objective_spreads.csv":
            return _sort_by_columns(table, [*fixed, *_problem_sort_columns()])
        return _sort_by_columns(table, fixed)

    metric_scoped = _metric_scoped_sort(name, metric)
    if metric_scoped is not None:
        return _sort_by_columns(table, metric_scoped)

    if name == "missing_results.csv":
        return _sort_by_columns(table, _problem_sort_columns(prefix_solver=True))
    if name == f"pairwise_speedups_{metric}.csv":
        return _sort_pairwise_speedups(table)
    if name == f"performance_ratios_{metric}.csv":
        return _sort_by_row_numeric_max(table, ascending=False)
    if name == "problem_solver_comparison.csv":
        return _sort_problem_solver_comparison(table, metric=metric)
    if name == "status_matrix.csv":
        return table.sort_index()
    return _sort_by_columns(table, _problem_sort_columns(prefix_solver=True))


def _sort_solver_problem_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    status_rank = None
    if "status" in table:
        status_rank = table["status"].eq("optimal").astype(int)
    frame = table.copy()
    if status_rank is not None:
        frame["__status_rank__"] = status_rank
    sort_order = [
        ("__status_rank__", True),
        ("run_time_seconds", False),
        *_problem_sort_columns(),
    ]
    return _sort_by_columns(frame, sort_order).drop(
        columns=["__status_rank__"],
        errors="ignore",
    )


def _sort_by_columns(
    table: pd.DataFrame,
    order: list[tuple[str, bool]],
) -> pd.DataFrame:
    columns = [column for column, _ in order if column in table]
    if not columns:
        return table
    ascending = [ascending for column, ascending in order if column in table]
    reset_index = isinstance(table.index, pd.RangeIndex)
    sorted_table = table.sort_values(
        columns,
        ascending=ascending,
        kind="mergesort",
        na_position="last",
    )
    if reset_index:
        return sorted_table.reset_index(drop=True)
    return sorted_table


def _sort_by_row_numeric_max(
    table: pd.DataFrame,
    *,
    ascending: bool,
) -> pd.DataFrame:
    numeric = table.apply(lambda column: pd.to_numeric(column, errors="coerce"))
    if numeric.empty:
        return table
    fill_value = math.inf if ascending else -math.inf
    order = numeric.max(axis=1).fillna(fill_value).sort_values(
        ascending=ascending,
        kind="mergesort",
    )
    return table.loc[order.index]


def _sort_pairwise_speedups(table: pd.DataFrame) -> pd.DataFrame:
    column = "geomean_speedup_a_over_b"
    if column not in table:
        return _sort_by_columns(table, [("solver_a", True), ("solver_b", True)])
    frame = table.copy()
    speedup = pd.to_numeric(frame[column], errors="coerce")
    frame["__effect_size__"] = speedup.map(
        lambda value: abs(math.log(value)) if value and math.isfinite(value) else None
    )
    sorted_table = _sort_by_columns(
        frame,
        [
            ("__effect_size__", False),
            ("common_successes", False),
            ("solver_a", True),
            ("solver_b", True),
        ],
    )
    return sorted_table.drop(columns=["__effect_size__"], errors="ignore")


def _sort_problem_solver_comparison(
    table: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    metric_columns = [
        column for column in table.columns if str(column).endswith(f"__{metric}")
    ]
    if not metric_columns:
        return _sort_by_columns(table, _problem_sort_columns())
    frame = table.copy()
    frame["__max_solver_metric__"] = frame[metric_columns].apply(
        lambda column: pd.to_numeric(column, errors="coerce"),
    ).max(axis=1)
    sorted_table = _sort_by_columns(
        frame,
        [("__max_solver_metric__", False), *_problem_sort_columns()],
    )
    return sorted_table.drop(columns=["__max_solver_metric__"], errors="ignore")


def _problem_sort_columns(
    *,
    prefix_solver: bool = False,
) -> list[tuple[str, bool]]:
    columns = []
    if prefix_solver:
        columns.append(("solver_id", True))
    columns.extend([("dataset", True), ("problem", True)])
    return columns


def _executive_summary_text(
    *,
    dataset_entries: list[dict],
    results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    metric: str,
) -> str:
    solver_count = results["solver_id"].nunique() if "solver_id" in results else 0
    problem_count = _problem_count(results)
    dataset_count = max(1, len(dataset_entries))
    summary = (
        f"This run compared {solver_count} solver variant"
        f"{'' if solver_count == 1 else 's'} on {problem_count} problem"
        f"{'' if problem_count == 1 else 's'} from {dataset_count} dataset"
        f"{'' if dataset_count == 1 else 's'}, producing {len(results)} result row"
        f"{'' if len(results) == 1 else 's'}. The primary metric is `{metric}`."
    )
    geomean = tables.get(f"shifted_geomean_{metric}.csv", pd.DataFrame())
    if not geomean.empty and metric in geomean:
        ranked = geomean.sort_values(metric, na_position="last")
        best = ranked.iloc[0]
        summary += (
            f" The lowest penalized shifted geomean is `{best['solver_id']}` "
            f"at `{_format_cell(best[metric])}`."
        )
    completion = tables.get("completion.csv", pd.DataFrame())
    if not completion.empty:
        missing = _int_sum(completion, "missing")
        unexpected = _int_sum(completion, "unexpected")
        duplicates = _int_sum(completion, "duplicate_rows")
        if missing == 0 and unexpected == 0 and duplicates == 0:
            summary += " Completion checks found no missing, unexpected, or duplicate rows."
        else:
            summary += (
                " Completion checks found "
                f"{missing} missing, {unexpected} unexpected, and "
                f"{duplicates} duplicate row entries."
            )
    return summary


def _run_scope_table(
    *,
    run_dir: Path,
    manifest: dict,
    config: dict,
    dataset_label: str,
    metric: str,
    results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    solver_count = results["solver_id"].nunique() if "solver_id" in results else 0
    completion = tables.get("completion.csv", pd.DataFrame())
    rows = [
        ("Run name", config.get("name") or manifest.get("run_id") or run_dir.name),
        ("Run directory", str(run_dir)),
        ("Created at UTC", manifest.get("created_at_utc") or _unknown()),
        ("Datasets", dataset_label),
        ("Primary metric", metric),
        ("Result rows", len(results)),
        ("Problems with results", _problem_count(results)),
        ("Solver variants", solver_count),
        ("solver_benchmarks version", BENCHMARK_VERSION),
    ]
    if not completion.empty:
        rows.extend(
            [
                ("Missing expected rows", _int_sum(completion, "missing")),
                ("Unexpected rows", _int_sum(completion, "unexpected")),
                ("Duplicate rows", _int_sum(completion, "duplicate_rows")),
            ]
        )
    rows.append(("Config hash", config.get("config_hash", _unknown())))
    return pd.DataFrame(rows, columns=["field", "value"])


def _headline_solver_performance(
    *,
    results: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    config: dict,
    metric: str,
) -> pd.DataFrame:
    metrics = tables.get("solver_metrics.csv", pd.DataFrame())
    if metrics.empty:
        return pd.DataFrame()

    columns = [
        column
        for column in [
            "solver_id",
            "completed",
            "success_count",
            "failure_count",
            "success_rate",
            "failure_rate",
        ]
        if column in metrics
    ]
    table = metrics[columns].copy()
    solver_names = _solver_name_by_id(config)
    if "solver_id" in table:
        table.insert(
            1,
            "solver",
            table["solver_id"].map(solver_names).fillna(""),
        )

    geomean = tables.get(f"shifted_geomean_{metric}.csv", pd.DataFrame())
    if not geomean.empty and metric in geomean:
        table = table.merge(
            geomean[["solver_id", metric]].rename(
                columns={metric: "penalized_shifted_geomean"}
            ),
            on="solver_id",
            how="left",
        )
    success_geomean = tables.get(
        f"shifted_geomean_{metric}_success_only.csv",
        pd.DataFrame(),
    )
    if not success_geomean.empty and metric in success_geomean:
        table = table.merge(
            success_geomean[["solver_id", metric]].rename(
                columns={metric: "success_only_shifted_geomean"}
            ),
            on="solver_id",
            how="left",
        )

    metric_aggregates = _metric_aggregates(results, metric)
    if not metric_aggregates.empty:
        table = table.merge(metric_aggregates, on="solver_id", how="left")

    sort_columns = []
    ascending = []
    if "penalized_shifted_geomean" in table:
        sort_columns.append("penalized_shifted_geomean")
        ascending.append(True)
    if "success_rate" in table:
        sort_columns.append("success_rate")
        ascending.append(False)
    median_column = f"median_{metric}"
    if median_column in table:
        sort_columns.append(median_column)
        ascending.append(True)
    if sort_columns:
        table = table.sort_values(sort_columns, ascending=ascending, na_position="last")
    return table.reset_index(drop=True)


def _metric_aggregates(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    if results.empty or metric not in results or "solver_id" not in results:
        return pd.DataFrame()
    values = results[["solver_id", metric]].copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    grouped = values.groupby("solver_id", dropna=False, observed=True)[metric]
    return grouped.agg(["median", "sum", "max"]).reset_index().rename(
        columns={
            "median": f"median_{metric}",
            "sum": f"total_{metric}",
            "max": f"max_{metric}",
        }
    )


def _configured_solvers_table(config: dict) -> pd.DataFrame:
    solvers = pd.DataFrame(config.get("solvers", []))
    if solvers.empty:
        return solvers
    if "settings" in solvers:
        solvers["settings"] = solvers["settings"].map(_compact_json)
    columns = [
        column
        for column in ("id", "solver", "timeout_seconds", "settings")
        if column in solvers
    ]
    return solvers[columns]


def _software_versions_table(results: pd.DataFrame, config: dict) -> pd.DataFrame:
    solver_names = _solver_name_by_id(config)
    result_solver_ids = (
        set(_unique_series_values(results["solver_id"]))
        if "solver_id" in results
        else set()
    )
    solver_ids = sorted(set(solver_names) | result_solver_ids)
    rows = []
    for solver_id in solver_ids:
        subset = (
            results[results["solver_id"] == solver_id]
            if "solver_id" in results and not results.empty
            else pd.DataFrame()
        )
        row = {
            "solver_id": solver_id,
            "solver": solver_names.get(solver_id, ""),
            "solver_benchmarks": BENCHMARK_VERSION,
            "solver_packages": _package_versions_summary(subset),
            "python_versions": _unique_values(subset, "metadata.runtime.python_version"),
            "environment_ids": _unique_values(subset, "metadata.environment_id"),
            "platforms": _unique_values(subset, "metadata.runtime.platform"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _system_summary_lines(manifest: dict) -> list[str]:
    """Render the manifest's ``system`` block as a Markdown summary.

    The block is captured once at run-start and answers the implicit
    question "what hardware made these timings"; surface the most
    relevant fields up-front (CPU model and count, total RAM, OS,
    Python version) so a reviewer can size up the timing data
    without scrolling to the manifest. Older runs without a
    ``system`` block render as an empty section so downstream
    parsing doesn't break on legacy data.
    """
    system = manifest.get("system") or {}
    if not system:
        return []
    cpu = system.get("cpu") or {}
    memory = system.get("memory") or {}

    rows: list[tuple[str, str]] = []
    cpu_model = cpu.get("model")
    if cpu_model:
        rows.append(("CPU model", str(cpu_model)))
    logical = cpu.get("logical_count")
    physical = cpu.get("physical_count")
    if logical or physical:
        if physical and logical and physical != logical:
            rows.append(("CPU cores", f"{physical} physical / {logical} logical"))
        elif logical:
            rows.append(("CPU cores", f"{logical} logical"))
    max_freq = cpu.get("max_frequency_mhz")
    if max_freq:
        rows.append(("CPU max frequency", f"{float(max_freq):.0f} MHz"))
    total_mem = memory.get("total_bytes")
    if total_mem:
        rows.append(("Total RAM", _format_bytes(int(total_mem))))
    available_mem = memory.get("available_bytes")
    if available_mem:
        rows.append(("RAM available at start", _format_bytes(int(available_mem))))
    if system.get("platform"):
        rows.append(("OS", str(system["platform"])))
    if system.get("python_version"):
        rows.append(("Python", str(system["python_version"])))
    libs = system.get("library_versions") or {}
    if libs:
        formatted = ", ".join(
            f"{name} {version}" for name, version in sorted(libs.items()) if version
        )
        if formatted:
            rows.append(("Libraries", formatted))

    if not rows:
        return []

    lines = ["### System", "", "| Field | Value |", "| --- | --- |"]
    for field, value in rows:
        lines.append(f"| {field} | {value} |")
    lines.append("")
    return lines


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count as the largest sensible binary-prefix unit.

    Memory totals are typically reported in GiB / TiB; using SI
    prefixes (10^9) rounds 16 GiB DIMMs to the misleading 17.2 GB.
    """
    units = (("TiB", 1024**4), ("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024))
    for label, divisor in units:
        if num_bytes >= divisor:
            return f"{num_bytes / divisor:.1f} {label}"
    return f"{num_bytes} B"


def _source_config_section(run_dir: Path, manifest: dict) -> list[str]:
    lines = ["### Exact Source Config", ""]
    paths = _source_config_paths(run_dir)
    if paths:
        for path in paths:
            lines.extend(
                [
                    f"Source: `{path.name}`",
                    "",
                    f"```{_fence_language(path)}",
                    path.read_text(encoding="utf-8").rstrip(),
                    "```",
                    "",
                ]
            )
        return lines

    config = manifest.get("config", {})
    lines.extend(
        [
            "No copied source config file was found in this run directory. "
            "The normalized manifest config is shown instead.",
            "",
            "```json",
            json.dumps(config, indent=2, sort_keys=True, default=str),
            "```",
            "",
        ]
    )
    return lines


def _source_config_paths(run_dir: Path) -> list[Path]:
    candidates = [
        "run_config.yaml",
        "run_config.yml",
        "run_config.json",
        "environment_config.yaml",
        "environment_config.yml",
        "environment_config.json",
    ]
    return [run_dir / name for name in candidates if (run_dir / name).exists()]


def _fence_language(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".json":
        return "json"
    return ""


def _compact_json(value) -> str:
    if isinstance(value, float):
        try:
            if pd.isna(value):
                return ""
        except ValueError:
            pass
    if value in (None, ""):
        return ""
    return json.dumps(value, sort_keys=True, default=str)


def _solver_name_by_id(config: dict) -> dict[str, str]:
    return {
        str(solver.get("id")): str(solver.get("solver", ""))
        for solver in config.get("solvers", [])
        if solver.get("id") is not None
    }


def _package_versions_summary(results: pd.DataFrame) -> str:
    if results.empty:
        return ""
    prefix = "metadata.runtime.solver_package_versions."
    parts = []
    for column in sorted(column for column in results.columns if column.startswith(prefix)):
        package = column[len(prefix):]
        values = _unique_series_values(results[column])
        if values:
            parts.append(f"{package}={','.join(values)}")
    return "; ".join(parts)


def _unique_values(results: pd.DataFrame, column: str) -> str:
    if results.empty or column not in results:
        return ""
    return ", ".join(_unique_series_values(results[column]))


def _unique_series_values(series: pd.Series) -> list[str]:
    values = []
    for value in series.dropna().unique():
        text = str(value)
        if text:
            values.append(text)
    return sorted(values)


def _int_sum(table: pd.DataFrame, column: str) -> int:
    if table.empty or column not in table:
        return 0
    return int(pd.to_numeric(table[column], errors="coerce").fillna(0).sum())


def _dataframe_to_markdown(
    table: pd.DataFrame,
    *,
    max_rows: int,
    max_cols: int,
    source_link: str | None = None,
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
                f"of {total_rows} rows and {total_cols} columns. "
                f"{_full_table_note(source_link)}",
            ]
        )
    return rows


def _full_table_note(source_link: str | None) -> str:
    if source_link:
        return f"See [full CSV]({source_link}) for the full table."
    return "See the linked CSV for the full table."


def _plot_block(
    output_dir: Path,
    plot_outputs: list[Path],
    plots: list[tuple[str, str]],
    *,
    width: int = PLOT_IMAGE_WIDTH,
) -> list[str]:
    available = {path.name: path for path in plot_outputs}
    lines = []
    for filename, title in plots:
        path = available.get(filename)
        if path is None:
            continue
        relative = _relative(path, output_dir)
        lines.extend(
            [
                f"### {title}",
                "",
                (
                    f'<img src="{html.escape(relative, quote=True)}" '
                    f'alt="{html.escape(title, quote=True)}" width="{width}">'
                ),
                "",
            ]
        )
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


def _problem_count(results: pd.DataFrame) -> int:
    if "problem" not in results.columns:
        return 0
    if "dataset" in results.columns:
        return int(results[["dataset", "problem"]].drop_duplicates().shape[0])
    return int(results["problem"].nunique())


def _dataset_display_label(entry: dict) -> str:
    entry_id = str(entry.get("id") or entry.get("name") or _unknown())
    name = str(entry.get("name") or entry_id)
    if entry_id != name:
        return f"{entry_id} ({name})"
    return entry_id
