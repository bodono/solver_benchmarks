"""Plot generation for benchmark runs."""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import matplotlib

# Only force the headless ``Agg`` backend if the user hasn't already
# selected one. Importing this module from a notebook (or any process
# that already configured an interactive backend) would otherwise
# silently override that choice.
if matplotlib.get_backend().lower() not in {
    "agg",
    "pdf",
    "ps",
    "svg",
    "cairo",
    "module://matplotlib_inline.backend_inline",
}:
    matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.tables import (
    difficulty_scaling,
    failure_rates,
    performance_ratio_matrix,
    setup_solve_breakdown,
    status_matrix,
)
from solver_benchmarks.core import status


def write_analysis_plots(
    run_dir: str | Path,
    *,
    metric: str = "run_time_seconds",
    output_dir: str | Path | None = None,
) -> list[Path]:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(run_dir)
    if results.empty:
        return []

    paths = [
        _write_performance_profile(results, output_dir, metric),
        _write_shifted_geomean(results, output_dir, metric),
        _write_failure_rates(results, output_dir),
        _write_cactus(results, output_dir, metric),
        _write_pairwise_scatter(results, output_dir, metric),
        _write_performance_ratio_heatmap(results, output_dir, metric),
        _write_status_heatmap(results, output_dir),
        _write_kkt_residual_boxplot(results, output_dir),
        _write_kkt_residual_heatmap(results, output_dir),
        _write_kkt_accuracy_profile(results, output_dir),
        _write_difficulty_scaling(results, output_dir, metric),
        _write_setup_solve_breakdown(results, output_dir),
    ]
    return [path for path in paths if path is not None]


def _write_performance_profile(results, output_dir: Path, metric: str) -> Path | None:
    profile = performance_profile(results, metric=metric)
    if profile.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for column in profile.columns:
        if column == "tau":
            continue
        ax.plot(profile["tau"], profile[column], label=column, linewidth=2)
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Performance ratio tau")
    ax.set_ylabel("Fraction of problems solved")
    ax.set_title(f"Dolan-More Performance Profile ({metric})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    path = output_dir / f"performance_profile_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_shifted_geomean(results, output_dir: Path, metric: str) -> Path | None:
    geomean = shifted_geomean(results, metric=metric)
    if geomean.empty:
        return None

    geomean = geomean.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.barh(geomean["solver_id"], geomean[metric])
    ax.set_xlabel(metric)
    ax.set_title(f"Penalized Shifted Geometric Mean ({metric})")
    ax.grid(True, axis="x", alpha=0.25)

    path = output_dir / f"shifted_geomean_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_failure_rates(results, output_dir: Path) -> Path | None:
    failures = failure_rates(results)
    if failures.empty:
        return None

    failures = failures.sort_values("failure_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.barh(failures["solver_id"], failures["failure_rate"] * 100.0)
    ax.set_xlabel("Failure rate (%)")
    ax.set_xlim(0.0, 100.0)
    ax.set_title("Failure Rate")
    ax.grid(True, axis="x", alpha=0.25)

    path = output_dir / "failure_rates.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _unique_problem_count(results: pd.DataFrame) -> int:
    if "dataset" in results.columns:
        return max(1, results[["dataset", "problem"]].drop_duplicates().shape[0])
    return max(1, results["problem"].nunique())


def _write_cactus(results, output_dir: Path, metric: str) -> Path | None:
    if metric not in results:
        return None
    successful = results[results["status"].isin(status.SOLUTION_PRESENT)].copy()
    successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
    # Keep zero-duration successes (instantly-solved problems) so the
    # cactus curve correctly counts them; previously a strict ``> 0``
    # filter dropped these rows but the denominator still used the
    # total problem count, so a solver with one zero-time and one
    # nonzero-time success would top out at 0.5 instead of 1.0.
    # Negative or non-finite metric values are still dropped — those
    # would not have a meaningful place on a log-scale axis.
    successful = successful[np.isfinite(successful[metric]) & (successful[metric] >= 0.0)]
    if successful.empty:
        return None

    problem_count = _unique_problem_count(results)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    has_positive = False
    # On a log axis a literal zero would map to -inf. Floor zeros to a
    # tiny positive value so they're drawn at the leftmost edge of the
    # axis but still contribute one step to the curve.
    log_floor = 1.0e-12
    for solver_id, group in successful.groupby("solver_id", observed=True):
        raw = group[metric].to_numpy()
        if raw.size == 0:
            continue
        has_positive = has_positive or bool(np.any(raw > 0.0))
        values = np.sort(np.where(raw > 0.0, raw, log_floor))
        fractions = np.arange(1, len(values) + 1) / problem_count
        ax.step(values, fractions, where="post", label=solver_id, linewidth=2)
    if not ax.lines:
        plt.close(fig)
        return None
    if has_positive:
        ax.set_xscale("log")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(metric)
    ax.set_ylabel("Fraction of problems solved")
    ax.set_title(f"Cactus Plot ({metric})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()

    path = output_dir / f"cactus_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_pairwise_scatter(results, output_dir: Path, metric: str) -> Path | None:
    if metric not in results:
        return None
    successful = results[results["status"].isin(status.SOLUTION_PRESENT)].copy()
    successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
    successful = successful[np.isfinite(successful[metric]) & (successful[metric] > 0.0)]
    keys = ["dataset", "problem"] if "dataset" in successful.columns else ["problem"]
    index = keys[0] if len(keys) == 1 else keys
    pivot = successful.pivot_table(
        index=index,
        columns="solver_id",
        values=metric,
        aggfunc="first",
    )
    pairs = [
        (solver_a, solver_b)
        for solver_a, solver_b in combinations(sorted(pivot.columns), 2)
        if not pivot[[solver_a, solver_b]].dropna().empty
    ]
    if not pairs:
        return None

    cols = min(3, len(pairs))
    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(5 * cols, 4.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")

    for ax, (solver_a, solver_b) in zip(axes.ravel(), pairs):
        common = pivot[[solver_a, solver_b]].dropna()
        x = common[solver_a]
        y = common[solver_b]
        lower = min(float(x.min()), float(y.min()))
        upper = max(float(x.max()), float(y.max()))
        ax.scatter(x, y, alpha=0.7, s=18)
        ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(solver_a)
        ax.set_ylabel(solver_b)
        ax.set_title(f"{solver_a} vs {solver_b}")
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(f"Pairwise Solver Scatter ({metric})")
    path = output_dir / f"pairwise_scatter_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_status_heatmap(results, output_dir: Path) -> Path | None:
    matrix = status_matrix(results)
    if matrix.empty:
        return None

    statuses = sorted(
        {str(value) for value in matrix.to_numpy().ravel() if pd.notna(value)}
    )
    palette = _status_palette(statuses)
    # Use apply(map) instead of replace + astype: a stray cell whose
    # string form is not in `statuses` would otherwise leak through
    # replace() and crash astype(float). With map we get NaN for the
    # unrecognized cell, which the colormap renders as `set_bad`.
    encoding = {name: idx for idx, name in enumerate(statuses)}
    encoded = matrix.apply(lambda col: col.map(encoding)).astype(float)

    fig_height = min(24, max(4, 0.12 * len(matrix.index)))
    fig_width = min(18, max(6, 1.8 * len(matrix.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    colors = [palette[name] for name in statuses]
    cmap = matplotlib.colors.ListedColormap(colors)
    ax.imshow(encoded, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    if len(matrix.index) <= 60:
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels(_index_labels(matrix.index))
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{len(matrix.index)} problems")
    ax.set_title("Status Heatmap")
    legend = [
        mpatches.Patch(color=palette[name], label=name)
        for name in statuses
    ]
    ax.legend(handles=legend, bbox_to_anchor=(1.02, 1), loc="upper left")

    path = output_dir / "status_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_performance_ratio_heatmap(
    results,
    output_dir: Path,
    metric: str,
) -> Path | None:
    ratios = performance_ratio_matrix(results, metric=metric)
    if ratios.empty:
        return None

    encoded = np.log10(ratios.astype(float))
    fig_height = min(24, max(4, 0.12 * len(ratios.index)))
    fig_width = min(18, max(6, 1.8 * len(ratios.columns)))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    image = ax.imshow(encoded, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(np.arange(len(ratios.columns)))
    ax.set_xticklabels(ratios.columns, rotation=45, ha="right")
    if len(ratios.index) <= 60:
        ax.set_yticks(np.arange(len(ratios.index)))
        ax.set_yticklabels(_index_labels(ratios.index))
    else:
        ax.set_yticks([])
        ax.set_ylabel(f"{len(ratios.index)} problems")
    ax.set_title(f"Per-Problem Performance Ratio Heatmap ({metric})")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("log10(metric / best metric)")

    path = output_dir / f"performance_ratio_heatmap_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_kkt_residual_boxplot(results, output_dir: Path) -> Path | None:
    residual_fields = [
        ("kkt.primal_res_rel", "primal_res_rel"),
        ("kkt.dual_res_rel", "dual_res_rel"),
        ("kkt.comp_slack", "comp_slack"),
        ("kkt.duality_gap_rel", "duality_gap_rel"),
    ]
    available = [(col, label) for col, label in residual_fields if col in results]
    if not available:
        return None

    successful = results[results["status"].isin(status.SOLUTION_PRESENT)].copy()
    if successful.empty:
        return None

    solver_ids = sorted(successful["solver_id"].dropna().unique())
    if not solver_ids:
        return None

    cols = len(available)
    fig, axes = plt.subplots(
        1, cols, figsize=(4.5 * cols, 4.5), constrained_layout=True, squeeze=False
    )
    drew_any = False
    for ax, (col, label) in zip(axes.ravel(), available):
        data = []
        labels = []
        for solver_id in solver_ids:
            values = pd.to_numeric(
                successful[successful["solver_id"] == solver_id][col], errors="coerce"
            ).dropna()
            values = values[np.isfinite(values) & (values > 0.0)]
            if values.empty:
                continue
            data.append(np.log10(values.to_numpy()))
            labels.append(solver_id)
        if not data:
            ax.set_axis_off()
            ax.set_title(f"{label}\n(no data)")
            continue
        drew_any = True
        # tick_labels is the matplotlib >=3.9 spelling; the legacy
        # `labels` kwarg was deprecated in 3.9 and removed in 3.11.
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        rng = np.random.default_rng(42)
        for idx, arr in enumerate(data):
            xs = rng.normal(idx + 1, 0.07, arr.size)
            ax.scatter(xs, arr, alpha=0.45, s=12, color="#1f77b4", edgecolors="none")
        ax.set_title(label)
        ax.set_ylabel("log10(residual)")
        ax.grid(True, axis="y", alpha=0.25)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha("right")

    if not drew_any:
        plt.close(fig)
        return None

    fig.suptitle("KKT Residuals on Successful Solves")
    path = output_dir / "kkt_residual_boxplot.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


_KKT_RESIDUAL_FIELDS = (
    ("kkt.primal_res_rel", "primal_res_rel"),
    ("kkt.dual_res_rel", "dual_res_rel"),
    ("kkt.comp_slack", "comp_slack"),
    ("kkt.duality_gap_rel", "duality_gap_rel"),
)


def _write_kkt_residual_heatmap(results, output_dir: Path) -> Path | None:
    available = [(col, label) for col, label in _KKT_RESIDUAL_FIELDS if col in results]
    if not available:
        return None

    successful = results[results["status"].isin(status.SOLUTION_PRESENT)].copy()
    if successful.empty:
        return None

    pivots: list[tuple[str, pd.DataFrame]] = []
    keys = ["dataset", "problem"] if "dataset" in successful.columns else ["problem"]
    pivot_index = keys[0] if len(keys) == 1 else keys
    for col, label in available:
        successful[col] = pd.to_numeric(successful[col], errors="coerce")
        frame = successful[np.isfinite(successful[col]) & (successful[col] > 0.0)]
        if frame.empty:
            continue
        pivot = frame.pivot_table(
            index=pivot_index,
            columns="solver_id",
            values=col,
            aggfunc="first",
        ).sort_index()
        if pivot.empty:
            continue
        pivots.append((label, pivot))

    if not pivots:
        return None

    problem_count = max(len(pivot.index) for _, pivot in pivots)
    solver_count = max(len(pivot.columns) for _, pivot in pivots)
    panel_height = min(14, max(3.5, 0.14 * problem_count))
    panel_width = min(10, max(4.0, 1.2 * solver_count))
    cols = len(pivots)
    fig, axes = plt.subplots(
        1,
        cols,
        figsize=(panel_width * cols, panel_height),
        constrained_layout=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e7eb")
    for ax, (label, pivot) in zip(axes.ravel(), pivots):
        encoded = np.log10(pivot.astype(float))
        image = ax.imshow(encoded, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        if len(pivot.index) <= 60:
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(_index_labels(pivot.index))
        else:
            ax.set_yticks([])
            ax.set_ylabel(f"{len(pivot.index)} problems")
        ax.set_title(label)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log10(residual)")

    fig.suptitle("KKT Residual Heatmap (successful solves)")
    path = output_dir / "kkt_residual_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_kkt_accuracy_profile(results, output_dir: Path) -> Path | None:
    available = [(col, label) for col, label in _KKT_RESIDUAL_FIELDS if col in results]
    if not available:
        return None

    successful = results[results["status"].isin(status.SOLUTION_PRESENT)].copy()
    if successful.empty:
        return None

    solver_ids = sorted(successful["solver_id"].dropna().unique())
    if not solver_ids:
        return None

    problem_totals = {
        solver_id: int((results["solver_id"] == solver_id).sum())
        for solver_id in solver_ids
    }
    thresholds = np.logspace(-12, 0, 121)

    cols = len(available)
    fig, axes = plt.subplots(
        1, cols, figsize=(4.5 * cols, 4.5), constrained_layout=True, squeeze=False
    )
    drew_any = False
    for ax, (col, label) in zip(axes.ravel(), available):
        per_solver = []
        for solver_id in solver_ids:
            series = pd.to_numeric(
                successful.loc[successful["solver_id"] == solver_id, col], errors="coerce"
            )
            series = series[np.isfinite(series)].to_numpy()
            total = problem_totals.get(solver_id, 0)
            if total == 0 or series.size == 0:
                continue
            series = np.where(series <= 0.0, 1e-16, series)
            fractions = np.array([(series <= t).sum() / total for t in thresholds])
            per_solver.append((solver_id, fractions))
        if not per_solver:
            ax.set_axis_off()
            ax.set_title(f"{label}\n(no data)")
            continue
        drew_any = True
        for solver_id, fractions in per_solver:
            ax.plot(thresholds, fractions, label=solver_id, linewidth=2)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("residual threshold tau")
        ax.set_ylabel("fraction with residual <= tau")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize="small")

    if not drew_any:
        plt.close(fig)
        return None

    fig.suptitle("KKT Accuracy Profile (per solver, across all problems)")
    path = output_dir / "kkt_accuracy_profile.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_difficulty_scaling(results, output_dir: Path, metric: str) -> Path | None:
    table = difficulty_scaling(results, metric=metric)
    if table.empty:
        return None
    plottable = table.dropna(subset=["median_time"])
    if plottable.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    drew_any = False
    for solver_id, group in plottable.groupby("solver_id", observed=True):
        ordered = group.sort_values("size_bin")
        midpoints = 0.5 * (ordered["size_min"] + ordered["size_max"])
        ax.plot(
            midpoints,
            ordered["median_time"],
            marker="o",
            linewidth=2,
            label=str(solver_id),
        )
        drew_any = True
    if not drew_any:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    if (plottable["median_time"] > 0).any():
        ax.set_yscale("log")
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel(f"Median {metric} (successes)")
    ax.set_title(f"Difficulty Scaling ({metric})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize="small")

    path = output_dir / f"difficulty_scaling_{metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_setup_solve_breakdown(results, output_dir: Path) -> Path | None:
    table = setup_solve_breakdown(results)
    if table.empty:
        return None
    plottable = table.dropna(subset=["setup_median", "solve_median"])
    if plottable.empty:
        return None
    plottable = plottable.sort_values("total_median", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    solver_ids = plottable["solver_id"].astype(str).tolist()
    setup = plottable["setup_median"].to_numpy()
    solve = plottable["solve_median"].to_numpy()
    ax.barh(solver_ids, setup, color="#dc2626", label="setup")
    ax.barh(solver_ids, solve, left=setup, color="#2563eb", label="solve")
    ax.set_xlabel("Median time per solve (s)")
    ax.set_title("Setup vs Solve Time (median across successful solves)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower right", fontsize="small")

    path = output_dir / "setup_solve_breakdown.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _index_labels(index: pd.Index) -> list[str]:
    """Render a (possibly MultiIndex) row index as legible tick labels.

    Multi-dataset frames use a ``(dataset, problem)`` MultiIndex so that two
    datasets sharing a problem name remain distinct rows. Tuples render as
    ``ds_a/p1`` rather than the default ``('ds_a', 'p1')``.
    """
    if isinstance(index, pd.MultiIndex):
        return ["/".join(str(part) for part in entry) for entry in index]
    return [str(entry) for entry in index]


def _status_palette(statuses: list[str]) -> dict[str, str]:
    base = {
        "optimal": "#2f7d32",
        "optimal_inaccurate": "#f0b429",
        "primal_infeasible": "#8c6d31",
        "dual_infeasible": "#8c6d31",
        "primal_infeasible_inaccurate": "#b08d57",
        "dual_infeasible_inaccurate": "#b08d57",
        "primal_or_dual_infeasible": "#8c6d31",
        "max_iter_reached": "#d97706",
        "time_limit": "#b45309",
        "solver_error": "#b91c1c",
        "worker_error": "#7f1d1d",
        "skipped_unsupported": "#64748b",
        "missing": "#e5e7eb",
    }
    fallback = [
        "#2563eb",
        "#7c3aed",
        "#db2777",
        "#0891b2",
        "#4d7c0f",
    ]
    return {
        name: base.get(name, fallback[idx % len(fallback)])
        for idx, name in enumerate(statuses)
    }
