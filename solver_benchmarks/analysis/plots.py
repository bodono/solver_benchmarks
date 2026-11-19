"""Plot generation for benchmark runs."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
from solver_benchmarks.analysis.outcomes import success_mask as solved_mask
from solver_benchmarks.analysis.outcomes import with_expected_outcomes
from solver_benchmarks.analysis.penalties import geomean_time_limits
from solver_benchmarks.analysis.profiles import (
    TIME_METRICS,
    complete_results,
    drop_solvers_without_metric,
    performance_profile,
    profile_floor,
    shifted_geomean,
)
from solver_benchmarks.analysis.tables import (
    difficulty_scaling,
    expected_results,
    failure_rates,
    one_row_per_solve,
    optimality_successes,
    performance_ratio_matrix,
    setup_solve_breakdown,
    status_matrix,
    successful_metric_values,
)
from solver_benchmarks.analysis.verify import VerifyPolicy
from solver_benchmarks.core import status as status_codes

logger = logging.getLogger(__name__)

# Hard cap on the number of pairwise-scatter subplots in a single figure.
# Each pair becomes its own axes; matplotlib's constrained_layout solver
# (kiwisolver) is super-linear in the axes count and goes pathological
# well before the figure is human-readable. The cap is set so common
# multi-solver configs (~10 solvers ⇒ 45 pairs) still render — protection
# is targeted at the runaway case (100s+ solvers ⇒ 1000s of axes), not at
# normal benchmark sweeps. Past the cap, pairwise_speedups_*.csv remains
# the right artifact to inspect.
_PAIRWISE_SCATTER_MAX_PAIRS = 50

# Line styles of the per-solver line plots: matplotlib's ten default
# colours, then the same colours with the next line style (and marker), so
# up to 40 solvers each get a distinct look.
_LINE_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
_LINE_STYLES = ("-", "--", "-.", ":")
_LINE_MARKERS = ("o", "s", "^", "D")


def solver_styles(solver_ids: Iterable) -> dict[str, dict[str, str]]:
    """Colour, line style and marker of each solver in the line plots.

    Solvers are numbered in sorted order; the i-th gets colour ``i % 10``
    with line style and marker ``i // 10`` (cycling after 40 solvers), so
    the 11th..40th solvers do not look like the 1st..10th. A style depends
    only on the set of ids: ``write_analysis_plots`` builds one map from
    every solver of the run, recorded or planned, so a solver looks the
    same in every plot, including plots that draw only some solvers.
    """
    ids = sorted({str(solver_id) for solver_id in solver_ids})
    return {
        solver_id: {
            "color": _LINE_COLORS[index % len(_LINE_COLORS)],
            "linestyle": _LINE_STYLES[(index // len(_LINE_COLORS)) % len(_LINE_STYLES)],
            "marker": _LINE_MARKERS[(index // len(_LINE_COLORS)) % len(_LINE_MARKERS)],
        }
        for index, solver_id in enumerate(ids)
    }


def _line_style(styles: dict[str, dict[str, str]], solver_id, *, marker: bool = False) -> dict[str, Any]:
    """Plot keyword arguments of ``solver_id`` in ``styles``."""
    style = styles.get(str(solver_id), {})
    if not marker:
        style = {key: value for key, value in style.items() if key != "marker"}
    return style


def write_analysis_plots(
    run_dir: str | Path,
    *,
    metric: str = "run_time_seconds",
    output_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    data_root: str | Path | None = None,
    max_value: float | None = None,
    results: pd.DataFrame | None = None,
    expected: pd.DataFrame | None = None,
    geomean_options: dict | None = None,
    verify: bool | VerifyPolicy = True,
) -> list[Path]:
    """Write the analysis plots of a run.

    Statuses are KKT-verified as ``load_results`` verifies them, with
    ``verify`` choosing the rule (False: the solvers' own statuses).
    ``results``, ``expected`` and ``geomean_options`` let a caller that has
    them already (``bench report``) skip loading them again. Without a
    usable time limit (and no ``max_value``) the penalized time geomean is
    skipped with a warning; every other plot is still written. ``repo_root``
    and ``data_root`` locate the datasets listed for the planned solves (see
    ``expected_results``).

    Duplicated solves are resolved once, by ``metric``, and every plot draws
    that one row per solve. A plot of ``metric`` (see :func:`plot_names`)
    that is not drawn removes the file of its name left by an earlier call;
    plots of another metric and every other file are left alone.
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if results is None:
        results = load_results(run_dir, verify=verify)
    if results.empty:
        return []

    if expected is None:
        expected = expected_results(run_dir, repo_root=repo_root, data_root=data_root)
    if geomean_options is None:
        geomean_options = penalized_geomean_options(run_dir, metric=metric, max_value=max_value)
    scored = with_expected_outcomes(results)
    solves = one_row_per_solve(scored, metric=metric)
    # Planned solvers without rows are drawn too (the profile charges their
    # missing solves), so they get a style of their own.
    solver_ids = set(solves["solver_id"]) if "solver_id" in solves else set()
    if expected is not None and "solver_id" in expected:
        solver_ids |= set(expected["solver_id"])
    styles = solver_styles(solver_ids)
    paths = [
        _write_performance_profile(solves, output_dir, metric, expected=expected, styles=styles),
        _write_shifted_geomean(solves, output_dir, metric, expected=expected, geomean_options=geomean_options)
        if geomean_options.get("available", True)
        else None,
        _write_failure_rates(solves, output_dir, expected=expected),
        _write_cactus(solves, output_dir, metric, expected=expected, styles=styles),
        _write_pairwise_scatter(solves, output_dir, metric),
        _write_performance_ratio_heatmap(solves, output_dir, metric),
        _write_status_heatmap(solves, output_dir),
        _write_kkt_residual_boxplot(solves, output_dir),
        _write_kkt_residual_heatmap(solves, output_dir),
        _write_kkt_accuracy_profile(solves, output_dir, styles=styles),
        # Sizes problems from every row and resolves duplicates like solves.
        _write_difficulty_scaling(scored, output_dir, metric, styles=styles),
        _write_setup_solve_breakdown(solves, output_dir),
    ]
    written = [path for path in paths if path is not None]
    drawn = {path.name for path in written}
    for name in plot_names(metric):
        if name not in drawn:
            (output_dir / name).unlink(missing_ok=True)
    return written


def plot_names(metric: str) -> list[str]:
    """File names of every plot :func:`write_analysis_plots` can write for ``metric``."""
    return [
        f"performance_profile_{metric}.png",
        f"shifted_geomean_{metric}.png",
        "failure_rates.png",
        f"cactus_{metric}.png",
        f"pairwise_scatter_{metric}.png",
        f"performance_ratio_heatmap_{metric}.png",
        "status_heatmap.png",
        "kkt_residual_boxplot.png",
        "kkt_residual_heatmap.png",
        "kkt_accuracy_profile.png",
        f"difficulty_scaling_{metric}.png",
        "setup_solve_breakdown.png",
    ]


def penalized_geomean_options(run_dir: Path, *, metric: str, max_value: float | None) -> dict:
    """Penalty options for the penalized geomean of ``metric``.

    ``available`` is False, with the reason under ``unavailable_reason``,
    when ``metric`` is a time and the manifest records no usable limit and
    no ``max_value`` was given: the penalized time geomean is then left out
    rather than the whole report or plot set.
    """
    try:
        limits = geomean_time_limits(run_dir, metric=metric, max_value=max_value)
    except ValueError as exc:
        if metric not in TIME_METRICS or max_value is not None:
            raise
        logger.warning("Skipping the penalized %s geomean: %s", metric, exc)
        return {"max_value": None, "timeout_seconds": None, "available": False, "unavailable_reason": str(exc)}
    return {"max_value": max_value, "timeout_seconds": limits, "available": True}


def geomean_kwargs(options: dict | None) -> dict:
    """The ``shifted_geomean`` keyword arguments within penalty options."""
    options = options or {}
    return {key: options[key] for key in ("max_value", "timeout_seconds") if key in options}


def _write_performance_profile(
    results,
    output_dir: Path,
    metric: str,
    *,
    expected: pd.DataFrame | None = None,
    styles: dict[str, dict[str, str]] | None = None,
) -> Path | None:
    profile = performance_profile(results, metric=metric, expected=expected)
    if profile.empty:
        return None

    solver_columns = [column for column in profile.columns if column != "tau"]
    styles = styles if styles is not None else solver_styles(solver_columns)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for column in solver_columns:
        ax.plot(profile["tau"], profile[column], label=column, linewidth=2, **_line_style(styles, column))
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


def _write_shifted_geomean(results, output_dir: Path, metric: str, *, expected: pd.DataFrame | None = None, geomean_options: dict | None = None) -> Path | None:
    geomean = shifted_geomean(results, metric=metric, expected=expected, **geomean_kwargs(geomean_options))
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


def _write_failure_rates(results, output_dir: Path, *, expected: pd.DataFrame | None = None) -> Path | None:
    failures = failure_rates(results, expected=expected)
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


def _write_cactus(
    results,
    output_dir: Path,
    metric: str,
    *,
    expected: pd.DataFrame | None = None,
    styles: dict[str, dict[str, str]] | None = None,
) -> Path | None:
    """Fraction of problems solved within each budget of ``metric``.

    The denominator is the performance profile's: every problem of the
    comparison, planned solves never recorded included when ``expected`` is
    given. Values are floored at the profile's resolution for ``metric``
    (``profile_floor``), and zeros of metrics without one at 1e-12, so they
    sit at the left edge of the log axis.
    """
    if metric not in results:
        return None
    # One row per solve, as in the profile: duplicated attempts (merged
    # shards, resumes) would otherwise each add a step past 1.0.
    deduped = one_row_per_solve(results, metric=metric)
    successful = deduped[solved_mask(deduped)].copy()
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

    compared, compared_expected = drop_solvers_without_metric(results, metric, expected=expected)
    problem_count = _unique_problem_count(complete_results(compared, expected=compared_expected))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    has_positive = False
    # On a log axis a literal zero would map to -inf. Values below the
    # profile's resolution count as that resolution (one iteration, 0.01 s);
    # zeros of metrics without one are drawn at a tiny positive value, at
    # the leftmost edge of the axis. Either way each still adds a step.
    floor = profile_floor(metric)
    styles = styles if styles is not None else solver_styles(successful["solver_id"].unique())
    for solver_id, group in successful.groupby("solver_id", observed=True):
        raw = group[metric].to_numpy(dtype=float)
        if raw.size == 0:
            continue
        if floor > 0.0:
            values = np.maximum(raw, floor)
        else:
            values = np.where(raw > 0.0, raw, 1.0e-12)
        has_positive = has_positive or floor > 0.0 or bool(np.any(raw > 0.0))
        values = np.sort(values)
        fractions = np.arange(1, len(values) + 1) / problem_count
        ax.step(values, fractions, where="post", label=solver_id, linewidth=2, **_line_style(styles, solver_id))
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
    # Floored like the profile and the speedup table (see successful_metric_values).
    successful = successful_metric_values(results, metric, set(status_codes.SOLUTION_PRESENT))

    keys = ["dataset", "problem"] if "dataset" in successful.columns else ["problem"]
    index = keys[0] if len(keys) == 1 else keys
    pivot = successful.pivot_table(
        index=index,
        columns="solver_id",
        values=metric,
        aggfunc="first",
    )

    # Compute the *actual* number of comparable pairs (those with at least
    # one common non-NaN entry) before deciding to skip. The naive ``C(N, 2)``
    # upper bound would over-skip in sparse-coverage configs (e.g. partial
    # sweeps, multi-dataset runs where only a subset of solvers cover each
    # dataset) — there can be many successful solvers but only a handful
    # of comparable pairs. The dominant cost we want to avoid past the cap
    # is the per-pair ``pivot[[a,b]].dropna()`` enumeration (~16 s on the
    # 144-solver case), not the pivot itself; counting pairs from a single
    # boolean matrix product on the finite-mask matrix is O(n_solvers²)
    # and trivially fast.
    sorted_solvers = sorted(pivot.columns)
    if len(sorted_solvers) < 2:
        return None
    pivot = pivot[sorted_solvers]
    finite = pivot.notna().to_numpy()
    overlap = finite.astype(np.int32).T @ finite.astype(np.int32)
    upper_i, upper_j = np.triu_indices(overlap.shape[0], k=1)
    pair_has_common = overlap[upper_i, upper_j] > 0
    n_pairs = int(np.sum(pair_has_common))
    if n_pairs == 0:
        return None
    if n_pairs > _PAIRWISE_SCATTER_MAX_PAIRS:
        # Use warning level so users running `bench report` actually see
        # the message under default Python logging (the package does not
        # call logging.basicConfig, so info-level is silent).
        logger.warning(
            "Skipping pairwise scatter for %s: %d pairs exceeds cap of %d "
            "(see pairwise_speedups_%s.csv for the underlying data).",
            metric,
            n_pairs,
            _PAIRWISE_SCATTER_MAX_PAIRS,
            metric,
        )
        return None
    pairs = [
        (sorted_solvers[upper_i[k]], sorted_solvers[upper_j[k]])
        for k in range(len(upper_i))
        if pair_has_common[k]
    ]

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

    for ax, (solver_a, solver_b) in zip(axes.ravel(), pairs, strict=False):
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
    # unrecognized cell, which the colormap renders in its bad color.
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
    cmap = plt.get_cmap("viridis").with_extremes(bad="#e5e7eb")
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


# Exact-zero residuals (an easy problem solved to roundoff) are drawn at this
# floor on the log axes rather than dropped, the same floor in every KKT plot.
_KKT_ZERO_FLOOR = 1.0e-16

_KKT_RESIDUAL_FIELDS = (
    ("kkt.primal_res_rel", "primal_res_rel"),
    ("kkt.dual_res_rel", "dual_res_rel"),
    ("kkt.comp_slack", "comp_slack"),
    ("kkt.duality_gap_rel", "duality_gap_rel"),
)


def _kkt_fields(results: pd.DataFrame) -> list[tuple[str, str]]:
    """Residual columns the KKT plots draw, in order.

    Complementarity is drawn scaled (``comp_slack_rel``, on the same axis
    as the other relative residuals) when the records carry it, and as the
    absolute ``comp_slack`` for records written before it existed.
    """
    fields = list(_KKT_RESIDUAL_FIELDS)
    if "kkt.comp_slack_rel" in results:
        fields[2] = ("kkt.comp_slack_rel", "comp_slack_rel")
    return [(col, label) for col, label in fields if col in results]


def _kkt_plot_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Successful optimality claims, one per solve (certificates carry no
    residuals of these kinds)."""
    return optimality_successes(results, set(status_codes.SOLUTION_PRESENT))


def _floored_residuals(values: pd.Series) -> np.ndarray:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values) & (values >= 0.0)]
    return np.maximum(values, _KKT_ZERO_FLOOR)


def _write_kkt_residual_boxplot(results, output_dir: Path) -> Path | None:
    available = _kkt_fields(results)
    if not available:
        return None

    successful = _kkt_plot_rows(results)
    if successful.empty:
        return None

    groups = {str(solver_id): group for solver_id, group in successful.groupby("solver_id", observed=True)}
    solver_ids = sorted(groups)
    if not solver_ids:
        return None

    cols = len(available)
    fig, axes = plt.subplots(
        1, cols, figsize=(4.5 * cols, 4.5), constrained_layout=True, squeeze=False
    )
    drew_any = False
    for ax, (col, label) in zip(axes.ravel(), available, strict=True):
        data = []
        labels = []
        for solver_id in solver_ids:
            values = _floored_residuals(groups[solver_id][col])
            if values.size == 0:
                continue
            data.append(np.log10(values))
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
        ax.set_ylabel("log10(residual), zeros at -16")
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


def _write_kkt_residual_heatmap(results, output_dir: Path) -> Path | None:
    available = _kkt_fields(results)
    if not available:
        return None

    successful = _kkt_plot_rows(results).copy()
    if successful.empty:
        return None

    pivots: list[tuple[str, pd.DataFrame]] = []
    keys = ["dataset", "problem"] if "dataset" in successful.columns else ["problem"]
    pivot_index = keys[0] if len(keys) == 1 else keys
    for col, label in available:
        successful[col] = pd.to_numeric(successful[col], errors="coerce")
        frame = successful[np.isfinite(successful[col]) & (successful[col] >= 0.0)]
        if frame.empty:
            continue
        frame = frame.assign(**{col: frame[col].clip(lower=_KKT_ZERO_FLOOR)})
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
    cmap = plt.get_cmap("viridis").with_extremes(bad="#e5e7eb")
    for ax, (label, pivot) in zip(axes.ravel(), pivots, strict=True):
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
        cbar.set_label("log10(residual), zeros at -16")

    fig.suptitle("KKT Residual Heatmap (successful solves)")
    path = output_dir / "kkt_residual_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_kkt_accuracy_profile(
    results,
    output_dir: Path,
    *,
    styles: dict[str, dict[str, str]] | None = None,
) -> Path | None:
    available = _kkt_fields(results)
    if not available:
        return None

    successful = _kkt_plot_rows(results)
    if successful.empty:
        return None

    groups = {str(solver_id): group for solver_id, group in successful.groupby("solver_id", observed=True)}
    solver_ids = sorted(groups)
    if not solver_ids:
        return None

    # Denominator is the number of solves of the solver that produced
    # the residual being plotted (i.e. successful solves with a
    # finite KKT value). Previously we divided by total attempts which
    # conflated "solver failed" with "solver produced a poor
    # residual" — a solver that fails on 95% of problems but is
    # accurate on the 5% it solves would otherwise look like a 5%-rate
    # failure. The y-axis label now reflects the per-success
    # normalization explicitly.
    thresholds = np.logspace(-12, 0, 121)
    styles = styles if styles is not None else solver_styles(solver_ids)

    cols = len(available)
    fig, axes = plt.subplots(
        1, cols, figsize=(4.5 * cols, 4.5), constrained_layout=True, squeeze=False
    )
    drew_any = False
    for ax, (col, label) in zip(axes.ravel(), available, strict=True):
        per_solver = []
        for solver_id in solver_ids:
            series = _floored_residuals(groups[solver_id][col])
            if series.size == 0:
                continue
            sorted_series = np.sort(series)
            fractions = np.searchsorted(sorted_series, thresholds, side="right") / float(series.size)
            per_solver.append((solver_id, fractions))
        if not per_solver:
            ax.set_axis_off()
            ax.set_title(f"{label}\n(no data)")
            continue
        drew_any = True
        for solver_id, fractions in per_solver:
            ax.plot(thresholds, fractions, label=solver_id, linewidth=2, **_line_style(styles, solver_id))
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("residual threshold tau")
        ax.set_ylabel("fraction of successes with residual <= tau")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize="small")

    if not drew_any:
        plt.close(fig)
        return None

    fig.suptitle(
        "KKT Accuracy Profile (per solver, normalized by per-solver successes)"
    )
    path = output_dir / "kkt_accuracy_profile.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _write_difficulty_scaling(
    results,
    output_dir: Path,
    metric: str,
    *,
    styles: dict[str, dict[str, str]] | None = None,
) -> Path | None:
    table = difficulty_scaling(results, metric=metric)
    if table.empty:
        return None
    median_column = f"median_{metric}"
    # The row of problems of unknown size has no place on the size axis.
    plottable = table.dropna(subset=[median_column, "size_bin"])
    if plottable.empty:
        return None

    styles = styles if styles is not None else solver_styles(plottable["solver_id"].unique())
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    drew_any = False
    for solver_id, group in plottable.groupby("solver_id", observed=True):
        ordered = group.sort_values("size_bin")
        midpoints = 0.5 * (ordered["size_min"] + ordered["size_max"])
        ax.plot(
            midpoints,
            ordered[median_column],
            linewidth=2,
            label=str(solver_id),
            **_line_style(styles, solver_id, marker=True),
        )
        drew_any = True
    if not drew_any:
        plt.close(fig)
        return None
    ax.set_xscale("log")
    if (plottable[median_column] > 0).any():
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
