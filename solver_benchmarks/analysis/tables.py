"""Higher-level benchmark report tables."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solver_benchmarks.analysis.profiles import deduplicate_for_pivot
from solver_benchmarks.core import status
from solver_benchmarks.core.config import manifest_dataset_entries
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.datasets.base import filter_problem_specs_by_size


def solver_metrics(
    results: pd.DataFrame,
    *,
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    columns = [
        "solver_id",
        "completed",
        "success_count",
        "failure_count",
        "success_rate",
        "failure_rate",
        "run_time_total_seconds",
        "run_time_mean_seconds",
        "run_time_median_seconds",
        "run_time_max_seconds",
        "iterations_total",
        "iterations_mean",
        "iterations_median",
        "iterations_max",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for solver_id, group in results.groupby("solver_id", observed=True):
        total = len(group)
        successful = group["status"].isin(success_statuses)
        success_count = int(successful.sum())
        run_times = _numeric_column(group, "run_time_seconds")
        iterations = _numeric_column(group, "iterations")
        rows.append(
            {
                "solver_id": solver_id,
                "completed": int(total),
                "success_count": success_count,
                "failure_count": int(total - success_count),
                "success_rate": _rate(success_count, total),
                "failure_rate": _rate(total - success_count, total),
                "run_time_total_seconds": _sum(run_times),
                "run_time_mean_seconds": _mean(run_times),
                "run_time_median_seconds": _median(run_times),
                "run_time_max_seconds": _max(run_times),
                "iterations_total": _sum(iterations),
                "iterations_mean": _mean(iterations),
                "iterations_median": _median(iterations),
                "iterations_max": _max(iterations),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("solver_id")


KKT_RESIDUAL_COLUMNS: tuple[str, ...] = (
    "kkt.primal_res",
    "kkt.primal_res_rel",
    "kkt.dual_res",
    "kkt.dual_res_rel",
    "kkt.comp_slack",
    "kkt.primal_cone_res",
    "kkt.dual_cone_res",
    "kkt.duality_gap",
    "kkt.duality_gap_rel",
)

KKT_CERTIFICATE_COLUMNS: tuple[str, ...] = (
    "kkt.Aty_rel",
    "kkt.Px_rel",
    "kkt.qtx",
    "kkt.bty",
    "kkt.support",
    "kkt.primal_cone_res",
    "kkt.dual_cone_res",
    "kkt.valid",
)


def kkt_summary(
    results: pd.DataFrame,
    *,
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    """Per-solver summary of KKT residuals on successful solves."""
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    residual_fields = [
        "primal_res_rel",
        "dual_res_rel",
        "comp_slack",
        "primal_cone_res",
        "dual_cone_res",
        "duality_gap_rel",
    ]
    base_columns = ["solver_id", "success_count", "kkt_count", "kkt_missing"]
    stat_columns = [
        f"{field}_{stat}"
        for field in residual_fields
        for stat in ("median", "p95", "max")
    ]
    columns = base_columns + stat_columns
    if results.empty:
        return pd.DataFrame(columns=columns)

    successful = results[results["status"].isin(success_statuses)]
    if successful.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for solver_id, group in successful.groupby("solver_id", observed=True):
        row: dict[str, Any] = {
            "solver_id": solver_id,
            "success_count": int(len(group)),
        }
        any_present = pd.Series(False, index=group.index)
        for field in residual_fields:
            col = f"kkt.{field}"
            values = _numeric_column(group, col)
            any_present = any_present | (
                pd.to_numeric(group.get(col, pd.Series(dtype=float)), errors="coerce").notna()
                if col in group
                else pd.Series(False, index=group.index)
            )
            row[f"{field}_median"] = _median(values)
            row[f"{field}_p95"] = _quantile(values, 0.95)
            row[f"{field}_max"] = _max(values)
        row["kkt_count"] = int(any_present.sum())
        row["kkt_missing"] = int(len(group) - row["kkt_count"])
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values("solver_id")


def claimed_optimal_kkt_thresholds(
    results: pd.DataFrame,
    *,
    success_statuses: set[str] | None = None,
    residual_fields: tuple[str, ...] = (
        "kkt.primal_res_rel",
        "kkt.dual_res_rel",
        "kkt.duality_gap_rel",
    ),
    thresholds: tuple[float, ...] = (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2),
) -> pd.DataFrame:
    """Bucket claimed-optimal solves by their worst relative KKT residual.

    For each solver, computes the per-row maximum of the listed relative
    residual columns on rows whose status indicates a solution is present,
    then counts how many of those fall at or below each threshold. The
    ``count_above_max`` column flags claims of optimality whose worst
    relative residual exceeds the loosest threshold — a quick way to spot
    solvers labelling dubious solutions as ``optimal``.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    threshold_columns = [f"count_le_{t:.0e}" for t in thresholds]
    columns = [
        "solver_id",
        "claimed_optimal",
        "with_residuals",
        "missing_residuals",
        "worst_max",
        "worst_p95",
        *threshold_columns,
        "count_above_max",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    available_fields = [field for field in residual_fields if field in results]
    if not available_fields:
        return pd.DataFrame(columns=columns)
    claimed = results[results["status"].isin(success_statuses)].copy()
    if claimed.empty:
        return pd.DataFrame(columns=columns)
    numeric = claimed[available_fields].apply(pd.to_numeric, errors="coerce")
    claimed["__worst_kkt__"] = numeric.max(axis=1, skipna=True)

    rows = []
    for solver_id, group in claimed.groupby("solver_id", observed=True):
        worst = group["__worst_kkt__"].dropna()
        claimed_optimal = int(len(group))
        with_residuals = int(worst.size)
        missing_residuals = claimed_optimal - with_residuals
        if worst.empty:
            row = {
                "solver_id": solver_id,
                "claimed_optimal": claimed_optimal,
                "with_residuals": with_residuals,
                "missing_residuals": missing_residuals,
                "worst_max": None,
                "worst_p95": None,
                "count_above_max": 0,
                **{column: 0 for column in threshold_columns},
            }
        else:
            row = {
                "solver_id": solver_id,
                "claimed_optimal": claimed_optimal,
                "with_residuals": with_residuals,
                "missing_residuals": missing_residuals,
                "worst_max": float(worst.max()),
                "worst_p95": float(worst.quantile(0.95)),
                "count_above_max": int((worst > thresholds[-1]).sum()),
                **{
                    column: int((worst <= threshold).sum())
                    for column, threshold in zip(threshold_columns, thresholds)
                },
            }
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values("solver_id")


def difficulty_scaling(
    results: pd.DataFrame,
    *,
    size_field: str = "metadata.n",
    metric: str = "run_time_seconds",
    success_statuses: set[str] | None = None,
    bin_count: int = 4,
) -> pd.DataFrame:
    """Median ``metric`` per solver across problems binned by size.

    Bins are equal-population quantile buckets of ``size_field`` over the
    set of problems with finite sizes. Each row reports, for one
    ``(solver, bin)`` pair, the bin's size range, the number of attempts,
    the number of successes, and the median / p95 of ``metric`` across
    successful solves.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    columns = [
        "solver_id",
        "size_bin",
        "size_min",
        "size_max",
        "problem_count",
        "success_count",
        "median_time",
        "p95_time",
    ]
    if results.empty or size_field not in results or metric not in results:
        return pd.DataFrame(columns=columns)

    frame = results.copy()
    frame[size_field] = pd.to_numeric(frame[size_field], errors="coerce")
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    sized = frame[np.isfinite(frame[size_field])]
    if sized.empty:
        return pd.DataFrame(columns=columns)
    unique_sizes = sized[size_field].nunique()
    if unique_sizes < 2:
        return pd.DataFrame(columns=columns)
    bins = min(int(bin_count), int(unique_sizes))
    bin_index = pd.qcut(
        sized[size_field].astype(float), q=bins, duplicates="drop", labels=False
    )
    sized = sized.assign(__bin__=bin_index).dropna(subset=["__bin__"])
    sized["__bin__"] = sized["__bin__"].astype(int)

    rows = []
    for (solver_id, bin_id), group in sized.groupby(["solver_id", "__bin__"], observed=True):
        sizes_in_bin = group[size_field].astype(float)
        successful = group[group["status"].isin(success_statuses)]
        success_metric = pd.to_numeric(successful[metric], errors="coerce").dropna()
        rows.append(
            {
                "solver_id": solver_id,
                "size_bin": int(bin_id),
                "size_min": float(sizes_in_bin.min()),
                "size_max": float(sizes_in_bin.max()),
                "problem_count": int(len(group)),
                "success_count": int(len(successful)),
                "median_time": float(success_metric.median())
                if not success_metric.empty
                else None,
                "p95_time": float(success_metric.quantile(0.95))
                if not success_metric.empty
                else None,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["solver_id", "size_bin"])


def setup_solve_breakdown(
    results: pd.DataFrame,
    *,
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    """Per-solver setup vs solve time on successful solves.

    Many adapters split ``run_time_seconds`` into ``setup_time_seconds`` (KKT
    factorization, scaling, internal preprocessing) and
    ``solve_time_seconds`` (the iterative loop). For QP-style direct-method
    solvers the setup phase often dominates and a fast ``solve`` time is
    misleading without that context. Solvers that do not report a split
    appear with ``with_breakdown = 0`` and null statistics.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    columns = [
        "solver_id",
        "with_breakdown",
        "setup_median",
        "solve_median",
        "total_median",
        "setup_share_median",
        "setup_total",
        "solve_total",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    if (
        "setup_time_seconds" not in results
        or "solve_time_seconds" not in results
    ):
        return pd.DataFrame(columns=columns)
    successful = results[results["status"].isin(success_statuses)].copy()
    if successful.empty:
        return pd.DataFrame(columns=columns)
    successful["setup_time_seconds"] = pd.to_numeric(
        successful["setup_time_seconds"], errors="coerce"
    )
    successful["solve_time_seconds"] = pd.to_numeric(
        successful["solve_time_seconds"], errors="coerce"
    )

    rows = []
    for solver_id, group in successful.groupby("solver_id", observed=True):
        both = group.dropna(subset=["setup_time_seconds", "solve_time_seconds"])
        if both.empty:
            rows.append(
                {
                    "solver_id": solver_id,
                    "with_breakdown": 0,
                    "setup_median": None,
                    "solve_median": None,
                    "total_median": None,
                    "setup_share_median": None,
                    "setup_total": None,
                    "solve_total": None,
                }
            )
            continue
        setup = both["setup_time_seconds"]
        solve = both["solve_time_seconds"]
        total = setup + solve
        share = setup.divide(total.where(total > 0))
        rows.append(
            {
                "solver_id": solver_id,
                "with_breakdown": int(len(both)),
                "setup_median": float(setup.median()),
                "solve_median": float(solve.median()),
                "total_median": float(total.median()),
                "setup_share_median": float(share.median())
                if share.notna().any()
                else None,
                "setup_total": float(setup.sum()),
                "solve_total": float(solve.sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("solver_id")


def kkt_certificate_summary(
    results: pd.DataFrame,
    *,
    infeasible_statuses: set[str] | None = None,
) -> pd.DataFrame:
    """Per-solver validity counts for infeasibility certificates."""
    if infeasible_statuses is None:
        infeasible_statuses = {
            status.PRIMAL_INFEASIBLE,
            status.PRIMAL_INFEASIBLE_INACCURATE,
            status.DUAL_INFEASIBLE,
            status.DUAL_INFEASIBLE_INACCURATE,
        }
    columns = [
        "solver_id",
        "infeasible_count",
        "cert_count",
        "cert_valid",
        "cert_invalid",
        "Aty_rel_max",
        "Px_rel_max",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)
    infeasible = results[results["status"].isin(infeasible_statuses)]
    if infeasible.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for solver_id, group in infeasible.groupby("solver_id", observed=True):
        valid_series = group.get("kkt.valid")
        cert_rows = group if valid_series is None else group[valid_series.notna()]
        valid_count = int(valid_series.fillna(False).astype(bool).sum()) if valid_series is not None else 0
        aty = _numeric_column(group, "kkt.Aty_rel")
        px = _numeric_column(group, "kkt.Px_rel")
        rows.append(
            {
                "solver_id": solver_id,
                "infeasible_count": int(len(group)),
                "cert_count": int(len(cert_rows)),
                "cert_valid": valid_count,
                "cert_invalid": int(len(cert_rows) - valid_count),
                "Aty_rel_max": _max(aty),
                "Px_rel_max": _max(px),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("solver_id")


def failure_rates(
    results: pd.DataFrame,
    *,
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    columns = [
        "solver_id",
        "total",
        "success_count",
        "failure_count",
        "success_rate",
        "failure_rate",
        "statuses",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for solver_id, group in results.groupby("solver_id", observed=True):
        total = len(group)
        successful = group["status"].isin(success_statuses)
        success_count = int(successful.sum())
        status_counts = group["status"].value_counts(dropna=False).sort_index()
        rows.append(
            {
                "solver_id": solver_id,
                "total": int(total),
                "success_count": success_count,
                "failure_count": int(total - success_count),
                "success_rate": success_count / total if total else 0.0,
                "failure_rate": (total - success_count) / total if total else 0.0,
                "statuses": ", ".join(
                    f"{status_name}={count}"
                    for status_name, count in status_counts.items()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("solver_id")


def completion_summary(
    run_dir: str | Path,
    results: pd.DataFrame | None = None,
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    if results is None:
        from solver_benchmarks.analysis.load import load_results

        results = load_results(run_dir)
    manifest_path = run_dir / "manifest.json"
    columns = [
        "solver_id",
        "dataset",
        "expected",
        "completed",
        "missing",
        "unexpected",
        "duplicate_rows",
        "complete",
    ]
    if not manifest_path.exists():
        return pd.DataFrame(columns=columns)

    config = json.loads(manifest_path.read_text())["config"]
    expected_by_dataset = _expected_by_dataset(config, repo_root=repo_root)
    solvers = [solver["id"] for solver in config.get("solvers", [])]

    rows = []
    has_dataset_col = not results.empty and "dataset" in results.columns
    for solver_id in solvers:
        solver_rows = (
            results[results["solver_id"] == solver_id]
            if not results.empty
            else results
        )
        for dataset_name, expected_set in expected_by_dataset.items():
            if has_dataset_col:
                dataset_rows = solver_rows[solver_rows["dataset"] == dataset_name]
            else:
                # Legacy single-dataset run dirs without a populated dataset
                # column: every row belongs to the one configured dataset.
                dataset_rows = solver_rows
            completed_problems = (
                set(dataset_rows["problem"]) if not dataset_rows.empty else set()
            )
            duplicate_rows = (
                int(dataset_rows.duplicated(["problem", "solver_id"]).sum())
                if not dataset_rows.empty
                else 0
            )
            missing = len(expected_set - completed_problems)
            unexpected = len(completed_problems - expected_set)
            rows.append(
                {
                    "solver_id": solver_id,
                    "dataset": dataset_name,
                    "expected": len(expected_set),
                    "completed": len(completed_problems),
                    "missing": int(missing),
                    "unexpected": int(unexpected),
                    "duplicate_rows": duplicate_rows,
                    "complete": missing == 0
                    and unexpected == 0
                    and duplicate_rows == 0,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def missing_results(
    run_dir: str | Path,
    results: pd.DataFrame | None = None,
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    run_dir = Path(run_dir)
    if results is None:
        from solver_benchmarks.analysis.load import load_results

        results = load_results(run_dir)
    manifest_path = run_dir / "manifest.json"
    columns = ["solver_id", "dataset", "problem"]
    if not manifest_path.exists():
        return pd.DataFrame(columns=columns)

    config = json.loads(manifest_path.read_text())["config"]
    expected_by_dataset = _expected_by_dataset(config, repo_root=repo_root)
    has_dataset_col = not results.empty and "dataset" in results.columns
    rows = []
    for solver in config.get("solvers", []):
        solver_id = solver["id"]
        solver_rows = (
            results[results["solver_id"] == solver_id]
            if not results.empty
            else results
        )
        for dataset_name, expected_set in expected_by_dataset.items():
            if has_dataset_col:
                dataset_rows = solver_rows[solver_rows["dataset"] == dataset_name]
            else:
                dataset_rows = solver_rows
            completed = (
                set(dataset_rows["problem"]) if not dataset_rows.empty else set()
            )
            for problem in sorted(expected_set - completed):
                rows.append(
                    {
                        "solver_id": solver_id,
                        "dataset": dataset_name,
                        "problem": problem,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def pairwise_speedups(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    success_statuses: set[str] | None = None,
    tie_rtol: float = 1.0e-3,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    has_dataset = not results.empty and "dataset" in results.columns
    base_columns = [
        "solver_a",
        "solver_b",
        "common_successes",
        "a_wins",
        "b_wins",
        "ties",
        "median_speedup_a_over_b",
        "geomean_speedup_a_over_b",
    ]
    if has_dataset:
        winner_columns = [
            "biggest_a_win_dataset",
            "biggest_a_win_problem",
            "biggest_a_win_speedup",
            "biggest_b_win_dataset",
            "biggest_b_win_problem",
            "biggest_b_win_speedup",
        ]
    else:
        winner_columns = [
            "biggest_a_win_problem",
            "biggest_a_win_speedup",
            "biggest_b_win_problem",
            "biggest_b_win_speedup",
        ]
    columns = base_columns + winner_columns
    if results.empty or metric not in results:
        return pd.DataFrame(columns=columns)

    successful = results[results["status"].isin(success_statuses)].copy()
    successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
    successful = successful[np.isfinite(successful[metric]) & (successful[metric] > 0.0)]
    keys = _problem_keys(successful)
    index = keys[0] if len(keys) == 1 else keys
    successful = deduplicate_for_pivot(successful, keys, metric)
    pivot = successful.pivot_table(
        index=index,
        columns="solver_id",
        values=metric,
        aggfunc="first",
    )
    rows = []
    for solver_a, solver_b in combinations(sorted(pivot.columns), 2):
        common = pivot[[solver_a, solver_b]].dropna()
        if common.empty:
            rows.append(_empty_pairwise_row(solver_a, solver_b, has_dataset=has_dataset))
            continue
        speedup_a_over_b = common[solver_b] / common[solver_a]
        a_wins = speedup_a_over_b > 1.0 + tie_rtol
        b_wins = speedup_a_over_b < 1.0 / (1.0 + tie_rtol)
        ties = ~(a_wins | b_wins)
        biggest_a_key = speedup_a_over_b.idxmax()
        biggest_b_key = speedup_a_over_b.idxmin()
        row = {
            "solver_a": solver_a,
            "solver_b": solver_b,
            "common_successes": int(len(common)),
            "a_wins": int(a_wins.sum()),
            "b_wins": int(b_wins.sum()),
            "ties": int(ties.sum()),
            "median_speedup_a_over_b": float(speedup_a_over_b.median()),
            "geomean_speedup_a_over_b": float(
                np.exp(np.mean(np.log(speedup_a_over_b)))
            ),
        }
        if has_dataset:
            row["biggest_a_win_dataset"] = biggest_a_key[0]
            row["biggest_a_win_problem"] = biggest_a_key[1]
            row["biggest_b_win_dataset"] = biggest_b_key[0]
            row["biggest_b_win_problem"] = biggest_b_key[1]
        else:
            row["biggest_a_win_problem"] = biggest_a_key
            row["biggest_b_win_problem"] = biggest_b_key
        row["biggest_a_win_speedup"] = float(speedup_a_over_b.loc[biggest_a_key])
        row["biggest_b_win_speedup"] = float(1.0 / speedup_a_over_b.loc[biggest_b_key])
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def objective_spreads(
    results: pd.DataFrame,
    *,
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    has_dataset = not results.empty and "dataset" in results.columns
    columns = [
        *(("dataset",) if has_dataset else ()),
        "problem",
        "solver_count",
        "objective_min",
        "objective_max",
        "absolute_spread",
        "relative_spread",
        "solver_min",
        "solver_max",
    ]
    if results.empty or "objective_value" not in results:
        return pd.DataFrame(columns=columns)

    successful = results[results["status"].isin(success_statuses)].copy()
    successful["objective_value"] = pd.to_numeric(
        successful["objective_value"], errors="coerce"
    )
    successful = successful[np.isfinite(successful["objective_value"])]
    keys = _problem_keys(successful)
    index = keys[0] if len(keys) == 1 else keys
    successful = deduplicate_for_pivot(successful, keys, "objective_value")
    pivot = successful.pivot_table(
        index=index,
        columns="solver_id",
        values="objective_value",
        aggfunc="first",
    )
    rows = []
    for key, row in pivot.iterrows():
        values = row.dropna()
        if len(values) < 2:
            continue
        objective_min = float(values.min())
        objective_max = float(values.max())
        reference = max(1.0, abs(float(values.median())))
        absolute_spread = objective_max - objective_min
        record: dict[str, Any] = {}
        if has_dataset:
            record["dataset"] = key[0]
            record["problem"] = key[1]
        else:
            record["problem"] = key
        record.update(
            {
                "solver_count": int(len(values)),
                "objective_min": objective_min,
                "objective_max": objective_max,
                "absolute_spread": absolute_spread,
                "relative_spread": absolute_spread / reference,
                "solver_min": str(values.idxmin()),
                "solver_max": str(values.idxmax()),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["relative_spread", "absolute_spread"], ascending=False
    )


def slowest_solves(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    limit: int = 25,
) -> pd.DataFrame:
    keys = _problem_keys(results) if not results.empty else ["problem"]
    columns = [
        *keys,
        "solver_id",
        "status",
        metric,
        "iterations",
        "objective_value",
        "artifact_dir",
        "error",
    ]
    if results.empty or metric not in results:
        return pd.DataFrame(columns=columns)
    frame = results.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame = frame[np.isfinite(frame[metric])]
    available = [column for column in columns if column in frame]
    return frame.sort_values(metric, ascending=False)[available].head(limit)


def failures_with_successful_alternatives(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    keys = _problem_keys(results) if not results.empty else ["problem"]
    columns = [
        *keys,
        "solver_id",
        "status",
        "best_success_solver",
        f"best_success_{metric}",
        "artifact_dir",
        "error",
    ]
    if results.empty:
        return pd.DataFrame(columns=columns)

    frame = results.copy()
    if metric in frame:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    successes = frame[frame["status"].isin(success_statuses)].copy()
    failures = frame[~frame["status"].isin(success_statuses)].copy()
    if successes.empty or failures.empty:
        return pd.DataFrame(columns=columns)

    success_metric = metric if metric in successes else "solver_id"
    best_successes = successes.sort_values(success_metric).groupby(keys, observed=True).first()
    rows = []
    for _, failure in failures.iterrows():
        key = tuple(failure[col] for col in keys) if len(keys) > 1 else failure[keys[0]]
        if key not in best_successes.index:
            continue
        best = best_successes.loc[key]
        row = {col: failure[col] for col in keys}
        row.update(
            {
                "solver_id": failure["solver_id"],
                "status": failure["status"],
                "best_success_solver": best["solver_id"],
                f"best_success_{metric}": best.get(metric),
                "artifact_dir": failure.get("artifact_dir"),
                "error": failure.get("error"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _problem_keys(results: pd.DataFrame) -> list[str]:
    """Return the row-identity columns for problem-keyed tables.

    Multi-dataset runs can have the same ``problem`` name in different
    datasets (e.g. ``afiro`` in two LP bundles), so any pivot/merge that
    keys on ``problem`` alone silently collapses them and can attach the
    wrong dimensions or status. Use ``(dataset, problem)`` whenever the
    frame carries a ``dataset`` column; fall back to ``("problem",)`` for
    legacy frames or empty inputs.
    """
    if "dataset" in results.columns:
        return ["dataset", "problem"]
    return ["problem"]


def status_matrix(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    keys = _problem_keys(results)
    index = keys[0] if len(keys) == 1 else keys
    # Deterministic dedup before pivoting; if multiple rows exist for a
    # (problem, solver_id) we keep the alphabetically-first status as a
    # stable choice (since "status" itself is the value column there's
    # no metric to sort on).
    deduped = deduplicate_for_pivot(results, keys)
    return deduped.pivot_table(
        index=index,
        columns="solver_id",
        values="status",
        aggfunc="first",
        fill_value="missing",
    ).sort_index()


def problem_solver_comparison(
    results: pd.DataFrame,
    *,
    fields: tuple[str, ...] = (
        "status",
        "run_time_seconds",
        "iterations",
        "objective_value",
    ),
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    keys = _problem_keys(results)
    output = (
        results[keys]
        .dropna(subset=["problem"])
        .drop_duplicates()
        .sort_values(keys)
        .reset_index(drop=True)
    )
    dimensions = problem_dimensions(results)
    if not dimensions.empty:
        output = output.merge(dimensions, on=keys, how="left")

    for solver_id in sorted(results["solver_id"].dropna().unique()):
        solver_rows = results[results["solver_id"] == solver_id]
        for field in fields:
            if field not in solver_rows:
                continue
            lookup = (
                solver_rows.dropna(subset=["problem"])
                .drop_duplicates(subset=keys)
                .set_index(keys)[field]
            )
            output_keys = (
                output[keys[0]]
                if len(keys) == 1
                else list(zip(*[output[key] for key in keys]))
            )
            output[f"{solver_id}__{field}"] = [
                lookup.get(key) for key in output_keys
            ]
    return output


def solver_problem_tables(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if results.empty:
        return {}
    dimensions = problem_dimensions(results)
    keys = _problem_keys(results)
    tables = {}
    base_columns = [
        *keys,
        "problem_kind",
        "status",
        "run_time_seconds",
        "iterations",
        "objective_value",
        "kkt.primal_res_rel",
        "kkt.dual_res_rel",
        "kkt.comp_slack",
        "kkt.duality_gap_rel",
        "artifact_dir",
        "error",
    ]
    for solver_id, group in results.groupby("solver_id", observed=True):
        available = [column for column in base_columns if column in group]
        table = group[available].sort_values(keys).reset_index(drop=True)
        if not dimensions.empty:
            dimension_columns = dimensions.drop(
                columns=[
                    column
                    for column in dimensions.columns
                    if column in table and column not in keys
                ],
                errors="ignore",
            )
            table = table.merge(dimension_columns, on=keys, how="left")
        tables[str(solver_id)] = table
    return tables


def problem_dimensions(results: pd.DataFrame) -> pd.DataFrame:
    keys = _problem_keys(results)
    if results.empty:
        return pd.DataFrame(columns=keys)
    dimension_columns = [
        column
        for column in [
            "problem_kind",
            "metadata.n",
            "metadata.m",
            "metadata.nnz_p",
            "metadata.nnz_a",
            "metadata.rows",
            "metadata.cols",
        ]
        if column in results
    ]
    if not dimension_columns:
        unique = (
            results[keys]
            .dropna(subset=["problem"])
            .drop_duplicates()
            .sort_values(keys)
            .reset_index(drop=True)
        )
        return unique
    dimensions = (
        results[[*keys, *dimension_columns]]
        .drop_duplicates(subset=keys)
        .sort_values(keys)
        .reset_index(drop=True)
    )
    return dimensions.rename(
        columns={
            "metadata.n": "n",
            "metadata.m": "m",
            "metadata.nnz_p": "nnz_p",
            "metadata.nnz_a": "nnz_a",
            "metadata.rows": "rows",
            "metadata.cols": "cols",
        }
    )


def performance_ratio_matrix(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    if results.empty or metric not in results:
        return pd.DataFrame()
    successful = results[results["status"].isin(success_statuses)].copy()
    successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
    successful = successful[np.isfinite(successful[metric]) & (successful[metric] > 0.0)]
    keys = _problem_keys(successful)
    index = keys[0] if len(keys) == 1 else keys
    successful = deduplicate_for_pivot(successful, keys, metric)
    pivot = successful.pivot_table(
        index=index,
        columns="solver_id",
        values=metric,
        aggfunc="first",
    )
    if pivot.empty:
        return pivot
    best = pivot.min(axis=1)
    return pivot.divide(best, axis=0).sort_index()


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "value"


def _empty_pairwise_row(solver_a: str, solver_b: str, *, has_dataset: bool = False) -> dict:
    row = {
        "solver_a": solver_a,
        "solver_b": solver_b,
        "common_successes": 0,
        "a_wins": 0,
        "b_wins": 0,
        "ties": 0,
        "median_speedup_a_over_b": None,
        "geomean_speedup_a_over_b": None,
    }
    if has_dataset:
        row["biggest_a_win_dataset"] = None
        row["biggest_b_win_dataset"] = None
    row["biggest_a_win_problem"] = None
    row["biggest_a_win_speedup"] = None
    row["biggest_b_win_problem"] = None
    row["biggest_b_win_speedup"] = None
    return row


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sum(values: pd.Series):
    return None if values.empty else float(values.sum())


def _mean(values: pd.Series):
    return None if values.empty else float(values.mean())


def _median(values: pd.Series):
    return None if values.empty else float(values.median())


def _max(values: pd.Series):
    return None if values.empty else float(values.max())


def _quantile(values: pd.Series, q: float):
    return None if values.empty else float(values.quantile(q))


def _expected_by_dataset(
    config: dict, *, repo_root: str | Path | None = None
) -> dict[str, set[str]]:
    """Return ``{dataset_id: set_of_expected_problem_names}`` for a manifest.

    Handles both the new ``datasets:`` list shape and the legacy
    ``dataset: name`` + ``dataset_options`` shape via
    ``manifest_dataset_entries``. Keys are dataset *ids* (the per-entry
    identity stamped into result rows), which equal the registry ``name``
    unless the config gave the entry an explicit ``id``.
    """
    expected: dict[str, set[str]] = {}
    for entry in manifest_dataset_entries(config):
        dataset_cls = get_dataset(entry["name"])
        dataset = dataset_cls(
            repo_root=repo_root,
            **entry.get("dataset_options", {}),
        )
        specs = dataset.list_problems()
        problems = [problem.name for problem in specs]
        include = set(entry.get("include") or [])
        exclude = set(entry.get("exclude") or [])
        if include:
            problems = [name for name in problems if name in include]
        if exclude:
            problems = [name for name in problems if name not in exclude]
        specs_by_name = {problem.name: problem for problem in specs}
        filtered_specs = filter_problem_specs_by_size(
            [specs_by_name[name] for name in problems],
            entry.get("dataset_options", {}).get("max_size_mb"),
        )
        expected[entry["id"]] = {problem.name for problem in filtered_specs}
    return expected


def _expected_problem_names(
    config: dict, *, repo_root: str | Path | None = None
) -> list[str]:
    """Backward-compatible: union of expected problems across all datasets."""
    expected: set[str] = set()
    for problems in _expected_by_dataset(config, repo_root=repo_root).values():
        expected |= problems
    return sorted(expected)
