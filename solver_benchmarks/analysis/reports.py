"""Higher-level benchmark report tables."""

from __future__ import annotations

import json
import re
from pathlib import Path
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from solver_benchmarks.core import status
from solver_benchmarks.datasets import get_dataset


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
    for solver_id, group in results.groupby("solver_id"):
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
    for solver_id, group in successful.groupby("solver_id"):
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
    for solver_id, group in infeasible.groupby("solver_id"):
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
    for solver_id, group in results.groupby("solver_id"):
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
    expected_problems = _expected_problem_names(config, repo_root=repo_root)
    solvers = [solver["id"] for solver in config.get("solvers", [])]

    rows = []
    expected_set = set(expected_problems)
    for solver_id in solvers:
        solver_results = results[results["solver_id"] == solver_id] if not results.empty else results
        completed_problems = (
            set(solver_results["problem"]) if not solver_results.empty else set()
        )
        duplicate_rows = (
            int(solver_results.duplicated(["problem", "solver_id"]).sum())
            if not solver_results.empty
            else 0
        )
        missing = len(expected_set - completed_problems)
        unexpected = len(completed_problems - expected_set)
        rows.append(
            {
                "solver_id": solver_id,
                "expected": len(expected_problems),
                "completed": len(completed_problems),
                "missing": int(missing),
                "unexpected": int(unexpected),
                "duplicate_rows": duplicate_rows,
                "complete": missing == 0 and unexpected == 0 and duplicate_rows == 0,
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
    if not manifest_path.exists():
        return pd.DataFrame(columns=["solver_id", "problem"])

    config = json.loads(manifest_path.read_text())["config"]
    expected_problems = set(_expected_problem_names(config, repo_root=repo_root))
    rows = []
    for solver in config.get("solvers", []):
        solver_id = solver["id"]
        completed = (
            set(results[results["solver_id"] == solver_id]["problem"])
            if not results.empty
            else set()
        )
        for problem in sorted(expected_problems - completed):
            rows.append({"solver_id": solver_id, "problem": problem})
    return pd.DataFrame(rows, columns=["solver_id", "problem"])


def pairwise_speedups(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    success_statuses: set[str] | None = None,
    tie_rtol: float = 1.0e-3,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    columns = [
        "solver_a",
        "solver_b",
        "common_successes",
        "a_wins",
        "b_wins",
        "ties",
        "median_speedup_a_over_b",
        "geomean_speedup_a_over_b",
        "biggest_a_win_problem",
        "biggest_a_win_speedup",
        "biggest_b_win_problem",
        "biggest_b_win_speedup",
    ]
    if results.empty or metric not in results:
        return pd.DataFrame(columns=columns)

    successful = results[results["status"].isin(success_statuses)].copy()
    successful[metric] = pd.to_numeric(successful[metric], errors="coerce")
    successful = successful[np.isfinite(successful[metric]) & (successful[metric] > 0.0)]
    pivot = successful.pivot_table(
        index="problem",
        columns="solver_id",
        values=metric,
        aggfunc="first",
    )
    rows = []
    for solver_a, solver_b in combinations(sorted(pivot.columns), 2):
        common = pivot[[solver_a, solver_b]].dropna()
        if common.empty:
            rows.append(_empty_pairwise_row(solver_a, solver_b))
            continue
        speedup_a_over_b = common[solver_b] / common[solver_a]
        a_wins = speedup_a_over_b > 1.0 + tie_rtol
        b_wins = speedup_a_over_b < 1.0 / (1.0 + tie_rtol)
        ties = ~(a_wins | b_wins)
        biggest_a_problem = speedup_a_over_b.idxmax()
        biggest_b_problem = speedup_a_over_b.idxmin()
        rows.append(
            {
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
                "biggest_a_win_problem": biggest_a_problem,
                "biggest_a_win_speedup": float(speedup_a_over_b.loc[biggest_a_problem]),
                "biggest_b_win_problem": biggest_b_problem,
                "biggest_b_win_speedup": float(1.0 / speedup_a_over_b.loc[biggest_b_problem]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def objective_spreads(
    results: pd.DataFrame,
    *,
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    columns = [
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
    pivot = successful.pivot_table(
        index="problem",
        columns="solver_id",
        values="objective_value",
        aggfunc="first",
    )
    rows = []
    for problem, row in pivot.iterrows():
        values = row.dropna()
        if len(values) < 2:
            continue
        objective_min = float(values.min())
        objective_max = float(values.max())
        reference = max(1.0, abs(float(values.median())))
        absolute_spread = objective_max - objective_min
        rows.append(
            {
                "problem": problem,
                "solver_count": int(len(values)),
                "objective_min": objective_min,
                "objective_max": objective_max,
                "absolute_spread": absolute_spread,
                "relative_spread": absolute_spread / reference,
                "solver_min": str(values.idxmin()),
                "solver_max": str(values.idxmax()),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["relative_spread", "absolute_spread"], ascending=False
    )


def slowest_solves(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    limit: int = 25,
) -> pd.DataFrame:
    columns = [
        "problem",
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
    columns = [
        "problem",
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
    best_successes = successes.sort_values(success_metric).groupby("problem").first()
    rows = []
    for _, failure in failures.iterrows():
        problem = failure["problem"]
        if problem not in best_successes.index:
            continue
        best = best_successes.loc[problem]
        rows.append(
            {
                "problem": problem,
                "solver_id": failure["solver_id"],
                "status": failure["status"],
                "best_success_solver": best["solver_id"],
                f"best_success_{metric}": best.get(metric),
                "artifact_dir": failure.get("artifact_dir"),
                "error": failure.get("error"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def status_matrix(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    return results.pivot_table(
        index="problem",
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

    problems = sorted(results["problem"].dropna().unique())
    output = pd.DataFrame({"problem": problems})
    dimensions = problem_dimensions(results)
    if not dimensions.empty:
        output = output.merge(dimensions, on="problem", how="left")

    for solver_id in sorted(results["solver_id"].dropna().unique()):
        solver_rows = results[results["solver_id"] == solver_id]
        for field in fields:
            if field not in solver_rows:
                continue
            values = solver_rows.pivot_table(
                index="problem",
                values=field,
                aggfunc="first",
            )
            output[f"{solver_id}__{field}"] = output["problem"].map(values[field])
    return output


def solver_problem_tables(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if results.empty:
        return {}
    dimensions = problem_dimensions(results)
    tables = {}
    base_columns = [
        "problem",
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
    for solver_id, group in results.groupby("solver_id"):
        available = [column for column in base_columns if column in group]
        table = group[available].sort_values("problem").reset_index(drop=True)
        if not dimensions.empty:
            dimension_columns = dimensions.drop(
                columns=[column for column in dimensions.columns if column in table and column != "problem"],
                errors="ignore",
            )
            table = table.merge(dimension_columns, on="problem", how="left")
        tables[str(solver_id)] = table
    return tables


def problem_dimensions(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["problem"])
    dimension_columns = [
        column
        for column in [
            "dataset",
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
        return pd.DataFrame({"problem": sorted(results["problem"].dropna().unique())})
    dimensions = (
        results[["problem", *dimension_columns]]
        .drop_duplicates("problem")
        .sort_values("problem")
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
    pivot = successful.pivot_table(
        index="problem",
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


def _empty_pairwise_row(solver_a: str, solver_b: str) -> dict:
    return {
        "solver_a": solver_a,
        "solver_b": solver_b,
        "common_successes": 0,
        "a_wins": 0,
        "b_wins": 0,
        "ties": 0,
        "median_speedup_a_over_b": None,
        "geomean_speedup_a_over_b": None,
        "biggest_a_win_problem": None,
        "biggest_a_win_speedup": None,
        "biggest_b_win_problem": None,
        "biggest_b_win_speedup": None,
    }


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


def _expected_problem_names(config: dict, *, repo_root: str | Path | None = None) -> list[str]:
    dataset_cls = get_dataset(config["dataset"])
    dataset = dataset_cls(
        repo_root=repo_root,
        **config.get("dataset_options", {}),
    )
    problems = [problem.name for problem in dataset.list_problems()]
    include = set(config.get("include") or [])
    exclude = set(config.get("exclude") or [])
    if include:
        problems = [problem for problem in problems if problem in include]
    if exclude:
        problems = [problem for problem in problems if problem not in exclude]
    return problems
