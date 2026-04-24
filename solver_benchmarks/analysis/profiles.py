"""Performance profile calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solver_benchmarks.core import status


def performance_profile(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    success_statuses: set[str] | None = None,
    max_value: float = 1.0e6,
    n_tau: int = 1000,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    if results.empty:
        return pd.DataFrame()
    pivot = results.pivot_table(index="problem", columns="solver_id", values=metric, aggfunc="first")
    status_pivot = results.pivot_table(index="problem", columns="solver_id", values="status", aggfunc="first")
    values = pivot.copy()
    for solver_id in values.columns:
        failed = ~status_pivot[solver_id].isin(success_statuses)
        values.loc[failed, solver_id] = max_value
    best = values.min(axis=1)
    ratios = values.divide(best, axis=0)
    tau = np.logspace(0, 4, n_tau)
    profile = {"tau": tau}
    for solver_id in ratios.columns:
        profile[solver_id] = [(ratios[solver_id] <= t).mean() for t in tau]
    return pd.DataFrame(profile)


def shifted_geomean(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    shift: float = 10.0,
    success_statuses: set[str] | None = None,
    max_value: float = 1.0e6,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    rows = []
    for solver_id, group in results.groupby("solver_id"):
        values = group[metric].astype(float).to_numpy(copy=True)
        values[~group["status"].isin(success_statuses).to_numpy()] = max_value
        geomean = np.exp(np.mean(np.log(np.maximum(1.0, values + shift)))) - shift
        rows.append({"solver_id": solver_id, metric: geomean})
    return pd.DataFrame(rows).sort_values("solver_id")
