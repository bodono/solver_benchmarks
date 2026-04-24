"""Dolan-More performance profile and shifted geomean calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solver_benchmarks.core import status


DEFAULT_FAILURE_PENALTY = 1.0e3


def performance_profile(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    success_statuses: set[str] | None = None,
    max_value: float = DEFAULT_FAILURE_PENALTY,
    n_tau: int = 1000,
) -> pd.DataFrame:
    """Compute the Dolan-More performance profile.

    For each problem ``p`` and solver ``s``, this computes
    ``r[p, s] = metric[p, s] / min_s metric[p, s]`` after assigning
    ``max_value`` to unsuccessful solves. The returned curve is
    ``rho_s(tau) = fraction of problems with r[p, s] <= tau``.
    """
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
    max_value: float = DEFAULT_FAILURE_PENALTY,
    penalize_failures: bool = True,
) -> pd.DataFrame:
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    rows = []
    for solver_id, group in results.groupby("solver_id"):
        values = pd.to_numeric(group[metric], errors="coerce").to_numpy(copy=True)
        successful = group["status"].isin(success_statuses).to_numpy()
        success_count = int(successful.sum())
        failure_count = int(len(group) - success_count)
        if penalize_failures:
            values[~successful] = max_value
            values = np.nan_to_num(values, nan=max_value, posinf=max_value, neginf=max_value)
            mode = "penalized"
        else:
            values = values[successful]
            values = values[np.isfinite(values)]
            mode = "success_only"
        geomean = _shifted_geomean(values, shift)
        rows.append(
            {
                "solver_id": solver_id,
                metric: geomean,
                "mode": mode,
                "shift": shift,
                "max_value": max_value if penalize_failures else None,
                "success_count": success_count,
                "failure_count": failure_count,
            }
        )
    return pd.DataFrame(rows).sort_values("solver_id")


def _shifted_geomean(values: np.ndarray, shift: float) -> float:
    if values.size == 0:
        return np.nan
    return float(np.exp(np.mean(np.log(np.maximum(1.0, values + shift)))) - shift)
