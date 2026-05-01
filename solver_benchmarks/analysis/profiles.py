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
    tau_max: float | None = None,
) -> pd.DataFrame:
    """Compute the Dolan-More performance profile.

    For each problem ``p`` and solver ``s``, this computes
    ``r[p, s] = metric[p, s] / min_s metric[p, s]`` after assigning
    ``max_value`` to unsuccessful solves. The returned curve is
    ``rho_s(tau) = fraction of problems with r[p, s] <= tau``.

    Multi-dataset frames are pivoted on ``(dataset, problem)`` so that two
    datasets sharing a problem name (e.g. ``afiro`` in two LP bundles)
    contribute as two separate problems instead of being collapsed by
    ``aggfunc="first"``.

    Problems on which every solver failed are dropped before computing
    the per-problem best, so an "all solvers failed" row does not
    contribute a spurious ratio of 1.0 to every solver's curve at
    ``tau=1``. The default ``tau_max`` is derived from
    ``ratios.max()`` so the right tail of the curve is not silently
    clipped.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    if results.empty:
        return pd.DataFrame()
    keys = ["dataset", "problem"] if "dataset" in results.columns else ["problem"]
    index = keys[0] if len(keys) == 1 else keys
    pivot = results.pivot_table(index=index, columns="solver_id", values=metric, aggfunc="first")
    status_pivot = results.pivot_table(
        index=index, columns="solver_id", values="status", aggfunc="first"
    )
    values = pivot.copy()
    success_mask = pd.DataFrame(False, index=values.index, columns=values.columns)
    for solver_id in values.columns:
        succeeded = status_pivot[solver_id].isin(success_statuses)
        success_mask[solver_id] = succeeded
        values.loc[~succeeded, solver_id] = max_value
    # Drop problems where no solver succeeded — Dolan-Moré is undefined
    # there, and including them would inflate every curve at tau=1.
    any_success = success_mask.any(axis=1)
    if not any_success.any():
        return pd.DataFrame()
    values = values.loc[any_success]
    best = values.min(axis=1)
    ratios = values.divide(best, axis=0)
    if tau_max is None:
        # Pick the smallest power of 10 that covers the largest finite
        # ratio; fall back to 1e4 if every ratio is degenerate.
        finite_ratios = ratios.to_numpy()[np.isfinite(ratios.to_numpy())]
        if finite_ratios.size:
            tau_max = max(10.0, float(10 ** np.ceil(np.log10(finite_ratios.max() + 1.0))))
        else:
            tau_max = 1.0e4
    tau = np.logspace(0, np.log10(tau_max), n_tau)
    profile: dict[str, np.ndarray] = {"tau": tau}
    # Vectorize the per-tau fraction computation: sort each column once
    # and use searchsorted instead of an O(n_tau * n_problems) scan.
    for solver_id in ratios.columns:
        col = ratios[solver_id].to_numpy(dtype=float)
        col = col[np.isfinite(col)]
        if col.size == 0:
            profile[solver_id] = np.zeros_like(tau)
            continue
        sorted_col = np.sort(col)
        counts = np.searchsorted(sorted_col, tau, side="right")
        profile[solver_id] = counts / float(len(ratios))
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
    for solver_id, group in results.groupby("solver_id", observed=True):
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
