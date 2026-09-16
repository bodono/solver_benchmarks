"""Dolan-More performance profile and shifted geomean calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solver_benchmarks.core import status

DEFAULT_FAILURE_PENALTY = 1.0e3

# Per-metric defaults used by shifted_geomean
# when the caller hasn't pinned ``max_value`` and ``shift``. The
# original globals (1e3 / 10) made sense only for run_time_seconds;
# applying them to ``iterations`` or KKT residuals gave nonsense
# aggregates. The dispatch is opt-in: callers can still pass explicit
# values, and unknown metrics fall back to the run-time-style defaults.
_METRIC_DEFAULTS: dict[str, tuple[float, float]] = {
    "run_time_seconds": (1.0e3, 10.0),
    "setup_time_seconds": (1.0e3, 10.0),
    "solve_time_seconds": (1.0e3, 10.0),
    "iterations": (1.0e6, 100.0),
    "kkt.primal_res_rel": (1.0, 0.0),
    "kkt.dual_res_rel": (1.0, 0.0),
    "kkt.duality_gap_rel": (1.0, 0.0),
    "kkt.comp_slack": (1.0, 0.0),
}

# Finite profile ratios need a positive resolution for metrics that can
# legitimately report zero. Other metrics keep exact ratios unless the
# caller supplies a floor in that metric's units.
_PROFILE_FLOORS = {
    "run_time_seconds": 0.01,
    "setup_time_seconds": 0.01,
    "solve_time_seconds": 0.01,
    "iterations": 1.0,
}


def metric_defaults(metric: str) -> tuple[float, float]:
    """Return ``(failure_penalty, shift)`` defaults for ``metric``.

    Falls back to the run-time-style defaults for unknown metrics so
    aggregates remain useful for ad-hoc columns (e.g. wall-time
    derivatives).
    """
    return _METRIC_DEFAULTS.get(metric, (DEFAULT_FAILURE_PENALTY, 10.0))


def deduplicate_for_pivot(
    frame: pd.DataFrame,
    keys: list[str],
    metric: str | None = None,
    *,
    success_statuses: set[str] | None = None,
) -> pd.DataFrame:
    """Collapse duplicate ``(*keys, solver_id)`` rows to one per group.

    ``pivot_table(aggfunc="first")`` on a frame with duplicate
    ``(problem, solver_id)`` rows picks non-deterministically across
    pandas versions; sort first by status (successful rows beat failed
    rows) and then by the metric so we always keep the best
    *successful* row per ``(problem, solver_id)``. NaN metrics sort
    last so a successful numeric row wins over a NaN row.

    Pre-fix this sorted only by metric, which let a fast failure
    (``solver_error`` at 0.1s) win over a slow success (``optimal`` at
    1s) when the same solver had both a failed and a retried row, and
    ``performance_profile`` then penalized the kept row as failed.

    When ``metric`` is None or absent from the frame, the deduplication
    falls back to ``keep="first"`` ordering.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    subset_keys = [*keys, "solver_id"]
    if metric is not None and metric in frame.columns:
        sortable = frame.assign(
            __metric_for_dedup=pd.to_numeric(frame[metric], errors="coerce"),
        )
        # Status-failure flag sorts ascending: False (success) before
        # True (failure). When the status column is missing we treat
        # every row as successful so behavior matches the old code path.
        if "status" in sortable.columns:
            valid = np.isfinite(sortable["__metric_for_dedup"]) & sortable["__metric_for_dedup"].ge(0)
            sortable["__failure_for_dedup"] = ~(sortable["status"].isin(success_statuses) & valid)
            sort_columns = ["__failure_for_dedup", "__metric_for_dedup"]
        else:
            sort_columns = ["__metric_for_dedup"]
        sortable = sortable.sort_values(
            sort_columns, kind="stable", na_position="last"
        )
        deduped = sortable.drop_duplicates(subset=subset_keys, keep="first")
        drop_columns = [c for c in ("__metric_for_dedup", "__failure_for_dedup") if c in deduped.columns]
        return deduped.drop(columns=drop_columns)
    return frame.drop_duplicates(subset=subset_keys, keep="first")


def performance_profile(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    success_statuses: set[str] | None = None,
    max_value: float | None = None,
    n_tau: int = 1000,
    tau_max: float | None = None,
    expected: pd.DataFrame | None = None,
    min_value: float | None = None,
) -> pd.DataFrame:
    """Compute the Dolan-More performance profile.

    For each problem ``p`` and solver ``s``, this computes
    ``r[p, s] = metric[p, s] / min_successful metric[p, s]``. Failures,
    absent solves, and nonfinite/negative metrics have infinite ratios
    by default. An explicit finite ``max_value`` requests a penalty in
    metric units, raised to at least the largest successful value for each
    problem so a failure cannot outrank a success. The returned curve is
    ``rho_s(tau) = fraction of problems with r[p, s] <= tau``.

    Multi-dataset frames are pivoted on ``(dataset, problem)`` so that two
    datasets sharing a problem name (e.g. ``afiro`` in two LP bundles)
    contribute as two separate problems instead of being collapsed by
    ``aggfunc="first"``.

    Problems on which every solver failed remain in the denominator,
    with infinite ratios for every solver. Successful metrics are floored at
    ``min_value`` before taking ratios: by default 0.01 for time in seconds,
    1 for iterations, and 0 for other metrics. With a zero floor, zero metrics
    tie at ratio 1 when the best is zero; positive values then have infinite
    ratio. Supply a positive floor in the metric's units to avoid that case.
    The default ``tau_max`` covers the largest finite ratio.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    if max_value is None:
        max_value = float("inf")
    if np.isnan(max_value) or max_value < 0:
        raise ValueError("max_value must be nonnegative and not NaN")
    if min_value is None:
        min_value = _PROFILE_FLOORS.get(metric, 0.0)
    if not np.isfinite(min_value) or min_value < 0:
        raise ValueError("min_value must be finite and nonnegative")
    results = complete_results(results, expected=expected)
    if results.empty:
        return pd.DataFrame()
    keys = ["dataset", "problem"] if "dataset" in results.columns else ["problem"]
    index = keys[0] if len(keys) == 1 else keys
    # Pre-deduplicate so pivot_table(aggfunc="first") becomes deterministic
    # — best (lowest) metric wins for any duplicated (problem, solver_id),
    # but successful rows beat failed rows regardless of metric.
    deduped = deduplicate_for_pivot(
        results, keys, metric, success_statuses=success_statuses
    )
    deduped = deduped.assign(**{metric: pd.to_numeric(
        deduped.get(metric, pd.Series(np.nan, index=deduped.index)), errors="coerce"
    )})
    # Unlike pivot_table, pivot preserves rows and solver columns whose
    # metrics are all missing (e.g. an entire worker-error shard).
    values = deduped.pivot(index=index, columns="solver_id", values=metric)
    statuses = deduped.pivot(index=index, columns="solver_id", values="status")
    success_mask = statuses.isin(success_statuses) & np.isfinite(values) & values.ge(0)
    values = values.clip(lower=min_value)
    best = values.where(success_mask).min(axis=1)
    penalty = values.where(success_mask).max(axis=1).clip(lower=max_value)
    ratios = values.where(success_mask, penalty, axis=0).divide(best, axis=0)
    ratios.loc[best.isna()] = float("inf")
    zero_best = best.eq(0)
    ratios.loc[zero_best] = np.where(
        success_mask.loc[zero_best] & values.loc[zero_best].eq(0), 1.0, float("inf")
    )
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


def complete_results(
    results: pd.DataFrame, *, expected: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Represent unattempted solves as failures without changing recorded rows.

    ``expected`` contains ``problem``, ``solver_id``, and optionally ``dataset``
    columns. Its identities extend the observed sets to include entirely absent
    problems or solvers. The combined problem and solver sets are crossed so
    every retained solver is compared on the same population.
    """
    if expected is not None and expected.empty:
        expected = None
    if expected is None:
        if results.empty or "problem" not in results:
            return results.copy()
        keys = ["dataset", "problem"] if "dataset" in results else ["problem"]
        expected = results[keys].drop_duplicates().merge(
            results[["solver_id"]].drop_duplicates(), how="cross"
        )
    else:
        expected = expected.copy()
        if "dataset" in expected and not results.empty and "dataset" not in results:
            if expected["dataset"].nunique() > 1:
                raise ValueError("Dataset identities are required for a multi-dataset comparison")
            expected = expected.drop(columns="dataset")
        keys = ["dataset", "problem"] if "dataset" in expected else ["problem"]
        if "dataset" in results and "dataset" not in expected and not expected.empty:
            raise ValueError("Expected solves must include dataset identities")
    identities = [*keys, "solver_id"]
    universe = pd.concat([results.reindex(columns=identities), expected[identities]], ignore_index=True)
    expected = universe[keys].drop_duplicates().merge(
        universe[["solver_id"]].drop_duplicates(), how="cross"
    )
    if expected.empty:
        return results.copy()
    if results.empty:
        return expected.assign(status="not_attempted")
    missing = expected.merge(
        results[identities].drop_duplicates(), how="left", indicator=True
    ).query("_merge == 'left_only'").drop(columns="_merge")
    if missing.empty:
        return results.copy()
    return pd.concat([results, missing.assign(status="not_attempted")], ignore_index=True)


def shifted_geomean(
    results: pd.DataFrame,
    *,
    metric: str = "run_time_seconds",
    shift: float | None = None,
    success_statuses: set[str] | None = None,
    max_value: float | None = None,
    penalize_failures: bool = True,
    expected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-solver means over the expected comparison universe.

    Missing solves receive the failure penalty. Supply ``expected`` with
    problem/solver identities to include completely absent problems or solvers;
    otherwise the observed sets are crossed. Success-only means still omit
    failures, while their counts describe the same comparison universe.
    The penalty (default or explicit ``max_value``) is raised to the largest
    successful metric in this comparison, so each failure costs at least as
    much as every admitted success. The returned ``max_value`` records that effective
    penalty. Larger explicit penalties are retained.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    if max_value is None or shift is None:
        default_max, default_shift = metric_defaults(metric)
        if max_value is None:
            max_value = default_max
        if shift is None:
            shift = default_shift
    results = complete_results(results, expected=expected)
    if "problem" in results:
        keys = ["dataset", "problem"] if "dataset" in results else ["problem"]
        results = deduplicate_for_pivot(results, keys, metric, success_statuses=success_statuses)
    columns = ["solver_id", metric, "mode", "shift", "max_value", "success_count", "failure_count"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    numeric = pd.to_numeric(
        results.get(metric, pd.Series(np.nan, index=results.index)), errors="coerce"
    )
    valid_success = results["status"].isin(success_statuses) & np.isfinite(numeric) & numeric.ge(0)
    if penalize_failures:
        if np.isnan(max_value) or max_value < 0:
            raise ValueError("max_value must be nonnegative and not NaN")
        if valid_success.any():
            max_value = max(max_value, float(numeric[valid_success].max()))
    rows = []
    for solver_id, group in results.groupby("solver_id", observed=True):
        values = pd.to_numeric(
            group.get(metric, pd.Series(np.nan, index=group.index)), errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        successful = group["status"].isin(success_statuses).to_numpy() & np.isfinite(values) & (values >= 0)
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
    return pd.DataFrame(rows, columns=columns).sort_values("solver_id")


def _shifted_geomean(values: np.ndarray, shift: float) -> float:
    """Shifted geometric mean: ``exp(mean(log(v + shift))) - shift``.

    The textbook formula assumes ``v + shift > 0``. The previous
    implementation floored at ``1.0`` to avoid ``log(<=0)``, which was
    a no-op for run-time-style metrics (``shift = 10`` already keeps
    everything above 1) but clamped sub-unit KKT residuals to ``1.0``
    regardless of magnitude — making ``[1e-12, 1e-3]`` and ``[0.5,
    0.99]`` both report a geomean of ``1.0``. We now floor at the
    smallest positive float just to dodge ``log(0)``; for metrics with
    ``shift >= 1`` and non-negative values this is a no-op (the
    addition already keeps every element above the floor).
    """
    if values.size == 0:
        return np.nan
    floor = float(np.finfo(float).tiny)
    adjusted = np.maximum(floor, values + shift)
    return float(np.exp(np.mean(np.log(adjusted))) - shift)
