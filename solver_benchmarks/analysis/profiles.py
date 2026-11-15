"""Dolan-More performance profile and shifted geomean calculations."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd

from solver_benchmarks.analysis.outcomes import success_mask as solved_mask
from solver_benchmarks.core import status

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_PENALTY = 1.0e3
TIME_METRICS = frozenset({"run_time_seconds", "setup_time_seconds", "solve_time_seconds"})

# Shift of the KKT residual geomeans. Residuals are often exactly zero (an
# easy problem solved to roundoff); with no shift one zero sends the mean of
# logs towards log(tiny float) and the geomean to ~1e-150. A shift of 1e-12
# treats residuals below it as equally good, which is the resolution the
# KKT accuracy plots use too.
KKT_SHIFT = 1.0e-12

# Fixed defaults for non-time failure penalties and metric-specific shifts.
# Time geomeans require a declared limit or explicit penalty; historical time
# penalty values remain available through metric_defaults for compatibility.
# Unknown metrics retain the historical defaults until callers override them.
_METRIC_DEFAULTS: dict[str, tuple[float, float]] = {
    "run_time_seconds": (1.0e3, 10.0),
    "setup_time_seconds": (1.0e3, 10.0),
    "solve_time_seconds": (1.0e3, 10.0),
    "iterations": (1.0e6, 100.0),
    "kkt.primal_res_rel": (1.0, KKT_SHIFT),
    "kkt.dual_res_rel": (1.0, KKT_SHIFT),
    "kkt.duality_gap_rel": (1.0, KKT_SHIFT),
    "kkt.comp_slack_rel": (1.0, KKT_SHIFT),
    "kkt.comp_slack": (1.0, KKT_SHIFT),
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


def profile_floor(metric: str) -> float:
    """Resolution below which values of ``metric`` count as equal (0 if none).

    Every ratio view of a report (profile, ratio heatmap, pairwise speedups
    and scatter) floors successful values here before dividing.
    """
    return _PROFILE_FLOORS.get(metric, 0.0)


def metric_defaults(metric: str) -> tuple[float, float]:
    """Return ``(failure_penalty, shift)`` defaults for ``metric``.

    Falls back to the run-time-style defaults for unknown metrics so
    aggregates remain useful for ad-hoc columns (e.g. wall-time
    derivatives). Time geomeans use the shift but require a time limit or
    explicit penalty instead of using the historical failure-penalty default.
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

    When ``metric`` is None or absent from the frame, a successful row
    still wins over a failed one, and otherwise the first row in file order
    is kept. Every per-identity view of a report resolves duplicates this
    way, so they agree on which attempt a duplicated solve stands for.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    subset_keys = [*keys, "solver_id"]
    if (metric is None or metric not in frame.columns) and "status" in frame.columns and not frame.empty:
        failed = ~solved_mask(frame, success_statuses).to_numpy()
        order = np.argsort(failed, kind="stable")
        return frame.iloc[order].drop_duplicates(subset=subset_keys, keep="first")
    if metric is not None and metric in frame.columns:
        sortable = frame.assign(
            __metric_for_dedup=pd.to_numeric(frame[metric], errors="coerce"),
        )
        # Status-failure flag sorts ascending: False (success) before
        # True (failure). When the status column is missing we treat
        # every row as successful so behavior matches the old code path.
        if "status" in sortable.columns:
            valid = np.isfinite(sortable["__metric_for_dedup"]) & sortable["__metric_for_dedup"].ge(0)
            sortable["__failure_for_dedup"] = ~(solved_mask(sortable, success_statuses) & valid)
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
    metric units and must be strictly larger than every retained successful
    value after applying ``min_value``; otherwise ValueError is raised. This
    keeps failures from tying a success, including at ratio 1. The returned curve is
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
        min_value = profile_floor(metric)
    if not np.isfinite(min_value) or min_value < 0:
        raise ValueError("min_value must be finite and nonnegative")
    results, expected = drop_solvers_without_metric(results, metric, success_statuses, expected=expected)
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
    solved = deduped.assign(__solved=solved_mask(deduped, success_statuses)).pivot(
        index=index, columns="solver_id", values="__solved"
    )
    success_mask = solved.fillna(False).astype(bool) & np.isfinite(values) & values.ge(0)
    values = values.clip(lower=min_value)
    successful_values = values.where(success_mask)
    if np.isfinite(max_value) and successful_values.ge(max_value).any().any():
        raise ValueError(
            "Finite max_value must be strictly greater than every successful metric "
            "after applying min_value; choose a larger penalty or omit max_value "
            "for infinite failure ratios"
        )
    best = successful_values.min(axis=1)
    ratios = values.where(success_mask, max_value).divide(best, axis=0)
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


def drop_solvers_without_metric(
    results: pd.DataFrame,
    metric: str,
    success_statuses: set[str] | None = None,
    *,
    expected: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Leave out solvers that report no value of ``metric`` on any success.

    Several adapters never set ``solve_time_seconds`` or
    ``setup_time_seconds``; scoring their solved problems as failures on
    such a metric would rank a solver that solved everything with one that
    solved nothing. A solver with at least one success and the metric
    missing (null) on all of them is dropped from the comparison (and from
    ``expected``) with a warning. A solver that does report the metric is
    still charged for a success without a usable value (missing, negative
    or not finite), so a solver cannot skip its hard problems by not
    reporting them.
    """
    if results.empty or "solver_id" not in results or "status" not in results:
        return results, expected
    raw = results[metric] if metric in results else pd.Series(np.nan, index=results.index)
    solved = solved_mask(results, success_statuses)
    reported = solved & raw.notna()
    by_solver = pd.DataFrame({"solver_id": results["solver_id"], "solved": solved, "reported": reported})
    counts = by_solver.groupby("solver_id", observed=True)[["solved", "reported"]].sum()
    silent = sorted(str(solver) for solver in counts.index[(counts["solved"] > 0) & (counts["reported"] == 0)])
    if not silent:
        return results, expected
    _warn_once(
        f"Leaving {', '.join(silent)} out of the {metric} comparison: "
        "no successful solve reports a value for it"
    )
    keep = ~results["solver_id"].astype(str).isin(silent)
    if expected is not None and "solver_id" in expected:
        expected = expected[~expected["solver_id"].astype(str).isin(silent)]
    return results[keep], expected


@functools.lru_cache(maxsize=256)
def _warn_once(message: str) -> None:
    # A report computes the same comparison several times (tables, plots,
    # per-dataset slices); say it once.
    logger.warning(message)


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
    timeout_seconds: float | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute per-solver means over the expected comparison universe.

    Missing solves receive the failure penalty, as do successes without a
    value of ``metric`` from a solver that reports it elsewhere; solvers
    that never report it are left out (see
    :func:`drop_solvers_without_metric`). Supply ``expected`` with
    problem/solver identities to include completely absent problems or solvers;
    otherwise the observed sets are crossed. Success-only means still omit
    failures, while their counts describe the same comparison universe.
    For time metrics, the default failure penalty is three times
    ``timeout_seconds`` (a positive finite scalar or a mapping by dataset ID).
    A time limit or explicit ``max_value`` is required. An explicit penalty is
    preserved as given; other metrics retain their fixed metric defaults.
    Penalties never depend on observed successful values or solver membership.
    The returned ``max_value`` records the fixed penalty, or is null for a
    mixed-dataset comparison with different penalties; ``max_value_by_dataset``
    then records the applied penalties as a JSON mapping. Success-only means
    do not require a penalty or time limit.
    """
    if success_statuses is None:
        success_statuses = set(status.SOLUTION_PRESENT)
    if shift is None:
        _, shift = metric_defaults(metric)
    results, expected = drop_solvers_without_metric(results, metric, success_statuses, expected=expected)
    results = complete_results(results, expected=expected)
    if "problem" in results:
        keys = ["dataset", "problem"] if "dataset" in results else ["problem"]
        results = deduplicate_for_pivot(results, keys, metric, success_statuses=success_statuses)
    columns = ["solver_id", metric, "mode", "shift", "max_value", "success_count", "failure_count",
               "max_value_by_dataset"]
    if results.empty:
        return pd.DataFrame(columns=columns)
    results = results.assign(__failure_penalty=(
        _geomean_penalties(results, metric, max_value, timeout_seconds)
        if penalize_failures else np.nan
    ))
    rows = []
    for solver_id, group in results.groupby("solver_id", observed=True):
        values = pd.to_numeric(
            group.get(metric, pd.Series(np.nan, index=group.index)), errors="coerce"
        ).to_numpy(dtype=float, copy=True)
        successful = solved_mask(group, success_statuses).to_numpy() & np.isfinite(values) & (values >= 0)
        success_count = int(successful.sum())
        failure_count = int(len(group) - success_count)
        reported_max = None
        max_by_dataset = None
        if penalize_failures:
            penalties = group["__failure_penalty"].to_numpy(dtype=float)
            if np.any(values[successful] >= penalties[successful]):
                _warn_once(
                    f"{solver_id}: a successful solve's {metric} is at or above the failure "
                    "penalty, so failing costs no more than succeeding; choose a larger penalty"
                )
            values[~successful] = penalties[~successful]
            if len(np.unique(penalties)) == 1:
                reported_max = float(penalties[0])
            else:
                max_by_dataset = json.dumps(
                    {str(dataset): float(penalty) for dataset, penalty in
                     group.groupby("dataset", observed=True)["__failure_penalty"].first().items()},
                    sort_keys=True,
                )
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
                "max_value": reported_max,
                "success_count": success_count,
                "failure_count": failure_count,
                "max_value_by_dataset": max_by_dataset,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("solver_id")


def _geomean_penalties(
    results: pd.DataFrame,
    metric: str,
    max_value: float | None,
    timeout_seconds: float | Mapping[str, float] | None,
) -> pd.Series:
    if max_value is not None:
        if np.isnan(max_value) or max_value < 0:
            raise ValueError("max_value must be nonnegative and not NaN")
        return pd.Series(max_value, index=results.index, dtype=float)
    if metric not in TIME_METRICS:
        penalty, _ = metric_defaults(metric)
        return pd.Series(penalty, index=results.index, dtype=float)
    if timeout_seconds is None:
        raise ValueError("Time geomeans require timeout_seconds or an explicit max_value failure penalty")
    if isinstance(timeout_seconds, Mapping):
        penalties = {str(dataset): _time_failure_penalty(limit) for dataset, limit in timeout_seconds.items()}
        if "dataset" not in results:
            if len(penalties) != 1:
                raise ValueError("Dataset identities are required for per-dataset time limits")
            return pd.Series(next(iter(penalties.values())), index=results.index, dtype=float)
        values = results["dataset"].astype(str).map(penalties)
        if values.isna().any():
            missing = sorted(set(results.loc[values.isna(), "dataset"].astype(str)))
            raise ValueError(f"Missing time limit for datasets {missing}; supply timeout_seconds or max_value")
        return values
    return pd.Series(_time_failure_penalty(timeout_seconds), index=results.index, dtype=float)


def _time_failure_penalty(timeout_seconds: float) -> float:
    penalty = 3.0 * timeout_seconds
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0 or not np.isfinite(penalty):
        raise ValueError("timeout_seconds must be positive and finite, with a finite three-times penalty")
    return penalty


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
