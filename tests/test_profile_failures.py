import numpy as np
import pandas as pd
import pytest

from solver_benchmarks.analysis.profiles import performance_profile


def test_default_failure_never_enters_profile_or_becomes_fastest():
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "solved", "status": "optimal", "run_time_seconds": 1500.0},
        {"problem": "p", "solver_id": "failed", "status": "time_limit", "run_time_seconds": 1800.0},
    ])
    profile = performance_profile(frame, n_tau=3, tau_max=1e9)
    assert profile["solved"].tolist() == [1.0, 1.0, 1.0]
    assert profile["failed"].tolist() == [0.0, 0.0, 0.0]


def test_all_failed_rows_and_missing_metrics_are_retained():
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "a", "status": "worker_error", "run_time_seconds": None},
        {"problem": "q", "solver_id": "b", "status": "time_limit", "run_time_seconds": 1000.0},
    ])
    profile = performance_profile(frame, n_tau=3)
    assert list(profile.columns) == ["tau", "a", "b"]
    assert (profile[["a", "b"]] == 0).all().all()


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf, -1.0, "not-a-time"])
def test_invalid_success_metric_counts_as_failure(invalid):
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "a", "status": "optimal", "run_time_seconds": 2.0},
        {"problem": "p", "solver_id": "b", "status": "optimal", "run_time_seconds": invalid},
    ])
    profile = performance_profile(frame, n_tau=3)
    assert (profile["a"] == 1).all()
    assert (profile["b"] == 0).all()


def test_zero_iterations_keep_positive_successes_finite_and_failures_infinite():
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "a", "status": "optimal", "iterations": 0},
        {"problem": "p", "solver_id": "b", "status": "optimal", "iterations": 0},
        {"problem": "p", "solver_id": "c", "status": "optimal", "iterations": 5},
        {"problem": "p", "solver_id": "d", "status": "time_limit", "iterations": 0},
    ])
    profile = performance_profile(frame, metric="iterations", n_tau=3, tau_max=100)
    assert (profile[["a", "b"]] == 1).all().all()
    assert profile["c"].tolist() == [0, 1, 1]
    assert (profile["d"] == 0).all()


@pytest.mark.parametrize("metric", ["run_time_seconds", "iterations", "custom"])
def test_finite_penalty_cannot_outrank_any_success(metric):
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "fast", "status": "optimal", metric: 1500},
        {"problem": "p", "solver_id": "slow", "status": "optimal", metric: 3000},
        {"problem": "p", "solver_id": "failed", "status": "time_limit", metric: 1800},
        {"problem": "q", "solver_id": "failed", "status": "worker_error", metric: None},
    ])
    profile = performance_profile(frame, metric=metric, max_value=1000, n_tau=3, tau_max=4)
    assert profile["fast"].tolist() == [0.5, 0.5, 0.5]
    assert profile["slow"].tolist() == profile["failed"].tolist() == [0, 0.5, 0.5]
    higher = performance_profile(frame, metric=metric, max_value=6000, n_tau=3, tau_max=4)
    assert higher["failed"].tolist() == [0, 0, 0.5]


def test_time_floor_is_configurable_in_metric_units():
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "a", "status": "optimal", "run_time_seconds": 0},
        {"problem": "p", "solver_id": "b", "status": "optimal", "run_time_seconds": 0.1},
    ])
    profile = performance_profile(frame, n_tau=3, tau_max=100)
    assert profile["b"].tolist() == [0, 1, 1]
    floored = performance_profile(frame, min_value=0.1, n_tau=3, tau_max=100)
    assert floored["b"].tolist() == [1, 1, 1]
    exact = performance_profile(frame, min_value=0, n_tau=3, tau_max=100)
    assert exact["b"].tolist() == [0, 0, 0]
