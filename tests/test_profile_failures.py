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


def test_zero_metric_ties_have_ratio_one_without_nan():
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "a", "status": "optimal", "iterations": 0},
        {"problem": "p", "solver_id": "b", "status": "optimal", "iterations": 0},
        {"problem": "p", "solver_id": "c", "status": "optimal", "iterations": 1},
    ])
    profile = performance_profile(frame, metric="iterations", n_tau=3)
    assert (profile[["a", "b"]] == 1).all().all()
    assert (profile["c"] == 0).all()
