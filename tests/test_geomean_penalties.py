import json
import math

import numpy as np
import pandas as pd
import pytest

from solver_benchmarks.analysis.profiles import shifted_geomean


def test_fixed_time_penalty_does_not_change_when_a_slow_solver_is_removed():
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "a", "status": "optimal", "run_time_seconds": 1},
        {"problem": "q", "solver_id": "a", "status": "time_limit", "run_time_seconds": 300},
        {"problem": "p", "solver_id": "slow", "status": "optimal", "run_time_seconds": 290},
        {"problem": "q", "solver_id": "slow", "status": "optimal", "run_time_seconds": 295},
    ])
    together = shifted_geomean(frame, timeout_seconds=300).set_index("solver_id")
    alone = shifted_geomean(frame[frame.solver_id == "a"], timeout_seconds=300).set_index("solver_id")
    pd.testing.assert_series_equal(together.loc["a"], alone.loc["a"])
    assert together.loc["a", "max_value"] == 900
    assert together.loc["a", "run_time_seconds"] == pytest.approx(math.sqrt(11 * 910) - 10)


def test_dataset_limits_charge_missing_rows_and_remain_fixed_in_slices():
    limits = {"qp": 300, "sdp": 900, "lpbig": 1800}
    expected = pd.DataFrame([
        {"dataset": dataset, "problem": "p", "solver_id": solver}
        for dataset in limits for solver in ("a", "absent")
    ])
    frame = pd.DataFrame([
        {"dataset": "qp", "problem": "p", "solver_id": "a", "status": "optimal", "run_time_seconds": 1},
    ])
    gm = shifted_geomean(frame, expected=expected, timeout_seconds=limits).set_index("solver_id")
    assert gm["max_value"].isna().all()
    assert json.loads(gm.loc["a", "max_value_by_dataset"]) == {"qp": 900, "sdp": 2700, "lpbig": 5400}
    assert gm.loc["a", "run_time_seconds"] == pytest.approx((11 * 2710 * 5410) ** (1 / 3) - 10)
    assert gm.loc["absent", "run_time_seconds"] == pytest.approx((910 * 2710 * 5410) ** (1 / 3) - 10)
    assert gm.loc["a", "failure_count"] == 2
    assert gm.loc["absent", "failure_count"] == 3
    for dataset, limit in limits.items():
        sliced = shifted_geomean(frame[frame.dataset == dataset], expected=expected[expected.dataset == dataset],
                                 timeout_seconds=limits).set_index("solver_id")
        assert (sliced["max_value"] == 3 * limit).all()
        assert sliced["max_value_by_dataset"].isna().all()
        assert sliced.loc["absent", "run_time_seconds"] == pytest.approx(3 * limit)


@pytest.mark.parametrize("limits", [None, {"wrong-dataset": 300}, {"qp": np.nan}, 0])
def test_explicit_penalty_is_exact_and_bypasses_time_limit_metadata(limits):
    frame = pd.DataFrame([
        {"dataset": "qp", "problem": "p", "solver_id": "a", "status": "optimal", "run_time_seconds": 1000},
        {"dataset": "qp", "problem": "p", "solver_id": "b", "status": "time_limit", "run_time_seconds": 2000},
    ])
    gm = shifted_geomean(frame, timeout_seconds=limits, max_value=25).set_index("solver_id")
    assert (gm["max_value"] == 25).all()
    assert gm.loc["b", "run_time_seconds"] == pytest.approx(25)
    assert gm.loc["a", "run_time_seconds"] == pytest.approx(1000)


@pytest.mark.parametrize("metric", ["run_time_seconds", "setup_time_seconds", "solve_time_seconds"])
def test_time_metrics_require_fixed_penalty_or_limit(metric):
    frame = pd.DataFrame([{"solver_id": "a", "status": "optimal", metric: 1}])
    with pytest.raises(ValueError, match="require timeout_seconds or an explicit max_value"):
        shifted_geomean(frame, metric=metric)
    result = shifted_geomean(frame, metric=metric, penalize_failures=False)
    assert result[metric].iloc[0] == pytest.approx(1)
    assert result["max_value"].isna().all()
    assert shifted_geomean(pd.DataFrame(), metric=metric).empty


@pytest.mark.parametrize("limit", [0, -1, np.nan, np.inf, -np.inf, 1e308])
def test_time_limits_must_define_positive_finite_penalties(limit):
    frame = pd.DataFrame([{"solver_id": "a", "status": "worker_error"}])
    with pytest.raises(ValueError, match="positive and finite"):
        shifted_geomean(frame, timeout_seconds=limit)


def test_dataset_limit_mapping_rejects_unidentified_or_uncovered_rows():
    frame = pd.DataFrame([{"solver_id": "a", "status": "worker_error"}])
    with pytest.raises(ValueError, match="Dataset identities"):
        shifted_geomean(frame, timeout_seconds={"qp": 300, "sdp": 900})
    result = shifted_geomean(frame, timeout_seconds={"qp": 300})
    assert result["max_value"].iloc[0] == 900
    with pytest.raises(ValueError, match="Missing time limit for datasets"):
        shifted_geomean(frame.assign(dataset="sdp"), timeout_seconds={"qp": 300})


@pytest.mark.parametrize("metric,penalty", [("iterations", 1e6), ("kkt.primal_res_rel", 1.0)])
def test_non_time_penalties_stay_in_metric_units_and_ignore_time_limits(metric, penalty):
    frame = pd.DataFrame([
        {"problem": "p", "solver_id": "a", "status": "optimal", metric: 2 * penalty},
        {"problem": "p", "solver_id": "b", "status": "worker_error", metric: None},
    ])
    gm = shifted_geomean(frame, metric=metric, timeout_seconds=0).set_index("solver_id")
    assert (gm["max_value"] == penalty).all()
    assert gm.loc["b", metric] == pytest.approx(penalty)
