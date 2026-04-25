"""Integration tests: each adapter runs a small problem and reports KKT residuals."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.solvers import get_solver


SOLVER_SETTINGS = {
    "highs": {"verbose": False},
    "osqp": {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 10000, "polish": True},
    "proxqp": {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 10000},
    "piqp": {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 10000},
    "scs": {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iters": 5000},
    "clarabel": {"verbose": False},
    "qtqp": {"verbose": False},
    "pdlp": {"time_limit_sec": 10.0, "use_glop": False},
    "sdpa": {"verbose": False, "max_iter": 50, "optimality_tolerance": 1e-5},
}


def _small_qp():
    # min 0.5 (x1^2 + x2^2) + x1 + x2 s.t. -5 <= x1,x2 <= 5.
    # Unconstrained optimum at x* = (-1, -1), bounds inactive.
    return {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
        "n": 2,
        "m": 2,
        "obj_type": "min",
    }


def _small_lp():
    # min -x1 - x2 s.t. x1 + x2 <= 1, 0 <= x1, 0 <= x2. Optimum obj = -1.
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([-1.0, -1.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([-np.inf, 0.0, 0.0]),
        "u": np.array([1.0, np.inf, np.inf]),
        "n": 2,
        "m": 3,
        "obj_type": "min",
    }


def _infeasible_lp():
    # x >= 2 and x <= 1  =>  infeasible
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([0.0]),
        "A": sp.csc_matrix([[1.0], [1.0]]),
        "l": np.array([2.0, -np.inf]),
        "u": np.array([np.inf, 1.0]),
        "n": 1,
        "m": 2,
        "obj_type": "min",
    }


def _unbounded_lp():
    # min -x s.t. x >= 0 (no upper bound) -> unbounded below.
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([-1.0]),
        "A": sp.csc_matrix([[1.0]]),
        "l": np.array([0.0]),
        "u": np.array([np.inf]),
        "n": 1,
        "m": 1,
        "obj_type": "min",
    }


def _small_cone_lp():
    # min x s.t. x >= 1 in cone form A x + s = b, s >= 0.
    return {
        "P": None,
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[-1.0]]),
        "b": np.array([-1.0]),
        "r": 0.0,
        "n": 1,
        "m": 1,
        "cone": {"l": 1},
        "obj_type": "min",
    }


def _solve(solver_name: str, qp: dict, tmp_path: Path):
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} not installed")
    adapter = adapter_cls(SOLVER_SETTINGS.get(solver_name, {}))
    problem = ProblemData("test", "p", QP, qp)
    artifacts = tmp_path / solver_name
    artifacts.mkdir(parents=True, exist_ok=True)
    return adapter.solve(problem, artifacts)


def _solve_cone(solver_name: str, cone_problem: dict, tmp_path: Path):
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} not installed")
    adapter = adapter_cls(SOLVER_SETTINGS.get(solver_name, {}))
    problem = ProblemData("test", "p", CONE, cone_problem)
    artifacts = tmp_path / solver_name
    artifacts.mkdir(parents=True, exist_ok=True)
    return adapter.solve(problem, artifacts)


@pytest.mark.parametrize("solver_name", ["osqp", "scs", "clarabel", "qtqp", "highs", "proxqp", "piqp"])
def test_adapter_reports_kkt_for_small_qp(solver_name: str, tmp_path: Path):
    result = _solve(solver_name, _small_qp(), tmp_path)
    assert result.status == status.OPTIMAL, result.status
    assert result.kkt is not None
    for key in ("primal_res_rel", "dual_res_rel", "comp_slack"):
        assert key in result.kkt
        assert np.isfinite(result.kkt[key])
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4
    assert result.kkt["duality_gap_rel"] < 1e-4


@pytest.mark.parametrize("solver_name", ["osqp", "scs", "clarabel", "qtqp", "pdlp", "highs", "proxqp", "piqp"])
def test_adapter_reports_kkt_for_small_lp(solver_name: str, tmp_path: Path):
    result = _solve(solver_name, _small_lp(), tmp_path)
    assert result.status == status.OPTIMAL, result.status
    assert result.objective_value == pytest.approx(-1.0, abs=1e-4)
    assert result.kkt is not None
    assert np.isfinite(result.kkt["primal_res_rel"])
    assert np.isfinite(result.kkt["dual_res_rel"])
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


def test_sdpa_reports_kkt_for_small_cone_lp(tmp_path: Path):
    result = _solve_cone("sdpa", _small_cone_lp(), tmp_path)
    assert result.status == status.OPTIMAL, result.status
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "osqp"])
def test_adapter_reports_primal_infeasibility_certificate(solver_name: str, tmp_path: Path):
    result = _solve(solver_name, _infeasible_lp(), tmp_path)
    assert result.status in {
        status.PRIMAL_INFEASIBLE,
        status.PRIMAL_INFEASIBLE_INACCURATE,
    }, result.status
    assert result.kkt is not None
    assert result.kkt.get("certificate") == "primal_infeasible"
    # Either the cone-form (bty) or QP-form (support) witness must be negative.
    witness = result.kkt.get("bty", result.kkt.get("support"))
    assert witness is not None and witness < 0.0


@pytest.mark.parametrize(
    "status_val, expected",
    [
        (1, status.OPTIMAL),
        (2, status.OPTIMAL_INACCURATE),
        (-1, status.DUAL_INFEASIBLE),
        (-2, status.PRIMAL_INFEASIBLE),
        (-3, status.SOLVER_ERROR),
        (-4, status.SOLVER_ERROR),
        (-5, status.SOLVER_ERROR),
        (-6, status.DUAL_INFEASIBLE_INACCURATE),
        (-7, status.PRIMAL_INFEASIBLE_INACCURATE),
    ],
)
def test_scs_status_val_mapping(status_val: int, expected: str):
    from solver_benchmarks.solvers.scs_adapter import _map_scs_status

    assert _map_scs_status({"status_val": status_val, "status": ""}) == expected


@pytest.mark.parametrize("solver_name", ["scs", "clarabel"])
def test_adapter_reports_dual_infeasibility_certificate(solver_name: str, tmp_path: Path):
    result = _solve(solver_name, _unbounded_lp(), tmp_path)
    assert result.status in {
        status.DUAL_INFEASIBLE,
        status.DUAL_INFEASIBLE_INACCURATE,
    }, result.status
    assert result.kkt is not None
    assert result.kkt.get("certificate") == "dual_infeasible"
    assert result.kkt.get("qtx") is not None and result.kkt["qtx"] < 0.0
