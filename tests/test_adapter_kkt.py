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
    "cvxopt": {"verbose": False, "abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9},
    "ecos": {"verbose": False, "feastol": 1e-9, "abstol": 1e-9, "reltol": 1e-9},
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


def _small_sdp():
    # min trace(C X) s.t. trace(X) = 1, X in S^2_+, with C = diag(1, 2).
    # The PSD-cone variables are stored col-major lower:
    # x = [X[0,0], X[1,0], X[1,1]]. Optimum: X = diag(1, 0), value = 1.
    a = sp.csc_matrix(
        np.vstack(
            [
                np.array([[1.0, 0.0, 1.0]]),  # trace(X) = 1 (zero cone row)
                -np.eye(3),                    # -X + S = 0 with S in PSD-triangle
            ]
        )
    )
    return {
        "P": None,
        "q": np.array([1.0, 0.0, 2.0]),  # objective: X[0,0] + 2 X[1,1]
        "A": a,
        "b": np.array([1.0, 0.0, 0.0, 0.0]),
        "r": 0.0,
        "n": 3,
        "m": 4,
        "cone": {"z": 1, "s": [2]},
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


@pytest.mark.parametrize("solver_name", ["osqp", "scs", "clarabel", "qtqp", "highs", "proxqp", "piqp", "cvxopt", "ecos"])
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


@pytest.mark.parametrize("solver_name", ["osqp", "scs", "clarabel", "qtqp", "pdlp", "highs", "proxqp", "piqp", "cvxopt", "ecos"])
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


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "sdpa", "cvxopt"])
def test_solvers_agree_on_small_sdp_with_psd_cone(solver_name: str, tmp_path: Path):
    # Verifies that all conic adapters interpret the canonical PSD triangle vec
    # (col-major lower with √2 off-diagonal scaling) consistently. Regression
    # guard against the SDPLIB row-major-lower bug.
    result = _solve_cone(solver_name, _small_sdp(), tmp_path)
    assert result.status == status.OPTIMAL, result.status
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "osqp", "ecos"])
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


def test_osqp_infeasible_result_has_no_reported_objective(tmp_path: Path):
    result = _solve("osqp", _infeasible_lp(), tmp_path)

    assert result.status in {
        status.PRIMAL_INFEASIBLE,
        status.PRIMAL_INFEASIBLE_INACCURATE,
    }
    assert result.objective_value is None
    assert "obj_val" in result.info


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


@pytest.mark.parametrize(
    "raw_status, expected",
    [
        ("solved", status.OPTIMAL),
        ("infeasible", status.PRIMAL_INFEASIBLE),
        ("unbounded", status.DUAL_INFEASIBLE),
        ("hit_max_iter", status.MAX_ITER_REACHED),
        ("unfinished", status.SOLVER_ERROR),
        ("failed", status.SOLVER_ERROR),
        ("totally-unknown", status.SOLVER_ERROR),
    ],
)
def test_qtqp_status_mapping(raw_status: str, expected: str):
    from solver_benchmarks.solvers.qtqp_adapter import _map_qtqp_status

    assert _map_qtqp_status(raw_status) == expected


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "ecos"])
def test_adapter_reports_dual_infeasibility_certificate(solver_name: str, tmp_path: Path):
    result = _solve(solver_name, _unbounded_lp(), tmp_path)
    assert result.status in {
        status.DUAL_INFEASIBLE,
        status.DUAL_INFEASIBLE_INACCURATE,
    }, result.status
    assert result.kkt is not None
    assert result.kkt.get("certificate") == "dual_infeasible"
    assert result.kkt.get("qtx") is not None and result.kkt["qtx"] < 0.0


@pytest.mark.parametrize(
    "solver_name",
    ["osqp", "scs", "clarabel", "qtqp", "highs", "proxqp", "piqp", "gurobi", "mosek", "cplex"],
)
def test_adapter_does_not_add_r_offset_to_reported_objective(solver_name: str, tmp_path: Path):
    # Regression: every QP adapter must return ``c^T x + 0.5 x^T P x`` (no
    # offset) — ``solver_benchmarks.worker._reported_objective`` adds ``r``
    # exactly once. Adding it again inside an adapter would double-count.
    qp = _small_qp()
    qp["r"] = 42.0
    result = _solve(solver_name, qp, tmp_path)
    assert result.status == status.OPTIMAL, result.status
    # Optimum of min 0.5(x1^2 + x2^2) + x1 + x2 is x* = (-1, -1), value = -1.
    assert result.objective_value == pytest.approx(-1.0, abs=1e-4)


@pytest.mark.parametrize(
    "phase, errors, expected",
    [
        # sdpapinfo['phasevalue'] is in CLP form — pUNBD = primal unbounded
        # = dual infeasible; dUNBD = dual unbounded = primal infeasible.
        ("pdOPT", 0.0, status.OPTIMAL),
        ("pdFEAS", 0.0, status.OPTIMAL),
        ("pdFEAS", 1.0, status.OPTIMAL_INACCURATE),
        ("pINF_dFEAS", 0.0, status.PRIMAL_INFEASIBLE),
        ("pFEAS_dINF", 0.0, status.DUAL_INFEASIBLE),
        ("pUNBD", 0.0, status.DUAL_INFEASIBLE),
        ("dUNBD", 0.0, status.PRIMAL_INFEASIBLE),
        ("pdINF", 0.0, status.PRIMAL_OR_DUAL_INFEASIBLE),
        ("noINFO", 0.0, status.SOLVER_ERROR),
    ],
)
def test_sdpa_phase_mapping(phase: str, errors: float, expected: str):
    from solver_benchmarks.solvers.sdpa_adapter import _map_sdpa_status

    sdpap_info = {
        "phasevalue": phase,
        "primalError": errors,
        "dualError": errors,
        "dualityGap": errors,
    }
    sdpa_info = {"iteration": 0}
    assert _map_sdpa_status(sdpap_info, sdpa_info, {"maxIteration": 100}, 1.0e-5) == expected


@pytest.mark.parametrize("solver_name", ["cplex", "mosek"])
@pytest.mark.parametrize("quadratic", [False, True])
def test_commercial_adapter_kkt_original_rows(solver_name, quadratic, tmp_path):
    # Includes a skipped free row, equality, lower/upper bounds, and two
    # ranged rows active on opposite sides. Off-diagonal P tests symmetry.
    p = np.eye(5)
    p[0, 1] = p[1, 0] = 0.25
    qp = {
        "P": sp.csc_matrix(p if quadratic else np.zeros((5, 5))),
        "q": np.array([1.0, 2.0, -3.0, 2.0, -3.0]),
        "A": sp.csc_matrix(np.vstack([np.ones(5), np.eye(5)])),
        "l": np.array([-np.inf, 1.0, 0.0, -np.inf, 0.0, 0.0]),
        "u": np.array([np.inf, 1.0, np.inf, 1.0, 1.0, 1.0]),
    }
    result = _solve(solver_name, qp, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.kkt is not None
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert result.kkt[field] < 1e-6


@pytest.mark.parametrize("solver_name", ["cplex", "mosek"])
def test_commercial_adapter_infeasible_has_no_kkt(solver_name, tmp_path):
    result = _solve(solver_name, _infeasible_lp(), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.kkt is None


@pytest.mark.parametrize("solver_name", ["clarabel", "cplex", "highs", "mosek", "qpo3"])
def test_solver_error_retains_kkt(solver_name, tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module(f"solver_benchmarks.solvers.{solver_name}_adapter")
    # Numerical errors are not reproducible: keep a real point and report an error.
    reported = status.SOLVER_ERROR
    if solver_name == "qpo3":
        monkeypatch.setattr(module, "_STATUS", dict.fromkeys(module._STATUS, reported))
    else:
        monkeypatch.setattr(module, f"_map_{solver_name}_status", lambda *args: reported)
    result = _solve(solver_name, _small_qp(), tmp_path)
    assert result.status == reported
    assert result.kkt is not None
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert result.kkt[field] < 1e-6


@pytest.fixture(scope="module")
def limit_qp():
    # Dense positive-definite QP, large enough to exercise early termination.
    # Stay within CPLEX Community Edition's variable/constraint limits.
    n = 400
    rng = np.random.default_rng(0)
    b = rng.normal(size=(n, n))
    return {
        "P": sp.csc_matrix(b.T @ b / n + np.eye(n)),
        "q": -np.linspace(1.0, 3.0, n),
        "A": sp.eye(n, format="csc"),
        "l": np.zeros(n),
        "u": np.ones(n),
    }


@pytest.mark.parametrize("solver_name, settings", [
    ("clarabel", {"presolve_enable": False, "time_limit": 0.001}),
    ("qpo3", {"presolve": False, "time_limit": 0.001}),
    ("cplex", {"qpmethod": 4, "preprocessing.presolve": 0, "threads": 1, "time_limit": 0.1}),
    ("mosek", {"MSK_IPAR_PRESOLVE_USE": 0, "time_limit": 0.001}),
    ("highs", {"presolve": "off", "time_limit": 0.001}),
])
def test_time_limit_retains_kkt(solver_name, settings, limit_qp, tmp_path, monkeypatch):
    monkeypatch.setitem(SOLVER_SETTINGS, solver_name, settings)
    result = _solve(solver_name, limit_qp, tmp_path)
    assert result.status == status.TIME_LIMIT
    assert result.kkt is not None
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert np.isfinite(result.kkt[field])


@pytest.mark.parametrize("solver_name, settings", [
    ("clarabel", {"max_iter": 1, "presolve_enable": False}),
    ("qpo3", {"max_iter": 1, "presolve": False}),
    ("cplex", {"barrier.limits.iteration": 1, "qpmethod": 4, "preprocessing.presolve": 0}),
    ("mosek", {"MSK_IPAR_INTPNT_MAX_ITERATIONS": 1, "MSK_IPAR_PRESOLVE_USE": 0}),
    ("highs", {"qp_iteration_limit": 1, "presolve": "off"}),
])
def test_iteration_limit_retains_kkt(solver_name, settings, limit_qp, tmp_path, monkeypatch):
    monkeypatch.setitem(SOLVER_SETTINGS, solver_name, settings)
    result = _solve(solver_name, limit_qp, tmp_path)
    assert result.status == status.MAX_ITER_REACHED
    assert result.kkt is not None
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert np.isfinite(result.kkt[field])
