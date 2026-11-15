"""Cross-adapter result policy for the conic adapters (SCS, Clarabel, ECOS,
CVXOPT, SDPA).

Every adapter reports the same kind of event the same way, so status
breakdowns, ``kkt-verify --promote`` and objective listings compare solvers
rather than adapters:

* a solve stopped by its iteration or time limit is ``max_iter_reached`` /
  ``time_limit`` (the solver's own wording stays in ``info``), never
  ``optimal_inaccurate`` or an ``*_infeasible_inaccurate`` guess;
* solution KKT residuals are recorded for every status in
  ``status.ANY_FEASIBLE`` whenever the solver returned a point;
* ``objective_value`` is reported only for ``optimal`` and
  ``optimal_inaccurate``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.solvers import get_solver


def _adapter(solver_name: str, settings: dict):
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} not installed")
    return adapter_cls({"verbose": False, **settings})


def _solve(solver_name: str, kind: str, data: dict, settings: dict, tmp_path: Path):
    adapter = _adapter(solver_name, settings)
    artifacts = tmp_path / solver_name
    artifacts.mkdir(parents=True, exist_ok=True)
    return adapter.solve(ProblemData("test", "p", kind, data), artifacts)


def _feasible_cone_lp(seed: int = 0, n: int = 60, m: int = 120) -> dict:
    """A random LP ``min c'x s.t. Ax + s = b, s >= 0`` with a strictly
    feasible primal point and a strictly feasible dual point."""
    rng = np.random.default_rng(seed)
    a = sp.random(m, n, density=0.3, random_state=seed + 1, format="csc")
    b = a @ rng.standard_normal(n) + np.abs(rng.standard_normal(m))
    c = -a.T @ np.abs(rng.standard_normal(m))
    return {"P": None, "A": a, "b": b, "q": c, "r": 0.0, "cone": {"l": m}}


def _feasible_qp(seed: int = 0, n: int = 80) -> dict:
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, n))
    return {
        "P": sp.csc_matrix(g.T @ g / n + np.eye(n)),
        "q": -np.linspace(1.0, 3.0, n),
        "A": sp.eye(n, format="csc"),
        "l": np.zeros(n),
        "u": np.ones(n),
    }


def _infeasible_cone_lp() -> dict:
    # x >= 2 and x <= 1 in cone form: [-x; x] + s = [-2; 1], s >= 0.
    return {
        "P": None,
        "A": sp.csc_matrix([[-1.0], [1.0]]),
        "b": np.array([-2.0, 1.0]),
        "q": np.array([1.0]),
        "r": 0.0,
        "cone": {"l": 2},
    }


def _infeasible_qp_form_lp() -> dict:
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[1.0], [1.0]]),
        "l": np.array([2.0, -np.inf]),
        "u": np.array([np.inf, 1.0]),
    }


def _assert_finite_solution_kkt(result) -> None:
    assert result.kkt is not None, result.info
    assert result.kkt.get("certificate") is None
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert np.isfinite(result.kkt[field]), (field, result.kkt)


# Settings that stop each adapter after very few iterations.
ITERATION_CAPS = {
    "scs": {"max_iters": 3},
}


@pytest.mark.parametrize("solver_name", sorted(ITERATION_CAPS))
def test_iteration_limit_is_max_iter_with_kkt_and_no_objective(solver_name, tmp_path):
    result = _solve(
        solver_name, CONE, _feasible_cone_lp(), ITERATION_CAPS[solver_name], tmp_path
    )
    assert result.status == status.MAX_ITER_REACHED, (result.status, result.info)
    _assert_finite_solution_kkt(result)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs"])
def test_qp_iteration_limit_is_max_iter_with_kkt(solver_name, tmp_path):
    result = _solve(solver_name, QP, _feasible_qp(), ITERATION_CAPS[solver_name], tmp_path)
    assert result.status == status.MAX_ITER_REACHED, (result.status, result.info)
    _assert_finite_solution_kkt(result)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs"])
def test_time_limit_is_time_limit_with_kkt_and_no_objective(solver_name, tmp_path):
    settings = {"time_limit": 1e-6}
    if solver_name == "clarabel":
        settings["presolve_enable"] = False
    result = _solve(solver_name, CONE, _feasible_cone_lp(n=200, m=400), settings, tmp_path)
    assert result.status == status.TIME_LIMIT, (result.status, result.info)
    _assert_finite_solution_kkt(result)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs"])
def test_infeasible_cone_result_has_no_reported_objective(solver_name, tmp_path):
    result = _solve(solver_name, CONE, _infeasible_cone_lp(), {}, tmp_path)
    assert result.status in status.ANY_INFEASIBLE, (result.status, result.info)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs"])
def test_infeasible_qp_result_has_no_reported_objective(solver_name, tmp_path):
    result = _solve(solver_name, QP, _infeasible_qp_form_lp(), {}, tmp_path)
    assert result.status in status.ANY_INFEASIBLE, (result.status, result.info)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs"])
def test_optimal_cone_result_reports_objective(solver_name, tmp_path):
    result = _solve(solver_name, CONE, _feasible_cone_lp(n=10, m=20), {}, tmp_path)
    assert result.status == status.OPTIMAL, (result.status, result.info)
    assert result.objective_value is not None
    assert result.objective_value == pytest.approx(result.kkt["primal_obj"], rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# SCS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info, expected",
    [
        ({"status_val": 2, "status": "solved (inaccurate - reached max_iters)"},
         status.MAX_ITER_REACHED),
        ({"status_val": 2, "status": "solved (inaccurate - reached time_limit_secs)"},
         status.TIME_LIMIT),
        ({"status_val": -6, "status": "unbounded (inaccurate - reached max_iters)"},
         status.MAX_ITER_REACHED),
        ({"status_val": -7, "status": "infeasible (inaccurate - reached time_limit_secs)"},
         status.TIME_LIMIT),
        ({"status_val": -7, "status": "infeasible (inaccurate - reached max_iters)"},
         status.MAX_ITER_REACHED),
        # Text missing: the configured limits decide.
        ({"status_val": 2, "status": "", "iter": 50}, status.MAX_ITER_REACHED),
        ({"status_val": 2, "status": "", "iter": 7, "solve_time": 20.0}, status.TIME_LIMIT),
        ({"status_val": 2, "status": "", "iter": 7, "solve_time": 1.0},
         status.OPTIMAL_INACCURATE),
        # A real solution or certificate is never relabelled.
        ({"status_val": 1, "status": "solved", "iter": 50}, status.OPTIMAL),
        ({"status_val": -2, "status": "infeasible", "iter": 50}, status.PRIMAL_INFEASIBLE),
    ],
)
def test_scs_limit_statuses(info, expected):
    from solver_benchmarks.solvers.scs_adapter import _map_scs_status

    assert _map_scs_status(info, max_iters=50, time_limit_secs=0.01) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("solved (inaccurate)", status.OPTIMAL_INACCURATE),
        ("infeasible (inaccurate)", status.PRIMAL_INFEASIBLE_INACCURATE),
        ("unbounded (inaccurate)", status.DUAL_INFEASIBLE_INACCURATE),
        ("solved (inaccurate - reached max_iters)", status.MAX_ITER_REACHED),
    ],
)
def test_scs_text_fallback_keeps_inaccurate_inaccurate(text, expected):
    from solver_benchmarks.solvers.scs_adapter import _map_scs_status

    assert _map_scs_status({"status": text}) == expected


def test_scs_limit_keeps_raw_status(tmp_path):
    result = _solve("scs", CONE, _feasible_cone_lp(), {"max_iters": 3}, tmp_path)
    assert result.status == status.MAX_ITER_REACHED
    assert "reached max_iters" in result.info["raw_status"]
    assert result.info["status_val"] == 2


def test_scs_translates_max_iter_and_eps(tmp_path):
    result = _solve(
        "scs", CONE, _feasible_cone_lp(), {"max_iter": 3, "eps": 1e-3}, tmp_path
    )
    assert result.status == status.MAX_ITER_REACHED
    assert result.iterations == 3
    assert result.info["settings_translated"] == {
        "max_iters": 3,
        "eps_abs": 1e-3,
        "eps_rel": 1e-3,
    }


def test_scs_native_settings_win_over_aliases(tmp_path):
    from solver_benchmarks.solvers.scs_adapter import translate_settings

    settings = {"max_iter": 3, "max_iters": 9, "eps": 1e-3, "eps_abs": 1e-7}
    translated = translate_settings(settings)
    assert settings == {"max_iters": 9, "eps_abs": 1e-7, "eps_rel": 1e-3}
    assert translated == {"eps_rel": 1e-3}
