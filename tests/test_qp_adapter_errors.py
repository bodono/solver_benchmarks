"""A solver that fails on a problem is a solver_error, not a harness error.

An exception escaping ``adapter.solve`` becomes ``worker_error``, which the
runner treats as a harness failure and re-runs on every resume. A solver's
own deterministic failure on a problem (a non-convex P, l > u, non-finite
data) is recorded as ``solver_error`` with the message on ``info.error``;
configuration mistakes still raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.solvers import get_solver


def _solve(solver_name, qp, tmp_path: Path, settings=None):
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} not installed")
    adapter = adapter_cls({"verbose": False, **(settings or {})})
    return adapter.solve(ProblemData("test", "p", QP, qp), tmp_path)


def _qp(p_diag, l, u):
    rng = np.random.default_rng(2)
    n, m = len(p_diag), len(l)
    return {
        "P": sp.diags(np.asarray(p_diag, dtype=float), format="csc"),
        "q": rng.normal(size=n),
        "A": sp.csc_matrix(rng.normal(size=(m, n))),
        "l": np.asarray(l, dtype=float),
        "u": np.asarray(u, dtype=float),
    }


NONCONVEX = _qp([-5.0, 1.0, 1.0, 1.0], -np.ones(6), np.ones(6))
CROSSED_BOUNDS = _qp([1.0, 1.0, 1.0, 1.0], np.ones(6), -np.ones(6))


@pytest.mark.parametrize("solver_name, qp", [
    ("osqp", NONCONVEX),
    ("osqp", CROSSED_BOUNDS),
    ("qtqp", NONCONVEX),
])
def test_solver_failure_is_a_solver_error(solver_name, qp, tmp_path):
    result = _solve(solver_name, qp, tmp_path)
    assert result.status == status.SOLVER_ERROR
    assert result.info["error"]
    assert result.run_time_seconds is not None and result.run_time_seconds >= 0.0
    assert result.objective_value is None and result.kkt is None


def test_osqp_invalid_setting_value_still_raises(tmp_path):
    with pytest.raises(ValueError, match="OSQP rejected"):
        _solve("osqp", _qp([1.0, 1.0], -np.ones(3), np.ones(3)), tmp_path, {"eps_abs": -1.0})


SMALL_QP = _qp([1.0, 1.0, 1.0, 1.0], -np.ones(6), np.ones(6))
SMALL_LP = dict(SMALL_QP, P=sp.csc_matrix((4, 4)))


@pytest.mark.parametrize("settings", [{"eps_abs": -1.0}, {"max_iter": -5}])
@pytest.mark.parametrize("solver_name", ["osqp", "piqp", "proxqp", "qtqp", "cpo3", "pdlp", "highs"])
def test_invalid_setting_value_is_a_configuration_error(solver_name, settings, tmp_path):
    # Before, PIQP and PDLP returned these as solver_error rows (which resume
    # never retries), ProxQP accepted them and ran to max_iter, QTQP failed an
    # assert with no message, and cpo3 raised OverflowError for max_iter.
    qp = SMALL_LP if solver_name == "pdlp" else SMALL_QP
    with pytest.raises(ValueError):
        _solve(solver_name, qp, tmp_path, settings)


def test_piqp_invalid_settings_status_raises_naming_the_settings(tmp_path):
    with pytest.raises(ValueError, match="PIQP_INVALID_SETTINGS.*'reg_lower_limit'"):
        _solve("piqp", SMALL_QP, tmp_path, {"reg_lower_limit": -1.0})


def test_pdlp_invalid_parameter_raises_with_its_message(tmp_path):
    with pytest.raises(ValueError, match="iteration_limit must be non-negative"):
        _solve("pdlp", SMALL_LP, tmp_path, {"max_iter": -5})


@pytest.mark.parametrize("settings, name", [
    ({"max_centrality_correctors": -1}, "max_centrality_correctors"),
    ({"gmres_restart": 0}, "gmres_restart"),
    ({"step_size_scale": 1.0}, "step_size_scale"),
    ({"tol_gap_rel": -1e-9}, "tol_gap_rel"),
    ({"eps_rel": -1e-9}, r"tol_gap_rel.*eps_abs / eps_rel / eps"),
    ({"max_iter": 10.5}, "max_iter"),
])
def test_qtqp_out_of_range_setting_raises_naming_it(settings, name, tmp_path):
    # qtqp raises ValueError for these from inside solve(), the same call that
    # raises for problem data it cannot take; the adapter checks them first so
    # they stay configuration errors instead of becoming solver_error rows.
    with pytest.raises(ValueError, match=name):
        _solve("qtqp", SMALL_QP, tmp_path, settings)


def test_qtqp_gmres_restart_is_not_checked_without_gmres(tmp_path):
    pytest.importorskip("qtqp")
    result = _solve(
        "qtqp", SMALL_QP, tmp_path, {"refinement_strategy": "richardson", "gmres_restart": 0}
    )
    assert result.status == status.OPTIMAL


@pytest.mark.parametrize("settings, name", [
    ({"eps_rel": -1e-6}, "eps_rel"),
    ({"eps_duality_gap_abs": float("nan")}, "eps_duality_gap_abs"),
    ({"rho": 0.0}, "rho"),
    ({"max_iter": 0}, "max_iter"),
])
def test_proxqp_out_of_range_setting_raises_naming_it(settings, name, tmp_path):
    with pytest.raises(ValueError, match=name):
        _solve("proxqp", SMALL_QP, tmp_path, settings)


def test_cpo3_conversion_failure_is_a_value_error(tmp_path):
    with pytest.raises(ValueError, match="max_iters"):
        _solve("cpo3", SMALL_QP, tmp_path, {"max_iters": -5})
