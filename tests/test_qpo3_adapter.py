"""qpo3 adapter: settings translation (no solver needed) and, when qpo3 is installed,
solves of a small QP, an LP and an infeasible LP with verified certificates."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.solvers.qpo3_adapter import translate_settings


def test_aliases_map_to_qpo3_names_and_native_values_win():
    s = {"eps": 1e-6, "max_iter": 50}
    assert translate_settings(s) == {"tol_feas": 1e-6, "tol_gap_abs": 1e-6, "tol_gap_rel": 1e-6}
    assert s == {"tol_feas": 1e-6, "tol_gap_abs": 1e-6, "tol_gap_rel": 1e-6, "max_iters": 50}
    s = {"eps_abs": 1e-3, "eps_rel": 1e-4, "tol_feas": 1e-9}
    assert translate_settings(s) == {"tol_gap_abs": 1e-3, "tol_gap_rel": 1e-4}
    assert s["tol_feas"] == 1e-9


def _problem(name, qp):
    from solver_benchmarks.core.problem import QP, ProblemData
    return ProblemData("unit", name, QP, qp, metadata={})


def _adapter(**settings):
    from solver_benchmarks.solvers.qpo3_adapter import Qpo3SolverAdapter
    return Qpo3SolverAdapter(settings=settings)


def test_qpo3_solves_box_qp_and_lp():
    pytest.importorskip("qpo3")
    n = 3
    qp = {"P": sp.eye(n, format="csc"), "q": -np.ones(n), "A": sp.eye(n, format="csc"),
          "l": np.zeros(n), "u": np.full(n, 0.5)}
    result = _adapter(eps=1e-8, verbose=False).solve(_problem("box", qp), Path(tempfile.mkdtemp()))
    assert result.status == "optimal"
    assert np.allclose(result.objective_value, 3 * (0.5**2 / 2 - 0.5), atol=1e-6)
    assert result.kkt["primal_res_rel"] < 1e-6 and result.kkt["dual_res_rel"] < 1e-6
    assert result.info["settings_translated"] == {"tol_feas": 1e-8, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8}
    # LP: min x1 + x2 s.t. x1 + x2 >= 1, x >= 0  (P all zero is passed as an LP)
    lp = {"P": sp.csc_matrix((2, 2)), "q": np.ones(2), "A": sp.csc_matrix(np.vstack([np.ones((1, 2)), np.eye(2)])),
          "l": np.array([1.0, 0.0, 0.0]), "u": np.array([np.inf, np.inf, np.inf])}
    result = _adapter(eps=1e-8).solve(_problem("lp", lp), Path(tempfile.mkdtemp()))
    assert result.status == "optimal" and abs(result.objective_value - 1.0) < 1e-6


def test_qpo3_certifies_infeasible_lp():
    pytest.importorskip("qpo3")
    # x >= 1 and x <= 0
    lp = {"P": sp.csc_matrix((1, 1)), "q": np.ones(1), "A": sp.csc_matrix(np.array([[1.0], [1.0]])),
          "l": np.array([1.0, -np.inf]), "u": np.array([np.inf, 0.0])}
    result = _adapter(eps=1e-8).solve(_problem("infeasible", lp), Path(tempfile.mkdtemp()))
    assert result.status == "primal_infeasible"
    assert result.kkt["certificate"] == "primal_infeasible" and result.kkt["valid"]
