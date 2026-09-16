"""QTQP adapter: tolerance settings follow the installed qtqp API (0.0.7 uses
tol_feas/tol_gap_abs/tol_gap_rel; earlier releases used atol/rtol)."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.solvers.qtqp_adapter import _translate_tolerances

NEW = {"tol_feas", "tol_gap_abs", "tol_gap_rel", "max_iter", "linear_solver"}
OLD = {"atol", "rtol", "max_iter", "linear_solver"}


def test_legacy_atol_rtol_map_to_new_names():
    s = {"atol": 1e-6, "rtol": 1e-7, "max_iter": 50}
    translated = _translate_tolerances(s, NEW)
    assert s == {"tol_feas": 1e-6, "tol_gap_abs": 1e-6, "tol_gap_rel": 1e-7, "max_iter": 50}
    assert translated == {"tol_feas": 1e-6, "tol_gap_abs": 1e-6, "tol_gap_rel": 1e-7}


def test_harness_eps_aliases_map_to_new_names():
    s = {"eps": 1e-5}
    _translate_tolerances(s, NEW)
    assert s == {"tol_feas": 1e-5, "tol_gap_abs": 1e-5, "tol_gap_rel": 1e-5}


def test_explicit_new_names_win_over_aliases():
    s = {"tol_feas": 1e-9, "atol": 1e-3}
    _translate_tolerances(s, NEW)
    assert s["tol_feas"] == 1e-9 and s["tol_gap_abs"] == 1e-3 and "atol" not in s


def test_new_names_map_back_on_old_api():
    s = {"eps_abs": 1e-6, "eps_rel": 1e-7, "tol_feas": 1e-6}
    _translate_tolerances(s, OLD)
    assert s == {"atol": 1e-6, "rtol": 1e-7}


def test_adapter_solves_with_translated_settings():
    import tempfile
    from pathlib import Path

    pytest.importorskip("qtqp")
    from solver_benchmarks.core.problem import QP, ProblemData
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    n = 3
    qp = {"P": sp.eye(n, format="csc"), "q": -np.ones(n), "A": sp.eye(n, format="csc"), "l": np.zeros(n), "u": np.full(n, 0.5)}
    problem = ProblemData("unit", "box", QP, qp, metadata={})
    adapter = QTQPSolverAdapter(settings={"atol": 1e-8, "rtol": 1e-8, "verbose": False})
    result = adapter.solve(problem, Path(tempfile.mkdtemp()))
    assert result.status == "optimal"
    assert np.allclose(result.objective_value, 3 * (0.5 ** 2 / 2 - 0.5), atol=1e-5)
    assert result.info.get("settings_translated") == {"tol_feas": 1e-8, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8}
