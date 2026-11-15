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
