"""MOSEK adapter behaviour that does not need a MOSEK installation.

A small stand-in for the ``mosek`` module drives the adapter through license
failures, numerical errors and limit terminations.
"""

from __future__ import annotations

import enum
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.solvers.base import SolverUnavailable


class _Rescode(enum.Enum):
    ok = 0
    trm_max_time = 10001
    trm_max_iterations = 10000
    err_license = 1000
    err_license_expired = 1001
    err_missing_license_file = 1008
    err_license_max = 1016
    err_flexlm = 1014
    err_space = 1051


class _Solsta(enum.Enum):
    unknown = 0
    optimal = 1
    prim_and_dual_feas = 3
    prim_feas = 4
    prim_infeas_cer = 6
    dual_infeas_cer = 7
    integer_optimal = 14


class _Soltype(enum.Enum):
    itr = 0
    bas = 1


class _Error(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.errno = code
        self.msg = msg


def _fake_mosek(*, optimize, solsta=_Solsta.optimal, x=(-1.0, -1.0)):
    """A ``mosek`` stand-in whose ``Task.optimize`` runs ``optimize()``."""

    class Task:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __getattr__(self, name):
            # Model-building and parameter calls are accepted and ignored.
            return lambda *args, **kwargs: None

        def optimize(self):
            return optimize()

        def solutiondef(self, soltype):
            return soltype is _Soltype.itr

        def getsolsta(self, soltype):
            return solsta

        def getxx(self, soltype):
            return list(x)

        def gety(self, soltype):
            return [0.0, 0.0]

        def getdouinf(self, item):
            return 0.01

        def getintinf(self, item):
            return 7

        def getprimalobj(self, soltype):
            return -1.0

    class Env:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def Task(self):  # noqa: N802 - MOSEK's API name
            return Task()

        def set_Stream(self, *args):  # noqa: N802
            pass

    names = types.SimpleNamespace
    return types.SimpleNamespace(
        Env=Env,
        Error=_Error,
        rescode=_Rescode,
        solsta=_Solsta,
        soltype=_Soltype,
        iparam=names(log="log", num_threads="num_threads"),
        dparam=names(optimizer_max_time="optimizer_max_time"),
        dinfitem=names(optimizer_time="optimizer_time"),
        iinfitem=names(intpnt_iter="intpnt_iter"),
        boundkey=names(fx="fx", fr="fr", up="up", lo="lo", ra="ra"),
        objsense=names(minimize="minimize"),
        streamtype=names(log="log"),
    )


def _small_qp() -> ProblemData:
    qp = {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
    }
    return ProblemData("test", "qp", QP, qp)


def _solve(monkeypatch, fake, tmp_path: Path):
    from solver_benchmarks.solvers.mosek_adapter import MosekSolverAdapter

    monkeypatch.setitem(sys.modules, "mosek", fake)
    return MosekSolverAdapter({"verbose": False}).solve(_small_qp(), tmp_path)


@pytest.mark.parametrize(
    "code",
    [
        _Rescode.err_license,
        _Rescode.err_license_expired,
        _Rescode.err_missing_license_file,
        _Rescode.err_license_max,
        _Rescode.err_flexlm,
    ],
)
def test_mosek_license_failure_is_an_unavailable_solver(code, monkeypatch, tmp_path):
    """A license failure in optimize() would otherwise be recorded as MOSEK
    failing every problem (solver_error), counted against it in success
    rates; it is an environment error (worker_error with the message)."""

    def optimize():
        raise _Error(code, "License cannot be located.")

    with pytest.raises(SolverUnavailable, match="MOSEK license error"):
        _solve(monkeypatch, _fake_mosek(optimize=optimize), tmp_path)


def test_mosek_other_errors_stay_solver_errors(monkeypatch, tmp_path):
    def optimize():
        raise _Error(_Rescode.err_space, "Out of space.")

    result = _solve(monkeypatch, _fake_mosek(optimize=optimize), tmp_path)
    assert result.status == status.SOLVER_ERROR
    assert "Out of space" in result.info["error"]


def test_mosek_optimal_reports_objective(monkeypatch, tmp_path):
    result = _solve(monkeypatch, _fake_mosek(optimize=lambda: _Rescode.ok), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == -1.0
    assert result.kkt is not None


@pytest.mark.parametrize(
    "termination, expected",
    [(_Rescode.trm_max_time, status.TIME_LIMIT),
     (_Rescode.trm_max_iterations, status.MAX_ITER_REACHED)],
)
def test_mosek_limit_keeps_kkt_but_reports_no_objective(
    termination, expected, monkeypatch, tmp_path
):
    fake = _fake_mosek(optimize=lambda: termination, solsta=_Solsta.unknown, x=(-0.9, -1.1))
    result = _solve(monkeypatch, fake, tmp_path)
    assert result.status == expected
    assert result.kkt is not None
    assert result.objective_value is None
