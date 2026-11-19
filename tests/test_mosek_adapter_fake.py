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


def _fake_mosek(
    *, optimize, solsta=_Solsta.optimal, x=(-1.0, -1.0), y=(0.0, 0.0), record=None, reject=()
):
    """A ``mosek`` stand-in whose ``Task.optimize`` runs ``optimize()``.

    Parameters and row bounds the adapter sets go to ``record``; a parameter
    named in ``reject`` is refused with ``mosek.Error``.
    """
    record = {} if record is None else record
    record.setdefault("params", {})
    record.setdefault("bounds", [])

    def put_param(name, value):
        if name in reject:
            raise _Error(_Rescode.err_space, f"{name} is not a parameter")
        record["params"][name] = value

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

        def putintparam(self, param, value):
            put_param(param, value)

        putdouparam = putintparam

        def putnaintparam(self, name, value):
            put_param(name, value)

        putnadouparam = putnastrparam = putnaintparam

        def putconbound(self, row, key, lower, upper):
            record["bounds"].append((key, lower, upper))

        def solutiondef(self, soltype):
            return soltype is _Soltype.itr

        def getsolsta(self, soltype):
            return solsta

        def getxx(self, soltype):
            return list(x)

        def gety(self, soltype):
            return list(y)

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


# ---------------------------------------------------------------------------
# the shared settings


def _solve_with(monkeypatch, fake, tmp_path: Path, settings, qp=None):
    from solver_benchmarks.solvers.mosek_adapter import MosekSolverAdapter

    monkeypatch.setitem(sys.modules, "mosek", fake)
    problem = _small_qp() if qp is None else ProblemData("test", "qp", QP, qp)
    return MosekSolverAdapter({"verbose": False, **settings}).solve(problem, tmp_path)


def test_mosek_translates_max_iter_to_both_iteration_limits(monkeypatch, tmp_path):
    # Was "Invalid MOSEK parameter 'max_iter'" while Gurobi, CPLEX and HiGHS
    # translate it.
    record: dict = {}
    fake = _fake_mosek(optimize=lambda: _Rescode.ok, record=record)
    result = _solve_with(monkeypatch, fake, tmp_path, {"max_iter": 50})
    assert record["params"]["MSK_IPAR_INTPNT_MAX_ITERATIONS"] == 50
    assert record["params"]["MSK_IPAR_SIM_MAX_ITERATIONS"] == 50
    assert result.info["settings_translated"] == {
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 50,
        "MSK_IPAR_SIM_MAX_ITERATIONS": 50,
    }


def test_mosek_native_iteration_limit_wins_over_max_iter(monkeypatch, tmp_path):
    record: dict = {}
    fake = _fake_mosek(optimize=lambda: _Rescode.ok, record=record)
    result = _solve_with(
        monkeypatch, fake, tmp_path, {"max_iter": 50, "MSK_IPAR_INTPNT_MAX_ITERATIONS": 7}
    )
    assert record["params"]["MSK_IPAR_INTPNT_MAX_ITERATIONS"] == 7
    assert result.info["settings_translated"] == {"MSK_IPAR_SIM_MAX_ITERATIONS": 50}


@pytest.mark.parametrize("settings", [{"eps": 1e-6}, {"eps_abs": 1e-6, "eps_rel": 1e-6}])
def test_mosek_refuses_eps_aliases_naming_its_tolerances(monkeypatch, tmp_path, settings):
    from solver_benchmarks.solvers.base import SolverConfigError

    fake = _fake_mosek(optimize=lambda: _Rescode.ok)
    with pytest.raises(SolverConfigError, match="MSK_DPAR_INTPNT_QO_TOL_REL_GAP"):
        _solve_with(monkeypatch, fake, tmp_path, settings)


def test_mosek_parameter_it_refuses_is_a_configuration_error(monkeypatch, tmp_path):
    from solver_benchmarks.solvers.base import SolverConfigError

    fake = _fake_mosek(optimize=lambda: _Rescode.ok, reject={"MSK_IPAR_BOGUS"})
    with pytest.raises(SolverConfigError, match="MSK_IPAR_BOGUS"):
        _solve_with(monkeypatch, fake, tmp_path, {"MSK_IPAR_BOGUS": 1})


@pytest.mark.parametrize("settings, expected", [
    ({}, 1),
    ({"threads": 0}, 0),
    ({"threads": 4}, 4),
    ({"MSK_IPAR_NUM_THREADS": 2}, 2),
])
def test_mosek_threads_default_to_one(monkeypatch, tmp_path, settings, expected):
    record: dict = {}
    fake = _fake_mosek(optimize=lambda: _Rescode.ok, record=record)
    result = _solve_with(monkeypatch, fake, tmp_path, settings)
    assert result.info["threads"] == expected
    set_values = [
        value for name, value in record["params"].items()
        if name in ("num_threads", "MSK_IPAR_NUM_THREADS")
    ]
    assert set_values == [expected]


def test_mosek_uses_the_shared_infinity_cutoff(monkeypatch, tmp_path):
    # MOSEK kept its own strict 1e20 cutoff, so a near-sentinel bound became
    # a finite side of a ranged row.
    record: dict = {}
    fake = _fake_mosek(optimize=lambda: _Rescode.ok, record=record)
    qp = _small_qp().data
    qp = dict(qp, l=np.array([-9.999999999999998e19, -5.0]), u=np.array([5.0, 9.999999999999998e19]))
    _solve_with(monkeypatch, fake, tmp_path, {}, qp)
    assert [key for key, _, _ in record["bounds"]] == ["up", "lo"]


def test_mosek_nan_point_has_no_kkt(monkeypatch, tmp_path):
    fake = _fake_mosek(
        optimize=lambda: _Rescode.trm_max_time, solsta=_Solsta.unknown, x=(float("nan"), -1.0)
    )
    result = _solve_with(monkeypatch, fake, tmp_path, {})
    assert result.status == status.TIME_LIMIT
    assert result.kkt is None


# MOSEK reports a primal infeasibility certificate as the dual solution, with
# y = s_l^c - s_u^c, A'y = 0 and (l^c)'s_l^c - (u^c)'s_u^c > 0, and a dual
# infeasibility certificate as the primal solution (c'x < 0, Ax in the
# recession cone); see its "Infeasibility for linear optimization".


def _infeasible_lp():
    # Rows: x1 + x2 <= 1, x1 >= 1, a free row, x2 >= 1. MOSEK's certificate
    # puts s_u^c = 1 on the first row and s_l^c = 1 on the others: y = (-1,
    # 1, 0, 1).
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [5.0, 7.0], [0.0, 1.0]]),
        "l": np.array([-np.inf, 1.0, -np.inf, 1.0]),
        "u": np.array([1.0, np.inf, np.inf, np.inf]),
    }


def _unbounded_qp():
    # min x1^2 / 2 - x2 with x2 >= 0, -1 <= x1 <= 1: the ray (0, 1), with Px = 0.
    return {
        "P": sp.csc_matrix(np.diag([1.0, 0.0])),
        "q": np.array([0.0, -1.0]),
        "A": sp.csc_matrix([[0.0, 1.0], [1.0, 0.0]]),
        "l": np.array([0.0, -1.0]),
        "u": np.array([np.inf, 1.0]),
    }


@pytest.mark.parametrize("make, solsta, vectors, claim", [
    (_infeasible_lp, _Solsta.prim_infeas_cer, {"y": (-1.0, 1.0, 0.0, 1.0), "x": (0.0, 0.0)},
     status.PRIMAL_INFEASIBLE),
    (_unbounded_qp, _Solsta.dual_infeas_cer, {"x": (0.0, 1.0), "y": (0.0, 0.0)}, status.DUAL_INFEASIBLE),
])
@pytest.mark.parametrize("sign, correct", [(1.0, True), (-1.0, False)], ids=["documented sign", "flipped"])
def test_mosek_records_its_certificate(make, solsta, vectors, claim, sign, correct, monkeypatch, tmp_path):
    from solver_benchmarks.analysis.verify import classify

    vectors = {key: tuple(sign * v for v in value) for key, value in vectors.items()}
    fake = _fake_mosek(optimize=lambda: _Rescode.ok, solsta=solsta, **vectors)
    result = _solve_with(monkeypatch, fake, tmp_path, {}, make())
    assert result.status == claim
    assert result.objective_value is None
    assert result.kkt is not None and result.kkt["certificate"] == claim
    verdict = classify({"status": result.status, "kkt": result.kkt}, tolerance=lambda: 1e-6)
    if correct:
        assert (verdict.verdict, verdict.status) == ("kept", claim), verdict.reason
    else:
        assert (verdict.verdict, verdict.status) == ("demoted", f"{claim}_inaccurate")


@pytest.mark.parametrize("make, solsta, vectors, claim", [
    (_infeasible_lp, _Solsta.prim_infeas_cer, {"y": (-1.0, np.nan, 0.0, 1.0)}, status.PRIMAL_INFEASIBLE),
    (_unbounded_qp, _Solsta.dual_infeas_cer, {"x": (0.0, np.inf)}, status.DUAL_INFEASIBLE),
])
def test_mosek_certificate_that_is_not_finite_says_why(make, solsta, vectors, claim, monkeypatch, tmp_path):
    fake = _fake_mosek(optimize=lambda: _Rescode.ok, solsta=solsta, **vectors)
    result = _solve_with(monkeypatch, fake, tmp_path, {}, make())
    assert result.status == claim
    assert result.kkt is None
    assert result.info["certificate_unavailable"] == "MOSEK's certificate is not finite"


_INACCURATE = {status.PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE_INACCURATE,
               status.DUAL_INFEASIBLE: status.DUAL_INFEASIBLE_INACCURATE}


@pytest.mark.parametrize("make, solsta, vectors, claim", [
    (_infeasible_lp, _Solsta.prim_infeas_cer, {"y": (-1.0, np.nan, 0.0, 1.0)}, status.PRIMAL_INFEASIBLE_INACCURATE),
    (_unbounded_qp, _Solsta.dual_infeas_cer, {"x": (0.0, np.inf)}, status.DUAL_INFEASIBLE_INACCURATE),
    (_infeasible_lp, _Solsta.prim_infeas_cer, {"y": (-1.0, 1.0, 0.0, 1.0)}, status.PRIMAL_INFEASIBLE_INACCURATE),
    (_unbounded_qp, _Solsta.dual_infeas_cer, {"x": (0.0, 1.0)}, status.DUAL_INFEASIBLE_INACCURATE),
], ids=["primal, not finite", "dual, not finite", "primal", "dual"])
def test_mosek_inaccurate_claim_is_read_like_an_accurate_one(make, solsta, vectors, claim, monkeypatch, tmp_path):
    # No MOSEK status maps to an inaccurate certificate today; if one ever
    # does, its certificate is read and checked the same way, and one that is
    # not finite says why.
    from solver_benchmarks.solvers import mosek_adapter

    accurate = mosek_adapter._map_mosek_status
    monkeypatch.setattr(mosek_adapter, "_map_mosek_status",
                        lambda *args: (lambda mapped: _INACCURATE.get(mapped, mapped))(accurate(*args)))
    fake = _fake_mosek(optimize=lambda: _Rescode.ok, solsta=solsta, **vectors)
    result = _solve_with(monkeypatch, fake, tmp_path, {}, make())
    assert result.status == claim
    finite = all(np.isfinite(value).all() for value in vectors.values())
    if finite:
        assert result.kkt is not None and "certificate_unavailable" not in result.info
    else:
        assert result.kkt is None
        assert result.info["certificate_unavailable"] == "MOSEK's certificate is not finite"


class _Unreadable:
    """A solution vector MOSEK fails to return."""

    def __iter__(self):
        raise RuntimeError("solution not available")


@pytest.mark.parametrize("make, solsta, vectors, claim", [
    (_infeasible_lp, _Solsta.prim_infeas_cer, {"y": _Unreadable()}, status.PRIMAL_INFEASIBLE),
    (_unbounded_qp, _Solsta.dual_infeas_cer, {"x": _Unreadable()}, status.DUAL_INFEASIBLE),
])
def test_mosek_certificate_that_cannot_be_read_keeps_the_claim(make, solsta, vectors, claim, monkeypatch, tmp_path):
    # The claim is already made: an error reading its certificate costs the
    # certificate, not the row.
    fake = _fake_mosek(optimize=lambda: _Rescode.ok, solsta=solsta, **vectors)
    result = _solve_with(monkeypatch, fake, tmp_path, {}, make())
    assert result.status == claim
    assert result.kkt is None
    assert result.info["certificate_unavailable"] == "no certificate from MOSEK: solution not available"


def test_mosek_fake_is_available_and_a_broken_install_is_not(monkeypatch, tmp_path):
    import builtins

    from solver_benchmarks.solvers.base import SolverImportWarning
    from solver_benchmarks.solvers.mosek_adapter import MosekSolverAdapter

    monkeypatch.setitem(sys.modules, "mosek", types.ModuleType("mosek"))
    assert MosekSolverAdapter.is_available() is True

    real_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "mosek":
            raise OSError("libmosek64.so.10.1: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.warns(SolverImportWarning, match="libmosek64"):
        assert MosekSolverAdapter.is_available() is False
    with pytest.raises(SolverUnavailable, match="libmosek64"):
        MosekSolverAdapter({"verbose": False}).solve(_small_qp(), tmp_path)
