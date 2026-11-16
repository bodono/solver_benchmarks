"""Rules every installed adapter follows, walked over the registry.

- A setting name an adapter does not know, or a value it (or its solver)
  rejects, raises ``SolverConfigError`` (a ``ValueError``), never a solver
  failure or another exception type.
- ``info["raw_status"]`` is the solver's own status as a string.
- ``info["threads"]`` is the thread count the solver was given: one by default
  for solvers with a thread setting, ``None`` where the adapter cannot set it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.solvers import get_solver, list_solvers
from solver_benchmarks.solvers.base import SolverConfigError

# Commercial and GPU solvers are covered by the fake-module tests.
_FAKE_TESTED = {"gurobi", "cplex", "mosek", "cuopt"}
_ADAPTERS = [name for name in list_solvers() if name not in _FAKE_TESTED]


def _problem(solver_name: str) -> ProblemData:
    if solver_name == "sdpa":
        # min x0 + x1 s.t. x0 + x1 >= 1, x >= 0 in cone form.
        cone = {
            "P": None,
            "q": np.array([1.0, 1.0]),
            "A": sp.csc_matrix([[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]]),
            "b": np.array([-1.0, 0.0, 0.0]),
            "r": 0.0,
            "cone": {"l": 3},
        }
        return ProblemData("test", "lp", CONE, cone)
    qp = {
        "P": sp.csc_matrix((2, 2)) if solver_name == "pdlp" else sp.eye(2, format="csc"),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([1.0, 0.0, 0.0]),
        "u": np.array([np.inf, 5.0, 5.0]),
    }
    return ProblemData("test", "qp", QP, qp)


def _solve(solver_name: str, settings: dict, tmp_path: Path):
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} not installed")
    artifacts = tmp_path / solver_name
    artifacts.mkdir(parents=True, exist_ok=True)
    return adapter_cls({"verbose": False, **settings}).solve(_problem(solver_name), artifacts)


@pytest.mark.parametrize("solver_name", _ADAPTERS)
def test_unknown_setting_name_is_a_config_error(solver_name, tmp_path):
    # SCS and ECOS raised TypeError from the solver; the others a plain ValueError.
    with pytest.raises(SolverConfigError, match="definitely_not_a_setting"):
        _solve(solver_name, {"definitely_not_a_setting": 1}, tmp_path)


# One value each adapter refuses. A negative absolute tolerance is refused by
# every adapter that has one (HiGHS refuses the alias itself).
_BAD_VALUES = {name: {"eps_abs": -1.0} for name in _ADAPTERS}
_BAD_VALUES["highs"] = {"primal_feasibility_tolerance": -1.0}


@pytest.mark.parametrize("solver_name", _ADAPTERS)
def test_bad_setting_value_is_a_config_error(solver_name, tmp_path):
    with pytest.raises(SolverConfigError):
        _solve(solver_name, _BAD_VALUES[solver_name], tmp_path)


# A value of the wrong type. OSQP's binding raised a bare TypeError for it,
# Clarabel's and HiGHS's too for theirs.
_WRONG_TYPES = {name: {"eps_abs": "abc"} for name in _ADAPTERS}
_WRONG_TYPES["highs"] = {"presolve": [1]}
_WRONG_TYPES["clarabel"] = {"max_iter": "many"}
_WRONG_TYPES["osqp"] = {"polishing": "yes"}


@pytest.mark.parametrize("solver_name", _ADAPTERS)
def test_wrong_typed_setting_value_is_a_config_error(solver_name, tmp_path):
    with pytest.raises(SolverConfigError):
        _solve(solver_name, _WRONG_TYPES[solver_name], tmp_path)


@pytest.mark.parametrize("solver_name", _ADAPTERS)
def test_max_iter_follows_one_integer_rule(solver_name, tmp_path):
    # YAML reads max_iter: 1e3 as 1000.0 (a TypeError in OSQP); 1.5 and true
    # were truncated to 1 by HiGHS, PDLP, cpo3 and ECOS; null was an error in
    # some adapters and unset in others.
    assert _solve(solver_name, {"max_iter": 1e3}, tmp_path).status == status.OPTIMAL
    assert _solve(solver_name, {"max_iter": np.int64(1000)}, tmp_path).status == status.OPTIMAL
    assert _solve(solver_name, {"max_iter": None}, tmp_path).status == status.OPTIMAL
    for value in (1.5, True, "1000", float("inf"), -3):
        with pytest.raises(SolverConfigError, match="max_iter"):
            _solve(solver_name, {"max_iter": value}, tmp_path)


@pytest.mark.parametrize("alias", ["eps", "eps_abs", "eps_rel"])
@pytest.mark.parametrize("solver_name", _ADAPTERS)
def test_boolean_tolerance_alias_is_a_config_error(solver_name, alias, tmp_path):
    # OSQP, PIQP and PDLP took eps_abs: true as a tolerance of 1.0.
    with pytest.raises(SolverConfigError, match=alias):
        _solve(solver_name, {alias: True}, tmp_path)


@pytest.mark.parametrize("value", [True, "2", 2.5])
def test_threads_follow_the_integer_rule(value, tmp_path):
    with pytest.raises(SolverConfigError, match="threads"):
        _solve("osqp", {"threads": value}, tmp_path)


@pytest.mark.parametrize("solver_name", _ADAPTERS)
def test_shared_alias_errors_are_config_errors(solver_name, tmp_path):
    with pytest.raises(SolverConfigError, match="time_limit"):
        _solve(solver_name, {"time_limit": 0}, tmp_path)
    with pytest.raises(SolverConfigError, match="threads"):
        _solve(solver_name, {"threads": -1}, tmp_path)


@pytest.mark.parametrize("solver_name", _ADAPTERS)
def test_raw_status_is_a_string(solver_name, tmp_path):
    # PIQP, ProxQP and PDLP had no raw_status (cuOpt and the commercial
    # adapters are checked with their fakes).
    result = _solve(solver_name, {}, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert isinstance(result.info["raw_status"], str) and result.info["raw_status"]


# Adapters whose solver has a thread setting, and the value in effect for
# ``threads: 0`` (the solver's own default; PDLP's is one thread).
_THREAD_KNOBS = {"cpo3": 0, "clarabel": 0, "highs": 0, "piqp": 0, "pdlp": 1, "sdpa": 0}


@pytest.mark.parametrize("solver_name", _ADAPTERS)
def test_threads_default_to_one_and_are_recorded(solver_name, tmp_path):
    # cpo3, Clarabel, HiGHS and PIQP used every core unless threads was set,
    # while SDPA was pinned to one thread.
    result = _solve(solver_name, {}, tmp_path)
    if solver_name in _THREAD_KNOBS:
        assert result.info["threads"] == 1
        assert _solve(solver_name, {"threads": 2}, tmp_path).info["threads"] == 2
        zero = _solve(solver_name, {"threads": 0}, tmp_path)
        assert zero.info["threads"] == _THREAD_KNOBS[solver_name]
    elif solver_name == "scs" and result.info["threads"] is not None:
        pytest.skip("this SCS build has OpenMP threads")
    else:
        assert result.info["threads"] is None
        ignored = _solve(solver_name, {"threads": 2}, tmp_path)
        assert ignored.info["threads"] is None
        assert ignored.info["threads_ignored"] is True


def test_threads_reach_the_solvers(tmp_path, monkeypatch):
    clarabel = pytest.importorskip("clarabel")
    seen = {}
    real_solver = clarabel.DefaultSolver

    def recording_solver(p, q, a, b, cones, settings):
        seen["max_threads"] = settings.max_threads
        return real_solver(p, q, a, b, cones, settings)

    monkeypatch.setattr(clarabel, "DefaultSolver", recording_solver)
    _solve("clarabel", {}, tmp_path)
    assert seen["max_threads"] == 1
    _solve("clarabel", {"max_threads": 3}, tmp_path)
    assert seen["max_threads"] == 3


def test_cpo3_threads_reach_the_solver(tmp_path, monkeypatch):
    cpo3 = pytest.importorskip("cpo3")
    seen = []
    real_settings = cpo3.Settings

    def recording_settings(**kwargs):
        seen.append(kwargs.get("threads"))
        return real_settings(**kwargs)

    monkeypatch.setattr(cpo3, "Settings", recording_settings)
    monkeypatch.setattr(recording_settings, "__dataclass_fields__", real_settings.__dataclass_fields__, raising=False)
    _solve("cpo3", {}, tmp_path)
    _solve("cpo3", {"threads": 0}, tmp_path)
    assert seen == [1, 0]


def test_scs_nan_iterate_has_no_kkt(tmp_path, monkeypatch):
    # A non-finite iterate gave a KKT record of NaNs in SCS, Clarabel, ECOS,
    # CVXOPT, cpo3, HiGHS and the commercial adapters.
    scs = pytest.importorskip("scs")
    real_solve = scs.solve

    def nan_solve(data, cone, **settings):
        raw = real_solve(data, cone, **settings)
        raw["x"] = np.full_like(raw["x"], np.nan)
        return raw

    monkeypatch.setattr(scs, "solve", nan_solve)
    result = _solve("scs", {"max_iters": 2}, tmp_path)
    assert result.status == status.MAX_ITER_REACHED
    assert result.kkt is None


@pytest.mark.parametrize("solver_name, key", [
    ("scs", "max_iters"),
    ("cpo3", "max_iters"),
    ("ecos", "max_iters"),
    ("pdlp", "iteration_limit"),
    ("clarabel", "max_threads"),
])
def test_native_iteration_and_thread_names_follow_the_integer_rule(solver_name, key, tmp_path):
    # 1000.0 was refused while max_iter: 1000.0 was accepted, and true was 1.
    value = 2.0 if key == "max_threads" else 1000.0
    assert _solve(solver_name, {key: value}, tmp_path).status == status.OPTIMAL
    for bad in (True, 2.5, "1000", float("inf")):
        with pytest.raises(SolverConfigError, match=key):
            _solve(solver_name, {key: bad}, tmp_path)
