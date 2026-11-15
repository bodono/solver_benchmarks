"""Settings handling of the QP adapters.

A misspelled or unsupported setting must fail loudly (ValueError naming it)
rather than leave the solver on its defaults while the manifest records the
intended value, and the shared aliases (``eps`` / ``eps_abs`` / ``eps_rel``,
``max_iter``, ``time_limit``, ``threads``) are either translated to the
solver's own option or refused explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.solvers import get_solver


def _small_qp():
    return {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
    }


def _small_lp():
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([-1.0, -1.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([-np.inf, 0.0, 0.0]),
        "u": np.array([1.0, np.inf, np.inf]),
    }


def _solve(solver_name, settings, tmp_path: Path, qp=None):
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} not installed")
    problem = ProblemData("test", "p", QP, qp if qp is not None else _small_qp())
    return adapter_cls({"verbose": False, **settings}).solve(problem, tmp_path)


# --- HiGHS ------------------------------------------------------------------


@pytest.mark.parametrize("settings, name", [
    ({"time_limt": 5.0}, "time_limt"),
    ({"presolve": "of"}, "presolve"),
    ({"primal_feasibilty_tolerance": 1e-3}, "primal_feasibilty_tolerance"),
    ({"primal_feasibility_tolerance": "abc"}, "primal_feasibility_tolerance"),
])
def test_highs_rejects_unknown_or_invalid_options(settings, name, tmp_path):
    # highspy reports these by returning HighsStatus.kError, not by raising;
    # the adapter used to ignore the return value and solve at the defaults.
    with pytest.raises(ValueError, match=name):
        _solve("highs", settings, tmp_path)


@pytest.mark.parametrize("alias", ["eps", "eps_abs", "eps_rel"])
def test_highs_refuses_eps_aliases_and_names_its_own_tolerances(alias, tmp_path):
    with pytest.raises(ValueError, match="primal_feasibility_tolerance"):
        _solve("highs", {alias: 1e-9}, tmp_path)


@pytest.mark.parametrize("value", [-1, 0, "x"])
def test_highs_time_limit_goes_through_the_shared_validation(value, tmp_path):
    with pytest.raises(ValueError, match="time_limit"):
        _solve("highs", {"time_limit_sec": value}, tmp_path)


def test_highs_forwards_time_limit_aliases_as_seconds(tmp_path, monkeypatch):
    highspy = pytest.importorskip("highspy")
    solver = highspy.Highs()
    monkeypatch.setattr(highspy, "Highs", lambda: solver)
    # An int used to reach HiGHS's double-valued option and be rejected.
    result = _solve("highs", {"time_limit_secs": 7}, tmp_path)
    assert result.status == status.OPTIMAL
    assert solver.getOptionValue("time_limit")[1] == 7.0


def test_highs_forwards_threads_and_checks_the_result():
    from solver_benchmarks.solvers.highs_adapter import _configure_highs

    calls = {}

    class FakeHighs:
        def setOptionValue(self, key, value):
            calls[key] = value
            if key == "threads" and value > 64:
                return type("Status", (), {"name": "kError"})()
            return None

    _configure_highs(FakeHighs(), {"num_threads": 2})
    assert calls == {"threads": 2}
    with pytest.raises(ValueError, match="threads"):
        _configure_highs(FakeHighs(), {"threads": 128})


# --- eps alias and threads ---------------------------------------------------


@pytest.mark.parametrize("solver_name", ["osqp", "piqp", "proxqp", "pdlp"])
def test_eps_alias_sets_both_native_tolerances(solver_name, tmp_path):
    result = _solve(solver_name, {"eps": 1e-7}, tmp_path, _small_lp())
    assert result.status == status.OPTIMAL
    assert result.info["settings_translated"] == {"eps_abs": 1e-7, "eps_rel": 1e-7}


@pytest.mark.parametrize("solver_name", ["osqp", "piqp", "proxqp", "pdlp"])
def test_eps_alias_does_not_override_explicit_tolerances(solver_name, tmp_path):
    result = _solve(solver_name, {"eps": 1e-7, "eps_abs": 1e-9}, tmp_path, _small_lp())
    assert result.info["settings_translated"] == {"eps_rel": 1e-7}


def test_proxqp_rejects_unknown_settings_by_name(tmp_path):
    with pytest.raises(ValueError, match="eps_abss"):
        _solve("proxqp", {"eps_abss": 1e-6}, tmp_path)


@pytest.mark.parametrize("key", ["threads", "num_threads"])
def test_piqp_forwards_threads_to_num_threads(key, tmp_path, monkeypatch):
    pytest.importorskip("piqp")
    from solver_benchmarks.solvers import piqp_adapter

    seen = {}
    real = piqp_adapter._configure_piqp

    def recording(solver, settings):
        seen.update(settings)
        return real(solver, settings)

    monkeypatch.setattr(piqp_adapter, "_configure_piqp", recording)
    result = _solve("piqp", {key: 1}, tmp_path)
    assert seen["num_threads"] == 1
    assert "threads_ignored" not in result.info


def test_pdlp_forwards_threads_to_num_threads(tmp_path):
    pytest.importorskip("ortools")
    from solver_benchmarks.solvers.pdlp_adapter import _pdlp_parameters_from_settings

    settings = {"threads": 4}
    parameters = _pdlp_parameters_from_settings(settings)
    assert parameters.num_threads == 4
    assert settings == {}
    result = _solve("pdlp", {"threads": 2}, tmp_path, _small_lp())
    assert result.status == status.OPTIMAL
    assert "threads_ignored" not in result.info
