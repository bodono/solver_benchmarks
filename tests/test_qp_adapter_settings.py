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


@pytest.mark.parametrize("option", [
    "simplex_iteration_limit", "ipm_iteration_limit", "qp_iteration_limit", "pdlp_iteration_limit",
])
def test_highs_iteration_limits_follow_the_integer_rule(option, tmp_path, monkeypatch):
    # highspy refused YAML's 1e3 (a float) and parsed the string "10".
    highspy = pytest.importorskip("highspy")
    from solver_benchmarks.solvers.base import SolverConfigError

    seen = {}
    original = highspy.Highs.setOptionValue

    def record(self, key, value):
        seen[key] = value
        return original(self, key, value)

    monkeypatch.setattr(highspy.Highs, "setOptionValue", record)
    assert _solve("highs", {option: 1e3}, tmp_path).status == status.OPTIMAL
    assert seen[option] == 1000 and isinstance(seen[option], int)
    for value in (True, "10", 2.5, -1):
        with pytest.raises(SolverConfigError, match=option):
            _solve("highs", {option: value}, tmp_path)


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


@pytest.mark.parametrize("key", ["presolve", "run_crossover", "parallel"])
@pytest.mark.parametrize("value, word", [(False, "off"), (True, "on")])
def test_highs_reads_yaml_booleans_on_its_on_off_options(key, value, word, tmp_path, monkeypatch):
    # YAML 1.1 reads an unquoted `presolve: off` as False; HiGHS's on/off
    # options are strings and refuse a bool.
    highspy = pytest.importorskip("highspy")
    solver = highspy.Highs()
    monkeypatch.setattr(highspy, "Highs", lambda: solver)
    result = _solve("highs", {key: value}, tmp_path)
    assert result.status == status.OPTIMAL
    assert solver.getOptionValue(key)[1] == word


def test_highs_boolean_for_a_string_option_says_why_it_failed(tmp_path):
    with pytest.raises(ValueError, match="'solver'='on'.*unquoted on/off"):
        _solve("highs", {"solver": True}, tmp_path)


def test_highs_boolean_for_a_numeric_option_is_still_refused(tmp_path):
    with pytest.raises(ValueError, match="'random_seed'=True"):
        _solve("highs", {"random_seed": True}, tmp_path)


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
    expected = {"eps_abs": 1e-7, "eps_rel": 1e-7}
    if solver_name == "piqp":
        # PIQP's duality-gap tolerances follow the configured eps_abs / eps_rel.
        expected.update(eps_duality_gap_abs=1e-7, eps_duality_gap_rel=1e-7)
    assert result.info["settings_translated"] == expected


@pytest.mark.parametrize("solver_name", ["osqp", "piqp", "proxqp", "pdlp"])
def test_eps_alias_does_not_override_explicit_tolerances(solver_name, tmp_path):
    result = _solve(solver_name, {"eps": 1e-7, "eps_abs": 1e-9}, tmp_path, _small_lp())
    expected = {"eps_rel": 1e-7}
    if solver_name == "piqp":
        expected.update(eps_duality_gap_abs=1e-9, eps_duality_gap_rel=1e-7)
    assert result.info["settings_translated"] == expected


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


# --- ProxQP duality gap -------------------------------------------------------


def _one_variable_lp():
    # min x s.t. 1 <= x <= 2: optimum 1.
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[1.0]]),
        "l": np.array([1.0]),
        "u": np.array([2.0]),
    }


def test_proxqp_checks_the_duality_gap_by_default(tmp_path):
    # ProxSuite's own default tests only the primal and dual residuals, and
    # stopped here at x = 1.1 (10% off) calling it solved; Maros-Meszaros
    # QPTEST came back "optimal" 28% off.
    result = _solve("proxqp", {}, tmp_path, _one_variable_lp())
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)
    assert result.info["check_duality_gap"] is True
    # Derived from ProxQP's own eps_abs / eps_rel defaults (1e-5 and 0); with
    # eps_rel 0, eps_abs is the relative gap tolerance too.
    assert result.info["eps_duality_gap_abs"] == pytest.approx(1e-5)
    assert result.info["eps_duality_gap_rel"] == pytest.approx(1e-5)


def test_proxqp_duality_gap_tolerances_follow_eps(tmp_path):
    result = _solve("proxqp", {"eps_abs": 1e-7, "eps_rel": 1e-6}, tmp_path, _one_variable_lp())
    assert result.info["eps_duality_gap_abs"] == 1e-7
    assert result.info["eps_duality_gap_rel"] == 1e-6
    result = _solve("proxqp", {"eps": 1e-8, "eps_duality_gap_rel": 1e-3}, tmp_path, _one_variable_lp())
    assert result.info["eps_duality_gap_abs"] == 1e-8
    assert result.info["eps_duality_gap_rel"] == 1e-3
    # An explicit zero relative gap tolerance is kept.
    result = _solve("proxqp", {"eps_abs": 1e-9, "eps_duality_gap_rel": 0.0}, tmp_path, _one_variable_lp())
    assert result.info["eps_duality_gap_rel"] == 0.0


@pytest.mark.parametrize("settings", [{"eps_abs": 1e-9}, {"eps_abs": 1e-9, "eps_rel": 0.0}])
def test_proxqp_relative_gap_tolerance_falls_back_to_eps_abs(settings, tmp_path):
    # An absolute-only gap test at 1e-9 is below double precision once the
    # objective is large: Maros-Meszaros QSCAGR25 (objective 2e8) ran to
    # max_iter and QPCBOEI1 ended "primal infeasible"; with the relative part
    # at eps_abs both solve (QSCAGR25 in the 804 iterations it took without
    # the gap check).
    result = _solve("proxqp", settings, tmp_path, _one_variable_lp())
    assert result.status == status.OPTIMAL
    assert result.info["eps_duality_gap_abs"] == 1e-9
    assert result.info["eps_duality_gap_rel"] == 1e-9


def test_proxqp_duality_gap_check_can_be_turned_off(tmp_path):
    result = _solve("proxqp", {"check_duality_gap": False}, tmp_path, _one_variable_lp())
    assert result.status == status.OPTIMAL
    assert result.info["check_duality_gap"] is False
    assert "eps_duality_gap_abs" not in result.info


# --- max_iter translation is recorded -------------------------------------------


@pytest.mark.parametrize("solver_name, expected", [
    ("cpo3", {"max_iters": 77}),
    ("highs", {
        "simplex_iteration_limit": 77,
        "ipm_iteration_limit": 77,
        "qp_iteration_limit": 77,
        "pdlp_iteration_limit": 77,
    }),
    ("pdlp", {"iteration_limit": 77}),
    ("scs", {"max_iters": 77}),
    ("ecos", {"max_iters": 77}),
    ("cvxopt", {"maxiters": 77}),
])
def test_max_iter_translation_is_recorded(solver_name, expected, tmp_path):
    # cpo3, HiGHS and PDLP renamed max_iter without saying so on info.
    result = _solve(solver_name, {"max_iter": 77}, tmp_path, _small_lp())
    assert result.info["settings_translated"] == expected


def test_highs_native_iteration_limit_wins_over_max_iter(tmp_path):
    result = _solve("highs", {"max_iter": 77, "qp_iteration_limit": 5}, tmp_path, _small_lp())
    assert "qp_iteration_limit" not in result.info["settings_translated"]


# --- HiGHS: refused model data, KKT gating, iteration count ---------------------


def test_highs_refused_rows_are_a_solver_error(tmp_path):
    # addRows refuses a coefficient >= large_matrix_value (1e15); the model
    # was solved without its rows and the adapter then died on a NumPy shape
    # error that named nothing.
    qp = _small_qp()
    qp["A"] = sp.csc_matrix([[1e15, 0.0], [0.0, 1.0]])
    result = _solve("highs", {}, tmp_path, qp)
    assert result.status == status.SOLVER_ERROR
    assert "addRows" in result.info["error"]
    assert result.objective_value is None and result.kkt is None


def test_highs_refused_hessian_is_a_solver_error(tmp_path):
    # A refused Hessian left an LP that was solved and reported as the QP.
    qp = _small_qp()
    qp["P"] = sp.csc_matrix(np.diag([1e300, 1.0]))
    result = _solve("highs", {}, tmp_path, qp)
    assert result.status == status.SOLVER_ERROR
    assert "passHessian" in result.info["error"]


@pytest.mark.parametrize("qp, expected", [
    ({"P": sp.csc_matrix([[1.0, 0.0], [0.0, 0.0]]), "q": np.array([0.0, -1.0]),
      "A": sp.eye(2, format="csc"), "l": np.array([-1.0, 0.0]), "u": np.array([1.0, np.inf])},
     status.DUAL_INFEASIBLE),
    ({"P": sp.eye(1, format="csc"), "q": np.array([1.0]), "A": sp.csc_matrix([[1.0], [1.0]]),
      "l": np.array([2.0, -np.inf]), "u": np.array([np.inf, 1.0])},
     status.PRIMAL_INFEASIBLE),
])
def test_highs_infeasibility_claims_carry_no_solution_residuals(qp, expected, tmp_path):
    # HiGHS has no checkable certificate; residuals of its last point were
    # recorded under certificate statuses.
    result = _solve("highs", {}, tmp_path, qp)
    assert result.status == expected
    assert result.kkt is None


def test_highs_iterations_are_those_of_the_finishing_algorithm(tmp_path):
    # A QP runs simplex for phase 1 first; the max over algorithms reported
    # 42 iterations for a solve stopped by qp_iteration_limit = 2.
    rng = np.random.default_rng(0)
    n, m = 20, 15
    g = rng.standard_normal((n, n))
    qp = {
        "P": sp.csc_matrix(g @ g.T / n + 0.1 * np.eye(n)),
        "q": rng.standard_normal(n),
        "A": sp.csc_matrix(np.vstack([rng.standard_normal((m, n)), np.eye(n)])),
        "l": np.concatenate([-rng.random(m), -np.ones(n)]),
        "u": np.concatenate([rng.random(m), np.ones(n)]),
    }
    result = _solve("highs", {"max_iter": 2}, tmp_path, qp)
    assert result.status == status.MAX_ITER_REACHED
    assert result.info["qp_iteration_count"] == 2
    assert result.iterations == 2


def test_highs_iteration_choice_without_a_qp_phase():
    from types import SimpleNamespace

    from solver_benchmarks.solvers.highs_adapter import _highs_iterations

    # More crossover iterations than interior-point ones: max() would pick 40.
    ipm_with_crossover = SimpleNamespace(
        qp_iteration_count=0, ipm_iteration_count=9, pdlp_iteration_count=0,
        simplex_iteration_count=40,
    )
    assert _highs_iterations(ipm_with_crossover) == 9
    simplex = SimpleNamespace(
        qp_iteration_count=-1, ipm_iteration_count=0, pdlp_iteration_count=0,
        simplex_iteration_count=6,
    )
    assert _highs_iterations(simplex) == 6
    presolved = SimpleNamespace(
        qp_iteration_count=0, ipm_iteration_count=0, pdlp_iteration_count=0,
        simplex_iteration_count=0,
    )
    assert _highs_iterations(presolved) == 0
    errored = SimpleNamespace(
        qp_iteration_count=-1, ipm_iteration_count=-1, pdlp_iteration_count=-1,
        simplex_iteration_count=-1,
    )
    assert _highs_iterations(errored) is None


def test_highs_thread_count_can_change_between_solves_in_one_process(tmp_path):
    # HiGHS sizes one thread pool per process on its first run; a later run
    # asking for another size failed.
    pytest.importorskip("highspy")
    for threads in (1, 2, 1):
        result = _solve("highs", {"threads": threads}, tmp_path)
        assert result.status == status.OPTIMAL, threads
        assert result.info["threads"] == threads


# --- PIQP duality-gap tolerances follow eps_abs / eps_rel -----------------------


def _piqp_settings_seen(monkeypatch):
    pytest.importorskip("piqp")
    from solver_benchmarks.solvers import piqp_adapter

    seen = {}
    real = piqp_adapter._configure_piqp

    def recording(solver, settings):
        seen.update(settings)
        return real(solver, settings)

    monkeypatch.setattr(piqp_adapter, "_configure_piqp", recording)
    return seen


def test_piqp_gap_tolerances_follow_the_configured_tolerances(tmp_path, monkeypatch):
    # PIQP closed the gap to its own 1e-8 / 1e-9 whatever eps_abs / eps_rel
    # said (+40% iterations at a nominal 1e-4 on Maros-Meszaros).
    seen = _piqp_settings_seen(monkeypatch)
    result = _solve("piqp", {"eps_abs": 1e-4, "eps_rel": 1e-4}, tmp_path)
    assert result.status == status.OPTIMAL
    assert seen["eps_duality_gap_abs"] == 1e-4 and seen["eps_duality_gap_rel"] == 1e-4
    assert result.info["settings_translated"] == {
        "eps_duality_gap_abs": 1e-4, "eps_duality_gap_rel": 1e-4,
    }


def test_piqp_gap_tolerances_explicit_values_win(tmp_path, monkeypatch):
    seen = _piqp_settings_seen(monkeypatch)
    result = _solve(
        "piqp", {"eps": 1e-5, "eps_duality_gap_abs": 1e-9, "eps_duality_gap_rel": 1e-10}, tmp_path
    )
    assert seen["eps_duality_gap_abs"] == 1e-9 and seen["eps_duality_gap_rel"] == 1e-10
    assert result.info["settings_translated"] == {"eps_abs": 1e-5, "eps_rel": 1e-5}


def test_piqp_keeps_its_gap_defaults_without_tolerances(tmp_path, monkeypatch):
    seen = _piqp_settings_seen(monkeypatch)
    result = _solve("piqp", {}, tmp_path)
    assert "eps_duality_gap_abs" not in seen and "eps_duality_gap_rel" not in seen
    assert "settings_translated" not in result.info


# --- PIQP / ProxQP backend values -----------------------------------------------


@pytest.mark.parametrize("solver_name", ["piqp", "proxqp"])
@pytest.mark.parametrize("settings", [{"backend": "bogus"}, {"dense": "false"}, {"dense": 1}])
def test_backend_values_are_validated(solver_name, settings, tmp_path):
    # backend: <anything> ran the sparse backend and dense: 'false' the dense one.
    from solver_benchmarks.solvers.base import SolverConfigError

    with pytest.raises(SolverConfigError):
        _solve(solver_name, settings, tmp_path)


@pytest.mark.parametrize("solver_name", ["piqp", "proxqp"])
@pytest.mark.parametrize("settings", [{"backend": "dense"}, {"backend": "Sparse"}, {"dense": True}])
def test_valid_backend_values_solve(solver_name, settings, tmp_path):
    assert _solve(solver_name, settings, tmp_path).status == status.OPTIMAL


# --- QTQP: a missing linear-solver backend is an unavailable solver -------------


def test_qtqp_missing_backend_module_is_solver_unavailable(tmp_path):
    pytest.importorskip("qtqp")
    try:
        import qdldl  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        pytest.skip("qdldl is installed")
    from solver_benchmarks.solvers.base import SolverUnavailable

    with pytest.raises(SolverUnavailable, match="qdldl"):
        _solve("qtqp", {"linear_solver": "qdldl"}, tmp_path)


@pytest.mark.parametrize("error", [
    ImportError("No module named 'qdldl'"),
    OSError("libcholmod.so.5: cannot open shared object file"),
])
def test_qtqp_backend_that_does_not_load_is_solver_unavailable(monkeypatch, tmp_path, error):
    # qtqp loads its linear-solver backend when the solve starts; a native
    # library that does not load raises OSError rather than ImportError.
    qtqp = pytest.importorskip("qtqp")
    from solver_benchmarks.solvers.base import SolverUnavailable

    def solve(self, **kwargs):
        raise error

    monkeypatch.setattr(qtqp.QTQP, "solve", solve)
    with pytest.raises(SolverUnavailable, match="backend could not be imported"):
        _solve("qtqp", {}, tmp_path)
