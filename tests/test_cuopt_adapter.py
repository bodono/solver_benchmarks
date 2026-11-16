"""cuOpt adapter tests against a fake ``cuopt.linear_programming`` module.

cuOpt needs an NVIDIA GPU, so CI cannot run it. These tests pin the parts of
the adapter that were only ever checked by hand on a GPU
(``tools/bench_modal.py`` ``probe_cuopt``): the QP objective mapping
``Q = P / 2``, the dual sign used for the harness KKT residuals, the
termination-status table, and option forwarding. If a cuOpt release changes
one of these conventions, update the fake to match ``probe_cuopt``'s output
and the adapter with it.
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
from solver_benchmarks.solvers import cuopt_adapter
from solver_benchmarks.solvers.cuopt_adapter import CuOptSolverAdapter, _map_status


class _Termination(enum.Enum):
    Optimal = 1
    PrimalInfeasible = 2
    TimeLimit = 4


class _FakeSolution:
    def __init__(self, termination, x, y, objective):
        self._termination = termination
        self._x = x
        self._y = y
        self._objective = objective

    def get_termination_status(self):
        return self._termination

    def get_primal_solution(self):
        return self._x

    def get_dual_solution(self):
        return self._y

    def get_primal_objective(self):
        return self._objective

    def get_dual_objective(self):
        return self._objective

    def get_solve_time(self):
        return 0.25

    def get_lp_stats(self):
        return {"nb_iterations": np.int64(17)}


class _FakeCuOpt:
    """Records what the adapter hands cuOpt and returns a canned solution."""

    def __init__(self, solution: _FakeSolution):
        self.solution = solution
        self.model_calls: dict[str, tuple] = {}
        self.parameters: dict[str, object] = {}
        self.optimality_tolerance = None
        fake = self

        class DataModel:
            def __getattr__(self, name):
                if not name.startswith("set_"):
                    raise AttributeError(name)

                def record(*args):
                    fake.model_calls[name] = tuple(np.asarray(a).copy() for a in args)

                return record

        class SolverSettings:
            def set_optimality_tolerance(self, value):
                fake.optimality_tolerance = value

            def set_parameter(self, key, value):
                fake.parameters[key] = value

        def solve(model, settings):
            return fake.solution

        self.module = types.SimpleNamespace(
            data_model=types.SimpleNamespace(DataModel=DataModel),
            solver=types.SimpleNamespace(Solve=solve),
            solver_settings=types.SimpleNamespace(SolverSettings=SolverSettings),
        )


@pytest.fixture
def install_cuopt(monkeypatch):
    def install(solution: _FakeSolution) -> _FakeCuOpt:
        fake = _FakeCuOpt(solution)
        package = types.ModuleType("cuopt")
        package.linear_programming = fake.module
        monkeypatch.setitem(sys.modules, "cuopt", package)
        monkeypatch.setitem(sys.modules, "cuopt.linear_programming", fake.module)
        return fake

    return install


def _problem(P, q, A, l, u) -> ProblemData:
    return ProblemData(
        dataset_id="unit", name="unit", kind=QP,
        data={"P": sp.csc_matrix(P), "q": np.asarray(q, float), "A": sp.csc_matrix(A),
              "l": np.asarray(l, float), "u": np.asarray(u, float), "r": 0.0, "n": len(q), "m": len(l)},
    )


@pytest.mark.parametrize("name, expected", sorted(cuopt_adapter._STATUS_MAP.items()))
def test_every_documented_termination_name_maps(name, expected):
    assert _map_status(name) == expected
    # Enum reprs such as ``PDLPTerminationStatus.Optimal`` map on the member name.
    assert _map_status(f"PDLPTerminationStatus.{name}") == expected


@pytest.mark.parametrize("name, expected", [
    ("Primal_Infeasible", status.PRIMAL_INFEASIBLE),
    ("timelimit", status.TIME_LIMIT),
    ("Feasible_Found", status.OPTIMAL_INACCURATE),
    ("NumericalError", status.SOLVER_ERROR),
    ("PRIMAL_FEASIBLE", status.SOLVER_ERROR),
    ("7", status.SOLVER_ERROR),
])
def test_termination_names_match_case_and_underscore_insensitively(name, expected):
    assert _map_status(name) == expected


def test_qp_objective_is_passed_as_half_of_p(install_cuopt, tmp_path: Path):
    # min x1^2 + 2 x2^2 - 2 x1 - 4 x2, i.e. P = diag(2, 4): optimum x = (1, 1).
    fake = install_cuopt(_FakeSolution(_Termination.Optimal, [1.0, 1.0], [0.0, 0.0], -3.0))
    problem = _problem(np.diag([2.0, 4.0]), [-2.0, -4.0], np.eye(2), [-10.0, -10.0], [10.0, 10.0])

    result = CuOptSolverAdapter({"verbose": False}).solve(problem, tmp_path)

    data, indices, indptr = fake.model_calls["set_quadratic_objective_matrix"]
    q_matrix = sp.csr_matrix((data, indices, indptr), shape=(2, 2)).toarray()
    # cuOpt minimizes c'x + (1/2) x'(Q + Q')x, so Q = P / 2 gives the harness objective.
    np.testing.assert_allclose(q_matrix, np.diag([1.0, 2.0]))
    np.testing.assert_allclose(fake.model_calls["set_objective_coefficients"][0], [-2.0, -4.0])
    np.testing.assert_allclose(fake.model_calls["set_variable_lower_bounds"][0], [-np.inf, -np.inf])
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(-3.0)
    assert result.iterations == 17
    assert result.kkt["dual_res_rel"] == pytest.approx(0.0, abs=1e-12)


def test_lp_sends_no_quadratic_objective(install_cuopt, tmp_path: Path):
    fake = install_cuopt(_FakeSolution(_Termination.Optimal, [1.0], [1.0], 1.0))
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])

    CuOptSolverAdapter({"verbose": False}).solve(problem, tmp_path)

    assert "set_quadratic_objective_matrix" not in fake.model_calls


def test_kkt_residuals_use_the_negated_cuopt_dual(install_cuopt, tmp_path: Path):
    # min x s.t. 1 <= x <= 10: x = 1 and, in the harness convention
    # (y = lambda_u - lambda_l), y = -1. cuOpt reports the opposite sign.
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])
    install_cuopt(_FakeSolution(_Termination.Optimal, [1.0], [1.0], 1.0))

    default = CuOptSolverAdapter({"verbose": False}).solve(problem, tmp_path)
    overridden = CuOptSolverAdapter({"verbose": False, "dual_sign": 1.0}).solve(problem, tmp_path)

    assert default.kkt["dual_res_rel"] == pytest.approx(0.0, abs=1e-12)
    assert overridden.kkt["dual_res_rel"] > 0.5


def test_infeasible_termination_reports_no_point(install_cuopt, tmp_path: Path):
    install_cuopt(_FakeSolution(_Termination.PrimalInfeasible, [0.0], [0.0], 0.0))
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])

    result = CuOptSolverAdapter({"verbose": False}).solve(problem, tmp_path)

    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.objective_value is None
    assert result.kkt is None


def test_options_are_forwarded_to_cuopt_parameters(install_cuopt, tmp_path: Path):
    fake = install_cuopt(_FakeSolution(_Termination.TimeLimit, [1.0], [1.0], 1.0))
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])

    result = CuOptSolverAdapter({
        "verbose": False, "eps": 1e-4, "eps_rel": 1e-6, "time_limit": 30, "threads": 2,
        "method": 1,
    }).solve(problem, tmp_path)

    assert fake.optimality_tolerance == pytest.approx(1e-4)
    for kind in ("primal", "dual", "gap"):
        assert fake.parameters[f"relative_{kind}_tolerance"] == pytest.approx(1e-6)
        assert f"absolute_{kind}_tolerance" not in fake.parameters
    assert fake.parameters["time_limit"] == pytest.approx(30.0)
    assert fake.parameters["num_cpu_threads"] == 2
    assert fake.parameters["method"] == 1
    assert result.status == status.TIME_LIMIT


@pytest.mark.parametrize("alias", ["eps", "eps_abs", "eps_rel"])
def test_boolean_tolerance_alias_is_refused(install_cuopt, tmp_path: Path, alias):
    from solver_benchmarks.solvers.base import SolverConfigError

    install_cuopt(_FakeSolution(_Termination.Optimal, [1.0], [1.0], 1.0))
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])
    with pytest.raises(SolverConfigError, match=alias):
        CuOptSolverAdapter({"verbose": False, alias: True}).solve(problem, tmp_path)


def test_unavailable_without_cuopt(monkeypatch):
    monkeypatch.setitem(sys.modules, "cuopt", None)

    assert CuOptSolverAdapter.is_available() is False


def test_docstring_points_at_the_real_convention_check():
    # The module has no runtime probe; the conventions are checked by hand
    # on a GPU with tools/bench_modal.py and pinned by the tests above.
    doc = cuopt_adapter.__doc__
    assert "``_probe``" not in doc
    assert "probe_cuopt" in doc
