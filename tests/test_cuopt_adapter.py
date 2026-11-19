"""cuOpt adapter tests against a fake ``cuopt.linear_programming`` module.

cuOpt needs an NVIDIA GPU, so CI cannot run it. These tests pin the parts of
the adapter that were only ever checked by hand on a GPU
(``tools/bench_modal.py`` ``probe_cuopt``): the QP objective mapping
``Q = P / 2``, the dual sign used for the harness KKT residuals, the
termination-status table, and option forwarding. If a cuOpt release changes
one of these conventions, update the fake (``FakeCuOpt`` in conftest.py, shared
with test_commercial_adapters_fake.py) to match ``probe_cuopt``'s output and
the adapter with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.solvers import cuopt_adapter
from solver_benchmarks.solvers.cuopt_adapter import CuOptSolverAdapter, _map_status


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


def test_qp_objective_is_passed_as_half_of_p(fake_cuopt, tmp_path: Path):
    # min x1^2 + 2 x2^2 - 2 x1 - 4 x2, i.e. P = diag(2, 4): optimum x = (1, 1).
    fake = fake_cuopt.returns("Optimal", x=[1.0, 1.0], y=[0.0, 0.0], objective=-3.0)
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


def test_lp_sends_no_quadratic_objective(fake_cuopt, tmp_path: Path):
    fake = fake_cuopt.returns("Optimal", x=[1.0], y=[1.0], objective=1.0)
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])

    CuOptSolverAdapter({"verbose": False}).solve(problem, tmp_path)

    assert "set_quadratic_objective_matrix" not in fake.model_calls


def test_kkt_residuals_use_the_negated_cuopt_dual(fake_cuopt, tmp_path: Path):
    # min x s.t. 1 <= x <= 10: x = 1 and, in the harness convention
    # (y = lambda_u - lambda_l), y = -1. cuOpt reports the opposite sign.
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])
    fake_cuopt.returns("Optimal", x=[1.0], y=[1.0], objective=1.0)

    default = CuOptSolverAdapter({"verbose": False}).solve(problem, tmp_path)
    overridden = CuOptSolverAdapter({"verbose": False, "dual_sign": 1.0}).solve(problem, tmp_path)

    assert default.kkt["dual_res_rel"] == pytest.approx(0.0, abs=1e-12)
    assert overridden.kkt["dual_res_rel"] > 0.5


def test_infeasible_termination_reports_no_point(fake_cuopt, tmp_path: Path):
    fake_cuopt.returns("PrimalInfeasible", x=[0.0], y=[0.0], objective=0.0)
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])

    result = CuOptSolverAdapter({"verbose": False}).solve(problem, tmp_path)

    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.objective_value is None
    assert result.kkt is None


def test_options_are_forwarded_to_cuopt_parameters(fake_cuopt, tmp_path: Path):
    fake = fake_cuopt.returns("TimeLimit", x=[1.0], y=[1.0], objective=1.0)
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


@pytest.mark.parametrize("settings, expected", [({}, True), ({"infeasibility_detection": False}, False)],
                         ids=["default", "configured off"])
def test_pdlp_infeasibility_detection_is_on_unless_configured(fake_cuopt, tmp_path: Path, settings, expected):
    # cuOpt's PDLP never stops on an infeasible problem with its detection
    # off (cuOpt's default), which is right only for a problem known in
    # advance to be feasible: the adapter turns it on, and the row says what
    # the solve ran with.
    fake = fake_cuopt.returns("PrimalInfeasible", x=[1.0], y=[1.0], objective=1.0)
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])

    result = CuOptSolverAdapter({"verbose": False, **settings}).solve(problem, tmp_path)

    assert fake.parameters["infeasibility_detection"] is expected
    assert result.info["infeasibility_detection"] is expected
    assert result.status == status.PRIMAL_INFEASIBLE


def test_a_value_cuopt_cannot_read_as_a_bool_is_refused(fake_cuopt, tmp_path: Path):
    # cuOpt reads a bool parameter from str(value); "yes" is not one it takes.
    from solver_benchmarks.solvers.base import SolverConfigError

    fake_cuopt.returns("Optimal", x=[1.0], y=[1.0], objective=1.0)
    problem = _problem(np.zeros((1, 1)), [1.0], np.eye(1), [1.0], [10.0])
    with pytest.raises(SolverConfigError, match="infeasibility_detection"):
        CuOptSolverAdapter({"verbose": False, "infeasibility_detection": "yes"}).solve(problem, tmp_path)


@pytest.mark.parametrize("alias", ["eps", "eps_abs", "eps_rel"])
def test_boolean_tolerance_alias_is_refused(fake_cuopt, tmp_path: Path, alias):
    from solver_benchmarks.solvers.base import SolverConfigError

    fake_cuopt.returns("Optimal", x=[1.0], y=[1.0], objective=1.0)
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
