"""cpo3 adapter: settings translation and cone mapping (no solver needed) and, when
cpo3 is installed, solves of a small QP, an LP and an infeasible LP with verified
certificates, plus second-order cone problems when the build supports them."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.solvers.cpo3_adapter import _parse_cones, translate_settings


def test_aliases_map_to_cpo3_names_and_native_values_win():
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
    from solver_benchmarks.solvers.cpo3_adapter import Cpo3SolverAdapter

    return Cpo3SolverAdapter(settings=settings)


def test_cpo3_solves_box_qp_and_lp():
    pytest.importorskip("cpo3")
    n = 3
    qp = {
        "P": sp.eye(n, format="csc"),
        "q": -np.ones(n),
        "A": sp.eye(n, format="csc"),
        "l": np.zeros(n),
        "u": np.full(n, 0.5),
    }
    result = _adapter(eps=1e-8, verbose=False).solve(_problem("box", qp), Path(tempfile.mkdtemp()))
    assert result.status == "optimal"
    assert np.allclose(result.objective_value, 3 * (0.5**2 / 2 - 0.5), atol=1e-6)
    assert result.kkt["primal_res_rel"] < 1e-6 and result.kkt["dual_res_rel"] < 1e-6
    assert result.info["settings_translated"] == {
        "tol_feas": 1e-8,
        "tol_gap_abs": 1e-8,
        "tol_gap_rel": 1e-8,
    }
    # LP: min x1 + x2 s.t. x1 + x2 >= 1, x >= 0  (P all zero is passed as an LP)
    lp = {
        "P": sp.csc_matrix((2, 2)),
        "q": np.ones(2),
        "A": sp.csc_matrix(np.vstack([np.ones((1, 2)), np.eye(2)])),
        "l": np.array([1.0, 0.0, 0.0]),
        "u": np.array([np.inf, np.inf, np.inf]),
    }
    result = _adapter(eps=1e-8).solve(_problem("lp", lp), Path(tempfile.mkdtemp()))
    assert result.status == "optimal" and abs(result.objective_value - 1.0) < 1e-6


def test_cpo3_certifies_infeasible_lp():
    pytest.importorskip("cpo3")
    # x >= 1 and x <= 0
    lp = {
        "P": sp.csc_matrix((1, 1)),
        "q": np.ones(1),
        "A": sp.csc_matrix(np.array([[1.0], [1.0]])),
        "l": np.array([1.0, -np.inf]),
        "u": np.array([np.inf, 0.0]),
    }
    result = _adapter(eps=1e-8).solve(_problem("infeasible", lp), Path(tempfile.mkdtemp()))
    assert result.status == "primal_infeasible"
    assert result.kkt["certificate"] == "primal_infeasible" and result.kkt["valid"]


class _Cones:
    """Stand-in for the cpo3 module: descriptors become ``(kind, dim)`` tuples."""

    ZeroCone = staticmethod(lambda dim: ("zero", dim))
    NonnegativeCone = staticmethod(lambda dim: ("nonnegative", dim))
    SecondOrderCone = staticmethod(lambda dim: ("second_order", dim))


def test_cone_blocks_map_to_cpo3_cones_in_row_order():
    cones, cone_dict = _parse_cones({"f": 2, "l": 3, "q": [3, 4]}, 12, _Cones)
    assert cones == [("zero", 2), ("nonnegative", 3), ("second_order", 3), ("second_order", 4)]
    assert cone_dict == {"f": 2, "l": 3, "q": [3, 4]}
    # Any block order is kept, empty blocks occupy no rows, and a one-dimensional
    # second-order cone is passed as the nonnegative ray it is.
    cones, cone_dict = _parse_cones({"q": np.array([3, 0, 1]), "z": 0, "l": 2}, 6, _Cones)
    assert cones == [("second_order", 3), ("nonnegative", 1), ("nonnegative", 2)]
    assert cone_dict == {"q": [3, 1], "l": 2}
    assert list(cone_dict) == ["q", "l"]


def test_unsupported_cone_problems_are_skipped_not_failed():
    result = _parse_cones({"z": 1, "s": [2]}, 4, _Cones)
    assert result.status == "skipped_unsupported" and "'s'" in result.info["reason"]
    result = _parse_cones({"l": 2, "q": [3]}, 6, _Cones)
    assert result.status == "skipped_unsupported"
    assert (result.info["cone_rows"], result.info["a_rows"]) == (5, 6)


def _cone_problem(name, data):
    from solver_benchmarks.core.problem import CONE, ProblemData

    return ProblemData("unit", name, CONE, data, metadata={})


def _solve_cone(name, data, **settings):
    """Solve a cone problem, skipping when this cpo3 build has no second-order cones."""
    result = _adapter(eps=1e-8, verbose=False, **settings).solve(
        _cone_problem(name, data), Path(tempfile.mkdtemp())
    )
    if result.status == "skipped_unsupported":
        pytest.skip(result.info["reason"])
    return result


def test_cpo3_solves_second_order_cone_problems():
    pytest.importorskip("cpo3")
    # min x1 + x2 s.t. ||x|| <= 1: rows are s = (1, x1, x2) in a second-order cone.
    ball = {
        "P": None,
        "q": np.ones(2),
        "A": sp.csc_matrix([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
        "b": np.array([1.0, 0.0, 0.0]),
        "cone": {"q": [3]},
    }
    result = _solve_cone("ball", ball)
    assert result.status == "optimal"
    assert abs(result.objective_value + np.sqrt(2.0)) < 1e-6
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert result.kkt[field] < 1e-6

    # The same constraint with a quadratic objective: min 1/2 ||x||^2 - 2 (x1 + x2).
    result = _solve_cone("ball_qp", {**ball, "P": sp.eye(2, format="csc"), "q": -2 * np.ones(2)})
    assert result.status == "optimal"
    assert abs(result.objective_value - (0.5 - 2.0 * np.sqrt(2.0))) < 1e-6
    assert result.kkt["primal_res_rel"] < 1e-6 and result.kkt["dual_res_rel"] < 1e-6

    # Second-order blocks before a nonnegative one, with a one-dimensional cone:
    # min x1 + x2 + x3 s.t. ||(x1, x2)|| <= 1, x3 >= -2, x3 >= -5.
    mixed = {
        "P": None,
        "q": np.ones(3),
        "A": sp.csc_matrix(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        ),
        "b": np.array([1.0, 0.0, 0.0, 2.0, 5.0]),
        "cone": {"q": [3, 1], "l": 1},
    }
    result = _solve_cone("mixed", mixed)
    assert result.status == "optimal"
    assert abs(result.objective_value + np.sqrt(2.0) + 2.0) < 1e-6
    assert result.kkt["primal_res_rel"] < 1e-6 and result.kkt["dual_res_rel"] < 1e-6


def test_cpo3_certifies_infeasible_second_order_cone_problem():
    pytest.importorskip("cpo3")
    # x1 <= -1 and (x1, x2) in a second-order cone, which needs x1 >= |x2| >= 0.
    # Presolve is off so that the certificate comes from the cone iteration.
    infeasible = {
        "P": None,
        "q": np.array([0.0, 1.0]),
        "A": sp.csc_matrix([[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
        "b": np.array([-1.0, 0.0, 0.0]),
        "cone": {"l": 1, "q": [2]},
    }
    result = _solve_cone("infeasible_soc", infeasible, presolve=False)
    assert result.status == "primal_infeasible"
    assert result.kkt["certificate"] == "primal_infeasible" and result.kkt["valid"]


def test_cpo3_build_without_a_cone_reports_the_problem_as_unsupported(monkeypatch):
    cpo3 = pytest.importorskip("cpo3")

    def unimplemented(*args, **kwargs):
        raise NotImplementedError("second_order cone in block 0")

    monkeypatch.setattr(cpo3, "solve", unimplemented)
    data = {
        "P": None,
        "q": np.ones(2),
        "A": sp.csc_matrix([[0.0, 0.0], [-1.0, 0.0], [0.0, -1.0]]),
        "b": np.array([1.0, 0.0, 0.0]),
        "cone": {"q": [3]},
    }
    result = _adapter(verbose=False).solve(_cone_problem("ball", data), Path(tempfile.mkdtemp()))
    assert result.status == "skipped_unsupported"
    assert "second_order cone in block 0" in result.info["reason"]
