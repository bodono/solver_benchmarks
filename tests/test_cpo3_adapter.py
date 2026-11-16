"""cpo3 adapter: settings translation (no solver needed) and, when cpo3 is installed,
solves of small QPs, LPs and cone-form LPs, infeasibility certificates, settings
forwarding, linear-solver validation and the status table."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.solvers.base import SolverUnavailable
from solver_benchmarks.solvers.cpo3_adapter import translate_settings


def test_aliases_map_to_cpo3_names_and_native_values_win():
    s = {"eps": 1e-6, "max_iter": 50}
    assert translate_settings(s) == {
        "tol_feas": 1e-6, "tol_gap_abs": 1e-6, "tol_gap_rel": 1e-6, "max_iters": 50,
    }
    assert s == {"tol_feas": 1e-6, "tol_gap_abs": 1e-6, "tol_gap_rel": 1e-6, "max_iters": 50}
    s = {"eps_abs": 1e-3, "eps_rel": 1e-4, "tol_feas": 1e-9}
    assert translate_settings(s) == {"tol_gap_abs": 1e-3, "tol_gap_rel": 1e-4}
    assert s["tol_feas"] == 1e-9


def test_native_max_iters_wins_and_alias_is_dropped():
    s = {"max_iter": 50, "max_iters": 70}
    translate_settings(s)
    assert s == {"max_iters": 70}


def _problem(name, qp, kind=QP):
    return ProblemData("unit", name, kind, qp, metadata={})


def _adapter(**settings):
    from solver_benchmarks.solvers.cpo3_adapter import Cpo3SolverAdapter

    return Cpo3SolverAdapter(settings=settings)


def _box_qp(n=3):
    return {
        "P": sp.eye(n, format="csc"),
        "q": -np.ones(n),
        "A": sp.eye(n, format="csc"),
        "l": np.zeros(n),
        "u": np.full(n, 0.5),
    }


def test_cpo3_solves_box_qp_and_lp():
    pytest.importorskip("cpo3")
    result = _adapter(eps=1e-8, verbose=False).solve(
        _problem("box", _box_qp()), Path(tempfile.mkdtemp())
    )
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
    assert result.objective_value is None


def test_cpo3_certifies_unbounded_lp():
    pytest.importorskip("cpo3")
    # min -x s.t. x >= 0
    lp = {
        "P": sp.csc_matrix((1, 1)),
        "q": -np.ones(1),
        "A": sp.eye(1, format="csc"),
        "l": np.zeros(1),
        "u": np.full(1, np.inf),
    }
    result = _adapter().solve(_problem("unbounded", lp), Path(tempfile.mkdtemp()))
    assert result.status == "dual_infeasible"
    assert result.kkt["certificate"] == "dual_infeasible" and result.kkt["valid"]


def test_cpo3_records_release_info_fields_and_timing():
    pytest.importorskip("cpo3")
    result = _adapter().solve(_problem("box", _box_qp()), Path(tempfile.mkdtemp()))
    info = result.info
    for field in ("iters", "pobj", "dobj", "primary_solves", "refinement_solves",
                  "lsmr_iterations", "factorizations"):
        assert field in info, field
    assert info["linear_solver"] == "faer"
    # Earlier report columns stay populated from the renamed counters.
    assert info["kkt_solves"] == info["primary_solves"]
    assert info["refinement_passes"] == info["refinement_solves"]
    assert result.setup_time_seconds is None
    assert result.solve_time_seconds is not None and result.solve_time_seconds >= 0
    assert result.run_time_seconds >= 0


def _mixed_qp(seed=0, m=30, n=40):
    # Random convex QP with equality, one-sided and two-sided rows that
    # cpo3's presolve cannot solve on its own.
    rng = np.random.default_rng(seed)
    a = sp.random(m, n, density=0.3, random_state=rng, format="csc")
    x0 = rng.normal(size=n)
    ax0 = a @ x0
    l = ax0 - rng.uniform(0.1, 1.0, m)
    u = ax0 + rng.uniform(0.1, 1.0, m)
    l[:5] = u[:5] = ax0[:5]
    l[5:15] = -np.inf
    u[15:20] = np.inf
    g = rng.normal(size=(n, n))
    return {
        "P": sp.csc_matrix(g @ g.T / n + 1e-2 * np.eye(n)),
        "q": rng.normal(size=n),
        "A": a,
        "l": l,
        "u": u,
    }


def test_cpo3_interior_point_path_matches_clarabel():
    pytest.importorskip("cpo3")
    pytest.importorskip("clarabel")
    from solver_benchmarks.solvers.clarabel_adapter import ClarabelSolverAdapter

    qp = _mixed_qp()
    result = _adapter(presolve=False).solve(_problem("mixed", qp), Path(tempfile.mkdtemp()))
    assert result.status == status.OPTIMAL, result.info
    # The solve really ran interior-point iterations and factorizations.
    assert result.iterations > 0
    assert result.info["factorizations"] > 0 and result.info["primary_solves"] > 0
    assert result.info["kkt_solves"] == result.info["primary_solves"] > 0
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert result.kkt[field] < 1e-6, field
    reference = ClarabelSolverAdapter({"verbose": False}).solve(
        _problem("mixed", qp), Path(tempfile.mkdtemp())
    )
    assert reference.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(reference.objective_value, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize(
    "lp, expected, certificate",
    [
        # x >= 1 and x <= 0
        (
            {"q": np.ones(1), "A": np.array([[1.0], [1.0]]),
             "l": np.array([1.0, -np.inf]), "u": np.array([np.inf, 0.0])},
            status.PRIMAL_INFEASIBLE,
            "primal_infeasible",
        ),
        # min -x1 - x2 s.t. x1 - x2 <= 1, x >= 0
        (
            {"q": -np.ones(2), "A": np.array([[1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
             "l": np.array([-np.inf, 0.0, 0.0]), "u": np.array([1.0, np.inf, np.inf])},
            status.DUAL_INFEASIBLE,
            "dual_infeasible",
        ),
    ],
)
def test_cpo3_interior_point_certificates_without_presolve(lp, expected, certificate):
    pytest.importorskip("cpo3")
    n = lp["q"].size
    qp = {"P": sp.csc_matrix((n, n)), "q": lp["q"], "A": sp.csc_matrix(lp["A"]),
          "l": lp["l"], "u": lp["u"]}
    result = _adapter(presolve=False).solve(_problem("cert", qp), Path(tempfile.mkdtemp()))
    assert result.status == expected, result.info
    assert result.iterations > 0
    assert result.kkt["certificate"] == certificate and result.kkt["valid"]


def test_cpo3_solves_cone_form_lp_with_zero_and_nonnegative_cones():
    pytest.importorskip("cpo3")
    # min x1 + x2 s.t. x1 - x2 = 0 (zero cone), x >= 0.25 (nonnegative cone).
    cone_problem = {
        "P": None,
        "q": np.ones(2),
        "A": sp.csc_matrix(np.array([[1.0, -1.0], [-1.0, 0.0], [0.0, -1.0]])),
        "b": np.array([0.0, -0.25, -0.25]),
        "cone": {"z": 1, "l": 2, "q": [], "s": []},
    }
    result = _adapter().solve(_problem("cone_lp", cone_problem, CONE), Path(tempfile.mkdtemp()))
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(0.5, abs=1e-6)
    assert result.kkt["primal_res_rel"] < 1e-6 and result.kkt["dual_res_rel"] < 1e-6


@pytest.mark.parametrize(
    "cone, rows",
    [
        ({"l": 1, "q": [3]}, 4),
        ({"l": 1, "s": [2]}, 4),
        ({"l": 1, "ep": 1}, 4),
        ({"l": 1, "bl": [0.0], "bu": [1.0]}, 3),
    ],
)
def test_cpo3_skips_cones_it_cannot_solve(cone, rows):
    cone_problem = {
        "q": np.ones(1),
        "A": sp.csc_matrix(np.ones((rows, 1))),
        "b": np.zeros(rows),
        "cone": cone,
    }
    result = _adapter().solve(_problem("socp", cone_problem, CONE), Path(tempfile.mkdtemp()))
    assert result.status == status.SKIPPED_UNSUPPORTED
    assert "zero and nonnegative" in result.info["reason"]


def test_cpo3_forwards_time_limit_threads_and_iteration_aliases(monkeypatch):
    cpo3 = pytest.importorskip("cpo3")
    import solver_benchmarks.solvers.cpo3_adapter as module

    captured = {}
    real_solve = cpo3.solve

    def spy(problem, *, settings=None, warm_start=None):
        captured["settings"] = settings
        return real_solve(problem, settings=settings)

    monkeypatch.setattr(cpo3, "solve", spy)
    adapter = module.Cpo3SolverAdapter(
        {"time_limit_sec": 12.5, "num_threads": 1, "max_iter": 77, "linear_solver": "FAER"}
    )
    result = adapter.solve(_problem("box", _box_qp()), Path(tempfile.mkdtemp()))
    assert result.status == status.OPTIMAL
    settings = captured["settings"]
    assert settings.time_limit == 12.5
    assert settings.threads == 1
    assert settings.max_iters == 77
    assert settings.linear_solver == cpo3.LinearSolver.FAER
    assert settings.verbose is False


def test_cpo3_rejects_unknown_settings_and_linear_solvers():
    pytest.importorskip("cpo3")
    with pytest.raises(ValueError, match="does not accept"):
        _adapter(not_a_setting=1).solve(_problem("box", _box_qp()), Path(tempfile.mkdtemp()))
    with pytest.raises(ValueError, match="Unknown cpo3 linear_solver"):
        _adapter(linear_solver="qdldl").solve(_problem("box", _box_qp()), Path(tempfile.mkdtemp()))


def test_cpo3_refuses_unavailable_backend_before_solving(monkeypatch):
    cpo3 = pytest.importorskip("cpo3")
    from solver_benchmarks.solvers.cpo3_adapter import resolve_linear_solver

    monkeypatch.setattr(cpo3.LinearSolver, "is_available", lambda self: False)
    with pytest.raises(SolverUnavailable, match="'cholmod' is not available"):
        resolve_linear_solver("cholmod", cpo3)
    with pytest.raises(SolverUnavailable):
        _adapter(linear_solver="pardiso").solve(_problem("box", _box_qp()), Path(tempfile.mkdtemp()))


def test_cpo3_status_table_covers_every_release_status():
    cpo3 = pytest.importorskip("cpo3")
    from solver_benchmarks.solvers.cpo3_adapter import _STATUS

    expected = {
        "solved": status.OPTIMAL,
        "primal_infeasible": status.PRIMAL_INFEASIBLE,
        "dual_infeasible": status.DUAL_INFEASIBLE,
        "max_iterations": status.MAX_ITER_REACHED,
        "time_limit": status.TIME_LIMIT,
    }
    assert {member.value for member in cpo3.SolverStatus} == set(expected)
    assert _STATUS == expected


def test_cpo3_numerical_failure_is_solver_error(monkeypatch):
    cpo3 = pytest.importorskip("cpo3")

    def boom(problem, *, settings=None, warm_start=None):
        raise RuntimeError("numerical solver failure: factorization failed")

    monkeypatch.setattr(cpo3, "solve", boom)
    result = _adapter().solve(_problem("box", _box_qp()), Path(tempfile.mkdtemp()))
    assert result.status == status.SOLVER_ERROR
    assert "factorization failed" in result.info["error"]
    assert result.run_time_seconds is not None
