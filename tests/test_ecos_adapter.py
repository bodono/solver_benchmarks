"""Comprehensive coverage for the ECOS adapter.

ECOS is an interior-point conic solver: LP / SOCP / exponential cone.
Covers QP-as-LP solves (P=0), CONE-shape solves, infeasibility and
unboundedness branches, status mapping, settings forwarding, and
unavailable-module paths.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData

# ---------------------------------------------------------------------------
# Problem fixtures.
# ---------------------------------------------------------------------------


def _small_lp_in_qp_form() -> dict:
    """min -x1 - x2  s.t. x1 + x2 <= 1, x1, x2 >= 0. Optimum: -1."""
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([-1.0, -1.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([-np.inf, 0.0, 0.0]),
        "u": np.array([1.0, np.inf, np.inf]),
    }


def _qp_with_nonzero_p() -> dict:
    """A real QP — should be skipped because ECOS can't solve QPs."""
    return {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
    }


def _infeasible_lp() -> dict:
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([0.0]),
        "A": sp.csc_matrix([[1.0], [1.0]]),
        "l": np.array([2.0, -np.inf]),
        "u": np.array([np.inf, 1.0]),
    }


def _unbounded_lp() -> dict:
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([-1.0]),
        "A": sp.csc_matrix([[1.0]]),
        "l": np.array([0.0]),
        "u": np.array([np.inf]),
    }


def _small_cone_lp() -> dict:
    """min x s.t. x >= 1 in cone form (s = -1 - (-1) x = x - 1 >= 0)."""
    return {
        "P": None,
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[-1.0]]),
        "b": np.array([-1.0]),
        "cone": {"l": 1},
    }


def _cone_with_free_key() -> dict:
    """Legacy ``f`` cone key must merge into the zero cone."""
    return {
        "P": None,
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[1.0]]),
        "b": np.array([5.0]),
        "cone": {"f": 1},
    }


def _small_socp() -> dict:
    """min y s.t. x = 1, (y, x) in SOC of dim 2. Optimum: y = 1."""
    return {
        "P": None,
        "q": np.array([0.0, 1.0]),
        "A": sp.csc_matrix([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]),
        "b": np.array([1.0, 0.0, 0.0]),
        "cone": {"z": 1, "q": [2]},
    }


def _make_qp_problem(data: dict) -> ProblemData:
    return ProblemData("test", "p", QP, data)


def _make_cone_problem(data: dict) -> ProblemData:
    return ProblemData("test", "p", CONE, data)


# ---------------------------------------------------------------------------
# Registry + module availability.
# ---------------------------------------------------------------------------


def test_ecos_is_registered():
    from solver_benchmarks.solvers import get_solver

    cls = get_solver("ecos")
    assert cls.solver_name == "ecos"
    assert {QP, CONE} == cls.supported_problem_kinds


def test_ecos_runtime_metadata_includes_package_version():
    """Each ECOS result row must record the installed ``ecos``
    package version under ``solver_package_versions``; otherwise the
    report cannot tell which ECOS build produced the timings."""
    from solver_benchmarks.core.environment import SOLVER_PACKAGES, runtime_metadata

    assert "ecos" in SOLVER_PACKAGES
    md = runtime_metadata("ecos")
    versions = md["solver_package_versions"]
    assert "ecos" in versions
    # The value is the installed version string, or None when the
    # package is not installed (the import-skip path).
    assert versions["ecos"] is None or isinstance(versions["ecos"], str)


def test_ecos_is_available_when_module_present():
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    assert ECOSSolverAdapter.is_available() is True


def test_ecos_is_available_returns_false_when_module_missing(monkeypatch):
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "ecos":
            raise ModuleNotFoundError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert ECOSSolverAdapter.is_available() is False


def test_ecos_solve_raises_solver_unavailable_when_module_missing(
    tmp_path, monkeypatch
):
    from solver_benchmarks.solvers.base import SolverUnavailable
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "ecos":
            raise ModuleNotFoundError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    adapter = ECOSSolverAdapter({"verbose": False})
    with pytest.raises(SolverUnavailable, match="ecos"):
        adapter.solve(_make_qp_problem(_small_lp_in_qp_form()), tmp_path)


# ---------------------------------------------------------------------------
# Status mapping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exit_flag,expected",
    [
        (0, status.OPTIMAL),
        (1, status.PRIMAL_INFEASIBLE),
        (2, status.DUAL_INFEASIBLE),
        (10, status.OPTIMAL_INACCURATE),
        (11, status.PRIMAL_INFEASIBLE_INACCURATE),
        (12, status.DUAL_INFEASIBLE_INACCURATE),
        (-1, status.MAX_ITER_REACHED),
        (-2, status.SOLVER_ERROR),
        (-3, status.SOLVER_ERROR),
        (-4, status.SOLVER_ERROR),
        (-7, status.SOLVER_ERROR),
        # Unknown exit flag falls through to SOLVER_ERROR rather than crashing.
        (999, status.SOLVER_ERROR),
        # Missing exitFlag key.
        (None, status.SOLVER_ERROR),
    ],
)
def test_ecos_exit_flag_mapping(exit_flag, expected):
    from solver_benchmarks.solvers.ecos_adapter import _map_ecos_status

    info = {} if exit_flag is None else {"exitFlag": exit_flag}
    assert _map_ecos_status(info) == expected


def test_ecos_exit_flag_mapping_handles_non_int():
    """A garbage exitFlag value (e.g. a string) should not crash; fall
    through to SOLVER_ERROR."""
    from solver_benchmarks.solvers.ecos_adapter import _map_ecos_status

    assert _map_ecos_status({"exitFlag": "not-an-int"}) == status.SOLVER_ERROR


# ---------------------------------------------------------------------------
# QP path (LP only — non-zero P is skipped).
# ---------------------------------------------------------------------------


def test_ecos_solves_lp_in_qp_form_and_reports_kkt(tmp_path):
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(_small_lp_in_qp_form()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(-1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4
    assert result.kkt["duality_gap_rel"] < 1e-4


def test_ecos_solves_qp_via_socp_reformulation(tmp_path):
    """ECOS does not solve QPs natively; the adapter applies a standard
    SOCP epigraph reformulation so QP datasets like Maros-Meszaros run
    through ECOS. The result must be the actual ½ x'Px + q'x objective
    (not the surrogate t + q'x)."""
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter(
        {"verbose": False, "feastol": 1e-9, "abstol": 1e-9, "reltol": 1e-9}
    )
    result = adapter.solve(_make_qp_problem(_qp_with_nonzero_p()), tmp_path)
    assert result.status == status.OPTIMAL
    # min 0.5(x1^2 + x2^2) + x1 + x2 → optimum at x = (-1, -1), value = -1.
    assert result.objective_value == pytest.approx(-1.0, abs=1e-4)
    # Marker on info so callers can tell the SOCP path was used.
    assert result.info.get("socp_reformulation") is True
    assert "socp_t_value" in result.info
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


def test_ecos_solves_rank_deficient_psd_qp(tmp_path):
    """``P`` may be PSD but rank-deficient (e.g. diag(1, 0)). The
    adapter falls back from Cholesky to an eigendecomposition that
    drops zero eigenvalues; the SOCP gets dim ``rank(P) + 2``."""
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    qp = {
        "P": sp.csc_matrix(np.diag([1.0, 0.0])),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
    }
    adapter = ECOSSolverAdapter(
        {"verbose": False, "feastol": 1e-9, "abstol": 1e-9, "reltol": 1e-9}
    )
    result = adapter.solve(_make_qp_problem(qp), tmp_path)
    assert result.status == status.OPTIMAL
    # min 0.5*x1^2 + x1 + x2 with -5 <= x <= 5: x1 = -1, x2 = -5. Value = -5.5.
    assert result.objective_value == pytest.approx(-5.5, abs=1e-4)
    assert result.info.get("socp_reformulation") is True


def test_ecos_socp_reformulation_skips_non_psd_p(tmp_path):
    """If ``P`` is not symmetric or has a negative eigenvalue, the
    SOCP reformulation is invalid; the adapter must return
    SKIPPED_UNSUPPORTED with a clear reason."""
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    # Non-symmetric P.
    qp = {
        "P": sp.csc_matrix(np.array([[1.0, 1.0], [0.0, 1.0]])),
        "q": np.array([0.0, 0.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
    }
    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(qp), tmp_path)
    assert result.status == status.SKIPPED_UNSUPPORTED
    assert "PSD" in result.info["reason"] or "psd" in result.info["reason"].lower()


def test_ecos_qp_with_nonzero_p_helper_detects_nnz():
    """``_qp_has_nonzero_p`` is the gate that triggers the SOCP
    reformulation path; verify directly."""
    from solver_benchmarks.solvers.ecos_adapter import _qp_has_nonzero_p

    assert _qp_has_nonzero_p({"P": sp.csc_matrix(np.eye(2))}) is True
    assert _qp_has_nonzero_p({"P": sp.csc_matrix((2, 2))}) is False
    assert _qp_has_nonzero_p({"P": None}) is False
    assert _qp_has_nonzero_p({}) is False


def test_ecos_psd_square_root_returns_cholesky_for_pd():
    """For positive-definite P, the Cholesky path returns a square
    upper-triangular factor with R'R = P."""
    from solver_benchmarks.solvers.ecos_adapter import _psd_square_root

    p = sp.csc_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    r = _psd_square_root(p)
    assert r.shape == (2, 2)
    reconstructed = r.T @ r
    assert np.allclose(reconstructed, p.toarray())


def test_ecos_psd_square_root_drops_zero_eigenvalues():
    """For rank-deficient PSD P, the eigendecomposition path returns
    a ``rank(P) × n`` factor that satisfies R'R = P."""
    from solver_benchmarks.solvers.ecos_adapter import _psd_square_root

    p = sp.csc_matrix(np.diag([4.0, 0.0, 1.0]))
    r = _psd_square_root(p)
    assert r.shape == (2, 3)  # rank 2, 3 columns
    reconstructed = r.T @ r
    assert np.allclose(reconstructed, p.toarray())


def test_ecos_psd_square_root_returns_none_for_nonsymmetric():
    """A non-symmetric ``P`` (within tolerance) is rejected — the
    adapter then surfaces SKIPPED_UNSUPPORTED to the caller."""
    from solver_benchmarks.solvers.ecos_adapter import _psd_square_root

    p = sp.csc_matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))
    assert _psd_square_root(p) is None


def test_ecos_psd_square_root_rejects_indefinite_p():
    """Pre-fix the eigendecomposition fallback dropped *negative*
    eigenvalues alongside the near-zero ones, so an indefinite QP
    like ``diag([1, -1])`` was silently sent to ECOS as a rank-1
    PSD relaxation. The reviewer flagged this as High severity:
    the user gets back what looks like a valid solve of a different
    (convex) problem. The fix is to refuse indefinite ``P`` so the
    adapter surfaces SKIPPED_UNSUPPORTED instead."""
    from solver_benchmarks.solvers.ecos_adapter import _psd_square_root

    indefinite = sp.csc_matrix(np.diag([1.0, -1.0]))
    assert _psd_square_root(indefinite) is None


def test_ecos_psd_square_root_rejects_negative_definite_p():
    """All-negative eigenvalues = ``-P`` is PSD; the QP has the
    wrong curvature direction and must not be accepted."""
    from solver_benchmarks.solvers.ecos_adapter import _psd_square_root

    neg_def = sp.csc_matrix(np.diag([-1.0, -2.0]))
    assert _psd_square_root(neg_def) is None


def test_ecos_skips_indefinite_qp_via_solve_path(tmp_path):
    """End-to-end: feeding an indefinite ``P`` to the adapter goes
    through ``_qp_to_ecos_via_socp`` which uses ``_psd_square_root``;
    the adapter must return SKIPPED_UNSUPPORTED rather than secretly
    solving the convex relaxation."""
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    qp = {
        "P": sp.csc_matrix(np.diag([1.0, -1.0])),
        "q": np.array([0.0, 0.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
    }
    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(qp), tmp_path)
    assert result.status == status.SKIPPED_UNSUPPORTED
    assert "PSD" in result.info["reason"] or "psd" in result.info["reason"].lower()


def test_ecos_socp_path_handles_missing_primal_x(monkeypatch, tmp_path):
    """If ECOS returns without a primal ``x`` (max-iter, numerical
    failure, an infeasibility certificate it computed without a
    primal), the SOCP post-processing must NOT crash on
    ``raw['x'][n_x]``. Pre-fix we unconditionally indexed there.
    Stub ECOS to return ``x=None`` and verify the adapter surfaces
    a status without raising."""
    pytest.importorskip("ecos")
    import sys as _sys
    from types import SimpleNamespace as _NS

    captured = {}

    def fake_solve(c, G, h, dims, A=None, b=None, **kwargs):
        captured["called"] = True
        return {
            # No 'x' key at all — simulating a status with no primal.
            "y": None,
            "z": None,
            "s": None,
            # exitFlag=-1 is MAX_ITER_REACHED in our mapping.
            "info": {"exitFlag": -1},
        }

    fake_ecos = _NS(solve=fake_solve)
    _sys.modules["ecos"] = fake_ecos
    try:
        from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

        adapter = ECOSSolverAdapter({"verbose": False})
        result = adapter.solve(_make_qp_problem(_qp_with_nonzero_p()), tmp_path)
    finally:
        _sys.modules.pop("ecos", None)
    assert captured.get("called")
    # Status maps from exitFlag=-1.
    assert result.status == status.MAX_ITER_REACHED
    # Objective is None (no primal); socp_reformulation flag still
    # set since we did go through the SOCP path.
    assert result.objective_value is None
    assert result.info.get("socp_reformulation") is True
    # No socp_t_value because there was no primal vector.
    assert "socp_t_value" not in result.info


def test_ecos_lp_via_qp_form_reports_iteration_count(tmp_path):
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(_small_lp_in_qp_form()), tmp_path)
    assert result.status == status.OPTIMAL
    assert isinstance(result.iterations, int)
    assert result.iterations >= 1


def test_ecos_records_setup_and_solve_times(tmp_path):
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(_small_lp_in_qp_form()), tmp_path)
    assert result.status == status.OPTIMAL
    # ECOS' info["timing"] dict has tsetup / tsolve in seconds.
    assert result.setup_time_seconds is not None and result.setup_time_seconds >= 0
    assert result.solve_time_seconds is not None and result.solve_time_seconds >= 0


def test_ecos_lp_primal_infeasibility_returns_certificate(tmp_path):
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(_infeasible_lp()), tmp_path)
    assert result.status in {
        status.PRIMAL_INFEASIBLE,
        status.PRIMAL_INFEASIBLE_INACCURATE,
    }
    # Objective is suppressed for infeasibility-certificate statuses.
    assert result.objective_value is None
    assert result.kkt is not None
    assert result.kkt.get("certificate") == "primal_infeasible"


def test_ecos_lp_unboundedness_returns_dual_certificate(tmp_path):
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(_unbounded_lp()), tmp_path)
    assert result.status in {
        status.DUAL_INFEASIBLE,
        status.DUAL_INFEASIBLE_INACCURATE,
    }
    assert result.objective_value is None
    assert result.kkt is not None
    assert result.kkt.get("certificate") == "dual_infeasible"


# ---------------------------------------------------------------------------
# CONE path.
# ---------------------------------------------------------------------------


def test_ecos_solves_cone_lp(tmp_path):
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(_small_cone_lp()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


def test_ecos_solves_socp(tmp_path):
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(_small_socp()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


def test_ecos_merges_legacy_free_cone_key(tmp_path):
    """The ``f`` (free) cone key folds into the zero cone."""
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(_cone_with_free_key()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(5.0, abs=1e-4)


def test_ecos_skips_unsupported_cone_keys(tmp_path):
    """Unknown / SDP cone keys are not in ECOS' vocabulary; the
    adapter must skip cleanly rather than crash."""
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    cone = {
        "P": None,
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[1.0]]),
        "b": np.array([0.0]),
        "cone": {"s": [2]},  # SDP cone is not supported by ECOS
    }
    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(cone), tmp_path)
    assert result.status == status.SKIPPED_UNSUPPORTED
    assert "cone keys" in result.info["reason"]


def test_ecos_skips_cone_with_nonzero_p(tmp_path):
    """A CONE-shape problem with ``P`` non-zero is also a QP and the
    adapter must skip; otherwise it would silently solve the LP
    relaxation of a QP."""
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    cone = {
        "P": sp.csc_matrix(np.eye(1)),
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[-1.0]]),
        "b": np.array([-1.0]),
        "cone": {"l": 1},
    }
    adapter = ECOSSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(cone), tmp_path)
    assert result.status == status.SKIPPED_UNSUPPORTED


# ---------------------------------------------------------------------------
# Settings forwarding: time_limit / threads ignored, kwargs through.
# ---------------------------------------------------------------------------


def test_ecos_records_time_limit_and_threads_ignored(tmp_path):
    pytest.importorskip("ecos")
    from solver_benchmarks.solvers.ecos_adapter import ECOSSolverAdapter

    adapter = ECOSSolverAdapter(
        {"verbose": False, "time_limit": 30.0, "threads": 4}
    )
    result = adapter.solve(_make_qp_problem(_small_lp_in_qp_form()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.info.get("time_limit_ignored") is True
    assert result.info.get("time_limit_seconds") == 30.0
    assert result.info.get("threads_ignored") is True
    assert result.info.get("threads_requested") == 4


def test_ecos_forwards_native_kwargs_to_solver(monkeypatch, tmp_path):
    """Settings that are not popped (e.g. ``feastol``, ``abstol``,
    ``max_iters``) must reach ecos.solve unchanged."""
    pytest.importorskip("ecos")
    import sys as _sys

    import solver_benchmarks.solvers.ecos_adapter as ecos_mod

    captured: dict = {}

    def fake_solve(c, G, h, dims, A=None, b=None, **kwargs):
        captured.update(kwargs)
        captured["dims"] = dims
        return {
            "x": None,
            "y": None,
            "z": None,
            "s": None,
            "info": {"exitFlag": 0, "iter": 1, "pcost": 0.0},
        }

    fake_ecos = SimpleNamespace(solve=fake_solve)
    _sys.modules["ecos"] = fake_ecos
    try:
        adapter = ecos_mod.ECOSSolverAdapter(
            {"verbose": False, "feastol": 1e-9, "abstol": 1e-9, "max_iters": 200}
        )
        adapter.solve(_make_qp_problem(_small_lp_in_qp_form()), tmp_path)
    finally:
        _sys.modules.pop("ecos", None)
    assert captured.get("feastol") == 1e-9
    assert captured.get("abstol") == 1e-9
    assert captured.get("max_iters") == 200
    assert captured.get("verbose") is False


# ---------------------------------------------------------------------------
# _qp_lp_to_ecos and _cone_to_ecos shape contracts.
# ---------------------------------------------------------------------------


def test_qp_lp_to_ecos_layout():
    """The transform must produce equality-first, inequality-second
    rows aligned with ECOS' (A, b) for equalities and (G, h) for
    inequalities."""
    from solver_benchmarks.solvers.ecos_adapter import _qp_lp_to_ecos

    # 1-row equality + 2-row inequalities.
    qp = {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([0.0, 0.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([0.0, 0.0, 0.0]),
        "u": np.array([0.0, np.inf, np.inf]),  # first row eq, others lower-bound
    }
    data, dims, cone_dict = _qp_lp_to_ecos(qp)
    assert data["A"].shape[0] == 1   # one equality
    assert data["G"].shape[0] >= 2   # at least the two lower-bound rows
    assert dims["q"] == []
    assert dims["e"] == 0
    assert cone_dict.get("z") == 1
    assert cone_dict.get("l") == data["G"].shape[0]


def test_cone_to_ecos_layout():
    from solver_benchmarks.solvers.ecos_adapter import _cone_to_ecos

    cone = _small_socp()
    data, dims, cone_dict = _cone_to_ecos(cone)
    assert data["A"].shape[0] == 1
    assert data["G"].shape[0] == 2
    assert dims == {"l": 0, "q": [2], "e": 0}
    assert cone_dict == {"z": 1, "q": [2]}


def test_cone_to_ecos_skipped_when_p_present():
    from solver_benchmarks.solvers.ecos_adapter import _cone_to_ecos

    cone = {
        "P": sp.csc_matrix(np.eye(1)),
        "q": np.array([0.0]),
        "A": sp.csc_matrix([[1.0]]),
        "b": np.array([0.0]),
        "cone": {"l": 1},
    }
    data, dims, cone_dict = _cone_to_ecos(cone)
    # First element is the SolverResult sentinel for SKIPPED_UNSUPPORTED.
    assert data.status == status.SKIPPED_UNSUPPORTED
    assert dims is None and cone_dict is None
