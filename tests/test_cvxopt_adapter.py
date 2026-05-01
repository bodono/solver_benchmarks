"""Comprehensive coverage for the CVXOPT adapter.

CVXOPT is an interior-point solver supporting LP / QP / SOCP / SDP.
Tests cover all problem-shape branches, status mapping, settings
forwarding (with the global-options save/restore contract), the
PSD-vec layout conversion in both directions, and the unavailable-
module paths.
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


def _small_qp() -> dict:
    """min 0.5(x1^2 + x2^2) + x1 + x2 s.t. -5 <= x1, x2 <= 5. Optimum: -1."""
    return {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
    }


def _small_lp_in_qp_form() -> dict:
    """min -x1 - x2 s.t. x1+x2 <= 1, x>=0. Optimum: -1."""
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([-1.0, -1.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([-np.inf, 0.0, 0.0]),
        "u": np.array([1.0, np.inf, np.inf]),
    }


def _small_cone_lp() -> dict:
    return {
        "P": None,
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[-1.0]]),
        "b": np.array([-1.0]),
        "cone": {"l": 1},
    }


def _small_socp() -> dict:
    """min y s.t. x = 1, (y, x) in SOC. Optimum: y = 1."""
    return {
        "P": None,
        "q": np.array([0.0, 1.0]),
        "A": sp.csc_matrix([[1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]),
        "b": np.array([1.0, 0.0, 0.0]),
        "cone": {"z": 1, "q": [2]},
    }


def _small_sdp() -> dict:
    """min trace(diag(1,2) X) s.t. trace(X) = 1, X in S^2_+. Optimum: 1."""
    return {
        "P": None,
        "q": np.array([1.0, 0.0, 2.0]),
        "A": sp.csc_matrix(
            np.vstack([np.array([[1.0, 0.0, 1.0]]), -np.eye(3)])
        ),
        "b": np.array([1.0, 0.0, 0.0, 0.0]),
        "cone": {"z": 1, "s": [2]},
    }


def _make_qp_problem(data: dict) -> ProblemData:
    return ProblemData("test", "p", QP, data)


def _make_cone_problem(data: dict) -> ProblemData:
    return ProblemData("test", "p", CONE, data)


# ---------------------------------------------------------------------------
# Registry + module availability.
# ---------------------------------------------------------------------------


def test_cvxopt_is_registered():
    from solver_benchmarks.solvers import get_solver

    cls = get_solver("cvxopt")
    assert cls.solver_name == "cvxopt"
    assert {QP, CONE} == cls.supported_problem_kinds


def test_cvxopt_is_available_when_module_present():
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    assert CVXOPTSolverAdapter.is_available() is True


def test_cvxopt_is_available_returns_false_when_module_missing(monkeypatch):
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "cvxopt":
            raise ModuleNotFoundError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert CVXOPTSolverAdapter.is_available() is False


def test_cvxopt_solve_raises_solver_unavailable_when_module_missing(
    tmp_path, monkeypatch
):
    from solver_benchmarks.solvers.base import SolverUnavailable
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in ("cvxopt", "cvxopt.solvers"):
            raise ModuleNotFoundError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    adapter = CVXOPTSolverAdapter({"verbose": False})
    with pytest.raises(SolverUnavailable, match="cvxopt"):
        adapter.solve(_make_qp_problem(_small_qp()), tmp_path)


# ---------------------------------------------------------------------------
# Status mapping.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_status,expected",
    [
        ("optimal", status.OPTIMAL),
        ("primal infeasible", status.PRIMAL_INFEASIBLE),
        ("dual infeasible", status.DUAL_INFEASIBLE),
        # Unknown status without bound info → MAX_ITER_REACHED.
        ("unknown", status.MAX_ITER_REACHED),
        # An unrecognized status string falls through to SOLVER_ERROR.
        ("garbage_value", status.SOLVER_ERROR),
        ("", status.SOLVER_ERROR),
    ],
)
def test_cvxopt_status_mapping(raw_status, expected):
    from solver_benchmarks.solvers.cvxopt_adapter import _map_cvxopt_status

    raw = {"status": raw_status}
    assert _map_cvxopt_status(raw) == expected


def test_cvxopt_unknown_with_small_relgap_maps_to_optimal_inaccurate():
    """CVXOPT's "unknown" with a tight relative gap is the practical
    "I gave up but the answer looks fine" case. Map it to
    OPTIMAL_INACCURATE so the downstream analysis treats the row as
    a low-confidence success rather than a max-iter timeout."""
    from solver_benchmarks.solvers.cvxopt_adapter import _map_cvxopt_status

    raw = {"status": "unknown", "relative gap": 1e-5}
    assert _map_cvxopt_status(raw) == status.OPTIMAL_INACCURATE


def test_cvxopt_unknown_with_large_primal_infeas_maps_to_pinf_inaccurate():
    from solver_benchmarks.solvers.cvxopt_adapter import _map_cvxopt_status

    raw = {"status": "unknown", "primal infeasibility": 100.0}
    assert _map_cvxopt_status(raw) == status.PRIMAL_INFEASIBLE_INACCURATE


def test_cvxopt_unknown_with_large_dual_infeas_maps_to_dinf_inaccurate():
    from solver_benchmarks.solvers.cvxopt_adapter import _map_cvxopt_status

    raw = {"status": "unknown", "dual infeasibility": 100.0}
    assert _map_cvxopt_status(raw) == status.DUAL_INFEASIBLE_INACCURATE


# ---------------------------------------------------------------------------
# QP solve.
# ---------------------------------------------------------------------------


def test_cvxopt_solves_qp_with_kkt(tmp_path):
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    adapter = CVXOPTSolverAdapter(
        {"verbose": False, "abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9}
    )
    result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(-1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4
    assert result.kkt["duality_gap_rel"] < 1e-4


def test_cvxopt_solves_lp_in_qp_form(tmp_path):
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    adapter = CVXOPTSolverAdapter(
        {"verbose": False, "abstol": 1e-9, "reltol": 1e-9, "feastol": 1e-9}
    )
    result = adapter.solve(_make_qp_problem(_small_lp_in_qp_form()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(-1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


def test_cvxopt_records_iteration_count(tmp_path):
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    adapter = CVXOPTSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    assert result.status == status.OPTIMAL
    assert isinstance(result.iterations, int)
    assert result.iterations >= 1


# ---------------------------------------------------------------------------
# CONE path: LP, SOCP, SDP.
# ---------------------------------------------------------------------------


def test_cvxopt_solves_cone_lp(tmp_path):
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    adapter = CVXOPTSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(_small_cone_lp()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)


def test_cvxopt_solves_socp(tmp_path):
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    adapter = CVXOPTSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(_small_socp()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-3
    assert result.kkt["dual_res_rel"] < 1e-3


def test_cvxopt_solves_sdp_with_psd_layout_conversion(tmp_path):
    """The PSD-cone layout differs between this codebase (canonical
    n*(n+1)/2 with √2 off-diagonals) and CVXOPT (BLAS unpacked 'L',
    n*n column-major). The transform must round-trip cleanly so the
    KKT residuals come back small."""
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    adapter = CVXOPTSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(_small_sdp()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-3
    assert result.kkt["dual_res_rel"] < 1e-3


def test_cvxopt_skips_exponential_cone(tmp_path):
    """CVXOPT does not support the exponential cone; the adapter must
    skip cleanly rather than try to forward an unrecognized cone key."""
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    cone = {
        "P": None,
        "q": np.array([1.0, 1.0, 1.0]),
        "A": sp.csc_matrix(-np.eye(3)),
        "b": np.array([0.0, 0.0, 0.0]),
        "cone": {"ep": 1},
    }
    adapter = CVXOPTSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(cone), tmp_path)
    assert result.status == status.SKIPPED_UNSUPPORTED
    assert "exponential" in result.info["reason"].lower()


def test_cvxopt_handles_legacy_free_cone_key(tmp_path):
    """``f`` must merge into the zero cone, mirroring the SCS / ECOS
    behavior."""
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    cone = {
        "P": None,
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[1.0]]),
        "b": np.array([5.0]),
        "cone": {"f": 1},
    }
    adapter = CVXOPTSolverAdapter({"verbose": False})
    result = adapter.solve(_make_cone_problem(cone), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Settings forwarding (global-options save/restore contract).
# ---------------------------------------------------------------------------


def test_cvxopt_options_are_restored_after_solve(tmp_path):
    """CVXOPT's options dict is global state. The adapter must
    snapshot and restore it so concurrent solves don't leak knobs
    into each other."""
    pytest.importorskip("cvxopt")
    import cvxopt.solvers as cs

    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    cs.options.clear()
    cs.options["sentinel_external"] = "preserved"

    adapter = CVXOPTSolverAdapter({"verbose": True, "abstol": 1e-9, "max_iters": 50})
    adapter.solve(_make_qp_problem(_small_qp()), tmp_path)

    # The user's external option survived; adapter knobs are gone.
    assert cs.options.get("sentinel_external") == "preserved"
    assert "abstol" not in cs.options
    assert "show_progress" not in cs.options


def test_cvxopt_max_iter_alias_routes_to_maxiters(tmp_path, monkeypatch):
    """``max_iter`` and ``max_iters`` are cross-adapter spellings; the
    adapter must rewrite them to CVXOPT's native ``maxiters``."""
    pytest.importorskip("cvxopt")
    import cvxopt.solvers as cs

    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    captured: dict = {}

    real_coneqp = cs.coneqp

    def spy(*args, **kwargs):
        captured["maxiters"] = cs.options.get("maxiters")
        return real_coneqp(*args, **kwargs)

    monkeypatch.setattr(cs, "coneqp", spy)
    adapter = CVXOPTSolverAdapter({"verbose": False, "max_iter": 42})
    adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    assert captured["maxiters"] == 42


def test_cvxopt_records_time_limit_and_threads_ignored(tmp_path):
    """CVXOPT exposes neither knob; both must show up on info as
    ignored markers."""
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    adapter = CVXOPTSolverAdapter(
        {"verbose": False, "time_limit": 30.0, "threads": 4}
    )
    result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    assert result.info.get("time_limit_ignored") is True
    assert result.info.get("time_limit_seconds") == 30.0
    assert result.info.get("threads_ignored") is True
    assert result.info.get("threads_requested") == 4


# ---------------------------------------------------------------------------
# Numerical-failure handling.
# ---------------------------------------------------------------------------


def test_cvxopt_converts_arithmetic_error_into_solver_error(tmp_path):
    """Some infeasible LPs push CVXOPT's IPM into a sqrt-of-negative
    domain error. The adapter must catch that and return
    ``SOLVER_ERROR`` rather than letting the exception escape."""
    pytest.importorskip("cvxopt")
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    qp = {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([0.0]),
        "A": sp.csc_matrix([[1.0], [1.0]]),
        "l": np.array([2.0, -np.inf]),
        "u": np.array([np.inf, 1.0]),
    }
    adapter = CVXOPTSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(qp), tmp_path)
    # Either SOLVER_ERROR (numerical) or PRIMAL_INFEASIBLE depending on
    # CVXOPT version; both are correct outcomes for this problem.
    assert result.status in {status.SOLVER_ERROR, status.PRIMAL_INFEASIBLE}
    if result.status == status.SOLVER_ERROR:
        assert "solver_error" in result.info or result.info.get("solver_error")


def test_cvxopt_solver_error_via_stub(tmp_path, monkeypatch):
    """Force a numerical exception from coneqp and verify the adapter
    surfaces SOLVER_ERROR instead of letting the exception escape."""
    pytest.importorskip("cvxopt")
    import cvxopt.solvers as cs

    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPTSolverAdapter

    def crash(*args, **kwargs):
        raise ArithmeticError("simulated CVXOPT factorization failure")

    monkeypatch.setattr(cs, "coneqp", crash)
    adapter = CVXOPTSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    assert result.status == status.SOLVER_ERROR
    assert "ArithmeticError" in result.info.get("solver_error", "")


# ---------------------------------------------------------------------------
# PSD layout transform: round-trip property.
# ---------------------------------------------------------------------------


def test_psd_triangle_to_full_inverse_round_trips():
    """``_psd_full_to_triangle`` must invert ``_psd_triangle_to_full``
    on canonical inputs (preserves √2 off-diagonal scaling)."""
    from solver_benchmarks.solvers.cvxopt_adapter import (
        _psd_full_to_triangle,
        _psd_triangle_to_full,
    )

    for n in (1, 2, 3, 5):
        forward = _psd_triangle_to_full(n)
        backward = _psd_full_to_triangle(n)
        # Random canonical PSD-triangle vector.
        tri = np.random.RandomState(n).randn(n * (n + 1) // 2)
        assert backward @ (forward @ tri) == pytest.approx(tri, abs=1e-12)


def test_psd_triangle_to_full_diagonal_unscaled_off_diagonal_inv_sqrt2():
    """The canonical → BLAS transform leaves diagonal entries unscaled
    and splits each off-diagonal canonical entry across symmetric
    positions, each scaled by ``1/√2``."""
    from solver_benchmarks.solvers.cvxopt_adapter import _psd_triangle_to_full

    transform = _psd_triangle_to_full(2).toarray()
    # n=2: canonical [X11, X21*√2, X22] → BLAS [X[0,0], X[1,0], X[0,1], X[1,1]]
    # row 0 (BLAS X[0,0]) gets canonical[0] (X11) directly.
    assert transform[0, 0] == pytest.approx(1.0)
    # row 3 (BLAS X[1,1]) gets canonical[2] directly.
    assert transform[3, 2] == pytest.approx(1.0)
    # rows 1 and 2 (X[1,0] and X[0,1]) each get canonical[1] / √2.
    assert transform[1, 1] == pytest.approx(1.0 / np.sqrt(2.0))
    assert transform[2, 1] == pytest.approx(1.0 / np.sqrt(2.0))


# ---------------------------------------------------------------------------
# _qp_to_cvxopt and _cone_to_cvxopt shape contracts.
# ---------------------------------------------------------------------------


def test_qp_to_cvxopt_layout(tmp_path):
    pytest.importorskip("cvxopt")
    import cvxopt

    from solver_benchmarks.solvers.cvxopt_adapter import _qp_to_cvxopt

    qp = _small_lp_in_qp_form()
    data, dims, cone_dict = _qp_to_cvxopt(qp, cvxopt)
    assert dims == {"l": 3, "q": [], "s": []}
    assert cone_dict == {"l": 3}  # all three rows are inequality
    assert data["P"].size == (2, 2)
    assert data["q"].size == (2, 1)


def test_cone_to_cvxopt_layout_for_socp():
    pytest.importorskip("cvxopt")
    import cvxopt

    from solver_benchmarks.solvers.cvxopt_adapter import _cone_to_cvxopt

    data, dims, cone_dict = _cone_to_cvxopt(_small_socp(), cvxopt)
    assert dims == {"l": 0, "q": [2], "s": []}
    assert cone_dict == {"z": 1, "q": [2]}


def test_cone_to_cvxopt_layout_for_sdp_expands_psd_block_to_blas():
    """The canonical 3-entry PSD block must expand to 4 BLAS rows
    (n*n for n=2)."""
    pytest.importorskip("cvxopt")
    import cvxopt

    from solver_benchmarks.solvers.cvxopt_adapter import _cone_to_cvxopt

    data, dims, cone_dict = _cone_to_cvxopt(_small_sdp(), cvxopt)
    assert dims["s"] == [2]
    # G has 4 rows (one PSD block of dim 2 → 2*2=4 BLAS entries).
    assert data["G"].size[0] == 4


def test_cone_to_cvxopt_skips_unsupported_cone_keys():
    pytest.importorskip("cvxopt")
    import cvxopt

    from solver_benchmarks.solvers.cvxopt_adapter import _cone_to_cvxopt

    cone = {
        "P": None,
        "q": np.array([1.0, 1.0, 1.0]),
        "A": sp.csc_matrix(-np.eye(3)),
        "b": np.array([0.0, 0.0, 0.0]),
        "cone": {"ep": 1},
    }
    result = _cone_to_cvxopt(cone, cvxopt)
    assert isinstance(result, type(_cone_to_cvxopt(cone, cvxopt)))  # SolverResult
    # Direct shape check.
    from solver_benchmarks.core.result import SolverResult

    assert isinstance(result, SolverResult)
    assert result.status == status.SKIPPED_UNSUPPORTED


# ---------------------------------------------------------------------------
# _flatten_info smoke test.
# ---------------------------------------------------------------------------


def test_flatten_info_keeps_scalar_fields():
    from solver_benchmarks.solvers.cvxopt_adapter import _flatten_info

    raw = {
        "status": "optimal",
        "gap": 1e-10,
        "relative gap": 1e-9,
        "primal objective": 1.5,
        "dual objective": 1.5,
        "primal infeasibility": 1e-8,
        "dual infeasibility": 1e-8,
        "primal slack": 0.1,
        "dual slack": 0.1,
        "iterations": 7,
        # Should be dropped (the cvxopt.matrix vectors).
        "x": SimpleNamespace(),
        "y": SimpleNamespace(),
    }
    info = _flatten_info(raw)
    assert info["status"] == "optimal"
    assert info["iterations"] == 7
    assert info["primal objective"] == pytest.approx(1.5)
    assert "x" not in info and "y" not in info
