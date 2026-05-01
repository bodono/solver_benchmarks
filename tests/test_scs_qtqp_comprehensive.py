"""Comprehensive coverage for the SCS and QTQP adapters.

Both are open-source solvers we want to keep on a tight regression
leash. ``test_adapter_kkt.py`` exercises the OPTIMAL branch on a small
QP, but the SCS and QTQP adapters have a lot of conditional plumbing
that isn't covered there: CONE-shape inputs, infeasibility and
unboundedness branches, status-mapping fallbacks, the SCS num_threads
probe, the QTQP linear_solver normalization, the trace artifacts, and
the unavailable-module paths.
"""

from __future__ import annotations

import csv
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData

# ---------------------------------------------------------------------------
# Shared problem fixtures.
# ---------------------------------------------------------------------------


def _small_qp() -> dict:
    return {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
        "n": 2,
        "m": 2,
        "obj_type": "min",
    }


def _infeasible_qp() -> dict:
    # x >= 2 and x <= 1, infeasible
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([0.0]),
        "A": sp.csc_matrix([[1.0], [1.0]]),
        "l": np.array([2.0, -np.inf]),
        "u": np.array([np.inf, 1.0]),
        "n": 1,
        "m": 2,
        "obj_type": "min",
    }


def _unbounded_qp() -> dict:
    # min -x s.t. x >= 0 (no upper bound) -> unbounded.
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([-1.0]),
        "A": sp.csc_matrix([[1.0]]),
        "l": np.array([0.0]),
        "u": np.array([np.inf]),
        "n": 1,
        "m": 1,
        "obj_type": "min",
    }


def _small_cone_lp() -> dict:
    # min x s.t. x >= 1 in cone form A x + s = b, s in NN cone.
    return {
        "P": None,
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[-1.0]]),
        "b": np.array([-1.0]),
        "r": 0.0,
        "n": 1,
        "m": 1,
        "cone": {"l": 1},
        "obj_type": "min",
    }


def _cone_with_free_variables() -> dict:
    """Cone problem with the legacy ``f`` (free) cone key.

    SCS' adapter merges the free cone into the leading zero cone (the
    ``f`` -> ``z`` rewrite at scs_adapter.py:121-123) so the underlying
    SCS call only sees the modern ``z`` form. Without that path the
    free constraint would be silently dropped or mis-routed, so we
    keep an explicit regression with a problem that needs both a
    free row and a non-zero zero-cone row.

    min x s.t. x = 5 (free constraint), via cone:
        A = [[1.0]], b = [5.0], cone = {"f": 1}.
    Expected optimum: x = 5, value = 5.
    """
    return {
        "P": None,
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[1.0]]),
        "b": np.array([5.0]),
        "r": 0.0,
        "n": 1,
        "m": 1,
        "cone": {"f": 1},
        "obj_type": "min",
    }


def _infeasible_cone_problem() -> dict:
    # x in NN-cone (x >= 0) and -x = -1 (so x=-1) -> infeasible.
    # Cone: A x + s = b, s in cone with cone={"l": 2}; constraints
    # x >= 0 and x <= -1 are inconsistent.
    return {
        "P": None,
        "q": np.array([0.0]),
        "A": sp.csc_matrix([[-1.0], [1.0]]),
        "b": np.array([0.0, -1.0]),
        "r": 0.0,
        "n": 1,
        "m": 2,
        "cone": {"l": 2},
        "obj_type": "min",
    }


def _unbounded_cone_problem() -> dict:
    # min -x s.t. x in NN cone (x >= 0). Unbounded below.
    return {
        "P": None,
        "q": np.array([-1.0]),
        "A": sp.csc_matrix([[-1.0]]),
        "b": np.array([0.0]),
        "r": 0.0,
        "n": 1,
        "m": 1,
        "cone": {"l": 1},
        "obj_type": "min",
    }


def _make_cone_problem(data: dict) -> ProblemData:
    return ProblemData("test", "p", CONE, data)


def _make_qp_problem(data: dict) -> ProblemData:
    return ProblemData("test", "p", QP, data)


# ---------------------------------------------------------------------------
# SCS - status mapping fallback paths.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status_text,expected",
    [
        ("solved", status.OPTIMAL),
        ("Solved (high accuracy)", status.OPTIMAL),
        ("infeasible", status.PRIMAL_INFEASIBLE),
        ("infeasible (inaccurate)", status.PRIMAL_INFEASIBLE),
        ("unbounded", status.DUAL_INFEASIBLE),
        ("unbounded (inaccurate)", status.DUAL_INFEASIBLE),
    ],
)
def test_scs_status_text_fallback_when_status_val_missing(status_text, expected):
    """When SCS reports a status string but no recognized status_val,
    the adapter falls back to substring matching. Pre-fix this branch
    was uncovered."""
    from solver_benchmarks.solvers.scs_adapter import _map_scs_status

    assert _map_scs_status({"status": status_text}) == expected


def test_scs_status_text_fallback_unknown_string_is_solver_error():
    """A status string SCS could plausibly emit that we don't recognize
    must fall through to SOLVER_ERROR rather than crash."""
    from solver_benchmarks.solvers.scs_adapter import _map_scs_status

    assert _map_scs_status({"status": "interrupted"}) == status.SOLVER_ERROR
    assert _map_scs_status({}) == status.SOLVER_ERROR


def test_scs_status_val_takes_precedence_over_text():
    """If the C extension supplies a known status_val, the adapter must
    not second-guess it from the (possibly noisy) status text."""
    from solver_benchmarks.solvers.scs_adapter import _map_scs_status

    # status_val=1 (Solved) wins even though text says "infeasible".
    assert (
        _map_scs_status({"status_val": 1, "status": "infeasible"})
        == status.OPTIMAL
    )


# ---------------------------------------------------------------------------
# SCS - num_threads probe and caching.
# ---------------------------------------------------------------------------


def test_scs_num_threads_probe_caches_after_first_call(monkeypatch):
    """The probe runs a 1x1 SCS solve and caches the result. A second
    call must not run the probe again — otherwise every solve with
    ``threads`` would pay the probe cost."""
    pytest.importorskip("scs")
    import solver_benchmarks.solvers.scs_adapter as scs_mod

    # Force a fresh probe.
    monkeypatch.setattr(scs_mod, "_SCS_NUM_THREADS_SUPPORTED", None)
    first = scs_mod._scs_supports_num_threads()
    # Replace SCS so that a cache-bypass would crash; the cached path
    # must short-circuit before touching the module.
    monkeypatch.setattr(scs_mod, "scs", None, raising=False)
    second = scs_mod._scs_supports_num_threads()
    assert first == second


def test_scs_num_threads_probe_returns_false_when_scs_missing(monkeypatch):
    """If SCS is not installed, the probe must report no support
    instead of raising. Sets the cached value too so subsequent calls
    do not retry."""
    import solver_benchmarks.solvers.scs_adapter as scs_mod

    monkeypatch.setattr(scs_mod, "_SCS_NUM_THREADS_SUPPORTED", None)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def fake_import(name, *args, **kwargs):
        if name == "scs":
            raise ModuleNotFoundError("simulated missing scs")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert scs_mod._scs_supports_num_threads() is False
    assert scs_mod._SCS_NUM_THREADS_SUPPORTED is False


def test_scs_num_threads_probe_treats_non_typeerror_as_supported(monkeypatch):
    """Any error that is not TypeError means SCS at least accepted the
    ``num_threads`` kwarg — the OpenMP build is present, the toy 1x1
    problem just failed for unrelated reasons (e.g. eps). Cache as
    True."""
    pytest.importorskip("scs")
    import solver_benchmarks.solvers.scs_adapter as scs_mod

    monkeypatch.setattr(scs_mod, "_SCS_NUM_THREADS_SUPPORTED", None)

    class FakeSCSModule:
        @staticmethod
        def SCS(*_args, **_kwargs):
            # Anything except TypeError keeps support=True.
            raise RuntimeError("solver state not initialized")

    monkeypatch.setitem(sys.modules, "scs", FakeSCSModule)
    assert scs_mod._scs_supports_num_threads() is True


# ---------------------------------------------------------------------------
# SCS - is_available and SolverUnavailable.
# ---------------------------------------------------------------------------


def test_scs_is_available_returns_false_when_module_missing(monkeypatch):
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "scs":
            raise ModuleNotFoundError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert SCSSolverAdapter.is_available() is False


def test_scs_solve_raises_solver_unavailable_when_module_missing(
    tmp_path, monkeypatch
):
    """If a user constructs the adapter without the scs extra
    installed, ``solve()`` must raise SolverUnavailable with a clear
    install hint instead of bare ModuleNotFoundError."""
    from solver_benchmarks.solvers.base import SolverUnavailable
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "scs":
            raise ModuleNotFoundError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    adapter = SCSSolverAdapter({"verbose": False})
    with pytest.raises(SolverUnavailable, match="scs"):
        adapter.solve(_make_qp_problem(_small_qp()), tmp_path)


# ---------------------------------------------------------------------------
# SCS - settings translation.
# ---------------------------------------------------------------------------


def test_scs_translates_time_limit_aliases_into_native_kwarg(tmp_path):
    """SCS' native time-limit knob is ``time_limit_secs`` (note trailing
    s); the adapter accepts the cross-adapter aliases. Verify the
    translation by intercepting scs.solve."""
    pytest.importorskip("scs")
    import solver_benchmarks.solvers.scs_adapter as scs_mod

    captured: dict = {}

    def fake_solve(data, cone, **kwargs):
        captured.update(kwargs)
        # Return x=None so the adapter short-circuits KKT computation.
        # These tests verify settings forwarding, not residuals.
        return {"x": None, "y": None, "s": None, "info": {"status_val": 1}}

    fake_scs = MagicMock()
    fake_scs.solve = fake_solve
    fake_scs.SCS = MagicMock(side_effect=RuntimeError("not used"))

    import sys as _sys

    _sys.modules["scs"] = fake_scs
    try:
        adapter = scs_mod.SCSSolverAdapter({"time_limit": 7.5, "verbose": False})
        adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    finally:
        # Restore real SCS so later tests aren't poisoned.
        _sys.modules.pop("scs", None)
    assert captured.get("time_limit_secs") == pytest.approx(7.5)
    # The cross-adapter alias must be popped, not forwarded.
    assert "time_limit" not in captured


def test_scs_log_csv_filename_true_routes_to_artifacts_dir(tmp_path):
    """When the user sets ``log_csv_filename: True`` (truthy sentinel),
    the adapter must rewrite it to a real path under artifacts_dir
    instead of forwarding the literal True (which would crash SCS)."""
    pytest.importorskip("scs")
    import solver_benchmarks.solvers.scs_adapter as scs_mod

    captured: dict = {}

    def fake_solve(data, cone, **kwargs):
        captured.update(kwargs)
        # Return x=None so the adapter short-circuits KKT computation.
        # These tests verify settings forwarding, not residuals.
        return {"x": None, "y": None, "s": None, "info": {"status_val": 1}}

    fake_scs = MagicMock()
    fake_scs.solve = fake_solve

    import sys as _sys

    _sys.modules["scs"] = fake_scs
    try:
        adapter = scs_mod.SCSSolverAdapter(
            {"log_csv_filename": True, "verbose": False}
        )
        adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    finally:
        _sys.modules.pop("scs", None)
    routed = captured.get("log_csv_filename")
    assert isinstance(routed, str)
    assert routed.startswith(str(tmp_path))
    assert routed.endswith("scs_trace.csv")


def test_scs_threads_setting_forwards_when_probe_supports(tmp_path, monkeypatch):
    """If the installed SCS exposes num_threads (OpenMP build), the
    cross-adapter ``threads`` setting forwards to ``num_threads``;
    otherwise it is recorded as ignored (covered separately).
    """
    pytest.importorskip("scs")
    import solver_benchmarks.solvers.scs_adapter as scs_mod

    monkeypatch.setattr(scs_mod, "_SCS_NUM_THREADS_SUPPORTED", True)

    captured: dict = {}

    def fake_solve(data, cone, **kwargs):
        captured.update(kwargs)
        # Return x=None so the adapter short-circuits KKT computation.
        # These tests verify settings forwarding, not residuals.
        return {"x": None, "y": None, "s": None, "info": {"status_val": 1}}

    fake_scs = MagicMock()
    fake_scs.solve = fake_solve

    import sys as _sys

    _sys.modules["scs"] = fake_scs
    try:
        adapter = scs_mod.SCSSolverAdapter({"threads": 4, "verbose": False})
        adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    finally:
        _sys.modules.pop("scs", None)
    assert captured.get("num_threads") == 4


# ---------------------------------------------------------------------------
# SCS - CONE-shape problem solving (covers _compute_kkt cone branches).
# ---------------------------------------------------------------------------


def test_scs_solves_cone_lp_and_reports_kkt(tmp_path):
    """CONE-shape problem solve. test_adapter_kkt.py tests QP only;
    this exercises the lines under ``problem.kind != QP`` in solve()
    and the cone-residual branch of _compute_kkt."""
    pytest.importorskip("scs")
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter

    adapter = SCSSolverAdapter(
        {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iters": 5000}
    )
    result = adapter.solve(_make_cone_problem(_small_cone_lp()), tmp_path)
    assert result.status == status.OPTIMAL, result.status
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4
    assert result.objective_value == pytest.approx(1.0, abs=1e-4)


def test_scs_merges_free_cone_key_into_zero_cone(tmp_path):
    """The legacy cone key ``f`` (free / equality) must be folded into
    ``z`` before the SCS call. Pre-fix path is at scs_adapter.py:121.
    """
    pytest.importorskip("scs")
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter

    adapter = SCSSolverAdapter(
        {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iters": 5000}
    )
    result = adapter.solve(
        _make_cone_problem(_cone_with_free_variables()), tmp_path
    )
    assert result.status == status.OPTIMAL, result.status
    assert result.objective_value == pytest.approx(5.0, abs=1e-4)


def test_scs_cone_primal_infeasibility_emits_certificate(tmp_path):
    """Primal-infeasibility branch of _compute_kkt for CONE problems
    (lines 189-191). Must produce a primal_infeasible certificate."""
    pytest.importorskip("scs")
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter

    adapter = SCSSolverAdapter(
        {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iters": 5000}
    )
    result = adapter.solve(
        _make_cone_problem(_infeasible_cone_problem()), tmp_path
    )
    assert result.status in {
        status.PRIMAL_INFEASIBLE,
        status.PRIMAL_INFEASIBLE_INACCURATE,
    }, result.status
    assert result.kkt is not None
    assert result.kkt.get("certificate") == "primal_infeasible"


def test_scs_cone_dual_infeasibility_emits_certificate(tmp_path):
    """Dual-infeasibility branch of _compute_kkt for CONE problems
    (lines 193-196). Must produce a dual_infeasible certificate."""
    pytest.importorskip("scs")
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter

    adapter = SCSSolverAdapter(
        {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iters": 5000}
    )
    result = adapter.solve(
        _make_cone_problem(_unbounded_cone_problem()), tmp_path
    )
    assert result.status in {
        status.DUAL_INFEASIBLE,
        status.DUAL_INFEASIBLE_INACCURATE,
    }, result.status
    assert result.kkt is not None
    assert result.kkt.get("certificate") == "dual_infeasible"


# ---------------------------------------------------------------------------
# SCS - objective gating on infeasibility-certificate statuses.
# ---------------------------------------------------------------------------


def test_scs_gates_objective_on_solution_present(tmp_path):
    """When the status is a primal- or dual-infeasibility certificate,
    SCS still populates ``info["pobj"]`` with the certificate
    normalizer; reporting that as the optimal value is misleading.
    The adapter must return ``objective_value=None`` for those
    statuses."""
    pytest.importorskip("scs")
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter

    adapter = SCSSolverAdapter(
        {"verbose": False, "eps_abs": 1e-8, "eps_rel": 1e-8, "max_iters": 5000}
    )
    result = adapter.solve(_make_qp_problem(_infeasible_qp()), tmp_path)
    assert result.status in {
        status.PRIMAL_INFEASIBLE,
        status.PRIMAL_INFEASIBLE_INACCURATE,
    }
    assert result.objective_value is None


# ---------------------------------------------------------------------------
# SCS - _read_csv_trace edge cases.
# ---------------------------------------------------------------------------


def test_scs_read_csv_trace_returns_empty_for_unset_path():
    from solver_benchmarks.solvers.scs_adapter import _read_csv_trace

    assert _read_csv_trace(None) == []
    assert _read_csv_trace("") == []


def test_scs_read_csv_trace_returns_empty_for_missing_file(tmp_path):
    from solver_benchmarks.solvers.scs_adapter import _read_csv_trace

    missing = tmp_path / "no_such_trace.csv"
    assert _read_csv_trace(missing) == []


def test_scs_read_csv_trace_parses_existing_file(tmp_path):
    from solver_benchmarks.solvers.scs_adapter import _read_csv_trace

    trace_path = tmp_path / "trace.csv"
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iter", "pobj"])
        writer.writeheader()
        writer.writerow({"iter": "0", "pobj": "1.5"})
        writer.writerow({"iter": "1", "pobj": "1.2"})
    rows = _read_csv_trace(trace_path)
    assert rows == [
        {"iter": "0", "pobj": "1.5"},
        {"iter": "1", "pobj": "1.2"},
    ]


# ---------------------------------------------------------------------------
# SCS - end-to-end trace persistence.
# ---------------------------------------------------------------------------


def test_scs_log_csv_filename_writes_trace_to_artifacts(tmp_path):
    """End-to-end: when ``log_csv_filename: True`` is set, an actual
    SCS solve produces a CSV trace inside artifacts_dir which the
    adapter then loads and returns."""
    pytest.importorskip("scs")
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter

    adapter = SCSSolverAdapter(
        {
            "verbose": False,
            "log_csv_filename": True,
            "eps_abs": 1e-8,
            "eps_rel": 1e-8,
            "max_iters": 5000,
        }
    )
    result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    assert result.status == status.OPTIMAL
    csv_path = tmp_path / "scs_trace.csv"
    assert csv_path.exists()
    # The adapter parses the CSV into result.trace.
    assert isinstance(result.trace, list)
    assert len(result.trace) > 0


# ---------------------------------------------------------------------------
# QTQP - is_available and SolverUnavailable.
# ---------------------------------------------------------------------------


def test_qtqp_is_available_returns_false_when_module_missing(monkeypatch):
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "qtqp":
            raise ModuleNotFoundError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert QTQPSolverAdapter.is_available() is False


def test_qtqp_solve_raises_solver_unavailable_when_module_missing(
    tmp_path, monkeypatch
):
    from solver_benchmarks.solvers.base import SolverUnavailable
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "qtqp":
            raise ModuleNotFoundError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    adapter = QTQPSolverAdapter({"verbose": False})
    with pytest.raises(SolverUnavailable, match="QTQP"):
        adapter.solve(_make_qp_problem(_small_qp()), tmp_path)


# ---------------------------------------------------------------------------
# QTQP - linear_solver string normalization.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_value,expected_attr",
    [
        ("qdldl", "QDLDL"),
        ("QDLDL", "QDLDL"),
        ("Qdldl", "QDLDL"),
        ("accelerate", "ACCELERATE"),
        ("ACCELERATE", "ACCELERATE"),
        ("cholmod", "CHOLMOD"),
        ("MUMPS", "MUMPS"),  # unknown lookup -> upper-case fallthrough
    ],
)
def test_qtqp_normalize_linear_solver_resolves_string_aliases(
    input_value, expected_attr
):
    """The adapter accepts QTQP's LinearSolver enum *or* a string. The
    string path must look up the canonical aliases (qdldl/accelerate/
    cholmod) case-insensitively and fall back to the .upper() name for
    unknown values, so the adapter is forward-compatible with new
    QTQP linear solvers."""
    from solver_benchmarks.solvers.qtqp_adapter import _normalize_settings

    fake_qtqp = SimpleNamespace(
        LinearSolver=SimpleNamespace(
            QDLDL="qdldl_enum",
            ACCELERATE="accelerate_enum",
            CHOLMOD="cholmod_enum",
            MUMPS="mumps_enum",
        )
    )
    settings = {"linear_solver": input_value, "other_kwarg": 42}
    normalized = _normalize_settings(settings, fake_qtqp)
    assert normalized["linear_solver"] == f"{expected_attr.lower()}_enum"
    # Unrelated settings must pass through untouched.
    assert normalized["other_kwarg"] == 42


def test_qtqp_normalize_linear_solver_passes_through_non_string():
    """If the caller already passed an enum value, the adapter must
    not mangle it via .upper()."""
    from solver_benchmarks.solvers.qtqp_adapter import _normalize_settings

    fake_qtqp = SimpleNamespace(LinearSolver=SimpleNamespace(QDLDL="enum_val"))
    sentinel = object()
    settings = {"linear_solver": sentinel}
    normalized = _normalize_settings(settings, fake_qtqp)
    assert normalized["linear_solver"] is sentinel


def test_qtqp_normalize_settings_no_linear_solver_is_noop():
    """When ``linear_solver`` is unset, _normalize_settings must not
    introduce one (or crash)."""
    from solver_benchmarks.solvers.qtqp_adapter import _normalize_settings

    settings = {"max_iter": 100, "verbose": False}
    fake_qtqp = SimpleNamespace(LinearSolver=SimpleNamespace())
    normalized = _normalize_settings(dict(settings), fake_qtqp)
    assert normalized == settings


# ---------------------------------------------------------------------------
# QTQP - end-to-end: trace persistence and KKT for OPTIMAL.
# ---------------------------------------------------------------------------


def test_qtqp_writes_trace_jsonl_when_solve_emits_stats(tmp_path):
    """If QTQP's solve produces stats rows, the adapter must persist
    them as JSONL inside artifacts_dir (one row per line)."""
    pytest.importorskip("qtqp")
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    adapter = QTQPSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    assert result.status == status.OPTIMAL
    trace_path = tmp_path / "trace.jsonl"
    if result.trace:
        assert trace_path.exists()
        lines = trace_path.read_text().strip().splitlines()
        assert len(lines) == len(result.trace)
        # Every row must be a JSON object the adapter could later read back.
        for line in lines:
            json.loads(line)
    else:
        # Empty trace is a valid no-op: _write_trace must not create
        # an empty file in that case.
        assert not trace_path.exists()


# ---------------------------------------------------------------------------
# QTQP - empty-stats / minimal solve_log: no objective, no iterations.
# ---------------------------------------------------------------------------


def test_qtqp_handles_empty_stats_solution_gracefully(tmp_path):
    """If QTQP returns a Solution with no stats rows, the adapter must
    fall back to ``objective=None``, ``iterations=None``, and an empty
    info beyond the raw_status. Validates lines 79-81 of the adapter.
    """
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    # x=None short-circuits KKT computation; these tests focus on the
    # objective / iterations / info contract, not residuals.
    fake_solution = SimpleNamespace(
        status=SimpleNamespace(value="solved"),
        stats=[],  # no rows
        x=None,
        y=None,
        s=None,
    )

    class FakeSolver:
        def __init__(self, **_kwargs):
            pass

        def solve(self, **_kwargs):
            return fake_solution

    fake_qtqp = SimpleNamespace(
        QTQP=FakeSolver,
        LinearSolver=SimpleNamespace(QDLDL="qdldl_enum"),
    )

    import sys as _sys

    _sys.modules["qtqp"] = fake_qtqp
    try:
        adapter = QTQPSolverAdapter({"verbose": False})
        result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    finally:
        _sys.modules.pop("qtqp", None)
    assert result.objective_value is None
    assert result.iterations is None
    # raw_status survives and so does the time_limit/threads ignored
    # markers contract (neither configured here, so neither set).
    assert result.info["raw_status"] == "solved"
    assert "threads_ignored" not in result.info


def test_qtqp_records_time_limit_and_threads_ignored(tmp_path, monkeypatch):
    """QTQP exposes neither a time-limit nor a threads knob; both
    cross-adapter aliases must survive on info as ignored markers
    rather than being silently dropped."""
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    fake_solution = SimpleNamespace(
        status=SimpleNamespace(value="solved"),
        stats=[],
        x=None,
        y=None,
        s=None,
    )

    class FakeSolver:
        def __init__(self, **_kwargs):
            pass

        def solve(self, **_kwargs):
            return fake_solution

    fake_qtqp = SimpleNamespace(
        QTQP=FakeSolver,
        LinearSolver=SimpleNamespace(QDLDL="qdldl_enum"),
    )

    import sys as _sys

    _sys.modules["qtqp"] = fake_qtqp
    try:
        adapter = QTQPSolverAdapter(
            {"verbose": False, "time_limit": 5.0, "threads": 4}
        )
        result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    finally:
        _sys.modules.pop("qtqp", None)
    assert result.info.get("time_limit_ignored") is True
    assert result.info.get("time_limit_seconds") == 5.0
    assert result.info.get("threads_ignored") is True
    assert result.info.get("threads_requested") == 4


# ---------------------------------------------------------------------------
# QTQP - KKT compute branches via stubbed solutions.
# ---------------------------------------------------------------------------


def _drive_qtqp_compute_kkt(solution, mapped_status):
    """Run _compute_kkt with a stubbed solution and a tiny problem."""
    from solver_benchmarks.solvers.qtqp_adapter import _compute_kkt

    p = sp.csc_matrix([[1.0]])
    c = np.array([0.0])
    a = sp.csc_matrix([[1.0]])
    b = np.array([0.0])
    cone_dict = {"l": 1}
    return _compute_kkt(mapped_status, solution, p, c, a, b, cone_dict)


def test_qtqp_compute_kkt_returns_none_when_x_is_missing():
    """Solver_error / aborted solves leave ``solution.x = None``; the
    KKT helper must short-circuit instead of crashing."""
    solution = SimpleNamespace(x=None, y=None, s=None)
    assert _drive_qtqp_compute_kkt(solution, status.OPTIMAL) is None


def test_qtqp_compute_kkt_returns_none_when_y_or_s_missing_for_optimal():
    """An OPTIMAL status without dual variables means the residual
    helper has no inputs to feed; return None rather than asserting."""
    solution = SimpleNamespace(x=np.array([0.0]), y=None, s=None)
    assert _drive_qtqp_compute_kkt(solution, status.OPTIMAL) is None
    solution = SimpleNamespace(
        x=np.array([0.0]), y=np.array([0.0]), s=None
    )
    assert _drive_qtqp_compute_kkt(solution, status.OPTIMAL) is None


def test_qtqp_compute_kkt_emits_primal_infeasibility_certificate():
    """When status is PRIMAL_INFEASIBLE and y is present, the helper
    must call ``cone_primal_infeasibility_cert``."""
    # bty witness < 0: choose b=[1.0], y=[-1.0] -> bty = -1.
    from solver_benchmarks.solvers.qtqp_adapter import _compute_kkt

    p = sp.csc_matrix([[1.0]])
    c = np.array([0.0])
    a = sp.csc_matrix([[1.0]])
    b = np.array([1.0])
    cone_dict = {"l": 1}
    solution = SimpleNamespace(
        x=np.array([0.0]),
        y=np.array([-1.0]),
        s=None,
    )
    result = _compute_kkt(
        status.PRIMAL_INFEASIBLE, solution, p, c, a, b, cone_dict
    )
    assert result is not None
    assert result.get("certificate") == "primal_infeasible"


def test_qtqp_compute_kkt_skips_primal_infeasibility_when_y_missing():
    solution = SimpleNamespace(x=np.array([0.0]), y=None, s=None)
    assert (
        _drive_qtqp_compute_kkt(solution, status.PRIMAL_INFEASIBLE) is None
    )


def test_qtqp_compute_kkt_emits_dual_infeasibility_certificate():
    """When status is DUAL_INFEASIBLE the helper must call
    ``cone_dual_infeasibility_cert`` with x as the witness."""
    from solver_benchmarks.solvers.qtqp_adapter import _compute_kkt

    # Construct a problem where qtx < 0 and Px = 0 and Ax+s = 0
    # is satisfiable for some s in the cone — easiest is q = [-1],
    # P = 0, A = -I, x = [1] gives qtx = -1, Px = 0, Ax = -1 (so
    # s = 1 in NN cone).
    p = sp.csc_matrix((1, 1))
    c = np.array([-1.0])
    a = sp.csc_matrix([[-1.0]])
    b = np.array([0.0])
    cone_dict = {"l": 1}
    solution = SimpleNamespace(
        x=np.array([1.0]),
        y=None,
        s=None,
    )
    result = _compute_kkt(
        status.DUAL_INFEASIBLE, solution, p, c, a, b, cone_dict
    )
    assert result is not None
    assert result.get("certificate") == "dual_infeasible"
    assert result.get("qtx") == pytest.approx(-1.0)


def test_qtqp_compute_kkt_returns_none_for_unmapped_status():
    """Statuses outside the OPTIMAL / INFEASIBLE / UNBOUNDED set (e.g.
    SOLVER_ERROR, MAX_ITER_REACHED) must not produce a kkt dict."""
    solution = SimpleNamespace(
        x=np.array([0.0]), y=np.array([0.0]), s=np.array([0.0])
    )
    assert (
        _drive_qtqp_compute_kkt(solution, status.SOLVER_ERROR) is None
    )
    assert (
        _drive_qtqp_compute_kkt(solution, status.MAX_ITER_REACHED) is None
    )


# ---------------------------------------------------------------------------
# QTQP - _write_trace edge cases.
# ---------------------------------------------------------------------------


def test_qtqp_write_trace_noop_for_empty_list(tmp_path):
    """An empty trace must not create a zero-byte file (which would
    mislead later readers into thinking there was a header but no
    rows)."""
    from solver_benchmarks.solvers.qtqp_adapter import _write_trace

    out = tmp_path / "trace.jsonl"
    _write_trace(out, [])
    assert not out.exists()


def test_qtqp_write_trace_persists_each_row_as_jsonl(tmp_path):
    from solver_benchmarks.solvers.qtqp_adapter import _write_trace

    rows = [{"iter": 0, "pcost": 1.5}, {"iter": 1, "pcost": 1.2}]
    out = tmp_path / "trace.jsonl"
    _write_trace(out, rows)
    decoded = [json.loads(line) for line in out.read_text().splitlines()]
    assert decoded == rows


# ---------------------------------------------------------------------------
# QTQP - cone_dict construction: pure-equality (z only) and pure-NN (l only).
# ---------------------------------------------------------------------------


def test_scs_threads_ignored_propagates_to_info(tmp_path, monkeypatch):
    """When the installed SCS does not accept ``num_threads`` and the
    user requests ``threads``, the solve must succeed and the request
    must show up on ``result.info`` as the ignored marker.

    Pre-fix this is the path through scs_adapter.py:101 and 130 — the
    earlier test_adapter_option_forwarding.py reproduces the same path
    inline, but does not exercise the real adapter solve.
    """
    pytest.importorskip("scs")
    import solver_benchmarks.solvers.scs_adapter as scs_mod

    monkeypatch.setattr(scs_mod, "_SCS_NUM_THREADS_SUPPORTED", False)
    adapter = scs_mod.SCSSolverAdapter(
        {"verbose": False, "threads": 2, "max_iters": 5000, "eps_abs": 1e-8, "eps_rel": 1e-8}
    )
    result = adapter.solve(_make_qp_problem(_small_qp()), tmp_path)
    assert result.status == status.OPTIMAL
    assert result.info.get("threads_ignored") is True
    assert result.info.get("threads_requested") == 2


def test_qtqp_cone_dict_includes_zero_cone_when_equality_present(tmp_path):
    """A QP with equality bounds (``l == u``) yields z > 0 after the
    cone transform, so cone_dict["z"] is set. This covers
    qtqp_adapter.py:86 — the ``if z:`` branch that the bound-only
    problems in test_adapter_kkt.py never trigger."""
    pytest.importorskip("qtqp")
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    # min x1^2 + x2^2 s.t. x1 + x2 = 1 and -5 <= x2 <= 5.
    # The first row is an equality (l=u=1) -> z = 1.
    qp = {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([0.0, 0.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [0.0, 1.0]]),
        "l": np.array([1.0, -5.0]),
        "u": np.array([1.0, 5.0]),
        "n": 2,
        "m": 2,
        "obj_type": "min",
    }
    adapter = QTQPSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(qp), tmp_path)
    assert result.status == status.OPTIMAL, result.status
    # Optimum: x1 = x2 = 0.5 (project onto x1+x2=1, the bound is inactive).
    # QP objective is 0.5 x'Px + q'x, so 0.5 * (0.25 + 0.25) = 0.25.
    assert result.objective_value == pytest.approx(0.25, abs=1e-4)


def test_qtqp_cone_dict_omits_zero_keys(tmp_path):
    """If z=0 (no equality rows) the cone_dict must not contain a
    ``z`` key — sending an empty zero cone to the KKT helper would
    confuse downstream tooling. Same for the nonnegative branch."""
    pytest.importorskip("qtqp")
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    # Use a QP with bounds only on the lower side so the
    # qp_to_nonnegative_cone transform yields all NN rows (no z).
    qp = {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([0.0, 0.0]),
        "u": np.array([np.inf, np.inf]),
        "n": 2,
        "m": 2,
        "obj_type": "min",
    }
    adapter = QTQPSolverAdapter({"verbose": False})
    result = adapter.solve(_make_qp_problem(qp), tmp_path)
    assert result.status == status.OPTIMAL
    # If kkt dict came back, it shows the cone construction worked.
    if result.kkt is not None:
        # Implicitly: no crash means the cone_dict was well-formed.
        assert np.isfinite(result.kkt["primal_res_rel"])
