"""Per-adapter status mapping table tests.

The audit flagged that most adapters had no direct test for their
``_map_*_status`` function — only the OPTIMAL branch was exercised
indirectly through ``test_adapter_reports_kkt_for_small_qp``. A
regression in any non-optimal branch (timeout, infeasible, etc.)
would have shipped silently.

Each test below imports the adapter behind ``importorskip`` and
walks the full mapping table.
"""

from __future__ import annotations

import pytest

from solver_benchmarks.core import status


def test_highs_status_mapping_table_covers_all_model_statuses():
    highspy = pytest.importorskip("highspy")
    from solver_benchmarks.solvers.highs_adapter import _map_highs_status

    model = highspy.HighsModelStatus
    expected = {
        model.kOptimal: status.OPTIMAL,
        model.kInfeasible: status.PRIMAL_INFEASIBLE,
        model.kUnbounded: status.DUAL_INFEASIBLE,
        model.kUnboundedOrInfeasible: status.PRIMAL_OR_DUAL_INFEASIBLE,
        model.kIterationLimit: status.MAX_ITER_REACHED,
        model.kTimeLimit: status.TIME_LIMIT,
    }
    for input_status, expected_canonical in expected.items():
        assert _map_highs_status(input_status, highspy) == expected_canonical
    # An unrecognized (made-up) status value falls through to
    # SOLVER_ERROR rather than crashing.
    assert _map_highs_status(-99999, highspy) == status.SOLVER_ERROR


def test_qtqp_status_mapping_table():
    pytest.importorskip("qtqp")
    from solver_benchmarks.solvers.qtqp_adapter import _map_qtqp_status

    expected = {
        "solved": status.OPTIMAL,
        "infeasible": status.PRIMAL_INFEASIBLE,
        "unbounded": status.DUAL_INFEASIBLE,
        "hit_max_iter": status.MAX_ITER_REACHED,
        "unfinished": status.SOLVER_ERROR,
        "failed": status.SOLVER_ERROR,
    }
    for raw, expected_canonical in expected.items():
        assert _map_qtqp_status(raw) == expected_canonical
    assert _map_qtqp_status("does_not_exist") == status.SOLVER_ERROR


def test_proxqp_status_mapping_table():
    proxsuite = pytest.importorskip("proxsuite")
    from solver_benchmarks.solvers.proxqp_adapter import _map_proxqp_status

    proxqp = proxsuite.proxqp
    expected = {
        proxqp.PROXQP_SOLVED: status.OPTIMAL,
        proxqp.PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: status.OPTIMAL_INACCURATE,
        proxqp.PROXQP_MAX_ITER_REACHED: status.MAX_ITER_REACHED,
        proxqp.PROXQP_PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE,
        proxqp.PROXQP_DUAL_INFEASIBLE: status.DUAL_INFEASIBLE,
    }
    for raw, expected_canonical in expected.items():
        assert _map_proxqp_status(raw, proxsuite) == expected_canonical


def test_piqp_status_mapping_table():
    piqp = pytest.importorskip("piqp")
    from solver_benchmarks.solvers.piqp_adapter import _map_piqp_status

    expected = {
        piqp.PIQP_SOLVED: status.OPTIMAL,
        piqp.PIQP_MAX_ITER_REACHED: status.MAX_ITER_REACHED,
        piqp.PIQP_PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE,
        piqp.PIQP_DUAL_INFEASIBLE: status.DUAL_INFEASIBLE,
    }
    for raw, expected_canonical in expected.items():
        assert _map_piqp_status(raw, piqp) == expected_canonical


def test_mosek_status_mapping_keeps_optimum_when_time_limit_also_fired():
    """Audit-driven regression: MOSEK could legitimately conclude with
    solsta=optimal but also raise trm_max_time when its internal
    bookkeeping ticked over after the solve. The adapter must report
    OPTIMAL, not TIME_LIMIT, in that case."""
    mosek = pytest.importorskip("mosek")
    from solver_benchmarks.solvers.mosek_adapter import _map_mosek_status

    # solsta.optimal + trm_max_time -> OPTIMAL, not TIME_LIMIT.
    assert (
        _map_mosek_status(mosek.solsta.optimal, mosek.rescode.trm_max_time, mosek)
        == status.OPTIMAL
    )
    # solsta.unknown + trm_max_time -> TIME_LIMIT (the original
    # behavior, preserved for the non-optimal case).
    assert (
        _map_mosek_status(mosek.solsta.unknown, mosek.rescode.trm_max_time, mosek)
        == status.TIME_LIMIT
    )
    # solsta.unknown + trm_max_iterations -> MAX_ITER_REACHED.
    assert (
        _map_mosek_status(
            mosek.solsta.unknown, mosek.rescode.trm_max_iterations, mosek
        )
        == status.MAX_ITER_REACHED
    )
    # solsta.optimal + ok termination -> OPTIMAL.
    assert (
        _map_mosek_status(mosek.solsta.optimal, mosek.rescode.ok, mosek)
        == status.OPTIMAL
    )
    # solsta.prim_infeas_cer overrides termination code.
    assert (
        _map_mosek_status(
            mosek.solsta.prim_infeas_cer, mosek.rescode.trm_max_time, mosek
        )
        == status.PRIMAL_INFEASIBLE
    )


def test_pdlp_status_mapping_table():
    pytest.importorskip("ortools")
    from ortools.pdlp import solve_log_pb2

    from solver_benchmarks.solvers.pdlp_adapter import _map_status

    termination = solve_log_pb2.TerminationReason

    class _FakeSolveLog:
        def __init__(self, reason):
            self.termination_reason = reason

    expected = {
        termination.TERMINATION_REASON_OPTIMAL: status.OPTIMAL,
        termination.TERMINATION_REASON_PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE,
        termination.TERMINATION_REASON_DUAL_INFEASIBLE: status.DUAL_INFEASIBLE,
        termination.TERMINATION_REASON_PRIMAL_OR_DUAL_INFEASIBLE: status.PRIMAL_OR_DUAL_INFEASIBLE,
        termination.TERMINATION_REASON_TIME_LIMIT: status.TIME_LIMIT,
        termination.TERMINATION_REASON_ITERATION_LIMIT: status.MAX_ITER_REACHED,
        termination.TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT: status.MAX_ITER_REACHED,
    }
    for reason, expected_canonical in expected.items():
        assert _map_status(_FakeSolveLog(reason)) == expected_canonical


def test_clarabel_status_mapping_table():
    """Walk every Clarabel status string through the production
    mapping helper so a regression in any non-optimal branch surfaces
    here instead of only when that branch is hit at runtime.

    Pre-fix this test only re-derived the mapping in a local dict and
    never called any adapter code, so renaming a canonical value or
    dropping a key from the production mapping would not have failed.
    """
    pytest.importorskip("clarabel")
    from solver_benchmarks.solvers.clarabel_adapter import _map_clarabel_status

    expected = {
        "Solved": status.OPTIMAL,
        "AlmostSolved": status.OPTIMAL_INACCURATE,
        "PrimalInfeasible": status.PRIMAL_INFEASIBLE,
        "AlmostPrimalInfeasible": status.PRIMAL_INFEASIBLE_INACCURATE,
        "DualInfeasible": status.DUAL_INFEASIBLE,
        "AlmostDualInfeasible": status.DUAL_INFEASIBLE_INACCURATE,
        "MaxIterations": status.MAX_ITER_REACHED,
        "MaxTime": status.TIME_LIMIT,
    }
    for raw, expected_canonical in expected.items():
        assert _map_clarabel_status(raw) == expected_canonical
    # Unknown / future statuses fall through to SOLVER_ERROR rather
    # than crashing.
    assert _map_clarabel_status("ThisStatusDoesNotExist") == status.SOLVER_ERROR
    assert _map_clarabel_status("") == status.SOLVER_ERROR
