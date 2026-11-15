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


# A solve stopped by its iteration or time limit is reported as
# max_iter_reached / time_limit even when the solver salvages the last (or
# best) iterate as "solved inaccurate" / "almost solved"; the solver's own
# word stays on info["raw_status"]. Scoring treats optimal_inaccurate and the
# limit statuses alike (neither counts as solved; kkt-verify --promote can
# promote either), so this changes the label, not the score.


def test_osqp_inaccurate_statuses_at_a_limit_map_to_that_limit():
    osqp = pytest.importorskip("osqp")
    from solver_benchmarks.solvers.osqp_adapter import _map_osqp_status

    for name in (
        "OSQP_SOLVED_INACCURATE",
        "OSQP_PRIMAL_INFEASIBLE_INACCURATE",
        "OSQP_DUAL_INFEASIBLE_INACCURATE",
    ):
        value = osqp.constant(name)
        assert _map_osqp_status(value, osqp, hit_iteration_limit=True) == status.MAX_ITER_REACHED
        assert _map_osqp_status(value, osqp, hit_iteration_limit=False) == status.TIME_LIMIT
    expected = {
        "OSQP_SOLVED": status.OPTIMAL,
        "OSQP_MAX_ITER_REACHED": status.MAX_ITER_REACHED,
        "OSQP_TIME_LIMIT_REACHED": status.TIME_LIMIT,
        "OSQP_PRIMAL_INFEASIBLE": status.PRIMAL_INFEASIBLE,
        "OSQP_DUAL_INFEASIBLE": status.DUAL_INFEASIBLE,
        "OSQP_NON_CVX": status.SOLVER_ERROR,
    }
    for name, canonical in expected.items():
        assert _map_osqp_status(osqp.constant(name), osqp, hit_iteration_limit=False) == canonical
    assert _map_osqp_status(-12345, osqp, hit_iteration_limit=False) == status.SOLVER_ERROR


def test_osqp_solved_inaccurate_at_max_iter_is_reported_as_max_iter(tmp_path):
    # OSQP only reports "solved inaccurate" when it hits max_iter or its time
    # limit and the final iterate passes its 10x-looser check. Sweep max_iter
    # on a random QP until that happens.
    pytest.importorskip("osqp")
    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.core.problem import QP, ProblemData
    from solver_benchmarks.solvers.osqp_adapter import OSQPSolverAdapter

    rng = np.random.default_rng(0)
    n, m = 50, 80
    b = rng.normal(size=(n, n))
    qp = {
        "P": sp.csc_matrix(b.T @ b / n + 0.1 * np.eye(n)),
        "q": rng.normal(size=n),
        "A": sp.csc_matrix(rng.normal(size=(m, n))),
        "l": -rng.uniform(size=m),
        "u": rng.uniform(size=m),
    }
    problem = ProblemData("t", "p", QP, qp)
    seen = 0
    for max_iter in range(200, 400, 7):
        settings = {
            "verbose": False, "max_iter": max_iter, "check_termination": 1,
            "eps_abs": 1e-6, "eps_rel": 1e-6,
        }
        result = OSQPSolverAdapter(settings).solve(problem, tmp_path)
        if result.info["raw_status"] == "solved inaccurate":
            seen += 1
            assert result.status == status.MAX_ITER_REACHED
            assert result.iterations == max_iter
            assert result.objective_value is None
            assert result.kkt is not None
    if not seen:
        pytest.skip("this OSQP build never stopped at 'solved inaccurate' here")


def test_qtqp_almost_solved_maps_by_why_the_solve_stopped():
    from solver_benchmarks.solvers.qtqp_adapter import _map_qtqp_status

    # Best-iterate salvage after running out of iterations: a limit status.
    assert _map_qtqp_status("almost_solved", hit_iteration_limit=True) == status.MAX_ITER_REACHED
    # Salvage after a numerical breakdown: the near-solution is inaccurate.
    assert _map_qtqp_status("almost_solved", hit_iteration_limit=False) == status.OPTIMAL_INACCURATE
    assert _map_qtqp_status("almost_solved") == status.OPTIMAL_INACCURATE
    assert _map_qtqp_status("hit_max_iter") == status.MAX_ITER_REACHED


@pytest.mark.parametrize("last_attempt, expected", [
    (9, status.MAX_ITER_REACHED),
    (4, status.OPTIMAL_INACCURATE),
])
def test_qtqp_adapter_reports_almost_solved_at_max_iter_as_max_iter(
    monkeypatch, tmp_path, last_attempt, expected
):
    import sys
    from types import SimpleNamespace

    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.core.problem import QP, ProblemData
    from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter

    class FakeQTQP:
        def __init__(self, a, b, c, z, p):
            self.n, self.m = a.shape[1], a.shape[0]

        def solve(self, max_iter=100, **kwargs):
            # qtqp keeps the zero-based attempt index on ``it``.
            self.it = last_attempt
            return SimpleNamespace(
                status=SimpleNamespace(value="almost_solved"),
                stats=[],
                iterations=last_attempt,
                x=np.zeros(self.n),
                y=np.zeros(self.m),
                s=np.ones(self.m),
            )

    monkeypatch.setitem(sys.modules, "qtqp", SimpleNamespace(QTQP=FakeQTQP))
    qp = {
        "P": sp.eye(2, format="csc"),
        "q": np.ones(2),
        "A": sp.eye(2, format="csc"),
        "l": -np.ones(2),
        "u": np.ones(2),
    }
    result = QTQPSolverAdapter({"max_iter": 10}).solve(ProblemData("t", "p", QP, qp), tmp_path)
    assert result.status == expected
    assert result.info["raw_status"] == "almost_solved"


def test_proxqp_closest_primal_feasible_is_not_an_infeasibility_verdict(tmp_path):
    # ProxSuite documents PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE as "the
    # closest (in L2 sense) feasible problem is solved", and its sparse
    # backend returns it for a feasible problem too, with the true solution.
    # So it stays optimal_inaccurate (not primal_infeasible_inaccurate), and
    # the KKT check of the returned point decides whether kkt-verify promotes.
    pytest.importorskip("proxsuite")
    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.core.problem import QP, ProblemData
    from solver_benchmarks.solvers.proxqp_adapter import ProxQPSolverAdapter

    qp = {
        "P": sp.eye(2, format="csc"),
        "q": np.ones(2),
        "A": sp.eye(2, format="csc"),
        "l": -5 * np.ones(2),
        "u": 5 * np.ones(2),
    }
    settings = {
        "verbose": False,
        "primal_infeasibility_solving": True,
        "check_duality_gap": False,
    }
    result = ProxQPSolverAdapter(settings).solve(ProblemData("t", "p", QP, qp), tmp_path)
    assert result.info["status"].endswith("PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE")
    assert result.status == status.OPTIMAL_INACCURATE
    assert result.kkt["primal_res_rel"] < 1e-4 and result.kkt["dual_res_rel"] < 1e-4
