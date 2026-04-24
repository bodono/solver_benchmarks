"""OSQP adapter."""

from __future__ import annotations

from pathlib import Path
import time

import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from .base import SolverAdapter, SolverUnavailable


class OSQPSolverAdapter(SolverAdapter):
    solver_name = "osqp"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import osqp  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import osqp
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the osqp extra to use OSQP") from exc

        qp = problem.qp
        settings = dict(self.settings)
        settings.setdefault("verbose", False)
        p = sp.csc_matrix(qp["P"])
        a = sp.csc_matrix(qp["A"])

        solver = osqp.OSQP()
        start = time.perf_counter()
        solver.setup(p, qp["q"], a, qp["l"], qp["u"], **settings)
        raw = solver.solve()
        elapsed = time.perf_counter() - start

        status_map = {
            osqp.constant("OSQP_SOLVED"): status.OPTIMAL,
            osqp.constant("OSQP_SOLVED_INACCURATE"): status.OPTIMAL_INACCURATE,
            osqp.constant("OSQP_MAX_ITER_REACHED"): status.MAX_ITER_REACHED,
            osqp.constant("OSQP_PRIMAL_INFEASIBLE"): status.PRIMAL_INFEASIBLE,
            osqp.constant("OSQP_DUAL_INFEASIBLE"): status.DUAL_INFEASIBLE,
            osqp.constant("OSQP_TIME_LIMIT_REACHED"): status.TIME_LIMIT,
        }
        mapped = status_map.get(raw.info.status_val, status.SOLVER_ERROR)
        info = {
            key: getattr(raw.info, key, None)
            for key in [
                "status",
                "status_val",
                "obj_val",
                "prim_res",
                "dual_res",
                "iter",
                "rho_updates",
                "rho_estimate",
                "setup_time",
                "solve_time",
                "update_time",
                "polish_time",
                "run_time",
            ]
            if hasattr(raw.info, key)
        }
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(raw.info.obj_val),
            iterations=_maybe_int(raw.info.iter),
            run_time_seconds=elapsed,
            setup_time_seconds=_maybe_float(getattr(raw.info, "setup_time", None)),
            solve_time_seconds=_maybe_float(getattr(raw.info, "solve_time", None)),
            info=info,
            extra={
                "status_polish": getattr(raw.info, "status_polish", None),
                "rho_updates": getattr(raw.info, "rho_updates", None),
                "rho_estimate": getattr(raw.info, "rho_estimate", None),
            },
        )


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
