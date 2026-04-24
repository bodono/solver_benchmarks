"""Clarabel adapter."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone
from .base import SolverAdapter, SolverUnavailable


class ClarabelSolverAdapter(SolverAdapter):
    solver_name = "clarabel"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import clarabel  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import clarabel
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the clarabel extra to use Clarabel") from exc

        qp = problem.qp
        a, b, z = qp_to_nonnegative_cone(qp)
        p = sp.csc_matrix(qp["P"])
        q = np.asarray(qp["q"], dtype=float)
        cones = [clarabel.ZeroConeT(z), clarabel.NonnegativeConeT(a.shape[0] - z)]
        settings = clarabel.DefaultSettings()
        settings.verbose = bool(self.settings.get("verbose", False))
        for key, value in self.settings.items():
            if key == "verbose":
                continue
            if hasattr(settings, key):
                setattr(settings, key, value)

        start = time.perf_counter()
        solver = clarabel.DefaultSolver(p, q, sp.csc_matrix(a), b, cones, settings)
        solution = solver.solve()
        elapsed = time.perf_counter() - start
        mapped = {
            "Solved": status.OPTIMAL,
            "AlmostSolved": status.OPTIMAL_INACCURATE,
            "PrimalInfeasible": status.PRIMAL_INFEASIBLE,
            "AlmostPrimalInfeasible": status.PRIMAL_INFEASIBLE_INACCURATE,
            "DualInfeasible": status.DUAL_INFEASIBLE,
            "AlmostDualInfeasible": status.DUAL_INFEASIBLE_INACCURATE,
            "MaxIterations": status.MAX_ITER_REACHED,
            "MaxTime": status.TIME_LIMIT,
        }.get(str(solution.status), status.SOLVER_ERROR)
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(getattr(solution, "obj_val", None)),
            iterations=_maybe_int(getattr(solution, "iterations", None)),
            run_time_seconds=elapsed,
            info={
                "raw_status": str(solution.status),
                "r_prim": getattr(solution, "r_prim", None),
                "r_dual": getattr(solution, "r_dual", None),
                "solve_time": getattr(solution, "solve_time", None),
            },
        )


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
