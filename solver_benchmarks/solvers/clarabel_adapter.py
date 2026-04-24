"""Clarabel adapter."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone
from .base import SolverAdapter, SolverUnavailable


class ClarabelSolverAdapter(SolverAdapter):
    solver_name = "clarabel"
    supported_problem_kinds = {QP, CONE}

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

        if problem.kind == QP:
            p, q, a, b, cones = _qp_data(problem.qp, clarabel)
        elif problem.kind == CONE:
            native = _cone_data(problem.cone, clarabel)
            if isinstance(native, SolverResult):
                return native
            p, q, a, b, cones = native
        else:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": f"Clarabel does not support problem kind {problem.kind!r}"},
            )

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


def _qp_data(qp: dict, clarabel):
    a, b, z = qp_to_nonnegative_cone(qp)
    p = sp.csc_matrix(qp["P"])
    q = np.asarray(qp["q"], dtype=float)
    cones = []
    if z:
        cones.append(clarabel.ZeroConeT(z))
    nonnegative = a.shape[0] - z
    if nonnegative:
        cones.append(clarabel.NonnegativeConeT(nonnegative))
    return p, q, sp.csc_matrix(a), b, cones


def _cone_data(cone_problem: dict, clarabel):
    a = sp.csc_matrix(cone_problem["A"])
    b = np.asarray(cone_problem["b"], dtype=float)
    q = np.asarray(cone_problem["q"], dtype=float)
    p = cone_problem.get("P")
    if p is None:
        p = sp.csc_matrix((a.shape[1], a.shape[1]))
    else:
        p = sp.csc_matrix(p)

    parsed = _parse_cones(dict(cone_problem["cone"]), a.shape[0], clarabel)
    if isinstance(parsed, SolverResult):
        return parsed
    cones, keep_rows = parsed
    if len(keep_rows) != a.shape[0]:
        a = a[keep_rows, :]
        b = b[keep_rows]
    return p, q, sp.csc_matrix(a), b, cones


def _parse_cones(cone: dict, row_count: int, clarabel):
    cones = []
    keep_rows: list[int] = []
    row = 0
    for name, value in cone.items():
        if name in ("f", "z"):
            dim = int(value)
            if dim:
                cones.append(clarabel.ZeroConeT(dim))
                keep_rows.extend(range(row, row + dim))
            row += dim
            continue
        if name == "l":
            dim = int(value)
            if dim:
                cones.append(clarabel.NonnegativeConeT(dim))
                keep_rows.extend(range(row, row + dim))
            row += dim
            continue
        if name == "q":
            for dim in _as_list(value):
                dim = int(dim)
                if dim:
                    cones.append(clarabel.SecondOrderConeT(dim))
                    keep_rows.extend(range(row, row + dim))
                row += dim
            continue
        if name == "s":
            for dim in _as_list(value):
                dim = int(dim)
                triangle_dim = dim * (dim + 1) // 2
                if dim:
                    cones.append(clarabel.PSDTriangleConeT(dim))
                    keep_rows.extend(range(row, row + triangle_dim))
                row += triangle_dim
            continue
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={"reason": f"Clarabel adapter does not support cone key {name!r}"},
        )

    if row != row_count:
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={
                "reason": "Clarabel cone dimensions do not match A rows",
                "cone_rows": row,
                "a_rows": row_count,
            },
        )
    return cones, keep_rows


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return value
    return [value]
