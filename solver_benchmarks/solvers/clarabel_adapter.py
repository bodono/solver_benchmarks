"""Clarabel adapter."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone
from solver_benchmarks.transforms.psd import (
    cone_row_perm_canonical_to_row_major,
    cone_vec_row_major_to_canonical,
)
from .base import SolverAdapter, SolverUnavailable, settings_with_defaults


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
            p, q, a, b, cones, cone_dict = _qp_data(problem.qp, clarabel)
        elif problem.kind == CONE:
            native = _cone_data(problem.cone, clarabel)
            if isinstance(native, SolverResult):
                return native
            p, q, a, b, cones, cone_dict = native
        else:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": f"Clarabel does not support problem kind {problem.kind!r}"},
            )

        normalized_settings = settings_with_defaults(self.settings)
        settings = clarabel.DefaultSettings()
        settings.verbose = bool(normalized_settings.get("verbose"))
        for key, value in normalized_settings.items():
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
        kkt_dict = _compute_kkt(
            mapped, solution, problem, p, q, a, b, cone_dict
        )
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
            kkt=kkt_dict,
        )


def _compute_kkt(mapped_status, solution, problem, p, q, a, b, cone_dict):
    x = getattr(solution, "x", None)
    y = getattr(solution, "z", None)
    s_slack = getattr(solution, "s", None)
    if x is None:
        return None

    # Clarabel returns y, s in row-major lower PSD ordering and the matrices
    # ``a``/``b`` passed in were already permuted to that ordering. The KKT
    # helpers expect canonical (col-major lower) layout, so for cone problems
    # with PSD blocks we rebuild the canonical ``a``/``b`` from the original
    # problem and permute Clarabel's ``y``/``s`` back.
    if problem.kind == CONE and "s" in cone_dict:
        a_canonical = sp.csc_matrix(problem.cone["A"])
        b_canonical = np.asarray(problem.cone["b"], dtype=float)
        a_for_kkt = a_canonical
        b_for_kkt = b_canonical
        y_for_kkt = (
            cone_vec_row_major_to_canonical(np.asarray(y, dtype=float), cone_dict)
            if y is not None
            else None
        )
        s_for_kkt = (
            cone_vec_row_major_to_canonical(np.asarray(s_slack, dtype=float), cone_dict)
            if s_slack is not None
            else None
        )
    else:
        a_for_kkt = a
        b_for_kkt = b
        y_for_kkt = np.asarray(y, dtype=float) if y is not None else None
        s_for_kkt = np.asarray(s_slack, dtype=float) if s_slack is not None else None

    if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
        if y_for_kkt is None or s_for_kkt is None:
            return None
        return kkt.cone_residuals(
            p, q, a_for_kkt, b_for_kkt, cone_dict, x, y_for_kkt, s_for_kkt
        )
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        if y_for_kkt is None:
            return None
        return kkt.cone_primal_infeasibility_cert(
            a_for_kkt, b_for_kkt, cone_dict, y_for_kkt
        )
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        return kkt.cone_dual_infeasibility_cert(p, q, a_for_kkt, cone_dict, x)
    return None


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)


def _qp_data(qp: dict, clarabel):
    a, b, z = qp_to_nonnegative_cone(qp)
    p = sp.csc_matrix(qp["P"])
    q = np.asarray(qp["q"], dtype=float)
    cones = []
    cone_dict: dict = {}
    if z:
        cones.append(clarabel.ZeroConeT(z))
        cone_dict["z"] = int(z)
    nonnegative = a.shape[0] - z
    if nonnegative:
        cones.append(clarabel.NonnegativeConeT(nonnegative))
        cone_dict["l"] = int(nonnegative)
    return p, q, sp.csc_matrix(a), b, cones, cone_dict


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
    cones, keep_rows, cone_dict = parsed
    if len(keep_rows) != a.shape[0]:
        a = a[keep_rows, :]
        b = b[keep_rows]
    if "s" in cone_dict:
        # Convert s-cone rows from canonical col-major lower PSD ordering to
        # the row-major lower ordering expected by Clarabel's PSDTriangleConeT.
        row_perm = cone_row_perm_canonical_to_row_major(cone_dict, a.shape[0])
        a = a[row_perm, :]
        b = b[row_perm]
    return p, q, sp.csc_matrix(a), b, cones, cone_dict


def _parse_cones(cone: dict, row_count: int, clarabel):
    cones = []
    keep_rows: list[int] = []
    cone_dict: dict = {}
    row = 0
    for name, value in cone.items():
        if name in ("f", "z"):
            dim = int(value)
            if dim:
                cones.append(clarabel.ZeroConeT(dim))
                keep_rows.extend(range(row, row + dim))
                cone_dict["z"] = cone_dict.get("z", 0) + dim
            row += dim
            continue
        if name == "l":
            dim = int(value)
            if dim:
                cones.append(clarabel.NonnegativeConeT(dim))
                keep_rows.extend(range(row, row + dim))
                cone_dict["l"] = cone_dict.get("l", 0) + dim
            row += dim
            continue
        if name == "q":
            for dim in _as_list(value):
                dim = int(dim)
                if dim:
                    cones.append(clarabel.SecondOrderConeT(dim))
                    keep_rows.extend(range(row, row + dim))
                    cone_dict.setdefault("q", []).append(dim)
                row += dim
            continue
        if name == "s":
            for dim in _as_list(value):
                dim = int(dim)
                triangle_dim = dim * (dim + 1) // 2
                if dim:
                    cones.append(clarabel.PSDTriangleConeT(dim))
                    keep_rows.extend(range(row, row + triangle_dim))
                    cone_dict.setdefault("s", []).append(dim)
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
    return cones, keep_rows, cone_dict


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return value
    return [value]
