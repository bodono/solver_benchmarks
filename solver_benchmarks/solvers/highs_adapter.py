"""HiGHS adapter."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import (
    SolverAdapter,
    SolverUnavailable,
    pop_threads,
    settings_with_defaults,
)


class HighsSolverAdapter(SolverAdapter):
    solver_name = "highs"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import highspy  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import highspy
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the highs extra to use HiGHS") from exc

        qp = problem.qp
        p = sp.csc_matrix(qp["P"])
        q = np.asarray(qp["q"], dtype=float)
        a = sp.csr_matrix(qp["A"])
        l = np.asarray(qp["l"], dtype=float)
        u = np.asarray(qp["u"], dtype=float)
        n = int(qp.get("n", q.shape[0]))
        m = int(qp.get("m", a.shape[0]))

        solver = highspy.Highs()
        settings = settings_with_defaults(self.settings)
        _configure_highs(solver, settings)
        inf = solver.getInfinity()

        start = time.perf_counter()
        solver.addCols(
            n,
            q,
            np.full(n, -inf),
            np.full(n, inf),
            0,
            np.zeros(n + 1, dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=float),
        )
        solver.addRows(
            m,
            _highs_bounds(l, -inf),
            _highs_bounds(u, inf),
            int(a.nnz),
            a.indptr.astype(np.int32),
            a.indices.astype(np.int32),
            a.data.astype(float),
        )
        if p.nnz:
            lower = sp.tril(p, format="csc")
            solver.passHessian(
                n,
                int(lower.nnz),
                int(highspy.HessianFormat.kTriangular),
                lower.indptr.astype(np.int32),
                lower.indices.astype(np.int32),
                lower.data.astype(float),
            )
        raw_status = solver.run()
        elapsed = time.perf_counter() - start

        model_status = solver.getModelStatus()
        mapped = _map_highs_status(model_status, highspy)
        info = solver.getInfo()
        solution = solver.getSolution()
        y = -np.asarray(solution.row_dual, dtype=float) if solution.dual_valid else None
        kkt_dict = None
        if mapped == status.OPTIMAL and solution.value_valid and y is not None:
            kkt_dict = kkt.qp_residuals(
                p,
                q,
                a,
                l,
                u,
                np.asarray(solution.col_value, dtype=float),
                y,
            )
        return SolverResult(
            status=mapped,
            objective_value=float(info.objective_function_value)
            if mapped in status.SOLUTION_PRESENT
            else None,
            iterations=_highs_iterations(info),
            run_time_seconds=elapsed,
            info={
                "raw_status": str(model_status),
                "api_status": str(raw_status),
                "simplex_iteration_count": getattr(info, "simplex_iteration_count", None),
                "ipm_iteration_count": getattr(info, "ipm_iteration_count", None),
                "qp_iteration_count": getattr(info, "qp_iteration_count", None),
                "pdlp_iteration_count": getattr(info, "pdlp_iteration_count", None),
                "primal_solution_status": getattr(info, "primal_solution_status", None),
                "dual_solution_status": getattr(info, "dual_solution_status", None),
                "objective_function_value": getattr(info, "objective_function_value", None),
                "version": solver.version(),
            },
            kkt=kkt_dict,
        )


def _configure_highs(solver, settings: dict) -> None:
    if "verbose" in settings:
        solver.setOptionValue("output_flag", bool(settings.pop("verbose")))
    for source, target in [
        ("time_limit_sec", "time_limit"),
        ("time_limit_secs", "time_limit"),
        ("time_limit", "time_limit"),
    ]:
        if source in settings:
            solver.setOptionValue(target, settings.pop(source))
    threads = pop_threads(settings)
    if threads is not None:
        try:
            solver.setOptionValue("threads", int(threads))
        except Exception:
            pass
    if "max_iter" in settings:
        # HiGHS uses a separate iteration cap per algorithm. Forward
        # the user's max_iter to all of them so an LP solved by IPM,
        # a QP solved by the QP code, or a PDLP run is also capped.
        max_iter = settings.pop("max_iter")
        for option in (
            "simplex_iteration_limit",
            "ipm_iteration_limit",
            "qp_iteration_limit",
            "pdlp_iteration_limit",
        ):
            try:
                solver.setOptionValue(option, max_iter)
            except Exception:
                # Older highspy versions lack one of these options;
                # ignore so the available ones still take effect.
                pass
    for key, value in settings.items():
        solver.setOptionValue(key, value)


def _highs_bounds(values, inf_value: float) -> np.ndarray:
    bounds = np.asarray(values, dtype=float).copy()
    if inf_value > 0:
        bounds[bounds >= 1.0e20] = inf_value
    else:
        bounds[bounds <= -1.0e20] = inf_value
    return bounds


def _map_highs_status(model_status, highspy) -> str:
    model = highspy.HighsModelStatus
    mapping = {
        model.kOptimal: status.OPTIMAL,
        model.kInfeasible: status.PRIMAL_INFEASIBLE,
        model.kUnbounded: status.DUAL_INFEASIBLE,
        model.kUnboundedOrInfeasible: status.PRIMAL_OR_DUAL_INFEASIBLE,
        model.kIterationLimit: status.MAX_ITER_REACHED,
        model.kTimeLimit: status.TIME_LIMIT,
    }
    # Optional model statuses depending on highspy build. Use getattr so
    # the mapping works across versions.
    for attr_name, canonical in (
        ("kInterrupt", status.TIME_LIMIT),
        ("kModelEmpty", status.SOLVER_ERROR),
        ("kPresolveError", status.SOLVER_ERROR),
        ("kSolveError", status.SOLVER_ERROR),
        ("kPostsolveError", status.SOLVER_ERROR),
        ("kMemoryLimit", status.SOLVER_ERROR),
        ("kSolutionLimit", status.MAX_ITER_REACHED),
    ):
        attr = getattr(model, attr_name, None)
        if attr is not None:
            mapping[attr] = canonical
    return mapping.get(model_status, status.SOLVER_ERROR)


def _highs_iterations(info) -> int | None:
    values = [
        getattr(info, "qp_iteration_count", 0),
        getattr(info, "ipm_iteration_count", 0),
        getattr(info, "simplex_iteration_count", 0),
        getattr(info, "pdlp_iteration_count", 0),
    ]
    return int(max(values)) if values else None
