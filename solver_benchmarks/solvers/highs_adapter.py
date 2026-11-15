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
    EPS_ALIASES,
    SolverAdapter,
    SolverUnavailable,
    pop_threads,
    pop_time_limit,
    refuse_settings,
    reported_objective,
    settings_with_defaults,
)
from .qp_split import free_row_mask, normalize_qp_bounds


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
        l, u = normalize_qp_bounds(qp["l"], qp["u"])
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
        row_lower = np.where(np.isneginf(l), -inf, l)
        row_upper = np.where(np.isposinf(u), inf, u)
        # Free rows can trigger heap corruption in HiPO 1.15.1.
        keep = ~free_row_mask(l, u)
        a_highs = a[keep]
        solver.addRows(
            a_highs.shape[0],
            row_lower[keep],
            row_upper[keep],
            int(a_highs.nnz),
            a_highs.indptr.astype(np.int32),
            a_highs.indices.astype(np.int32),
            a_highs.data.astype(float),
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
        y = None
        if solution.dual_valid:
            y = np.zeros(m)
            y[keep] = -np.asarray(solution.row_dual, dtype=float)
        kkt_dict = None
        if solution.value_valid and y is not None:
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
            objective_value=reported_objective(mapped, info.objective_function_value),
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


# HiGHS caps each algorithm separately; ``max_iter`` sets all of them.
_ITERATION_LIMIT_OPTIONS = (
    "simplex_iteration_limit",
    "ipm_iteration_limit",
    "qp_iteration_limit",
    "pdlp_iteration_limit",
)


def _configure_highs(solver, settings: dict) -> None:
    """Apply ``settings`` to a ``highspy.Highs`` instance.

    highspy reports an unknown option or a bad value by returning
    ``HighsStatus.kError`` rather than raising, so every return value is
    checked and a rejected setting raises ValueError naming it.
    """
    refuse_settings(
        settings,
        EPS_ALIASES,
        "HiGHS",
        "it has no absolute/relative tolerance pair; set its own tolerances, e.g. "
        "primal_feasibility_tolerance, dual_feasibility_tolerance, "
        "ipm_optimality_tolerance",
    )
    if "verbose" in settings:
        _set_option(solver, "output_flag", bool(settings.pop("verbose")))
    time_limit = pop_time_limit(settings)
    if time_limit is not None:
        _set_option(solver, "time_limit", float(time_limit))
    threads = pop_threads(settings)
    if threads is not None:
        _set_option(solver, "threads", int(threads))
    if "max_iter" in settings:
        max_iter = settings.pop("max_iter")
        accepted = 0
        for option in _ITERATION_LIMIT_OPTIONS:
            try:
                result = solver.setOptionValue(option, int(max_iter))
            except Exception:
                # Older highspy builds lack some of these options; the
                # available ones still take effect.
                continue
            if not _is_error(result):
                accepted += 1
        if not accepted:
            raise ValueError(f"Invalid HiGHS setting 'max_iter'={max_iter!r}")
    for key, value in settings.items():
        _set_option(solver, key, value)


def _set_option(solver, key: str, value) -> None:
    if _is_error(solver.setOptionValue(key, value)):
        raise ValueError(
            f"Invalid HiGHS setting {key!r}={value!r}: unknown option or value "
            "rejected by HiGHS (see its log)"
        )


def _is_error(result) -> bool:
    return getattr(result, "name", None) == "kError"


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
    """Iterations of the algorithm that ran (HiGHS keeps -1 for the others,
    and for all of them when the solve errored before iterating)."""
    values = [
        getattr(info, "qp_iteration_count", None),
        getattr(info, "ipm_iteration_count", None),
        getattr(info, "simplex_iteration_count", None),
        getattr(info, "pdlp_iteration_count", None),
    ]
    counts = [int(value) for value in values if value is not None and value >= 0]
    return max(counts) if counts else None
