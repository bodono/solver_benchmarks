"""HiGHS adapter.

``iterations`` is the count of the algorithm that finished the solve: the QP
solver's when it ran, else interior point's, else PDLP's, else simplex's (a QP
runs simplex for phase 1 first, and an interior-point LP solve may end with
simplex crossover iterations); every algorithm's count stays on ``info``.
"""

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
    IMPORT_ERRORS,
    SolverAdapter,
    SolverConfigError,
    coerce_int_setting,
    finite_point,
    module_available,
    pop_max_iter,
    pop_threads,
    pop_time_limit,
    refuse_settings,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    threads_or_default,
)
from .qp_split import free_row_mask, normalize_qp_bounds


class HighsSolverAdapter(SolverAdapter):
    solver_name = "highs"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("highspy")

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import highspy
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "highspy", "Install with the highs extra to use HiGHS", exc
            ) from exc

        qp = problem.qp
        p = sp.csc_matrix(qp["P"])
        q = np.asarray(qp["q"], dtype=float)
        a = sp.csr_matrix(qp["A"])
        l, u = normalize_qp_bounds(qp["l"], qp["u"])
        n = int(qp.get("n", q.shape[0]))
        m = int(qp.get("m", a.shape[0]))

        solver = highspy.Highs()
        settings = settings_with_defaults(self.settings)
        threads, translated = _configure_highs(solver, settings)
        reset_scheduler = getattr(type(solver), "resetGlobalScheduler", None)
        if reset_scheduler is not None:
            # HiGHS keeps one thread pool per process, sized by the first
            # run; a later run asking for another size fails, and threads 0
            # would reuse whatever pool an earlier run left. The runner
            # starts a fresh worker per solve, so this only matters for
            # in-process use, and resetting a pool that does not exist yet
            # costs nothing.
            reset_scheduler(True)
        inf = solver.getInfinity()

        start = time.perf_counter()
        build = {}
        build["addCols"] = solver.addCols(
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
        build["addRows"] = solver.addRows(
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
            build["passHessian"] = solver.passHessian(
                n,
                int(lower.nnz),
                int(highspy.HessianFormat.kTriangular),
                lower.indptr.astype(np.int32),
                lower.indices.astype(np.int32),
                lower.data.astype(float),
            )
        rejected = [name for name, result in build.items() if _is_error(result)]
        if rejected:
            # HiGHS refuses data it cannot take (a coefficient at or above
            # large_matrix_value, an infinite entry, a wrong-signed infinite
            # bound) by returning an error and leaving the model without it.
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start,
                info={
                    "raw_status": "error",
                    "error": f"HiGHS rejected the model in {', '.join(rejected)} (see its log)",
                    "threads": threads,
                    **({"settings_translated": translated} if translated else {}),
                },
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
        x = np.asarray(solution.col_value, dtype=float) if solution.value_valid else None
        kkt_dict = None
        # HiGHS reports no certificate the harness can check, so only a
        # status that may carry a solution gets residuals.
        if mapped in status.ANY_FEASIBLE and finite_point(x, y):
            kkt_dict = kkt.qp_residuals(p, q, a, l, u, x, y)
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
                "threads": threads,
                **({"settings_translated": translated} if translated else {}),
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


def _configure_highs(solver, settings: dict) -> tuple[int, dict]:
    """Apply ``settings`` to a ``highspy.Highs`` instance.

    highspy reports an unknown option or a bad value by returning
    ``HighsStatus.kError`` rather than raising, so every return value is
    checked and a rejected setting raises SolverConfigError naming it.
    ``threads`` is one unless configured (0: HiGHS's own choice). Returns the
    thread count and the iteration limits ``max_iter`` set, for ``info``.
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
    threads = threads_or_default(pop_threads(settings))
    _set_option(solver, "threads", threads)
    translated: dict = {}
    # The native limits follow the shared integer rule (highspy refuses an
    # integer-valued float such as YAML's 1e3 and parses a string).
    for option in _ITERATION_LIMIT_OPTIONS:
        if settings.get(option) is not None:
            settings[option] = coerce_int_setting(option, settings[option], minimum=0)
    max_iter = pop_max_iter(settings, minimum=0)
    if max_iter is not None:
        for option in _ITERATION_LIMIT_OPTIONS:
            if option in settings:
                continue  # a native limit given alongside wins
            try:
                result = solver.setOptionValue(option, max_iter)
            except Exception:
                # Older highspy builds lack some of these options; the
                # available ones still take effect.
                continue
            if not _is_error(result):
                translated[option] = max_iter
        if not translated and not any(option in settings for option in _ITERATION_LIMIT_OPTIONS):
            raise SolverConfigError(f"Invalid HiGHS setting 'max_iter'={max_iter!r}")
    for key, value in settings.items():
        _set_option(solver, key, value)
    return threads, translated


def _set_option(solver, key: str, value) -> None:
    hint = ""
    if isinstance(value, bool) and _is_string_option(solver, key):
        # YAML 1.1 reads unquoted on/off (and yes/no) as booleans, but
        # HiGHS's on/off options (presolve, run_crossover, parallel, ranging)
        # are strings that refuse a bool.
        value = "on" if value else "off"
        hint = " (an unquoted on/off in YAML is read as a boolean)"
    try:
        result = solver.setOptionValue(key, value)
    except TypeError as exc:  # a value of a type the binding cannot take
        raise SolverConfigError(f"Invalid HiGHS setting {key!r}={value!r}: {exc}") from exc
    if _is_error(result):
        raise SolverConfigError(
            f"Invalid HiGHS setting {key!r}={value!r}{hint}: unknown option or value "
            "rejected by HiGHS (see its log)"
        )


def _is_string_option(solver, key: str) -> bool:
    get_type = getattr(solver, "getOptionType", None)
    if get_type is None:
        return False
    try:
        result = get_type(key)
    except Exception:
        return False
    option_type = result[-1] if isinstance(result, tuple) else result
    return getattr(option_type, "name", None) == "kString"


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
    """Iterations of the algorithm that finished the solve.

    The first algorithm with a positive count, in the order QP, interior
    point, PDLP, simplex: a QP's phase-1 simplex iterations and an LP's
    crossover iterations come before or after the algorithm that did the
    solve. HiGHS keeps -1 for algorithms that did not run (and for all of
    them when the solve errored before iterating); 0 when every algorithm
    that ran took no iteration (e.g. presolve solved the problem).
    """
    values = [
        getattr(info, "qp_iteration_count", None),
        getattr(info, "ipm_iteration_count", None),
        getattr(info, "pdlp_iteration_count", None),
        getattr(info, "simplex_iteration_count", None),
    ]
    counts = [int(value) for value in values if value is not None and value >= 0]
    positive = [count for count in counts if count > 0]
    if positive:
        return positive[0]
    return 0 if counts else None
