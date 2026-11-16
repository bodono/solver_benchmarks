"""IBM CPLEX adapter.

A license failure raised by the solve (the Community Edition that ``pip
install cplex`` ships with refuses larger problems inside ``solve()``, error
1016) is a broken environment, not CPLEX failing the problem: it raises
``SolverUnavailable``. Any other CPLEX error from the solve is
``solver_error``; other exceptions are left to raise. ``threads`` is one
unless configured (0: CPLEX's own choice). ``info["raw_status"]`` is CPLEX's
status string, ``info["raw_status_code"]`` its integer code.
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
    SolverUnavailable,
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
from .qp_split import normalize_qp_bounds


class CPLEXSolverAdapter(SolverAdapter):
    solver_name = "cplex"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("cplex")

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import cplex
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "cplex", "Install with the cplex extra to use CPLEX", exc
            ) from exc

        settings = dict(self.settings)
        use_ranged_rows = settings.pop("use_ranged_rows", False)
        if not isinstance(use_ranged_rows, bool):
            raise SolverConfigError("use_ranged_rows must be a boolean")

        qp = problem.qp
        q = np.asarray(qp["q"], dtype=float)
        p = sp.coo_matrix(qp["P"])
        a = sp.csr_matrix(qp["A"])
        l, u = normalize_qp_bounds(qp["l"], qp["u"])
        n = int(qp.get("n", q.shape[0]))
        model = cplex.Cplex()
        # Always end() the model so the C-level state and license token
        # are released deterministically, even when an exception fires.
        try:
            model.objective.set_sense(model.objective.sense.minimize)
            translated, threads = _configure_cplex(model, settings)
            infinity = cplex.infinity
            model.variables.add(
                obj=q.tolist(),
                lb=[-infinity] * n,
                ub=[infinity] * n,
                names=[f"x_{idx}" for idx in range(n)],
            )
            constraint_rows = _add_cplex_constraints(
                model, cplex, a, l, u, infinity, use_ranged_rows=use_ranged_rows
            )
            if p.nnz:
                model.objective.set_quadratic_coefficients(
                    [(int(i), int(j), float(v)) for i, j, v in zip(p.row, p.col, p.data)]
                )

            start = time.perf_counter()
            try:
                model.solve()
                resolved = _resolve_infeasible_or_unbounded(model, settings)
            except _cplex_error_type(cplex) as exc:
                if _is_license_error(exc):
                    raise SolverUnavailable(f"CPLEX license error: {exc}") from exc
                return SolverResult(
                    status=status.SOLVER_ERROR,
                    run_time_seconds=time.perf_counter() - start,
                    info={
                        "raw_status": "error",
                        "error": str(exc),
                        "use_ranged_rows": use_ranged_rows,
                        "threads": threads,
                    },
                )
            elapsed = time.perf_counter() - start
            raw_status = model.solution.get_status()
            mapped = _map_cplex_status(raw_status, model)
            iteration_counts = _cplex_iterations(model)
            objective_value = None
            kkt_dict = None
            if mapped in status.ANY_FEASIBLE:
                # A returned iterate may exist even with an error or limit status.
                try:
                    x = np.asarray(model.solution.get_values(), dtype=float)
                    objective_value = model.solution.get_objective_value()
                    duals = np.asarray(model.solution.get_dual_values(), dtype=float)
                except cplex.exceptions.CplexSolverError:
                    pass  # No complete primal/dual point is available.
                else:
                    y = np.zeros(a.shape[0])
                    # CPLEX uses the opposite dual sign; sum any split rows.
                    np.add.at(y, constraint_rows, -duals)
                    if finite_point(x, y):
                        kkt_dict = kkt.qp_residuals(p, q, a, l, u, x, y)
            return SolverResult(
                status=mapped,
                kkt=kkt_dict,
                objective_value=reported_objective(mapped, objective_value),
                iterations=iteration_counts["iterations"],
                run_time_seconds=elapsed,
                info={
                    "raw_status": _status_string(model, raw_status),
                    "raw_status_code": raw_status,
                    "threads": threads,
                    "status_string": _status_string(model, raw_status),
                    "problem_type": model.problem_type[model.get_problem_type()],
                    "use_ranged_rows": use_ranged_rows,
                    "simplex_iterations": iteration_counts["simplex_iterations"],
                    "barrier_iterations": iteration_counts["barrier_iterations"],
                    **({"settings_translated": translated} if translated else {}),
                    **({"inf_or_unbd_resolved": True} if resolved else {}),
                },
            )
        finally:
            try:
                model.end()
            except Exception:
                pass


def _add_cplex_constraints(
    model, cplex, a, l, u, infinity: float, *, use_ranged_rows: bool = False
) -> list[int]:
    constraint_rows = []
    lin_expr = []
    senses = []
    rhs = []
    names = []
    range_values = []
    for row in range(a.shape[0]):
        start, end = a.indptr[row], a.indptr[row + 1]
        expr = cplex.SparsePair(
            ind=[int(col) for col in a.indices[start:end]],
            val=[float(val) for val in a.data[start:end]],
        )
        l_val = float(l[row])
        u_val = float(u[row])
        if l_val <= -1.0e20 and u_val >= 1.0e20:
            continue
        if abs(l_val - u_val) <= 1.0e-10:
            constraint_rows.append(row)
            lin_expr.append(expr)
            senses.append("E")
            rhs.append(u_val)
            names.append(f"c_{row}_eq")
            range_values.append(0.0)
            continue
        # A negative CPLEX range reverses the interval rather than making it
        # infeasible, so keep contradictory bounds as separate inequalities.
        if use_ranged_rows and -1.0e20 < l_val < u_val < 1.0e20:
            constraint_rows.append(row)
            lin_expr.append(expr)
            senses.append("R")
            rhs.append(l_val)
            names.append(f"c_{row}_range")
            range_values.append(u_val - l_val)
            continue
        if l_val > -1.0e20:
            constraint_rows.append(row)
            lin_expr.append(expr)
            senses.append("G")
            rhs.append(l_val)
            names.append(f"c_{row}_lb")
            range_values.append(0.0)
        if u_val < 1.0e20:
            constraint_rows.append(row)
            lin_expr.append(expr)
            senses.append("L")
            rhs.append(u_val)
            names.append(f"c_{row}_ub")
            range_values.append(0.0)
    if lin_expr:
        options = {"range_values": range_values} if use_ranged_rows else {}
        model.linear_constraints.add(
            lin_expr=lin_expr, senses=senses, rhs=rhs, names=names, **options
        )
    return constraint_rows


def _configure_cplex(model, settings: dict) -> tuple[dict, int]:
    """Apply ``settings`` (dotted CPLEX parameter paths) to ``model``.

    Returns the aliases it translated and the thread count: ``max_iter``
    caps both the barrier and the simplex iteration counts unless either path
    is given, and ``threads`` (CPLEX's own name too) is one unless
    configured. The ``eps`` aliases are refused (CPLEX's tolerances are not an
    absolute/relative pair), and a path CPLEX does not have raises
    SolverConfigError naming it.
    """
    settings = settings_with_defaults(settings)
    refuse_settings(
        settings,
        EPS_ALIASES,
        "CPLEX",
        "set its own tolerances instead (barrier.convergetol, "
        "simplex.tolerances.feasibility, simplex.tolerances.optimality)",
    )
    translated: dict = {}
    max_iter = pop_max_iter(settings, minimum=0)
    if max_iter is not None:
        for path in ("barrier.limits.iteration", "simplex.limits.iterations"):
            if path not in settings:
                settings[path] = max_iter
                translated[path] = max_iter
    verbose = bool(settings.pop("verbose", False))
    if not verbose:
        model.set_results_stream(None)
        model.set_log_stream(None)
        model.set_warning_stream(None)
        model.set_error_stream(None)
    time_limit = pop_time_limit(settings)
    if time_limit is not None:
        model.parameters.timelimit.set(float(time_limit))
    threads = threads_or_default(pop_threads(settings))
    model.parameters.threads.set(threads)
    for key, value in settings.items():
        _set_cplex_parameter(model, key, value)
    return translated, threads


def _set_cplex_parameter(model, key: str, value) -> None:
    node = model.parameters
    try:
        for part in str(key).split("."):
            node = getattr(node, part)
        setter = node.set
    except AttributeError:
        raise SolverConfigError(f"Unknown CPLEX parameter {key!r}") from None
    try:
        setter(value)
    except Exception as exc:
        raise SolverConfigError(f"Invalid CPLEX parameter {key!r}={value!r}: {exc}") from exc


def _cplex_error_type(cplex):
    """CPLEX's base exception (``CplexError``; ``CplexSolverError`` derives from it)."""
    exceptions = cplex.exceptions
    return getattr(exceptions, "CplexError", exceptions.CplexSolverError)


# CPLEX error codes for license failures: Community Edition size limits
# (CPXERR_RESTRICTED_VERSION) and no valid license (CPXERR_ILOG_LICENSE).
_LICENSE_CODES = frozenset({1016, 32201})


def _is_license_error(exc: Exception) -> bool:
    # CplexError's args are (message, environment, error code).
    args = tuple(exc.args)
    code = args[2] if len(args) >= 3 else None
    text = str(exc).lower()
    return code in _LICENSE_CODES or "community edition" in text or "licen" in text


def _status_string(model, raw_status) -> str:
    try:
        return str(model.solution.get_status_string(raw_status))
    except Exception:
        return str(raw_status)


def _map_cplex_status(raw_status: int, model) -> str:
    sol_status = model.solution.status
    mapping = {
        sol_status.optimal: status.OPTIMAL,
        sol_status.infeasible: status.PRIMAL_INFEASIBLE,
        sol_status.unbounded: status.DUAL_INFEASIBLE,
        sol_status.infeasible_or_unbounded: status.PRIMAL_OR_DUAL_INFEASIBLE,
        sol_status.abort_iteration_limit: status.MAX_ITER_REACHED,
        sol_status.abort_time_limit: status.TIME_LIMIT,
    }
    for name, canonical in [
        ("optimal_infeasible", status.OPTIMAL_INACCURATE),
        ("num_best", status.OPTIMAL_INACCURATE),
        ("abort_dettime_limit", status.TIME_LIMIT),
    ]:
        if hasattr(sol_status, name):
            mapping[getattr(sol_status, name)] = canonical
    return mapping.get(raw_status, status.SOLVER_ERROR)


def _resolve_infeasible_or_unbounded(model, settings: dict) -> bool:
    """Re-solve without presolve reductions after infeasible_or_unbounded.

    CPLEX's presolve can prove only that a model has no optimum; turning its
    reductions off (preprocessing.reduce = 0) and re-solving gives the
    definite answer, so the row can carry an infeasibility or unboundedness
    status instead of primal_or_dual_infeasible, which never counts. Returns
    whether the model was re-solved (the elapsed time covers both solves).
    """
    status_codes = model.solution.status
    if model.solution.get_status() != status_codes.infeasible_or_unbounded:
        return False
    if settings.get("preprocessing.reduce") == 0:
        return False  # already off: the answer is final
    _set_cplex_parameter(model, "preprocessing.reduce", 0)
    model.solve()
    return True


def _cplex_iterations(model) -> dict:
    """Iteration counts; ``iterations`` is the count of the method that ran.

    ``get_num_iterations`` counts simplex (or crossover) iterations, which is
    0 for a QP solved by barrier without crossover, the default for
    continuous QPs; the barrier count is used whenever barrier ran.
    """
    progress = model.solution.progress

    def count(method) -> int | None:
        try:
            return int(method())
        except Exception:
            return None

    simplex = count(progress.get_num_iterations)
    barrier = count(getattr(progress, "get_num_barrier_iterations", lambda: None))
    return {
        "iterations": barrier if barrier else simplex,
        "simplex_iterations": simplex,
        "barrier_iterations": barrier,
    }
