"""IBM CPLEX adapter."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from .base import SolverAdapter, SolverUnavailable, settings_with_defaults


class CPLEXSolverAdapter(SolverAdapter):
    solver_name = "cplex"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cplex  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import cplex
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the cplex extra to use CPLEX") from exc

        qp = problem.qp
        q = np.asarray(qp["q"], dtype=float)
        p = sp.coo_matrix(qp["P"])
        a = sp.csr_matrix(qp["A"])
        l = np.asarray(qp["l"], dtype=float)
        u = np.asarray(qp["u"], dtype=float)
        n = int(qp.get("n", q.shape[0]))
        model = cplex.Cplex()
        model.objective.set_sense(model.objective.sense.minimize)
        _configure_cplex(model, self.settings)
        infinity = model.infinity
        model.variables.add(
            obj=q.tolist(),
            lb=[-infinity] * n,
            ub=[infinity] * n,
            names=[f"x_{idx}" for idx in range(n)],
        )
        _add_cplex_constraints(model, cplex, a, l, u, infinity)
        if p.nnz:
            model.objective.set_quadratic_coefficients(
                [(int(i), int(j), float(v)) for i, j, v in zip(p.row, p.col, p.data)]
            )

        start = time.perf_counter()
        try:
            model.solve()
        except Exception as exc:
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start,
                info={"error": str(exc)},
            )
        elapsed = time.perf_counter() - start
        raw_status = model.solution.get_status()
        mapped = _map_cplex_status(raw_status, model)
        return SolverResult(
            status=mapped,
            objective_value=model.solution.get_objective_value()
            if mapped in status.SOLUTION_PRESENT
            else None,
            iterations=_cplex_iterations(model),
            run_time_seconds=elapsed,
            info={
                "raw_status": raw_status,
                "status_string": model.solution.get_status_string(raw_status),
                "problem_type": model.problem_type[model.get_problem_type()],
            },
        )


def _add_cplex_constraints(model, cplex, a, l, u, infinity: float) -> None:
    lin_expr = []
    senses = []
    rhs = []
    names = []
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
            lin_expr.append(expr)
            senses.append("E")
            rhs.append(u_val)
            names.append(f"c_{row}_eq")
            continue
        if l_val > -1.0e20:
            lin_expr.append(expr)
            senses.append("G")
            rhs.append(l_val)
            names.append(f"c_{row}_lb")
        if u_val < 1.0e20:
            lin_expr.append(expr)
            senses.append("L")
            rhs.append(u_val)
            names.append(f"c_{row}_ub")
    if lin_expr:
        model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=names)


def _configure_cplex(model, settings: dict) -> None:
    settings = settings_with_defaults(settings)
    verbose = bool(settings.pop("verbose", True))
    if not verbose:
        model.set_results_stream(None)
        model.set_log_stream(None)
        model.set_warning_stream(None)
        model.set_error_stream(None)
    for source in ("time_limit_sec", "time_limit"):
        if source in settings:
            model.parameters.timelimit.set(float(settings.pop(source)))
    for key, value in settings.items():
        _set_cplex_parameter(model, key, value)


def _set_cplex_parameter(model, key: str, value) -> None:
    node = model.parameters
    for part in str(key).split("."):
        node = getattr(node, part)
    node.set(value)


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
        ("abort_obj_limit", status.TIME_LIMIT),
    ]:
        if hasattr(sol_status, name):
            mapping[getattr(sol_status, name)] = canonical
    return mapping.get(raw_status, status.SOLVER_ERROR)


def _cplex_iterations(model) -> int | None:
    try:
        return int(model.solution.progress.get_num_iterations())
    except Exception:
        return None
