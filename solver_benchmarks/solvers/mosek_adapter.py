"""MOSEK adapter."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import (
    SolverAdapter,
    SolverUnavailable,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)


class MosekSolverAdapter(SolverAdapter):
    solver_name = "mosek"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import mosek  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import mosek
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the mosek extra to use MOSEK") from exc

        qp = problem.qp
        q = np.asarray(qp["q"], dtype=float)
        p_mat = sp.tril(sp.coo_matrix(qp["P"]), format="coo")
        a_mat = sp.coo_matrix(qp["A"])
        n = int(qp.get("n", q.shape[0]))
        m = int(qp.get("m", a_mat.shape[0]))
        lower = np.asarray(qp["l"], dtype=float)
        upper = np.asarray(qp["u"], dtype=float)

        # Wrap MOSEK Env / Task in `with` blocks so the C-level resources
        # and the streamprinter callback are released deterministically.
        # Otherwise long batch runs leak license tokens until GC fires.
        with mosek.Env() as env, env.Task() as task:
            _configure_mosek(task, env, self.settings, mosek)

            task.appendcons(m)
            task.appendvars(n)
            # Set linear costs and free variable bounds in bulk; the
            # previous per-variable putvarbound loop was O(n) Python calls.
            if n:
                task.putclist(list(range(n)), q.tolist())
                task.putvarboundsliceconst(0, n, mosek.boundkey.fr, -np.inf, np.inf)

            task.putaijlist(a_mat.row, a_mat.col, a_mat.data)
            for row in range(m):
                bound_key, l_val, u_val = _mosek_bound(lower[row], upper[row], mosek)
                task.putconbound(row, bound_key, l_val, u_val)

            if p_mat.nnz:
                task.putqobj(p_mat.row, p_mat.col, p_mat.data)
            task.putobjsense(mosek.objsense.minimize)

            start = time.perf_counter()
            try:
                termination_code = task.optimize()
            except Exception as exc:
                return SolverResult(
                    status=status.SOLVER_ERROR,
                    run_time_seconds=time.perf_counter() - start,
                    info={"error": str(exc)},
                )
            elapsed = time.perf_counter() - start

            soltype = _resolve_mosek_soltype(task, mosek)
            raw_status = task.getsolsta(soltype)
            mapped = _map_mosek_status(raw_status, termination_code, mosek)
            solver_reported_runtime = task.getdouinf(mosek.dinfitem.optimizer_time)
            iterations = task.getintinf(mosek.iinfitem.intpnt_iter)
            # Include OPTIMAL_INACCURATE in the objective gate to match
            # the CPLEX / Gurobi / SCS treatment landed in this PR;
            # MOSEK exposes a usable primal objective for prim_feas /
            # prim_and_dual_feas (which map to OPTIMAL_INACCURATE).
            objective_present = mapped in status.SOLUTION_PRESENT or (
                mapped == status.OPTIMAL_INACCURATE
            )
            objective = task.getprimalobj(soltype) if objective_present else None
            return SolverResult(
                status=mapped,
                objective_value=None if objective is None else float(objective),
                iterations=int(iterations) if iterations is not None else None,
                run_time_seconds=elapsed,
                info={
                    "raw_status": str(raw_status),
                    "termination_code": str(termination_code),
                    "solver": "mosek",
                    "solver_reported_runtime": float(solver_reported_runtime),
                    "soltype": str(soltype),
                },
            )


def _configure_mosek(task, env, settings: dict, mosek) -> None:
    settings = settings_with_defaults(settings)
    if settings.pop("verbose", False):
        def streamprinter(text):
            print(text, end="")

        env.set_Stream(mosek.streamtype.log, streamprinter)
        task.set_Stream(mosek.streamtype.log, streamprinter)
    else:
        task.putintparam(mosek.iparam.log, 0)

    time_limit = pop_time_limit(settings)
    if time_limit is not None:
        task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit))
    threads = pop_threads(settings)
    if threads is not None and threads >= 1:
        task.putintparam(mosek.iparam.num_threads, int(threads))

    settings.pop("high_accuracy", None)
    for key, value in settings.items():
        if isinstance(key, str):
            _handle_mosek_str_param(task, key.strip(), value)
        else:
            _handle_mosek_enum_param(task, key, value, mosek)


def _mosek_bound(lower: float, upper: float, mosek):
    l_val = float(lower) if lower > -1.0e20 else -np.inf
    u_val = float(upper) if upper < 1.0e20 else np.inf
    if abs(l_val - u_val) <= 1.0e-10:
        return mosek.boundkey.fx, l_val, u_val
    if l_val == -np.inf and u_val == np.inf:
        return mosek.boundkey.fr, l_val, u_val
    if l_val == -np.inf:
        return mosek.boundkey.up, l_val, u_val
    if u_val == np.inf:
        return mosek.boundkey.lo, l_val, u_val
    return mosek.boundkey.ra, l_val, u_val


def _map_mosek_status(raw_status, termination_code, mosek) -> str:
    """Map MOSEK's (solsta, termination_code) pair to the canonical status.

    Solsta is checked first: if the optimizer found an optimum *before*
    the time/iteration limit fired, we should report OPTIMAL rather than
    discarding it as TIME_LIMIT.
    """
    solsta = mosek.solsta
    solsta_mapping = {
        solsta.optimal: status.OPTIMAL,
        solsta.integer_optimal: status.OPTIMAL,
        solsta.prim_and_dual_feas: status.OPTIMAL_INACCURATE,
        solsta.prim_feas: status.OPTIMAL_INACCURATE,
        solsta.prim_infeas_cer: status.PRIMAL_INFEASIBLE,
        solsta.dual_infeas_cer: status.DUAL_INFEASIBLE,
        solsta.unknown: status.SOLVER_ERROR,
    }
    canonical_solsta = solsta_mapping.get(raw_status)
    # If MOSEK actually concluded with a useful solsta, keep it even if
    # the optimizer also raised a time/iter termination code.
    if canonical_solsta in {status.OPTIMAL, status.PRIMAL_INFEASIBLE, status.DUAL_INFEASIBLE}:
        return canonical_solsta
    if termination_code == mosek.rescode.trm_max_time:
        return status.TIME_LIMIT
    if termination_code == mosek.rescode.trm_max_iterations:
        return status.MAX_ITER_REACHED
    if canonical_solsta is not None:
        return canonical_solsta
    return status.SOLVER_ERROR


def _resolve_mosek_soltype(task, mosek):
    """Pick the MOSEK solution slot most likely to be populated.

    MOSEK exposes separate slots for interior-point (itr), basic (bas),
    and integer (itg) solutions. Hard-coding `itr` made the adapter
    return ``unknown`` on simplex/MIP solves. Try in priority order and
    fall back to itr.
    """
    soltype = mosek.soltype
    candidates = (soltype.itg, soltype.bas, soltype.itr)
    for candidate in candidates:
        try:
            sta = task.getsolsta(candidate)
        except Exception:
            continue
        if sta != mosek.solsta.unknown:
            return candidate
    return soltype.itr


def _handle_mosek_str_param(task, param: str, value) -> None:
    if param.startswith("MSK_DPAR_"):
        task.putnadouparam(param, value)
    elif param.startswith("MSK_IPAR_"):
        task.putnaintparam(param, value)
    elif param.startswith("MSK_SPAR_"):
        task.putnastrparam(param, value)
    else:
        raise ValueError(f"Invalid MOSEK parameter {param!r}")


def _handle_mosek_enum_param(task, param, value, mosek) -> None:
    if isinstance(param, mosek.dparam):
        task.putdouparam(param, value)
    elif isinstance(param, mosek.iparam):
        task.putintparam(param, value)
    elif isinstance(param, mosek.sparam):
        task.putstrparam(param, value)
    else:
        raise ValueError(f"Invalid MOSEK parameter {param!r}")
