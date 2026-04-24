"""Optional commercial solver adapters."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.core import status
from .base import SolverAdapter, SolverUnavailable


class GurobiSolverAdapter(SolverAdapter):
    solver_name = "gurobi"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import gurobipy  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import gurobipy as grb
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the gurobi extra to use GUROBI") from exc

        qp = problem.qp
        q = np.asarray(qp["q"], dtype=float)
        p_mat = sp.coo_matrix(qp["P"])
        a_mat = sp.csr_matrix(qp["A"])
        n = int(qp.get("n", q.shape[0]))
        m = int(qp.get("m", a_mat.shape[0]))
        lower = _finite_bounds(qp["l"], -grb.GRB.INFINITY, lower=True)
        upper = _finite_bounds(qp["u"], grb.GRB.INFINITY, lower=False)

        model = grb.Model(f"{problem.dataset_id}/{problem.name}")
        _configure_gurobi(model, self.settings, grb)

        variables = [model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY) for _ in range(n)]
        model.update()

        for row in range(m):
            start, end = a_mat.indptr[row], a_mat.indptr[row + 1]
            expr = grb.LinExpr(a_mat.data[start:end], [variables[j] for j in a_mat.indices[start:end]])
            l_val = lower[row]
            u_val = upper[row]
            if l_val <= -grb.GRB.INFINITY and u_val >= grb.GRB.INFINITY:
                continue
            if abs(l_val - u_val) <= 1.0e-10:
                model.addConstr(expr == u_val)
            elif l_val <= -grb.GRB.INFINITY:
                model.addConstr(expr <= u_val)
            elif u_val >= grb.GRB.INFINITY:
                model.addConstr(expr >= l_val)
            else:
                model.addRange(expr, l_val, u_val)

        objective = grb.QuadExpr()
        for row, col, value in zip(p_mat.row, p_mat.col, p_mat.data):
            objective.add(0.5 * float(value) * variables[row] * variables[col])
        objective.add(grb.LinExpr(q, variables))
        objective.addConstant(float(qp.get("r", 0.0)))
        model.setObjective(objective, grb.GRB.MINIMIZE)

        start = time.perf_counter()
        try:
            model.optimize()
        except Exception as exc:
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start,
                info={"error": str(exc)},
            )

        mapped = _map_gurobi_status(model.Status, grb)
        return SolverResult(
            status=mapped,
            objective_value=float(model.ObjVal) if mapped in status.SOLUTION_PRESENT else None,
            iterations=_maybe_int(getattr(model, "BarIterCount", None) or getattr(model, "IterCount", None)),
            run_time_seconds=float(getattr(model, "Runtime", time.perf_counter() - start)),
            info={
                "raw_status": int(model.Status),
                "solver": "gurobi",
                "version": ".".join(str(part) for part in grb.gurobi.version()),
            },
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

        env = mosek.Env()
        task = env.Task()
        _configure_mosek(task, env, self.settings, mosek)

        task.appendcons(m)
        task.appendvars(n)
        for col, value in enumerate(q):
            task.putcj(col, float(value))
            task.putvarbound(col, mosek.boundkey.fr, -np.inf, np.inf)

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

        soltype = mosek.soltype.itr
        raw_status = task.getsolsta(soltype)
        mapped = _map_mosek_status(raw_status, termination_code, mosek)
        run_time = task.getdouinf(mosek.dinfitem.optimizer_time)
        iterations = task.getintinf(mosek.iinfitem.intpnt_iter)
        objective = task.getprimalobj(soltype) if mapped in status.SOLUTION_PRESENT else None
        return SolverResult(
            status=mapped,
            objective_value=None if objective is None else float(objective),
            iterations=int(iterations) if iterations is not None else None,
            run_time_seconds=float(run_time),
            info={
                "raw_status": str(raw_status),
                "termination_code": str(termination_code),
                "solver": "mosek",
            },
        )


def _finite_bounds(values, inf_value: float, *, lower: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if lower:
        arr[arr <= -1.0e20] = inf_value
    else:
        arr[arr >= 1.0e20] = inf_value
    return arr


def _configure_gurobi(model, settings: dict, grb) -> None:
    if not settings.get("verbose", False):
        model.setParam("OutputFlag", 0)
    if "time_limit" in settings:
        model.setParam("TimeLimit", float(settings["time_limit"]))
    if "time_limit_sec" in settings:
        model.setParam("TimeLimit", float(settings["time_limit_sec"]))
    ignored = {"verbose", "time_limit", "time_limit_sec", "high_accuracy"}
    for key, value in settings.items():
        if key not in ignored:
            model.setParam(key, value)


def _map_gurobi_status(raw_status: int, grb) -> str:
    return {
        grb.GRB.OPTIMAL: status.OPTIMAL,
        grb.GRB.INFEASIBLE: status.PRIMAL_INFEASIBLE,
        grb.GRB.UNBOUNDED: status.DUAL_INFEASIBLE,
        grb.GRB.INF_OR_UNBD: status.PRIMAL_OR_DUAL_INFEASIBLE,
        grb.GRB.ITERATION_LIMIT: status.MAX_ITER_REACHED,
        grb.GRB.TIME_LIMIT: status.TIME_LIMIT,
    }.get(raw_status, status.SOLVER_ERROR)


def _configure_mosek(task, env, settings: dict, mosek) -> None:
    if settings.get("verbose", False):
        def streamprinter(text):
            print(text, end="")

        env.set_Stream(mosek.streamtype.log, streamprinter)
        task.set_Stream(mosek.streamtype.log, streamprinter)
    else:
        task.putintparam(mosek.iparam.log, 0)

    if "time_limit" in settings:
        task.putdouparam(mosek.dparam.optimizer_max_time, float(settings["time_limit"]))
    if "time_limit_sec" in settings:
        task.putdouparam(mosek.dparam.optimizer_max_time, float(settings["time_limit_sec"]))

    ignored = {"verbose", "time_limit", "time_limit_sec", "high_accuracy"}
    for key, value in settings.items():
        if key in ignored:
            continue
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
    if termination_code == mosek.rescode.trm_max_time:
        return status.TIME_LIMIT
    return {
        mosek.solsta.optimal: status.OPTIMAL,
        mosek.solsta.integer_optimal: status.OPTIMAL,
        mosek.solsta.prim_feas: status.OPTIMAL_INACCURATE,
        mosek.solsta.prim_infeas_cer: status.PRIMAL_INFEASIBLE,
        mosek.solsta.dual_infeas_cer: status.DUAL_INFEASIBLE,
        mosek.solsta.unknown: status.SOLVER_ERROR,
    }.get(raw_status, status.SOLVER_ERROR)


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


def _maybe_int(value) -> int | None:
    return None if value is None else int(value)
