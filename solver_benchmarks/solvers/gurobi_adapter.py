"""GUROBI adapter."""

from __future__ import annotations

import os
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
    pop_time_limit,
    settings_with_defaults,
)


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

        env = _create_environment(grb)
        try:
            model = grb.Model(f"{problem.dataset_id}/{problem.name}", env=env)
            try:
                _configure_gurobi(model, self.settings, grb)

                variables = [
                    model.addVar(lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY)
                    for _ in range(n)
                ]
                model.update()

                constraints = []
                constraint_rows = []
                for row in range(m):
                    start, end = a_mat.indptr[row], a_mat.indptr[row + 1]
                    expr = grb.LinExpr(
                        a_mat.data[start:end],
                        [variables[j] for j in a_mat.indices[start:end]],
                    )
                    l_val = lower[row]
                    u_val = upper[row]
                    if l_val <= -grb.GRB.INFINITY and u_val >= grb.GRB.INFINITY:
                        continue
                    if abs(l_val - u_val) <= 1.0e-10:
                        constraint = model.addConstr(expr == u_val)
                    elif l_val <= -grb.GRB.INFINITY:
                        constraint = model.addConstr(expr <= u_val)
                    elif u_val >= grb.GRB.INFINITY:
                        constraint = model.addConstr(expr >= l_val)
                    else:
                        constraint = model.addRange(expr, l_val, u_val)
                    constraints.append(constraint)
                    constraint_rows.append(row)

                objective = grb.QuadExpr()
                for row, col, value in zip(p_mat.row, p_mat.col, p_mat.data):
                    objective.add(0.5 * float(value) * variables[row] * variables[col])
                objective.add(grb.LinExpr(q, variables))
                # The ``r`` constant offset is added by ``solver_benchmarks.worker``;
                # adding it here as well would double-count it for Gurobi solves.
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
                elapsed = time.perf_counter() - start

                mapped = _map_gurobi_status(model.Status, grb)
                kkt_dict = None
                if mapped in status.ANY_FEASIBLE:
                    kkt_dict = _solution_kkt(
                        model, grb, qp, variables, constraints, constraint_rows
                    )
                return SolverResult(
                    status=mapped,
                    kkt=kkt_dict,
                    objective_value=(
                        kkt_dict["primal_obj"] if kkt_dict is not None else None
                    ),
                    iterations=_maybe_int(
                        getattr(model, "BarIterCount", None)
                        or getattr(model, "IterCount", None)
                    ),
                    run_time_seconds=elapsed,
                    info={
                        "raw_status": int(model.Status),
                        "solver": "gurobi",
                        "version": ".".join(str(part) for part in grb.gurobi.version()),
                        "solver_reported_runtime": _maybe_float(
                            getattr(model, "Runtime", None)
                        ),
                    },
                )
            finally:
                # Releases the C-level Model storage. Without this, long
                # batch runs accumulate models held by the env until GC.
                model.dispose()
        finally:
            env.dispose()


def _create_environment(grb):
    token_file = os.environ.get("GUROBI_WLS_TOKEN_FILE")
    if token_file is None:
        return grb.Env()
    # Credentials and tokens must not enter settings or benchmark artifacts.
    return grb.Env(params={
        "OutputFlag": 0,
        "WLSAccessID": "",
        "WLSSecret": "",
        "LicenseID": 0,
        "WLSToken": Path(token_file).read_text().strip(),
    })


def _solution_kkt(model, grb, qp, variables, constraints, constraint_rows):
    # Read original variables only: addRange introduces extra slack variables.
    # An early barrier stop can expose BarX/BarPi without a final X/Pi pair.
    for primal_attr, dual_attr in (("X", "Pi"), ("BarX", "BarPi")):
        try:
            x = np.asarray(model.getAttr(primal_attr, variables), dtype=float)
            duals = np.asarray(model.getAttr(dual_attr, constraints), dtype=float)
        except grb.GurobiError:
            continue
        y = np.zeros(qp["A"].shape[0])
        # Gurobi's row multipliers have the opposite sign to our convention.
        y[constraint_rows] = -duals
        return kkt.qp_residuals(qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x, y)
    return None


def _finite_bounds(values, inf_value: float, *, lower: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if lower:
        arr[arr <= -1.0e20] = inf_value
    else:
        arr[arr >= 1.0e20] = inf_value
    return arr


def _configure_gurobi(model, settings: dict, grb) -> None:
    settings = settings_with_defaults(settings)
    if not settings.pop("verbose", False):
        model.setParam("OutputFlag", 0)
    time_limit = pop_time_limit(settings)
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    threads = pop_threads(settings)
    if threads is not None:
        model.setParam("Threads", int(threads))
    settings.pop("high_accuracy", None)
    for key, value in settings.items():
        model.setParam(key, value)


def _map_gurobi_status(raw_status: int, grb) -> str:
    mapping = {
        grb.GRB.OPTIMAL: status.OPTIMAL,
        grb.GRB.INFEASIBLE: status.PRIMAL_INFEASIBLE,
        grb.GRB.UNBOUNDED: status.DUAL_INFEASIBLE,
        grb.GRB.INF_OR_UNBD: status.PRIMAL_OR_DUAL_INFEASIBLE,
        grb.GRB.ITERATION_LIMIT: status.MAX_ITER_REACHED,
        grb.GRB.TIME_LIMIT: status.TIME_LIMIT,
        grb.GRB.SUBOPTIMAL: status.OPTIMAL_INACCURATE,
    }
    # WORK_LIMIT and NODE_LIMIT exist only in newer Gurobi releases.
    for name, canonical in [
        ("WORK_LIMIT", status.TIME_LIMIT),
        ("NODE_LIMIT", status.MAX_ITER_REACHED),
    ]:
        if hasattr(grb.GRB, name):
            mapping[getattr(grb.GRB, name)] = canonical
    return mapping.get(raw_status, status.SOLVER_ERROR)


def _maybe_int(value) -> int | None:
    return None if value is None else int(value)


def _maybe_float(value) -> float | None:
    return None if value is None else float(value)
