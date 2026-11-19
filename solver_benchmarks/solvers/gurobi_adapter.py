"""GUROBI adapter.

A license failure raised by the solve (the size-limited license ``pip install
gurobipy`` ships with refuses larger models inside ``optimize()``) is a broken
environment, not Gurobi failing the problem: it raises ``SolverUnavailable``.
Any other ``GurobiError`` from the solve is ``solver_error``; other exceptions
are left to raise. ``Threads`` is one unless ``threads`` or ``Threads`` is
given (0: Gurobi's own choice). ``info["raw_status"]`` is the status name,
``info["raw_status_code"]`` Gurobi's integer code.
"""

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
    EPS_ALIASES,
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    SolverUnavailable,
    coerce_int_setting,
    finite_point,
    loosest_tolerance,
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

# Gurobi's documented defaults (parameter names are case-insensitive): primal
# and dual feasibility, and the barrier's relative gap. Not checked here
# (Gurobi needs a license). No certificate tolerance of its own.
GUROBI_DEFAULT_TOLERANCES = {"FeasibilityTol": 1e-6, "OptimalityTol": 1e-6, "BarConvTol": 1e-8}


class GurobiSolverAdapter(SolverAdapter):
    solver_name = "gurobi"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("gurobipy")

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        refuse_settings(settings, EPS_ALIASES, "Gurobi", "set its own tolerances instead")
        by_name = {str(key).lower(): value for key, value in settings.items()}
        given = {name: by_name.get(name.lower()) for name in GUROBI_DEFAULT_TOLERANCES}
        return loosest_tolerance(given, GUROBI_DEFAULT_TOLERANCES)

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import gurobipy as grb
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "gurobipy", "Install with the gurobi extra to use GUROBI", exc
            ) from exc

        qp = problem.qp
        q = np.asarray(qp["q"], dtype=float)
        p_mat = sp.coo_matrix(qp["P"])
        a_mat = sp.csr_matrix(qp["A"])
        n = int(qp.get("n", int(q.shape[0])))
        m = int(qp.get("m", a_mat.shape[0]))
        l, u = normalize_qp_bounds(qp["l"], qp["u"])
        lower = np.where(np.isneginf(l), -grb.GRB.INFINITY, l)
        upper = np.where(np.isposinf(u), grb.GRB.INFINITY, u)

        env = _create_environment(grb)
        try:
            model = grb.Model(f"{problem.dataset_id}/{problem.name}", env=env)
            try:
                translated, threads = _configure_gurobi(model, self.settings, grb)

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
                for row, col, value in zip(p_mat.row, p_mat.col, p_mat.data, strict=True):
                    objective.add(0.5 * float(value) * variables[row] * variables[col])
                objective.add(grb.LinExpr(q, variables))
                # The ``r`` constant offset is added by ``solver_benchmarks.worker``;
                # adding it here as well would double-count it for Gurobi solves.
                model.setObjective(objective, grb.GRB.MINIMIZE)

                start = time.perf_counter()
                try:
                    model.optimize()
                    resolved = _resolve_inf_or_unbd(model, grb, self.settings)
                except grb.GurobiError as exc:
                    if _is_license_error(exc):
                        raise SolverUnavailable(f"Gurobi license error: {exc}") from exc
                    return SolverResult(
                        status=status.SOLVER_ERROR,
                        run_time_seconds=time.perf_counter() - start,
                        info={"raw_status": "error", "error": str(exc), "threads": threads},
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
                    objective_value=reported_objective(
                        mapped, kkt_dict["primal_obj"] if kkt_dict is not None else None
                    ),
                    iterations=_maybe_int(
                        getattr(model, "BarIterCount", None)
                        or getattr(model, "IterCount", None)
                    ),
                    run_time_seconds=elapsed,
                    info={
                        "raw_status": _status_name(model.Status, grb),
                        "raw_status_code": int(model.Status),
                        "threads": threads,
                        "solver": "gurobi",
                        "version": ".".join(str(part) for part in grb.gurobi.version()),
                        "solver_reported_runtime": _maybe_float(
                            getattr(model, "Runtime", None)
                        ),
                        **({"settings_translated": translated} if translated else {}),
                        **({"inf_or_unbd_resolved": True} if resolved else {}),
                    },
                )
            finally:
                # Releases the C-level Model storage. Without this, long
                # batch runs accumulate models held by the env until GC.
                model.dispose()
        finally:
            env.dispose()


def _resolve_inf_or_unbd(model, grb, settings: dict) -> bool:
    """Tell an infeasible model from an unbounded one after INF_OR_UNBD.

    Gurobi's presolve dual reductions can prove only that a model has no
    optimum; its documented remedy is to set DualReductions=0 and
    reoptimize. Without it the row is primal_or_dual_infeasible, which never
    counts as a certificate. Returns whether the model was re-solved (the
    elapsed time covers both solves, as HiGHS's internal re-solve does).
    """
    if model.Status != grb.GRB.INF_OR_UNBD:
        return False
    for key, value in settings.items():
        if key.lower() == "dualreductions" and int(value) == 0:
            return False  # already off: the answer is final
    model.setParam("DualReductions", 0)
    model.optimize()
    return True


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
        if not finite_point(x, y):
            return None
        return kkt.qp_residuals(qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x, y)
    return None


def _configure_gurobi(model, settings: dict, grb) -> tuple[dict, int]:
    """Apply ``settings`` to ``model``; return the aliases it translated and
    the thread count.

    ``max_iter`` caps both the barrier and the simplex iteration counts
    (unless either is given by its Gurobi name). The ``eps`` aliases are
    refused: Gurobi's tolerances are not an absolute/relative pair. A
    parameter Gurobi rejects raises SolverConfigError naming it. ``Threads``
    is one unless ``threads`` or ``Threads`` is given.
    """
    settings = settings_with_defaults(settings)
    refuse_settings(
        settings,
        EPS_ALIASES,
        "Gurobi",
        "set its own tolerances instead (FeasibilityTol, OptimalityTol, BarConvTol)",
    )
    translated: dict = {}
    max_iter = pop_max_iter(settings, minimum=0)
    if max_iter is not None:
        given = {key.lower() for key in settings}
        for name in ("BarIterLimit", "IterationLimit"):
            if name.lower() not in given:
                settings[name] = max_iter
                translated[name] = max_iter
    if not settings.pop("verbose", False):
        model.setParam("OutputFlag", 0)
    time_limit = pop_time_limit(settings)
    if time_limit is not None:
        model.setParam("TimeLimit", float(time_limit))
    threads = pop_threads(settings)
    native_threads = [key for key in settings if key.lower() == "threads"]
    if native_threads:
        threads = coerce_int_setting(native_threads[0], settings[native_threads[0]], minimum=0)
    else:
        threads = threads_or_default(threads)
        settings["Threads"] = threads
    settings.pop("high_accuracy", None)
    for key, value in settings.items():
        try:
            model.setParam(key, value)
        except grb.GurobiError as exc:
            raise SolverConfigError(f"Invalid Gurobi parameter {key!r}={value!r}: {exc}") from exc
    return translated, threads


# Gurobi error codes for license failures: no valid license, and a model too
# large for the size-limited license that ships with the PyPI package.
_LICENSE_ERRNOS = frozenset({10009, 10010})


def _is_license_error(exc: Exception) -> bool:
    errno = getattr(exc, "errno", None)
    return errno in _LICENSE_ERRNOS or "license" in str(exc).lower()


# Gurobi's optimization status codes, by name (gurobipy's GRB.Status).
_STATUS_NAMES = (
    "LOADED",
    "OPTIMAL",
    "INFEASIBLE",
    "INF_OR_UNBD",
    "UNBOUNDED",
    "CUTOFF",
    "ITERATION_LIMIT",
    "NODE_LIMIT",
    "TIME_LIMIT",
    "SOLUTION_LIMIT",
    "INTERRUPTED",
    "NUMERIC",
    "SUBOPTIMAL",
    "INPROGRESS",
    "USER_OBJ_LIMIT",
    "WORK_LIMIT",
    "MEM_LIMIT",
)


def _status_name(code, grb) -> str:
    """Gurobi's name for a status code (e.g. ``"OPTIMAL"``), else the code."""
    holder = getattr(grb.GRB, "Status", grb.GRB)
    for name in _STATUS_NAMES:
        if getattr(holder, name, None) == code:
            return name
    return str(code)


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
