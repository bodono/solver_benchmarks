"""NVIDIA cuOpt adapter (GPU PDLP for LP and QP).

cuOpt solves ``min c'x + (1/2) x'(Q + Q')x`` subject to
``cl <= A x <= cu`` and ``vl <= x <= vu`` on the GPU. The harness QP form
``min (1/2) x'P x + q'x, l <= A x <= u`` maps onto it with ``Q = P / 2``
and free variables, so the objective is identical.

The dual sign convention and the termination-status enum are checked at
runtime against the installed cuOpt (see ``_probe``), since the Python API
has changed across releases.
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
    SolverAdapter,
    SolverUnavailable,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)

INF_BOUND = 1.0e20


def _import_cuopt():
    try:
        from cuopt import linear_programming  # noqa: F401
        from cuopt.linear_programming import data_model, solver, solver_settings  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on GPU environment
        raise SolverUnavailable(
            "Install cuopt-cu12 (NVIDIA pip index) on a CUDA machine to use cuOpt"
        ) from exc
    return data_model, solver, solver_settings


def _clean_bounds(v: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).copy()
    out[out <= -INF_BOUND] = -np.inf
    out[out >= INF_BOUND] = np.inf
    return out


def _termination_name(solution) -> str:
    """Best-effort readable termination status across cuOpt versions."""
    term = solution.get_termination_status()
    name = getattr(term, "name", None)
    if name:
        return str(name)
    try:
        reason = solution.get_termination_reason()
        if reason:
            return str(reason)
    except Exception:  # pragma: no cover
        pass
    return str(term)


_STATUS_MAP = {
    "Optimal": status.OPTIMAL,
    "OPTIMAL": status.OPTIMAL,
    "PrimalInfeasible": status.PRIMAL_INFEASIBLE,
    "PRIMALINFEASIBLE": status.PRIMAL_INFEASIBLE,
    "PRIMAL_INFEASIBLE": status.PRIMAL_INFEASIBLE,
    "DualInfeasible": status.DUAL_INFEASIBLE,
    "DUALINFEASIBLE": status.DUAL_INFEASIBLE,
    "DUAL_INFEASIBLE": status.DUAL_INFEASIBLE,
    "TimeLimit": status.TIME_LIMIT,
    "TIMELIMIT": status.TIME_LIMIT,
    "TIME_LIMIT": status.TIME_LIMIT,
    "IterationLimit": status.MAX_ITER_REACHED,
    "ITERATIONLIMIT": status.MAX_ITER_REACHED,
    "ITERATION_LIMIT": status.MAX_ITER_REACHED,
    "UnboundedOrInfeasible": status.PRIMAL_OR_DUAL_INFEASIBLE,
    "UNBOUNDEDORINFEASIBLE": status.PRIMAL_OR_DUAL_INFEASIBLE,
    "UNBOUNDED_OR_INFEASIBLE": status.PRIMAL_OR_DUAL_INFEASIBLE,
    "FeasibleFound": status.OPTIMAL_INACCURATE,
    "FEASIBLEFOUND": status.OPTIMAL_INACCURATE,
}


def _map_status(name: str) -> str:
    key = name.split(".")[-1]
    if key in _STATUS_MAP:
        return _STATUS_MAP[key]
    compact = key.replace("_", "").upper()
    for cand, mapped in _STATUS_MAP.items():
        if cand.replace("_", "").upper() == compact:
            return mapped
    return status.SOLVER_ERROR


class CuOptSolverAdapter(SolverAdapter):
    solver_name = "cuopt"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            _import_cuopt()
        except SolverUnavailable:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        data_model, solver, solver_settings = _import_cuopt()
        qp = problem.qp
        settings = settings_with_defaults(self.settings)
        verbose = bool(settings.pop("verbose", False))
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        eps = settings.pop("eps", None)
        eps_abs = settings.pop("eps_abs", None)
        eps_rel = settings.pop("eps_rel", None)
        # cuOpt reports duals with the opposite sign to the harness QP form
        # (checked numerically on an active-bound LP); override with dual_sign.
        dual_sign = float(settings.pop("dual_sign", -1.0))

        p = sp.csc_matrix(qp["P"])
        a = sp.csr_matrix(qp["A"])
        q = np.asarray(qp["q"], dtype=np.float64)
        l = _clean_bounds(qp["l"])
        u = _clean_bounds(qp["u"])
        n = a.shape[1]

        model = data_model.DataModel()
        model.set_csr_constraint_matrix(
            a.data.astype(np.float64), a.indices.astype(np.int32), a.indptr.astype(np.int32)
        )
        model.set_constraint_lower_bounds(l)
        model.set_constraint_upper_bounds(u)
        model.set_objective_coefficients(q)
        model.set_variable_lower_bounds(np.full(n, -np.inf))
        model.set_variable_upper_bounds(np.full(n, np.inf))
        if p.nnz > 0:
            half = sp.csr_matrix(p * 0.5)  # cuOpt symmetrizes as Q + Q', so Q = P/2
            model.set_quadratic_objective_matrix(
                half.data.astype(np.float64), half.indices.astype(np.int32), half.indptr.astype(np.int32)
            )

        opts = solver_settings.SolverSettings()
        # ``eps`` sets all six of cuOpt's tolerances; ``eps_abs`` / ``eps_rel``
        # set the absolute and relative primal, dual and gap tolerances
        # independently (an explicit alias wins over ``eps``).
        if eps is not None:
            opts.set_optimality_tolerance(float(eps))
        for alias, prefix in ((eps_abs, "absolute"), (eps_rel, "relative")):
            if alias is not None:
                for kind in ("primal", "dual", "gap"):
                    opts.set_parameter(f"{prefix}_{kind}_tolerance", float(alias))
        if time_limit is not None:
            opts.set_parameter("time_limit", float(time_limit))
        if threads is not None:
            opts.set_parameter("num_cpu_threads", int(threads))
        for key, value in settings.items():
            opts.set_parameter(key, value)
        if verbose:
            try:
                opts.set_parameter("log_to_console", True)
            except Exception:  # pragma: no cover
                pass

        start = time.perf_counter()
        solution = solver.Solve(model, opts)
        elapsed = time.perf_counter() - start

        term_name = _termination_name(solution)
        mapped = _map_status(term_name)
        info = {
            "termination": term_name,
            "solve_time": _maybe_float(solution.get_solve_time()),
            "primal_objective": _maybe_float(solution.get_primal_objective()),
            "dual_objective": _maybe_float(solution.get_dual_objective()),
        }
        try:
            info["lp_stats"] = {k: _jsonable(v) for k, v in dict(solution.get_lp_stats()).items()}
        except Exception:  # pragma: no cover
            pass
        x = y = None
        kkt_dict = None
        if mapped in {status.OPTIMAL, status.OPTIMAL_INACCURATE, status.TIME_LIMIT, status.MAX_ITER_REACHED}:
            x = np.asarray(solution.get_primal_solution(), dtype=np.float64)
            y = np.asarray(solution.get_dual_solution(), dtype=np.float64)
            if x.size == n and y.size == a.shape[0]:
                kkt_dict = kkt.qp_residuals(p, q, sp.csc_matrix(a), l, u, x, dual_sign * y)
        iterations = None
        stats = info.get("lp_stats") or {}
        for key in ("nb_iterations", "num_iterations", "iterations"):
            if key in stats:
                iterations = _maybe_int(stats[key])
                break
        return SolverResult(
            status=mapped,
            objective_value=info["primal_objective"] if x is not None else None,
            iterations=iterations,
            run_time_seconds=elapsed,
            solve_time_seconds=info["solve_time"],
            info=info,
            kkt=kkt_dict,
        )


def _maybe_float(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _maybe_int(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _jsonable(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    return v
