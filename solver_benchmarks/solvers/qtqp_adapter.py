"""QTQP adapter."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult, to_jsonable
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

from .base import (
    SolverAdapter,
    SolverUnavailable,
    mark_threads_ignored,
    mark_time_limit_ignored,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)


class QTQPSolverAdapter(SolverAdapter):
    solver_name = "qtqp"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import qtqp  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import qtqp
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install QTQP to use the QTQP adapter") from exc

        qp = problem.qp
        settings = settings_with_defaults(self.settings)
        # QTQP exposes neither a time-limit knob nor a thread-count
        # setting; record the configured values on info so callers can
        # detect they were ignored rather than silently dropping them.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        settings = _normalize_settings(settings, qtqp)
        a, b, z = qp_to_nonnegative_cone(qp)
        p = sp.csc_matrix(qp["P"])
        c = np.asarray(qp["q"], dtype=float)

        start = time.perf_counter()
        solver = qtqp.QTQP(a=sp.csc_matrix(a), b=b, c=c, z=z, p=p)
        solve_kwargs = dict(settings)
        if "collect_stats" in inspect.signature(solver.solve).parameters:
            solve_kwargs["collect_stats"] = True
        solution = solver.solve(**solve_kwargs)
        elapsed = time.perf_counter() - start

        raw_status = getattr(solution.status, "value", str(solution.status))
        trace = list(getattr(solution, "stats", []) or [])
        _write_trace(artifacts_dir / "trace.jsonl", trace)
        stats = pd.DataFrame(trace) if trace else pd.DataFrame()
        if not stats.empty:
            last = stats.tail(1).iloc[0]
            objective = _maybe_float(last.get("pcost"))
            iterations = _maybe_int(last.get("iter"))
            info = to_jsonable(last.to_dict())
        else:
            objective = None
            iterations = None
            info = {}

        mapped = _map_qtqp_status(raw_status)
        cone_dict: dict = {}
        if z:
            cone_dict["z"] = int(z)
        if a.shape[0] - z:
            cone_dict["l"] = int(a.shape[0] - z)
        kkt_dict = _compute_kkt(mapped, solution, p, c, a, b, cone_dict)
        result_info = {"raw_status": raw_status, **info}
        mark_time_limit_ignored(result_info, time_limit)
        mark_threads_ignored(result_info, threads)
        return SolverResult(
            status=mapped,
            objective_value=objective,
            iterations=iterations,
            run_time_seconds=elapsed,
            info=result_info,
            trace=[to_jsonable(row) for row in trace],
            kkt=kkt_dict,
        )


def _map_qtqp_status(raw_status) -> str:
    return {
        "solved": status.OPTIMAL,
        "infeasible": status.PRIMAL_INFEASIBLE,
        "unbounded": status.DUAL_INFEASIBLE,
        "hit_max_iter": status.MAX_ITER_REACHED,
        "unfinished": status.SOLVER_ERROR,
        "failed": status.SOLVER_ERROR,
    }.get(str(raw_status), status.SOLVER_ERROR)


def _compute_kkt(mapped_status, solution, p, c, a, b, cone_dict):
    x = getattr(solution, "x", None)
    y = getattr(solution, "y", None)
    s_slack = getattr(solution, "s", None)
    if x is None:
        return None
    if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
        if y is None or s_slack is None:
            return None
        return kkt.cone_residuals(p, c, a, b, cone_dict, x, y, s_slack)
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        if y is None:
            return None
        return kkt.cone_primal_infeasibility_cert(a, b, cone_dict, y)
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        return kkt.cone_dual_infeasibility_cert(p, c, a, cone_dict, x)
    return None


def _normalize_settings(settings: dict, qtqp_module):
    linear_solver = settings.get("linear_solver")
    if isinstance(linear_solver, str):
        lookup = {
            "qdldl": "QDLDL",
            "accelerate": "ACCELERATE",
            "cholmod": "CHOLMOD",
        }
        attr = lookup.get(linear_solver.lower(), linear_solver.upper())
        settings["linear_solver"] = getattr(qtqp_module.LinearSolver, attr)
    return settings


def _write_trace(path: Path, trace: list[dict]) -> None:
    if not trace:
        return
    with path.open("w") as handle:
        for row in trace:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
