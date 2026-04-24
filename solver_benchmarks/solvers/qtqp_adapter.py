"""QTQP adapter."""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult, to_jsonable
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone
from .base import SolverAdapter, SolverUnavailable


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
        settings = dict(self.settings)
        settings = _normalize_settings(settings, qtqp)
        a, b, z = qp_to_nonnegative_cone(qp)
        p = sp.csc_matrix(qp["P"])
        c = np.asarray(qp["q"], dtype=float)

        start = time.perf_counter()
        solver = qtqp.QTQP(a=sp.csc_matrix(a), b=b, c=c, z=z, p=p)
        solution = solver.solve(**settings, collect_stats=True)
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

        mapped = {
            "solved": status.OPTIMAL,
            "infeasible": status.PRIMAL_INFEASIBLE,
            "unbounded": status.DUAL_INFEASIBLE,
            "failed": status.MAX_ITER_REACHED,
        }.get(str(raw_status), status.SOLVER_ERROR)
        return SolverResult(
            status=mapped,
            objective_value=objective,
            iterations=iterations,
            run_time_seconds=elapsed,
            info={"raw_status": raw_status, **info},
            trace=[to_jsonable(row) for row in trace],
        )


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
    settings.setdefault("verbose", False)
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
