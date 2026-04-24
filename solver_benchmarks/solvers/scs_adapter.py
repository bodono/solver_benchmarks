"""SCS adapter."""

from __future__ import annotations

from pathlib import Path
import csv
import time

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_scs_box_cone
from .base import SolverAdapter, SolverUnavailable


class SCSSolverAdapter(SolverAdapter):
    solver_name = "scs"
    supported_problem_kinds = {QP, CONE}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import scs  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import scs
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the scs extra to use SCS") from exc

        settings = dict(self.settings)
        settings.setdefault("verbose", False)
        if settings.get("log_csv_filename") is True:
            settings["log_csv_filename"] = str(artifacts_dir / "scs_trace.csv")

        if problem.kind == QP:
            data, cone, _ = qp_to_scs_box_cone(problem.qp)
        else:
            cone_problem = problem.cone
            a = sp.csc_matrix(cone_problem["A"])
            p = cone_problem.get("P")
            if p is None:
                p = sp.csc_matrix((a.shape[1], a.shape[1]))
            data = {
                "P": sp.csc_matrix(p),
                "A": a,
                "b": np.asarray(cone_problem["b"], dtype=float),
                "c": np.asarray(cone_problem["q"], dtype=float),
            }
            cone = dict(cone_problem["cone"])
            free = int(cone.pop("f", 0))
            if free:
                cone["z"] = int(cone.get("z", 0)) + free

        start = time.perf_counter()
        raw = scs.solve(data, cone, **settings)
        elapsed = time.perf_counter() - start
        info = dict(raw.get("info", {}))
        mapped = _map_scs_status(info)
        trace = _read_csv_trace(settings.get("log_csv_filename"))
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(info.get("pobj")),
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_maybe_scs_seconds(info.get("setup_time")),
            solve_time_seconds=_maybe_scs_seconds(info.get("solve_time")),
            info=info,
            trace=trace,
        )


def _map_scs_status(info: dict) -> str:
    status_val = info.get("status_val")
    if status_val == 1:
        return status.OPTIMAL
    if status_val == 2:
        return status.OPTIMAL_INACCURATE
    if status_val == -1:
        return status.DUAL_INFEASIBLE
    if status_val == -2:
        return status.PRIMAL_INFEASIBLE
    if status_val in {-6, -7}:
        return status.MAX_ITER_REACHED
    text = str(info.get("status", "")).lower()
    if "solved" in text:
        return status.OPTIMAL
    if "infeasible" in text:
        return status.PRIMAL_INFEASIBLE
    if "unbounded" in text:
        return status.DUAL_INFEASIBLE
    return status.SOLVER_ERROR


def _read_csv_trace(path) -> list[dict]:
    if not path:
        return []
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    with trace_path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_scs_seconds(value):
    return None if value is None else float(value) / 1000.0


def _maybe_int(value):
    return None if value is None else int(value)
