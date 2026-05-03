"""qtqp.clarabel.Clarabel adapter.

This is the Clarabel-paper IPM implemented inside the ``qtqp`` package
(``qtqp.clarabel.Clarabel``), distinct from the standalone Rust-based
Clarabel registered under ``clarabel``. The class extends ``qtqp.QTQP``
and takes the same ``(a, b, c, z, p)`` constructor, so this adapter
reuses qtqp's CONE/QP input handling.
"""

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
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult, to_jsonable
from solver_benchmarks.core.storage import atomic_write_text
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

from .base import (
    SolverAdapter,
    SolverUnavailable,
    mark_threads_ignored,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)


class QTQPClarabelSolverAdapter(SolverAdapter):
    solver_name = "qtqp_clarabel"
    supported_problem_kinds = {QP, CONE}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import qtqp.clarabel  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import qtqp
            import qtqp.clarabel as qtqp_clarabel
        except ModuleNotFoundError as exc:
            raise SolverUnavailable(
                "Install QTQP to use the qtqp_clarabel adapter"
            ) from exc

        settings = settings_with_defaults(self.settings)
        # qtqp.clarabel.Clarabel.solve has no native thread knob. We pop
        # threads/time_limit so the cross-adapter aliases don't surface
        # as TypeErrors, then mark them on info if the user asked for
        # something we couldn't honor.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        if time_limit is not None:
            settings["time_limit_secs"] = float(time_limit)
        settings = _normalize_settings(settings, qtqp)

        if problem.kind == QP:
            qp = problem.qp
            a, b, z = qp_to_nonnegative_cone(qp)
            p = sp.csc_matrix(qp["P"])
            c = np.asarray(qp["q"], dtype=float)
        else:
            cone_problem = problem.cone
            cone_keys = dict(cone_problem["cone"])
            z = int(cone_keys.pop("z", 0))
            l_count = int(cone_keys.pop("l", 0))
            # qtqp.clarabel.Clarabel inherits qtqp.QTQP's input shape:
            # zero (z) and nonneg (l) cones only. The legacy `f` key is
            # not merged into `z` here either — see qtqp_adapter for the
            # rationale.
            if cone_keys:
                return SolverResult(
                    status=status.SKIPPED_UNSUPPORTED,
                    info={
                        "reason": (
                            "qtqp_clarabel only handles z/l cones; got extra keys "
                            f"{sorted(cone_keys)!r}"
                        )
                    },
                )
            a = sp.csc_matrix(cone_problem["A"])
            if a.shape[0] != z + l_count:
                return SolverResult(
                    status=status.SOLVER_ERROR,
                    info={
                        "reason": (
                            f"cone dims {{'z': {z}, 'l': {l_count}}} sum to "
                            f"{z + l_count} but A has {a.shape[0]} rows"
                        )
                    },
                )
            b = np.asarray(cone_problem["b"], dtype=float)
            c = np.asarray(cone_problem["q"], dtype=float)
            p_in = cone_problem.get("P")
            p = (
                sp.csc_matrix(p_in)
                if p_in is not None
                else sp.csc_matrix((a.shape[1], a.shape[1]))
            )

        start = time.perf_counter()
        solver = qtqp_clarabel.Clarabel(a=sp.csc_matrix(a), b=b, c=c, z=z, p=p)
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

        mapped = _map_clarabel_status(raw_status)
        cone_dict: dict = {}
        if z:
            cone_dict["z"] = int(z)
        if a.shape[0] - z:
            cone_dict["l"] = int(a.shape[0] - z)
        kkt_dict = _compute_kkt(mapped, solution, p, c, a, b, cone_dict)
        result_info = {"raw_status": raw_status, **info}
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


def _map_clarabel_status(raw_status) -> str:
    return {
        "solved": status.OPTIMAL,
        "infeasible": status.PRIMAL_INFEASIBLE,
        "unbounded": status.DUAL_INFEASIBLE,
        "hit_max_iter": status.MAX_ITER_REACHED,
        "hit_time_limit": status.TIME_LIMIT,
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
    initialization = settings.get("initialization")
    if isinstance(initialization, str):
        # Clarabel.solve restricts initialization to TRIVIAL or CVXOPT
        # (LEAST_SQUARES raises ValueError). We still convert the string
        # to an enum here so the user gets the upstream solver's
        # validation error rather than an "Unknown initialization"
        # surprise from a string-comparison mismatch.
        settings["initialization"] = getattr(
            qtqp_module.Initialization, initialization.upper()
        )
    return settings


def _write_trace(path: Path, trace: list[dict]) -> None:
    if not trace:
        path.unlink(missing_ok=True)
        return
    body = "".join(json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in trace)
    atomic_write_text(path, body)


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
