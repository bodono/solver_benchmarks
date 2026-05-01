"""OSQP adapter."""

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
    mark_threads_ignored,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)


class OSQPSolverAdapter(SolverAdapter):
    solver_name = "osqp"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import osqp  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import osqp
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the osqp extra to use OSQP") from exc

        qp = problem.qp
        settings = settings_with_defaults(self.settings)
        # Translate the cross-adapter aliases. OSQP's own knob is
        # ``time_limit`` (seconds, 0 = none); the alias `time_limit_sec`
        # is normalized here so users can use either spelling.
        time_limit = pop_time_limit(settings)
        if time_limit is not None:
            settings["time_limit"] = float(time_limit)
        # OSQP does not expose a thread count knob.
        threads = pop_threads(settings)
        p = sp.csc_matrix(qp["P"])
        a = sp.csc_matrix(qp["A"])

        solver = osqp.OSQP()
        start = time.perf_counter()
        solver.setup(p, qp["q"], a, qp["l"], qp["u"], **settings)
        raw = solver.solve()
        elapsed = time.perf_counter() - start

        status_map = {
            osqp.constant("OSQP_SOLVED"): status.OPTIMAL,
            osqp.constant("OSQP_SOLVED_INACCURATE"): status.OPTIMAL_INACCURATE,
            osqp.constant("OSQP_MAX_ITER_REACHED"): status.MAX_ITER_REACHED,
            osqp.constant("OSQP_PRIMAL_INFEASIBLE"): status.PRIMAL_INFEASIBLE,
            osqp.constant("OSQP_DUAL_INFEASIBLE"): status.DUAL_INFEASIBLE,
            osqp.constant("OSQP_TIME_LIMIT_REACHED"): status.TIME_LIMIT,
        }
        for name, canonical in [
            ("OSQP_PRIMAL_INFEASIBLE_INACCURATE", status.PRIMAL_INFEASIBLE_INACCURATE),
            ("OSQP_DUAL_INFEASIBLE_INACCURATE", status.DUAL_INFEASIBLE_INACCURATE),
            ("OSQP_NON_CVX", status.SOLVER_ERROR),
        ]:
            try:
                status_map[osqp.constant(name)] = canonical
            except (ValueError, KeyError, AttributeError):
                continue
        mapped = status_map.get(raw.info.status_val, status.SOLVER_ERROR)
        info = {
            key: getattr(raw.info, key, None)
            for key in [
                "status",
                "status_val",
                "obj_val",
                "prim_res",
                "dual_res",
                "iter",
                "rho_updates",
                "rho_estimate",
                "setup_time",
                "solve_time",
                "update_time",
                "polish_time",
                "run_time",
            ]
            if hasattr(raw.info, key)
        }
        kkt_dict = _compute_kkt(mapped, raw, qp, p, a)
        mark_threads_ignored(info, threads)
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(raw.info.obj_val),
            iterations=_maybe_int(raw.info.iter),
            run_time_seconds=elapsed,
            setup_time_seconds=_maybe_float(getattr(raw.info, "setup_time", None)),
            solve_time_seconds=_maybe_float(getattr(raw.info, "solve_time", None)),
            info=info,
            extra={
                "status_polish": getattr(raw.info, "status_polish", None),
                "rho_updates": getattr(raw.info, "rho_updates", None),
                "rho_estimate": getattr(raw.info, "rho_estimate", None),
            },
            kkt=kkt_dict,
        )


def _compute_kkt(mapped_status, raw, qp, p, a):
    q = np.asarray(qp["q"], dtype=float)
    l = np.asarray(qp["l"], dtype=float)
    u = np.asarray(qp["u"], dtype=float)
    if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
        x = getattr(raw, "x", None)
        y = getattr(raw, "y", None)
        if x is None or y is None:
            return None
        return kkt.qp_residuals(p, q, a, l, u, x, y)
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        cert = getattr(raw, "prim_inf_cert", None)
        if cert is None:
            return None
        return kkt.qp_primal_infeasibility_cert(a, l, u, cert)
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        cert = getattr(raw, "dual_inf_cert", None)
        if cert is None:
            return None
        return kkt.qp_dual_infeasibility_cert(p, q, a, l, u, cert)
    return None


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
