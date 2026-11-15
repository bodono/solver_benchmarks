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
    finite_point,
    mark_threads_ignored,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
)
from .qp_split import free_row_mask, normalize_qp_bounds, scatter_rows


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
        # OSQP counts a bound as infinite only beyond OSQP_INFTY * MIN_SCALING
        # (1e26), so the datasets' +-1e20 sentinel must become a true
        # infinity here; otherwise it is a finite bound that slows the solve
        # and breaks infeasibility detection. Rows with no finite bound are
        # left out (they would still enter OSQP's scaling) and get a zero
        # multiplier when the duals are mapped back.
        l, u = normalize_qp_bounds(qp["l"], qp["u"])
        kept = ~free_row_mask(l, u)
        a_kept = a if kept.all() else a[np.flatnonzero(kept), :]

        solver = osqp.OSQP()
        start = time.perf_counter()
        solver.setup(p, qp["q"], a_kept, l[kept], u[kept], **settings)
        raw = solver.solve()
        elapsed = time.perf_counter() - start

        iterations = _maybe_int(raw.info.iter)
        max_iter = getattr(getattr(solver, "settings", None), "max_iter", None)
        if max_iter is None:
            max_iter = settings.get("max_iter", _OSQP_DEFAULT_MAX_ITER)
        mapped = _map_osqp_status(
            raw.info.status_val,
            osqp,
            hit_iteration_limit=iterations is not None and iterations >= int(max_iter),
        )
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
        info["raw_status"] = getattr(raw.info, "status", None)
        kkt_dict = _compute_kkt(mapped, raw, qp, p, a, l, u, kept)
        mark_threads_ignored(info, threads)
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, raw.info.obj_val),
            iterations=iterations,
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


# OSQP's default iteration cap, used only if the solver does not expose its
# settings after setup.
_OSQP_DEFAULT_MAX_ITER = 4000
_INACCURATE_AT_LIMIT = (
    "OSQP_SOLVED_INACCURATE",
    "OSQP_PRIMAL_INFEASIBLE_INACCURATE",
    "OSQP_DUAL_INFEASIBLE_INACCURATE",
)


def _map_osqp_status(status_val, osqp, *, hit_iteration_limit: bool) -> str:
    """Canonical status for OSQP's ``info.status_val``.

    OSQP reports the ``*_INACCURATE`` statuses only after it stops on
    ``max_iter`` or ``time_limit`` and the final iterate passes its 10x looser
    check; the solve still ended on a limit, so it is reported as that limit
    (iteration count at ``max_iter``: max_iter_reached, else time_limit).
    """
    status_map = {
        osqp.constant("OSQP_SOLVED"): status.OPTIMAL,
        osqp.constant("OSQP_MAX_ITER_REACHED"): status.MAX_ITER_REACHED,
        osqp.constant("OSQP_PRIMAL_INFEASIBLE"): status.PRIMAL_INFEASIBLE,
        osqp.constant("OSQP_DUAL_INFEASIBLE"): status.DUAL_INFEASIBLE,
        osqp.constant("OSQP_TIME_LIMIT_REACHED"): status.TIME_LIMIT,
    }
    limit = status.MAX_ITER_REACHED if hit_iteration_limit else status.TIME_LIMIT
    for name, canonical in [
        *((name, limit) for name in _INACCURATE_AT_LIMIT),
        ("OSQP_NON_CVX", status.SOLVER_ERROR),
    ]:
        try:
            status_map[osqp.constant(name)] = canonical
        except (ValueError, KeyError, AttributeError, RuntimeError):
            continue
    return status_map.get(status_val, status.SOLVER_ERROR)


def _compute_kkt(mapped_status, raw, qp, p, a, l, u, kept):
    q = np.asarray(qp["q"], dtype=float)
    m = a.shape[0]
    if mapped_status in status.ANY_FEASIBLE:
        x = getattr(raw, "x", None)
        y = scatter_rows(m, kept, getattr(raw, "y", None))
        if not finite_point(x, y):
            return None
        return kkt.qp_residuals(p, q, a, l, u, x, y)
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        cert = scatter_rows(m, kept, getattr(raw, "prim_inf_cert", None))
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
