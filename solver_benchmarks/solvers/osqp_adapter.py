"""OSQP adapter."""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import (
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    finite_point,
    loosest_tolerance,
    mark_threads_ignored,
    module_available,
    normalize_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    translate_eps_alias,
)
from .qp_split import free_row_mask, normalize_qp_bounds, scatter_rows

# OSQP's defaults (checked on osqp 1.1.3): r <= eps_abs + eps_rel * norms.
OSQP_DEFAULT_TOLERANCES = {"eps_abs": 1e-3, "eps_rel": 1e-3}


class OSQPSolverAdapter(SolverAdapter):
    solver_name = "osqp"
    supported_problem_kinds = {QP}
    solver_modules = ("osqp",)

    @classmethod
    def is_available(cls) -> bool:
        return module_available("osqp")

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        translate_eps_alias(settings)
        return loosest_tolerance(settings, OSQP_DEFAULT_TOLERANCES)

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import osqp
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "osqp", "Install with the osqp extra to use OSQP", exc
            ) from exc

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
        normalize_max_iter(settings)
        translated = translate_eps_alias(settings)
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
        # Every OSQP status is mapped below, so the solve must not raise for
        # one: osqp 1.x raises OSQPException(status) for anything but "solved"
        # when raise_error is true (a default it announces will change).
        # Builds before 1.0 have no such argument and never raise. Looked up
        # before the clock starts: the signature costs tens of microseconds.
        solve_kwargs = _solve_kwargs(solver)
        start = time.perf_counter()
        try:
            solver.setup(p, qp["q"], a_kept, l[kept], u[kept], **settings)
        except ValueError as exc:
            # osqp's wrapper refuses an unknown setting name before the solve.
            if "setting" in str(exc).lower():
                raise SolverConfigError(f"OSQP rejected the settings: {exc}") from exc
            raise
        except TypeError as exc:
            # The settings binding refuses a value of the wrong type before
            # the solve; the problem data the adapter builds is well typed.
            raise SolverConfigError(
                f"OSQP rejected a setting value in {sorted(settings)}: {exc}"
            ) from exc
        except _osqp_exception(osqp) as exc:
            code = exc.args[0] if exc.args else None
            if code == _OSQP_SETTINGS_VALIDATION_ERROR:
                raise SolverConfigError(
                    "OSQP rejected a setting value (see its message in stdout.log): "
                    f"{sorted(settings)}"
                ) from exc
            # Data validation (e.g. l > u) and a non-convex P are the solver
            # failing on this problem, not a harness error.
            error_info = {
                "raw_status": "error",
                "error": f"OSQPException: {_osqp_error_text(code)}",
                "threads": None,
            }
            mark_threads_ignored(error_info, threads)
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start,
                info=error_info,
            )
        raw = solver.solve(**solve_kwargs)
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
        info: dict[str, Any] = {
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
        kkt_dict = _compute_kkt(
            _certificate_kind(raw.info.status_val, osqp) or mapped, raw, qp, p, a, l, u, kept
        )
        info["threads"] = None
        mark_threads_ignored(info, threads)
        if translated:
            info["settings_translated"] = translated
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
# OSQP's osqp_error_type, the code an OSQPException from setup() carries.
_OSQP_ERROR_NAMES = {
    1: "data validation",
    2: "settings validation",
    3: "linear system solver initialization",
    4: "non-convex problem",
    5: "memory allocation",
    6: "workspace not initialized",
    7: "algebra load",
    8: "file open",
    9: "codegen defines",
    10: "data not initialized",
    11: "function not implemented",
}
_OSQP_SETTINGS_VALIDATION_ERROR = 2


def _osqp_error_text(code) -> str:
    """``code`` with its osqp_error_type name, e.g. ``4 (non-convex problem)``."""
    name = _OSQP_ERROR_NAMES.get(code) if isinstance(code, int) else None
    return f"{code} ({name})" if name else str(code)


def _solve_kwargs(solver) -> dict[str, Any]:
    """``raise_error=False`` for OSQP builds whose solve() takes it."""
    try:
        parameters = inspect.signature(solver.solve).parameters
    except (TypeError, ValueError):
        return {}
    return {"raise_error": False} if "raise_error" in parameters else {}


def _osqp_exception(osqp):
    """The exception type osqp raises for solver errors (if this build has one)."""
    exc = getattr(osqp, "OSQPException", None)
    return exc if isinstance(exc, type) and issubclass(exc, Exception) else ()


# OSQP labels a limit-hit iterate that passes its looser check with these.
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


def _certificate_kind(status_val, osqp) -> str | None:
    """The certificate an ``*_INFEASIBLE_INACCURATE`` status carries, if any.

    Those statuses are reported as the limit the solve stopped on, and their
    ``x`` / ``y`` are NaN, but OSQP still returns the (inaccurate) certificate;
    its residuals are kept on ``kkt``. As recorded they do not count as
    success, since the row's status is a limit; the KKT verification
    (analysis/verify.py) promotes the row to the certificate's status when
    the certificate is fully checked and passes its tolerance.
    """
    for name, kind in (
        ("OSQP_PRIMAL_INFEASIBLE_INACCURATE", status.PRIMAL_INFEASIBLE_INACCURATE),
        ("OSQP_DUAL_INFEASIBLE_INACCURATE", status.DUAL_INFEASIBLE_INACCURATE),
    ):
        try:
            if status_val == osqp.constant(name):
                return kind
        except (ValueError, KeyError, AttributeError, RuntimeError):
            continue
    return None


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
        if not finite_point(cert):
            return None
        return kkt.qp_primal_infeasibility_cert(a, l, u, cert)
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        cert = getattr(raw, "dual_inf_cert", None)
        if not finite_point(cert):
            return None
        return kkt.qp_dual_infeasibility_cert(p, q, a, l, u, cert)
    return None


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
