"""SCS adapter.

Settings are SCS's own keyword arguments; the harness aliases ``max_iter``
(-> ``max_iters``) and ``eps`` (-> ``eps_abs`` / ``eps_rel``) are translated
and ``time_limit`` becomes ``time_limit_secs``. SCS reports an unknown name or
a bad value by raising from the same call that solves, so the adapter checks
the value ranges SCS enforces first and turns SCS's own refusal of a setting
into ``SolverConfigError``; any other exception from SCS (problem data it
cannot take: a non-convex ``P``, crossed bounds, non-finite entries, a problem
without rows) is the solver failing that problem, ``solver_error``.
"""

from __future__ import annotations

import csv
import math
import numbers
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_scs_box_cone, unbox_scs_dual

from .base import (
    IMPORT_ERRORS,
    SolverAdapter,
    SolverConfigError,
    coerce_int_setting,
    finite_point,
    is_setting_type_error,
    mark_threads_ignored,
    module_available,
    pop_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    translate_tolerance_aliases,
)

_SCS_NUM_THREADS_SUPPORTED: bool | None = None


def _scs_supports_num_threads() -> bool:
    """Detect whether the installed SCS build accepts ``num_threads``.

    Only the OpenMP-built variant accepts the kwarg; on a stock build
    SCS' C extension raises ``TypeError: 'num_threads' is an invalid
    keyword argument for this function``. We probe once with a 1x1
    problem and cache the result, so cross-adapter ``threads`` requests
    can be either forwarded or marked ignored without crashing the
    solve.
    """
    global _SCS_NUM_THREADS_SUPPORTED
    if _SCS_NUM_THREADS_SUPPORTED is not None:
        return _SCS_NUM_THREADS_SUPPORTED
    try:
        import scs as _scs
    except IMPORT_ERRORS:
        _SCS_NUM_THREADS_SUPPORTED = False
        return False
    data = {
        "P": sp.csc_matrix((1, 1)),
        "A": sp.csc_matrix(np.array([[1.0]])),
        "b": np.array([0.0]),
        "c": np.array([0.0]),
    }
    cone = {"z": 1}
    try:
        _scs.SCS(data, cone, num_threads=1, verbose=False)
    except TypeError:
        _SCS_NUM_THREADS_SUPPORTED = False
    except Exception:
        # Any other failure means SCS at least accepted the kwarg.
        _SCS_NUM_THREADS_SUPPORTED = True
    else:
        _SCS_NUM_THREADS_SUPPORTED = True
    return _SCS_NUM_THREADS_SUPPORTED


class SCSSolverAdapter(SolverAdapter):
    solver_name = "scs"
    supported_problem_kinds = {QP, CONE}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("scs")

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import scs
        except IMPORT_ERRORS as exc:
            raise solver_import_failed("scs", "Install with the scs extra to use SCS", exc) from exc

        settings = settings_with_defaults(self.settings)
        # Translate cross-adapter aliases. SCS' native time-limit knob
        # is ``time_limit_secs`` (note the trailing ``s``); accept the
        # other spellings via pop_time_limit.
        time_limit = pop_time_limit(settings)
        if time_limit is not None:
            settings["time_limit_secs"] = float(time_limit)
        translated = translate_settings(settings)
        threads = pop_threads(settings)
        threads_ignored = False
        if threads is not None:
            # SCS only accepts the ``num_threads`` kwarg on OpenMP
            # builds — on a stock build the C extension raises
            # TypeError. Probe once and either forward or mark ignored.
            if _scs_supports_num_threads():
                settings.setdefault("num_threads", threads)
            else:
                threads_ignored = True
        if settings.get("log_csv_filename") is True:
            settings["log_csv_filename"] = str(artifacts_dir / "scs_trace.csv")
        _check_setting_values(settings, translated)

        inv_perm = None
        if problem.kind == QP:
            data, cone, inv_perm = qp_to_scs_box_cone(problem.qp)
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
            exponents = cone.get("p", [])
            exponents = exponents if isinstance(exponents, (list, tuple)) else [exponents]
            if not all(_is_number(alpha) for alpha in exponents):
                # SCS's TypeError for it reads like a setting's; it is data.
                return SolverResult(
                    status=status.SOLVER_ERROR,
                    run_time_seconds=0.0,
                    info={
                        "raw_status": "error",
                        "error": f"power cone exponents must be numbers (got {exponents!r})",
                        "threads": None,
                    },
                )

        # SCS's C layer takes float64 data only (an integer-typed A or P is
        # refused with a TypeError); the other adapters convert as well.
        data["A"] = sp.csc_matrix(data["A"], dtype=float)
        if data.get("P") is not None:
            data["P"] = sp.csc_matrix(data["P"], dtype=float)
        data["b"] = np.asarray(data["b"], dtype=float)
        data["c"] = np.asarray(data["c"], dtype=float)
        threads_used = settings.get("num_threads") if not threads_ignored else None
        start = time.perf_counter()
        try:
            raw = scs.solve(data, cone, **settings)
        except (TypeError, ValueError) as exc:
            # SCS raises from the solving call both for a setting it refuses
            # (an unknown keyword, a value of the wrong type or range) and for
            # problem data it cannot take; only the first is a configuration
            # error.
            if isinstance(exc, TypeError) and is_setting_type_error(exc):
                raise SolverConfigError(f"SCS rejected the settings {sorted(settings)}: {exc}") from exc
            if isinstance(exc, ValueError) and _is_setting_error(exc, settings):
                raise SolverConfigError(f"SCS rejected a setting value: {exc}") from exc
            error_info = {
                "raw_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "threads": threads_used,
            }
            if threads_ignored:
                mark_threads_ignored(error_info, threads)
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start,
                info=error_info,
            )
        elapsed = time.perf_counter() - start
        info = dict(raw.get("info", {}))
        info["threads"] = threads_used
        if threads_ignored:
            mark_threads_ignored(info, threads)
        if translated:
            info["settings_translated"] = translated
        mapped = _map_scs_status(
            info,
            max_iters=settings.get("max_iters"),
            time_limit_secs=settings.get("time_limit_secs"),
        )
        info["raw_status"] = info.get("status")
        trace = _read_csv_trace(settings.get("log_csv_filename"))
        kkt_dict = _compute_kkt(problem, mapped, raw, cone, inv_perm)
        # The objective is reported only for a claimed solution. For an
        # infeasibility certificate SCS's pobj is the certificate's
        # normalizer, and for an iterate cut off by max_iters or
        # time_limit_secs it is the value at an arbitrary point; the KKT
        # record keeps that point's primal objective for kkt-verify.
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, info.get("pobj")),
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_maybe_scs_seconds(info.get("setup_time")),
            solve_time_seconds=_maybe_scs_seconds(info.get("solve_time")),
            info=info,
            trace=trace,
            kkt=kkt_dict,
        )


def _compute_kkt(problem, mapped_status, raw, cone, inv_perm):
    x = raw.get("x")
    y = raw.get("y")
    s = raw.get("s")
    if x is None:
        return None
    # A non-finite iterate has no residuals (None, not a record of NaNs).
    if problem.kind == QP:
        qp = problem.qp
        # Every status that may carry a usable point (including the
        # max_iters / time-limit iterates) gets solution residuals, so
        # ``kkt-verify --promote`` treats SCS like every other solver.
        if mapped_status in status.ANY_FEASIBLE and finite_point(x, y):
            y_qp = unbox_scs_dual(y, cone, inv_perm)
            return kkt.qp_residuals(
                qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x, y_qp
            )
        if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE} and finite_point(y):
            y_qp = unbox_scs_dual(y, cone, inv_perm)
            return kkt.qp_primal_infeasibility_cert(qp["A"], qp["l"], qp["u"], y_qp)
        if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE} and finite_point(x):
            return kkt.qp_dual_infeasibility_cert(
                qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x
            )
        return None
    cone_problem = problem.cone
    p = cone_problem.get("P")
    if p is None:
        p = sp.csc_matrix((sp.csc_matrix(cone_problem["A"]).shape[1],) * 2)
    if mapped_status in status.ANY_FEASIBLE and finite_point(x, y, s):
        return kkt.cone_residuals(
            p,
            cone_problem["q"],
            cone_problem["A"],
            cone_problem["b"],
            cone,
            x,
            y,
            s,
        )
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE} and finite_point(y):
        return kkt.cone_primal_infeasibility_cert(
            cone_problem["A"], cone_problem["b"], cone, y
        )
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE} and finite_point(x):
        return kkt.cone_dual_infeasibility_cert(
            p, cone_problem["q"], cone_problem["A"], cone, x
        )
    return None


# SCS 3.x returns its "inaccurate" codes only when it stops on max_iters or
# time_limit_secs: it then reports whichever of solved / infeasible /
# unbounded looks least bad and appends the limit to the status text, e.g.
# "solved (inaccurate - reached max_iters)".
_SCS_INACCURATE = {
    2: status.OPTIMAL_INACCURATE,
    -6: status.DUAL_INFEASIBLE_INACCURATE,
    -7: status.PRIMAL_INFEASIBLE_INACCURATE,
}


def _map_scs_status(
    info: dict,
    *,
    max_iters: int | None = None,
    time_limit_secs: float | None = None,
) -> str:
    """Translate SCS's ``info`` to a canonical status.

    A run cut off by ``max_iters`` or ``time_limit_secs`` maps to
    ``MAX_ITER_REACHED`` / ``TIME_LIMIT`` like every other adapter, never
    to ``optimal_inaccurate`` or an ``*_infeasible_inaccurate`` guess; the
    solver's own text stays on ``info`` (the adapter copies it to
    ``raw_status``). The limit is read from the status text and, when the
    text does not name it, from the iteration count and solve time against
    the configured limits.
    """
    # Codes from scs/include/glbopts.h.
    status_val = info.get("status_val")
    text = str(info.get("status", "")).lower()
    if status_val in _SCS_INACCURATE or "inaccurate" in text:
        limit = _scs_limit_status(info, text, max_iters, time_limit_secs)
        if limit is not None:
            return limit
    if status_val == 1:
        return status.OPTIMAL
    if status_val in _SCS_INACCURATE:
        return _SCS_INACCURATE[status_val]
    if status_val == -1:
        return status.DUAL_INFEASIBLE
    if status_val == -2:
        return status.PRIMAL_INFEASIBLE
    if status_val in {-3, -4, -5}:
        return status.SOLVER_ERROR
    # No recognised code: read the text, keeping "(inaccurate)" inaccurate.
    inaccurate = "inaccurate" in text
    if "solved" in text:
        return status.OPTIMAL_INACCURATE if inaccurate else status.OPTIMAL
    if "infeasible" in text:
        return status.PRIMAL_INFEASIBLE_INACCURATE if inaccurate else status.PRIMAL_INFEASIBLE
    if "unbounded" in text:
        return status.DUAL_INFEASIBLE_INACCURATE if inaccurate else status.DUAL_INFEASIBLE
    return status.SOLVER_ERROR


def _scs_limit_status(info, text, max_iters, time_limit_secs) -> str | None:
    if "reached max_iter" in text:
        return status.MAX_ITER_REACHED
    if "reached time_limit" in text:
        return status.TIME_LIMIT
    iterations = _maybe_int(info.get("iter"))
    if max_iters is not None and iterations is not None and iterations >= int(max_iters):
        return status.MAX_ITER_REACHED
    solve_ms = info.get("solve_time")
    if (
        time_limit_secs is not None
        and float(time_limit_secs) > 0
        and solve_ms is not None
        and float(solve_ms) / 1000.0 >= float(time_limit_secs)
    ):
        return status.TIME_LIMIT
    return None


def translate_settings(settings: dict) -> dict:
    """Map the harness aliases onto SCS's names, in place.

    ``max_iter`` becomes ``max_iters`` and ``eps`` supplies ``eps_abs`` and
    ``eps_rel`` (SCS's own names). An explicit native value always wins, an
    alias set to null counts as unset, and only settings actually inserted
    are reported. Any other unknown name is left for SCS, whose refusal the
    adapter raises as ``SolverConfigError`` naming it.
    """
    translated: dict = translate_tolerance_aliases(settings, ("eps_abs",), ("eps_rel",))
    max_iter = pop_max_iter(settings)
    if max_iter is not None and settings.get("max_iters") is None:
        settings["max_iters"] = max_iter
        translated["max_iters"] = max_iter
    if settings.get("max_iters") is not None:
        settings["max_iters"] = coerce_int_setting("max_iters", settings["max_iters"], minimum=1)
    return translated


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_integer(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _finite_at_least(bound: float, strict: bool):
    def test(value) -> bool:
        if not _is_number(value) or not math.isfinite(value):
            return False
        return value > bound if strict else value >= bound

    return test


# Ranges SCS enforces, checked before the solve so that a bad value is named
# as a configuration error instead of being raised from the solving call.
_SETTING_RANGES = {
    **{
        key: ("a finite number >= 0", _finite_at_least(0.0, strict=False))
        for key in ("eps_abs", "eps_rel", "eps_infeas", "time_limit_secs")
    },
    **{
        key: ("a finite number > 0", _finite_at_least(0.0, strict=True))
        for key in ("scale", "rho_x")
    },
    "alpha": ("in (0, 2)", lambda v: _is_number(v) and 0 < v < 2),
    "max_iters": ("an integer >= 1", lambda v: _is_integer(v) and v >= 1),
    "acceleration_lookback": ("an integer", _is_integer),
}


def _check_setting_values(settings: dict, translated: dict) -> None:
    """Raise SolverConfigError naming a setting whose value SCS would refuse."""
    for key, value in settings.items():
        if key in _SETTING_RANGES:
            description, test = _SETTING_RANGES[key]
            if not test(value):
                source = " (set from an alias)" if key in translated else ""
                raise SolverConfigError(
                    f"Invalid SCS setting {key!r}={value!r}{source}: must be {description}"
                )
    linear_solver = settings.get("linear_solver")
    if isinstance(linear_solver, str):
        try:
            import scs
        except IMPORT_ERRORS:  # pragma: no cover - solve() imported it
            return
        known = getattr(scs, "LinearSolver", None)
        if known is not None and linear_solver not in {member.value for member in known}:
            raise SolverConfigError(
                f"Unknown SCS linear_solver {linear_solver!r}; expected one of: "
                f"{', '.join(member.value for member in known)}"
            )


def _is_setting_error(exc: ValueError, settings: dict) -> bool:
    """Whether SCS's ValueError is its refusal of a setting value.

    SCS names the setting first ("max_iters must be positive", "'x' is not a
    valid LinearSolver"); its data errors name no setting ("ScsWork
    allocation error!", "m must be a positive integer").
    """
    text = str(exc)
    first = text.split(" ", 1)[0]
    return first in settings or "LinearSolver" in text


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
