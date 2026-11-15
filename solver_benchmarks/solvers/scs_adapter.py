"""SCS adapter."""

from __future__ import annotations

import csv
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
    SolverAdapter,
    SolverUnavailable,
    mark_threads_ignored,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
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
    except ModuleNotFoundError:
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

        start = time.perf_counter()
        raw = scs.solve(data, cone, **settings)
        elapsed = time.perf_counter() - start
        info = dict(raw.get("info", {}))
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
        objective_present = mapped in {status.OPTIMAL, status.OPTIMAL_INACCURATE}
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(info.get("pobj")) if objective_present else None,
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
    if problem.kind == QP:
        qp = problem.qp
        # Every status that may carry a usable point (including the
        # max_iters / time-limit iterates) gets solution residuals, so
        # ``kkt-verify --promote`` treats SCS like every other solver.
        if mapped_status in status.ANY_FEASIBLE and y is not None:
            y_qp = unbox_scs_dual(y, cone, inv_perm)
            return kkt.qp_residuals(
                qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x, y_qp
            )
        if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE} and y is not None:
            y_qp = unbox_scs_dual(y, cone, inv_perm)
            return kkt.qp_primal_infeasibility_cert(qp["A"], qp["l"], qp["u"], y_qp)
        if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
            return kkt.qp_dual_infeasibility_cert(
                qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x
            )
        return None
    cone_problem = problem.cone
    p = cone_problem.get("P")
    if p is None:
        p = sp.csc_matrix((sp.csc_matrix(cone_problem["A"]).shape[1],) * 2)
    if mapped_status in status.ANY_FEASIBLE and y is not None and s is not None:
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
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE} and y is not None:
        return kkt.cone_primal_infeasibility_cert(
            cone_problem["A"], cone_problem["b"], cone, y
        )
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
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
    ``eps_rel`` (SCS's own names). An explicit native value always wins,
    and only settings actually inserted are reported. Any other unknown
    name is left for SCS, which rejects it with a ``TypeError`` naming it.
    """
    translated: dict = {}
    max_iter = settings.pop("max_iter", None)
    if max_iter is not None and "max_iters" not in settings:
        settings["max_iters"] = int(max_iter)
        translated["max_iters"] = int(max_iter)
    eps = settings.pop("eps", None)
    if eps is not None:
        for key in ("eps_abs", "eps_rel"):
            if key not in settings:
                settings[key] = float(eps)
                translated[key] = float(eps)
    return translated


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
