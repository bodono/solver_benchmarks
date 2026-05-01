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
        mapped = _map_scs_status(info)
        trace = _read_csv_trace(settings.get("log_csv_filename"))
        kkt_dict = _compute_kkt(problem, mapped, raw, cone, inv_perm)
        # Gate objective_value on a solution-bearing status (including
        # OPTIMAL_INACCURATE). For infeasibility-certificate statuses
        # SCS still populates pobj, but reporting it as the objective
        # value is misleading because it's the certificate's normalizer,
        # not the optimal value.
        objective_present = mapped in status.SOLUTION_PRESENT or mapped == status.OPTIMAL_INACCURATE
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
        if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE} and y is not None:
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
    if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE} and y is not None and s is not None:
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


def _map_scs_status(info: dict) -> str:
    # Codes from scs/include/glbopts.h.
    status_val = info.get("status_val")
    if status_val == 1:
        return status.OPTIMAL
    if status_val == 2:
        return status.OPTIMAL_INACCURATE
    if status_val == -1:
        return status.DUAL_INFEASIBLE
    if status_val == -2:
        return status.PRIMAL_INFEASIBLE
    if status_val == -6:
        return status.DUAL_INFEASIBLE_INACCURATE
    if status_val == -7:
        return status.PRIMAL_INFEASIBLE_INACCURATE
    if status_val in {-3, -4, -5}:
        return status.SOLVER_ERROR
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
