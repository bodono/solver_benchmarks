"""qpo3 adapter.

qpo3 is a Rust interior-point (Mehrotra predictor-corrector) solver with Python
bindings for problems of the form ``min 1/2 x'Px + c'x`` s.t. ``Ax + s = b``,
``s`` in a product of zero and nonnegative cones, i.e. LPs and convex QPs. The
harness's QP form (``l <= Ax <= u``) is converted to that cone form the same way
as for QTQP. Settings follow Clarabel's names (``tol_feas``, ``tol_gap_abs``,
``tol_gap_rel``, ``tol_infeas_abs``, ``tol_infeas_rel``, ``max_iters``,
``time_limit``); the harness aliases ``eps`` / ``eps_abs`` / ``eps_rel`` are
mapped onto them and the mapping is recorded on ``info``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
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
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)

_STATUS = {
    "solved": status.OPTIMAL,
    "primal_infeasible": status.PRIMAL_INFEASIBLE,
    "dual_infeasible": status.DUAL_INFEASIBLE,
    "max_iterations": status.MAX_ITER_REACHED,
    "time_limit": status.TIME_LIMIT,
}
_INFO_FIELDS = (
    "iters", "solve_time", "pobj", "dobj", "pres", "dres", "comp", "presolve_rows_removed",
    "presolve_columns_removed", "presolve_time", "kkt_solves", "refinement_passes", "refinement_failures",
    "shift_escalations", "strict_fallbacks", "inertia_failures", "corrector_tries", "corrector_accepts",
    "linear_solver", "data_equilibration", "kkt_equilibration",
)


class Qpo3SolverAdapter(SolverAdapter):
    solver_name = "qpo3"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import qpo3  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import qpo3
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install qpo3 (pip install <path to the qpo3 checkout>) to use this adapter") from exc

        qp = problem.qp
        settings = settings_with_defaults(self.settings)
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)  # qpo3 has no thread setting
        translated = translate_settings(settings)
        if time_limit is not None:
            settings.setdefault("time_limit", float(time_limit))
        unknown = sorted(k for k in settings if k not in qpo3.Settings.__dataclass_fields__)
        if unknown:
            raise ValueError(f"qpo3.Settings does not accept {unknown}")
        qpo3_settings = qpo3.Settings(**settings)

        a, b, z = qp_to_nonnegative_cone(qp)
        a = sp.csc_matrix(a)
        p = sp.csc_matrix(qp["P"])
        c = np.asarray(qp["q"], dtype=float)
        cones = []
        if z:
            cones.append(qpo3.ZeroCone(int(z)))
        if a.shape[0] - z:
            cones.append(qpo3.NonnegativeCone(int(a.shape[0] - z)))
        qpo3_problem = qpo3.Problem(P=p if p.nnz else None, c=c, A=a, b=np.asarray(b, dtype=float), cones=cones)

        start = time.perf_counter()
        try:
            solution = qpo3.solve(qpo3_problem, settings=qpo3_settings)
        except RuntimeError as exc:  # numerical failures raise rather than returning a status
            elapsed = time.perf_counter() - start
            return SolverResult(status=status.SOLVER_ERROR, objective_value=None, iterations=None,
                                run_time_seconds=elapsed, info={"raw_status": "error", "error": str(exc)}, kkt=None)
        elapsed = time.perf_counter() - start

        raw_status = getattr(solution.status, "value", str(solution.status))
        mapped = _STATUS.get(str(raw_status), status.SOLVER_ERROR)
        info = solution.info
        result_info = {"raw_status": raw_status}
        for field in _INFO_FIELDS:
            value = getattr(info, field, None)
            if value is not None:
                result_info[field] = to_jsonable(getattr(value, "value", value))
        mark_threads_ignored(result_info, threads)
        if translated:
            result_info["settings_translated"] = translated
        cone_dict: dict = {}
        if z:
            cone_dict["z"] = int(z)
        if a.shape[0] - z:
            cone_dict["l"] = int(a.shape[0] - z)
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(getattr(info, "pobj", None)) if mapped in {status.OPTIMAL, status.OPTIMAL_INACCURATE} else None,
            iterations=_maybe_int(getattr(info, "iters", None)),
            run_time_seconds=elapsed,
            info=result_info,
            kkt=_compute_kkt(mapped, solution, p, c, a, b, cone_dict),
        )


def translate_settings(settings: dict) -> dict:
    """Map the harness aliases ``eps``, ``eps_abs``, ``eps_rel`` onto qpo3's names.

    ``eps_abs`` (or ``eps``) supplies ``tol_feas`` and ``tol_gap_abs``, ``eps_rel``
    (or ``eps``) supplies ``tol_gap_rel``; an explicit native value always wins,
    and only settings actually inserted are reported.
    """
    eps = settings.pop("eps", None)
    eps_abs = settings.pop("eps_abs", eps)
    eps_rel = settings.pop("eps_rel", eps)
    translated: dict = {}

    def put(key: str, value) -> None:
        if value is not None and key not in settings:
            settings[key] = float(value)
            translated[key] = float(value)

    put("tol_feas", eps_abs)
    put("tol_gap_abs", eps_abs)
    put("tol_gap_rel", eps_rel)
    if "max_iter" in settings and "max_iters" not in settings:
        settings["max_iters"] = int(settings.pop("max_iter"))
    return translated


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


def _maybe_float(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value):
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
