"""cpo3 adapter.

cpo3 is a Rust interior-point (Mehrotra predictor-corrector) solver with Python
bindings for problems of the form ``min 1/2 x'Px + c'x`` s.t. ``Ax + s = b``,
``s`` in a product of zero, nonnegative and second-order cones, i.e. LPs, convex
QPs and SOCPs. The harness's QP form (``l <= Ax <= u``) is converted to zero and
nonnegative cones the same way as for QTQP. Cone problems are passed through
natively: the harness's ``f``/``z``, ``l`` and ``q`` blocks map onto cpo3's
zero, nonnegative and second-order cones, and a problem with any other cone kind
is reported as unsupported. Second-order cones need a cpo3 build that implements
them; a build that does not also reports the problem as unsupported rather than
failing. Settings follow Clarabel's names (``tol_feas``, ``tol_gap_abs``,
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
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult, to_jsonable
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

from .base import (
    SolverAdapter,
    SolverUnavailable,
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
    "iters",
    "solve_time",
    "pobj",
    "dobj",
    "pres",
    "dres",
    "comp",
    "presolve_rows_removed",
    "presolve_columns_removed",
    "presolve_time",
    "factorizations",
    "kkt_solves",
    "refinement_passes",
    "primary_solves",
    "refinement_solves",
    "refinement_failures",
    "shift_escalations",
    "strict_fallbacks",
    "inertia_failures",
    "linear_solver",
    "data_equilibration",
    "kkt_equilibration",
)


class Cpo3SolverAdapter(SolverAdapter):
    solver_name = "cpo3"
    supported_problem_kinds = {QP, CONE}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import cpo3  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import cpo3
        except ModuleNotFoundError as exc:
            raise SolverUnavailable(
                "Install cpo3 (pip install <path to the cpo3 checkout>) to use this adapter"
            ) from exc

        settings = settings_with_defaults(self.settings)
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        if threads is not None:
            settings["threads"] = threads
        translated = translate_settings(settings)
        if time_limit is not None:
            settings.setdefault("time_limit", float(time_limit))
        unknown = sorted(k for k in settings if k not in cpo3.Settings.__dataclass_fields__)
        if unknown:
            raise ValueError(f"cpo3.Settings does not accept {unknown}")
        cpo3_settings = cpo3.Settings(**settings)

        if problem.kind == QP:
            p, c, a, b, cones, cone_dict = _qp_data(problem.qp, cpo3)
        elif problem.kind == CONE:
            native = _cone_data(problem.cone, cpo3)
            if isinstance(native, SolverResult):
                return native
            p, c, a, b, cones, cone_dict = native
        else:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": f"cpo3 does not support problem kind {problem.kind!r}"},
            )
        cpo3_problem = cpo3.Problem(P=p if p.nnz else None, c=c, A=a, b=b, cones=cones)

        start = time.perf_counter()
        try:
            solution = cpo3.solve(cpo3_problem, settings=cpo3_settings)
        except NotImplementedError as exc:
            # The installed cpo3 build accepts the cone descriptor but has no
            # implementation for it (e.g. second-order cones before they landed).
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": f"This cpo3 build does not support the problem: {exc}"},
            )
        except RuntimeError as exc:  # numerical failures raise rather than returning a status
            elapsed = time.perf_counter() - start
            return SolverResult(
                status=status.SOLVER_ERROR,
                objective_value=None,
                iterations=None,
                run_time_seconds=elapsed,
                info={"raw_status": "error", "error": str(exc)},
                kkt=None,
            )
        elapsed = time.perf_counter() - start

        raw_status = getattr(solution.status, "value", str(solution.status))
        mapped = _STATUS.get(str(raw_status), status.SOLVER_ERROR)
        info = solution.info
        result_info = {"raw_status": raw_status}
        for field in _INFO_FIELDS:
            value = getattr(info, field, None)
            if value is not None:
                result_info[field] = to_jsonable(getattr(value, "value", value))
        # Newer cpo3 versions renamed the linear-solve counters. Keep the native
        # fields and supply the legacy names used by existing benchmark reports.
        # Preserve explicit legacy values, including zero, when both are present.
        for legacy, current in (
            ("kkt_solves", "primary_solves"),
            ("refinement_passes", "refinement_solves"),
        ):
            if legacy not in result_info and current in result_info:
                result_info[legacy] = result_info[current]
        if translated:
            result_info["settings_translated"] = translated
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(getattr(info, "pobj", None))
            if mapped in {status.OPTIMAL, status.OPTIMAL_INACCURATE}
            else None,
            iterations=_maybe_int(getattr(info, "iters", None)),
            run_time_seconds=elapsed,
            info=result_info,
            kkt=_compute_kkt(mapped, solution, p, c, a, b, cone_dict),
        )


def _qp_data(qp: dict, cpo3):
    """Convert the harness QP form to cpo3's zero / nonnegative cone form."""
    a, b, z = qp_to_nonnegative_cone(qp)
    a = sp.csc_matrix(a)
    p = sp.csc_matrix(qp["P"])
    c = np.asarray(qp["q"], dtype=float)
    cones = []
    cone_dict: dict = {}
    if z:
        cones.append(cpo3.ZeroCone(int(z)))
        cone_dict["z"] = int(z)
    if a.shape[0] - z:
        cones.append(cpo3.NonnegativeCone(int(a.shape[0] - z)))
        cone_dict["l"] = int(a.shape[0] - z)
    return p, c, a, np.asarray(b, dtype=float), cones, cone_dict


def _cone_data(cone_problem: dict, cpo3):
    """Pass a harness cone problem to cpo3, or report why it cannot be."""
    a = sp.csc_matrix(cone_problem["A"])
    b = np.asarray(cone_problem["b"], dtype=float)
    c = np.asarray(cone_problem["q"], dtype=float)
    p = cone_problem.get("P")
    p = sp.csc_matrix((a.shape[1], a.shape[1])) if p is None else sp.csc_matrix(p)

    parsed = _parse_cones(dict(cone_problem["cone"]), a.shape[0], cpo3)
    if isinstance(parsed, SolverResult):
        return parsed
    cones, cone_dict = parsed
    return p, c, a, b, cones, cone_dict


def _parse_cones(cone: dict, row_count: int, cpo3):
    """Map the harness cone blocks, in row order, onto cpo3 cone descriptors.

    Returns the cpo3 cones and the cone dictionary used for KKT checks. The
    latter keeps the harness's keys and order so that it describes the rows of
    ``A`` exactly; only empty blocks, which occupy no rows, are dropped.
    """
    cones = []
    cone_dict: dict = {}
    rows = 0
    for name, value in cone.items():
        if name in ("f", "z", "l"):
            dim = int(value)
            if dim:
                descriptor = cpo3.NonnegativeCone if name == "l" else cpo3.ZeroCone
                cones.append(descriptor(dim))
                cone_dict[name] = dim
            rows += dim
            continue
        if name == "q":
            dims = [int(dim) for dim in _as_list(value) if int(dim)]
            for dim in dims:
                # A one-dimensional second-order cone is the nonnegative ray;
                # cpo3 requires second-order cones of dimension at least two.
                cones.append(cpo3.SecondOrderCone(dim) if dim > 1 else cpo3.NonnegativeCone(1))
            if dims:
                cone_dict["q"] = dims
            rows += sum(dims)
            continue
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={"reason": f"cpo3 adapter does not support cone key {name!r}"},
        )

    if rows != row_count:
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={
                "reason": "cpo3 cone dimensions do not match A rows",
                "cone_rows": rows,
                "a_rows": row_count,
            },
        )
    return cones, cone_dict


def _as_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def translate_settings(settings: dict) -> dict:
    """Map the harness aliases ``eps``, ``eps_abs``, ``eps_rel`` onto cpo3's names.

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
    if mapped_status in status.ANY_FEASIBLE:
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
