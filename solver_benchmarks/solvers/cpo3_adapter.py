"""cpo3 adapter.

cpo3 (https://github.com/cpo3-org/cpo3, ``pip install cpo3``) is a Rust
interior-point solver with Python bindings for problems of the form
``min 1/2 x'Px + c'x`` s.t. ``Ax + s = b``, ``s`` in a product of cones. The
0.0.1 release solves zero and nonnegative cones, i.e. LPs and convex QPs; it
already has second-order and PSD cone descriptors, but solving with them raises
``NotImplementedError``.

The harness's QP form (``l <= Ax <= u``) is converted to that cone form the
same way as for QTQP. Cone-form problems are passed through when they use only
zero and nonnegative cones and are skipped as unsupported otherwise.

Settings follow Clarabel's names (``tol_feas``, ``tol_gap_abs``,
``tol_gap_rel``, ``tol_infeas_abs``, ``tol_infeas_rel``, ``max_iters``,
``time_limit``); the harness aliases ``eps`` / ``eps_abs`` / ``eps_rel`` and
``max_iter`` are mapped onto them and the mapping is recorded on ``info``.
``linear_solver`` takes the backend name (``faer``, ``accelerate``,
``cholmod``, ``pardiso``); a backend this build or host cannot load is refused
before the solve instead of being reported as a numerical failure.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import (
    CONE,
    QP,
    ProblemData,
    cone_items_in_row_order,
    cone_row_count,
)
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
# Fields of cpo3.SolutionInfo recorded on ``info``. The first group is the
# 0.0.1 release; the second holds counters that only pre-release checkouts
# report, kept so runs against such a build still record them.
_INFO_FIELDS = (
    "iters",
    "solve_time",
    "pobj",
    "dobj",
    "pres",
    "dres",
    "comp",
    "warm_start_used",
    "cvxopt_init_used",
    "presolve_rows_removed",
    "presolve_columns_removed",
    "presolve_time",
    "factorizations",
    "primary_solves",
    "refinement_solves",
    "refinement_failures",
    "lsmr_iterations",
    "linear_solver",
    # Pre-release checkouts only.
    "kkt_solves",
    "refinement_passes",
    "shift_escalations",
    "strict_fallbacks",
    "inertia_failures",
    "data_equilibration",
    "kkt_equilibration",
)
# Cone keys cpo3 can solve. Any other key is refused unless it is empty.
_SUPPORTED_CONES = ("f", "z", "l")


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
                "Install with the cpo3 extra (pip install cpo3) to use cpo3"
            ) from exc

        if problem.kind == QP:
            p, c, a, b, cone_dict = _qp_data(problem.qp)
        elif problem.kind == CONE:
            native = _cone_data(problem.cone)
            if isinstance(native, SolverResult):
                return native
            p, c, a, b, cone_dict = native
        else:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": f"cpo3 does not support problem kind {problem.kind!r}"},
            )

        settings = settings_with_defaults(self.settings)
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        if threads is not None:
            settings["threads"] = threads
        translated = translate_settings(settings)
        if time_limit is not None:
            settings["time_limit"] = float(time_limit)
        unknown = sorted(k for k in settings if k not in cpo3.Settings.__dataclass_fields__)
        if unknown:
            raise ValueError(f"cpo3.Settings does not accept {unknown}")
        if "linear_solver" in settings:
            settings["linear_solver"] = resolve_linear_solver(settings["linear_solver"], cpo3)
        cpo3_settings = cpo3.Settings(**settings)

        cones: list[Any] = []
        if cone_dict.get("z"):
            cones.append(cpo3.ZeroCone(cone_dict["z"]))
        if cone_dict.get("l"):
            cones.append(cpo3.NonnegativeCone(cone_dict["l"]))

        # Time the problem construction with the solve, as the other adapters
        # time their solver's setup: cpo3.Problem copies and validates the data.
        start = time.perf_counter()
        try:
            cpo3_problem = cpo3.Problem(P=p if p.nnz else None, c=c, A=a, b=b, cones=cones)
            setup_elapsed = time.perf_counter() - start
            solution = cpo3.solve(cpo3_problem, settings=cpo3_settings)
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
        result_info: dict[str, Any] = {"raw_status": raw_status}
        for field in _INFO_FIELDS:
            value = getattr(info, field, None)
            if value is not None:
                result_info[field] = to_jsonable(getattr(value, "value", value))
        # cpo3 renamed the linear-solve counters before its release. Keep the
        # native fields and supply the earlier names used by existing reports
        # (which Clarabel also uses); an explicit earlier value, even zero, wins.
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
            setup_time_seconds=setup_elapsed,
            solve_time_seconds=_maybe_float(getattr(info, "solve_time", None)),
            info=result_info,
            kkt=_compute_kkt(mapped, solution, p, c, a, b, cone_dict),
        )


def translate_settings(settings: dict) -> dict:
    """Map the harness aliases ``eps``, ``eps_abs``, ``eps_rel`` onto cpo3's names.

    ``eps_abs`` (or ``eps``) supplies ``tol_feas`` and ``tol_gap_abs``, ``eps_rel``
    (or ``eps``) supplies ``tol_gap_rel``; an explicit native value always wins,
    and only settings actually inserted are reported. ``max_iter`` becomes
    ``max_iters`` unless that is given too.
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
    if "max_iter" in settings:
        max_iter = settings.pop("max_iter")
        if "max_iters" not in settings:
            settings["max_iters"] = int(max_iter)
    return translated


def resolve_linear_solver(value, cpo3_module):
    """Return the ``cpo3.LinearSolver`` named by ``value``.

    A name cpo3 does not know is a configuration error (``ValueError``). A
    backend this build or host cannot load raises ``SolverUnavailable`` here,
    before the solve: cpo3 would otherwise raise ``RuntimeError`` mid-solve and
    the row would be scored as a numerical failure of the solver.
    """
    linear_solver = cpo3_module.LinearSolver
    if isinstance(value, linear_solver):
        backend = value
    else:
        try:
            backend = linear_solver(str(value).strip().lower())
        except ValueError:
            known = ", ".join(member.value for member in linear_solver)
            raise ValueError(
                f"Unknown cpo3 linear_solver {value!r}; expected one of: {known}"
            ) from None
    if not backend.is_available():
        raise SolverUnavailable(
            f"cpo3 linear_solver {backend.value!r} is not available in this cpo3 "
            "build or on this host (see the cpo3 README on optional factorization "
            "backends; CPO3_CHOLMOD / CPO3_MKL_RT point it at the shared libraries)"
        )
    return backend


def _qp_data(qp: dict):
    a, b, z = qp_to_nonnegative_cone(qp)
    a = sp.csc_matrix(a)
    cone_dict: dict = {}
    if z:
        cone_dict["z"] = int(z)
    if a.shape[0] - z:
        cone_dict["l"] = int(a.shape[0] - z)
    return (
        sp.csc_matrix(qp["P"]),
        np.asarray(qp["q"], dtype=float),
        a,
        np.asarray(b, dtype=float),
        cone_dict,
    )


def _cone_data(cone_problem: dict):
    """Cone-form data for cpo3, or a skip result when a cone is unsupported.

    The harness stacks rows zero cone first, then nonnegative cone, the same
    order cpo3 takes its blocks in, so the rows pass through unchanged.
    """
    unsupported = sorted(
        key
        for key, value in cone_problem["cone"].items()
        if key not in _SUPPORTED_CONES and _cone_entry_has_rows(key, value)
    )
    if unsupported:
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={
                "reason": "cpo3 solves only zero and nonnegative cones; "
                f"this problem has {', '.join(repr(k) for k in unsupported)} cones"
            },
        )
    cone_dict: dict = {}
    for key, value in cone_items_in_row_order(cone_problem["cone"]):
        if key in ("f", "z") and int(value):
            cone_dict["z"] = cone_dict.get("z", 0) + int(value)
        elif key == "l" and int(value):
            cone_dict["l"] = int(value)
    a = sp.csc_matrix(cone_problem["A"])
    p = cone_problem.get("P")
    p = sp.csc_matrix((a.shape[1], a.shape[1])) if p is None else sp.csc_matrix(p)
    return (
        p,
        np.asarray(cone_problem["q"], dtype=float),
        a,
        np.asarray(cone_problem["b"], dtype=float),
        cone_dict,
    )


def _cone_entry_has_rows(key: str, value) -> bool:
    rows = cone_row_count({key: value})
    return rows is None or rows > 0


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
