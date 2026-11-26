"""Clarabel adapter.

QPs are converted to zero / nonnegative cone form; cone problems pass through
with their zero, nonnegative, second-order, PSD, exponential (``ep``) and
power (``p`` with ``0 < alpha < 1``) cones. Clarabel's exponential cone is the
canonical ``(x, y, z)`` with ``y exp(x / y) <= z`` and its power cone is
``x^alpha y^(1 - alpha) >= |z|``, the same orientation as SCS, so those rows
pass through unchanged. Dual exponential (``ed``) and dual power cones
(``p`` with negative alpha) are skipped as unsupported.

The harness aliases ``eps`` / ``eps_abs`` / ``eps_rel`` map onto ``tol_feas``,
``tol_gap_abs`` and ``tol_gap_rel`` (recorded in ``info.settings_translated``);
``max_iter`` and ``time_limit`` are Clarabel's own names. A negative or
non-finite tolerance is refused; so are ``tol_feas`` 0 and ``tol_gap_abs`` and
``tol_gap_rel`` both 0, which can never be met. One zero gap tolerance is a
valid absolute-only or relative-only test. ``threads`` sets ``max_threads``, one unless
configured (0: Clarabel's own choice).

Clarabel relabels a solve that stopped on ``max_iter`` or ``time_limit`` as
``AlmostSolved`` (or ``Almost*Infeasible``) when the last iterate meets its
reduced tolerances; such a stop is reported as ``max_iter_reached`` /
``time_limit`` like every other adapter's, with Clarabel's status kept in
``info.raw_status``. An ``Almost*`` status for any other stop (insufficient
progress, numerical trouble) stays ``*_inaccurate``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData, as_list, cone_items_in_row_order
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone
from solver_benchmarks.transforms.psd import (
    cone_row_perm_canonical_to_row_major,
    cone_vec_row_major_to_canonical,
)

from .base import (
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    SolverUnavailable,
    coerce_int_setting,
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
    threads_or_default,
    translate_tolerance_aliases,
)

# Clarabel's defaults (DefaultSettings, checked on clarabel 0.11.1): feasibility
# and absolute and relative gap tolerances.
CLARABEL_DEFAULT_TOLERANCES = {"tol_feas": 1e-8, "tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8}


class ClarabelSolverAdapter(SolverAdapter):
    solver_name = "clarabel"
    supported_problem_kinds = {QP, CONE}
    solver_modules = ("clarabel",)

    @classmethod
    def is_available(cls) -> bool:
        return module_available("clarabel")

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        translate_settings(settings)
        return loosest_tolerance(settings, CLARABEL_DEFAULT_TOLERANCES)

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import clarabel
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "clarabel", "Install with the clarabel extra to use Clarabel", exc
            ) from exc

        if problem.kind == QP:
            p, q, a, b, cones, cone_dict = _qp_data(problem.qp, clarabel)
        elif problem.kind == CONE:
            native = _cone_data(problem.cone, clarabel)
            if isinstance(native, SolverResult):
                return native
            p, q, a, b, cones, cone_dict = native
        else:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": f"Clarabel does not support problem kind {problem.kind!r}"},
            )

        normalized_settings = settings_with_defaults(self.settings)
        # Translate the cross-adapter aliases into Clarabel's native
        # knobs: time_limit_sec/time_limit -> time_limit (Clarabel
        # exposes it directly), threads -> max_threads/threads
        # (whichever the installed Clarabel build exposes).
        time_limit = pop_time_limit(normalized_settings)
        threads = pop_threads(normalized_settings)
        if time_limit is not None:
            normalized_settings["time_limit"] = float(time_limit)
        translated = translate_settings(normalized_settings)
        normalize_max_iter(normalized_settings, minimum=0)
        settings = clarabel.DefaultSettings()
        # Probe the real DefaultSettings instance: Clarabel renamed the knob
        # across versions and some builds expose neither. Injecting an
        # attribute that does not exist on the actual object would later
        # trip the "Invalid Clarabel settings" validation below.
        thread_attrs = [attr for attr in ("max_threads", "threads") if hasattr(settings, attr)]
        threads_used = None
        if thread_attrs:
            native = [normalized_settings[attr] for attr in thread_attrs if attr in normalized_settings]
            threads_used = (
                coerce_int_setting("max_threads", native[0], minimum=0)
                if native
                else threads_or_default(threads)
            )
            for attr in thread_attrs:
                normalized_settings[attr] = threads_used

        settings.verbose = bool(normalized_settings.get("verbose"))
        unknown_keys: list[str] = []
        for key, value in normalized_settings.items():
            if key == "verbose":
                continue
            if not hasattr(settings, key):
                unknown_keys.append(key)
                continue
            _check_setting_value(key, value, translated)
            try:
                setattr(settings, key, value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SolverConfigError(
                    f"Invalid Clarabel setting {key!r}={value!r}: {exc}"
                ) from exc
        if unknown_keys:
            raise SolverConfigError(
                f"Invalid Clarabel settings: {sorted(unknown_keys)!r}"
            )
        if getattr(settings, "tol_gap_abs", None) == 0 and getattr(settings, "tol_gap_rel", None) == 0:
            raise SolverConfigError(
                "Invalid Clarabel settings tol_gap_abs = tol_gap_rel = 0: the gap test "
                "(absolute or relative) could never pass"
            )

        start = time.perf_counter()
        try:
            solver = clarabel.DefaultSolver(p, q, sp.csc_matrix(a), b, cones, settings)
        except Exception as exc:
            # Clarabel checks enum-valued settings (direct_solve_method, ...)
            # when the solver is built and raises a bare Exception. A valid
            # backend whose library is not on this host ("Problem with mkl
            # solver (not available)") is a broken environment, not a typo.
            text = str(exc)
            if text.startswith("Bad settings") and "(not available)" in text:
                raise SolverUnavailable(f"Clarabel backend not available: {exc}") from exc
            if text.startswith("Bad settings"):
                raise SolverConfigError(f"Clarabel rejected a setting: {exc}") from exc
            raise
        solution = solver.solve()
        elapsed = time.perf_counter() - start
        mapped = _map_clarabel_status(
            str(solution.status),
            hit_iteration_limit=_at_least(getattr(solution, "iterations", None), settings.max_iter),
            hit_time_limit=_at_least(getattr(solution, "solve_time", None), settings.time_limit),
        )
        kkt_dict = _compute_kkt(
            mapped, solution, problem, p, q, a, b, cone_dict
        )
        info: dict = {
            "raw_status": str(solution.status),
            "r_prim": getattr(solution, "r_prim", None),
            "r_dual": getattr(solution, "r_dual", None),
            "solve_time": getattr(solution, "solve_time", None),
        }
        info.update(_linear_solver_counters(solver))
        info["threads"] = threads_used
        if not thread_attrs:
            mark_threads_ignored(info, threads)
        if translated:
            info["settings_translated"] = translated
        # Clarabel fills obj_val for every status; only a claimed solution
        # has an objective (the KKT record keeps an iterate's primal_obj).
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, getattr(solution, "obj_val", None)),
            iterations=_maybe_int(getattr(solution, "iterations", None)),
            run_time_seconds=elapsed,
            # Clarabel reports one time, its setup (equilibration, KKT
            # factorization set-up) included, and no separate setup time.
            solve_time_seconds=_maybe_float(getattr(solution, "solve_time", None)),
            info=info,
            kkt=kkt_dict,
        )


def translate_settings(settings: dict) -> dict:
    """Map the harness aliases ``eps``, ``eps_abs``, ``eps_rel`` onto Clarabel's
    names, in place, as the cpo3 and QTQP adapters do.

    ``eps_abs`` (or ``eps``) supplies ``tol_feas`` and ``tol_gap_abs``,
    ``eps_rel`` (or ``eps``) supplies ``tol_gap_rel``. An explicit native value
    always wins, and only settings actually inserted are reported.
    """
    return translate_tolerance_aliases(settings, ("tol_feas", "tol_gap_abs"), ("tol_gap_rel",))


def _check_setting_value(key: str, value, translated: dict) -> None:
    """Refuse a negative or non-finite tolerance, and ``tol_feas`` 0.

    ``tol_feas`` <= 0 can never be met (every problem ends ``AlmostSolved`` or
    worse). Any other tolerance may be 0: the gap test is absolute OR
    relative, so one zero gap tolerance (the shared ``eps_rel: 0``) is a
    valid configuration; both gap tolerances 0 are refused by the caller.
    """
    if not (key.startswith("tol_") or key.startswith("reduced_tol_")):
        return
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    source = " (set from eps_abs / eps_rel / eps)" if key in translated else ""
    if not (is_number and math.isfinite(value) and value >= 0):
        raise SolverConfigError(
            f"Invalid Clarabel setting {key!r}={value!r}{source}: must be a finite number >= 0"
        )
    if key == "tol_feas" and value == 0:
        raise SolverConfigError(
            f"Invalid Clarabel setting {key!r}={value!r}{source}: must be positive"
        )


def _at_least(value, limit) -> bool:
    try:
        return value is not None and limit is not None and float(value) >= float(limit)
    except (TypeError, ValueError):
        return False


def _linear_solver_counters(solver) -> dict:
    """Factorization / KKT-solve counters, when the installed Clarabel has them.

    ``DefaultSolver.get_info`` only exists from clarabel 0.11; older releases
    allowed by the extra would otherwise lose a finished solve to an
    AttributeError here.
    """
    get_info = getattr(solver, "get_info", None)
    if get_info is None:
        return {}
    linear_info = getattr(get_info(), "linsolver", None)
    counters: dict = {}
    for field in ("factorizations", "kkt_solves", "refinement_passes"):
        value = getattr(linear_info, field, None)
        if value is not None:
            counters[field] = value
    return counters


def _compute_kkt(mapped_status, solution, problem, p, q, a, b, cone_dict):
    x = getattr(solution, "x", None)
    y = getattr(solution, "z", None)
    s_slack = getattr(solution, "s", None)
    if x is None:
        return None

    # Clarabel returns y, s in row-major lower PSD ordering and the matrices
    # ``a``/``b`` passed in were already permuted to that ordering. The KKT
    # helpers expect canonical (col-major lower) layout, so for cone problems
    # with PSD blocks we rebuild the canonical ``a``/``b`` from the original
    # problem and permute Clarabel's ``y``/``s`` back.
    if problem.kind == CONE and "s" in cone_dict:
        a_canonical = sp.csc_matrix(problem.cone["A"])
        b_canonical = np.asarray(problem.cone["b"], dtype=float)
        a_for_kkt = a_canonical
        b_for_kkt = b_canonical
        y_for_kkt = (
            cone_vec_row_major_to_canonical(np.asarray(y, dtype=float), cone_dict)
            if y is not None
            else None
        )
        s_for_kkt = (
            cone_vec_row_major_to_canonical(np.asarray(s_slack, dtype=float), cone_dict)
            if s_slack is not None
            else None
        )
    else:
        a_for_kkt = a
        b_for_kkt = b
        y_for_kkt = np.asarray(y, dtype=float) if y is not None else None
        s_for_kkt = np.asarray(s_slack, dtype=float) if s_slack is not None else None

    if mapped_status in status.ANY_FEASIBLE:
        if not finite_point(x, y_for_kkt, s_for_kkt):
            return None
        return kkt.cone_residuals(
            p, q, a_for_kkt, b_for_kkt, cone_dict, x, y_for_kkt, s_for_kkt
        )
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        if not finite_point(y_for_kkt):
            return None
        return kkt.cone_primal_infeasibility_cert(
            a_for_kkt, b_for_kkt, cone_dict, y_for_kkt
        )
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        if not finite_point(x):
            return None
        return kkt.cone_dual_infeasibility_cert(p, q, a_for_kkt, cone_dict, x)
    return None


_CLARABEL_STATUS_MAP: dict[str, str] = {
    "Solved": status.OPTIMAL,
    "AlmostSolved": status.OPTIMAL_INACCURATE,
    "PrimalInfeasible": status.PRIMAL_INFEASIBLE,
    "AlmostPrimalInfeasible": status.PRIMAL_INFEASIBLE_INACCURATE,
    "DualInfeasible": status.DUAL_INFEASIBLE,
    "AlmostDualInfeasible": status.DUAL_INFEASIBLE_INACCURATE,
    "MaxIterations": status.MAX_ITER_REACHED,
    "MaxTime": status.TIME_LIMIT,
}


def _map_clarabel_status(
    raw_status: str, *, hit_iteration_limit: bool = False, hit_time_limit: bool = False
) -> str:
    """Translate a Clarabel ``solution.status`` string to canonical.

    An ``Almost*`` status from a solve that used up ``max_iter`` or
    ``time_limit`` is that limit (Clarabel relabels such a stop when the
    last iterate meets its reduced tolerances); otherwise the table applies.
    Exposed as a module-level helper so the per-status mapping table
    can be exercised directly in tests instead of via a real Clarabel
    solve, and so a regression in any non-optimal branch surfaces in
    a fast unit test rather than only when that branch is hit at
    runtime.
    """
    if raw_status.startswith("Almost"):
        if hit_iteration_limit:
            return status.MAX_ITER_REACHED
        if hit_time_limit:
            return status.TIME_LIMIT
    return _CLARABEL_STATUS_MAP.get(raw_status, status.SOLVER_ERROR)


def _maybe_int(value):
    return None if value is None else int(value)


def _maybe_float(value):
    return None if value is None else float(value)


def _qp_data(qp: dict, clarabel):
    a, b, z = qp_to_nonnegative_cone(qp)
    p = sp.csc_matrix(qp["P"])
    q = np.asarray(qp["q"], dtype=float)
    cones = []
    cone_dict: dict = {}
    if z:
        cones.append(clarabel.ZeroConeT(z))
        cone_dict["z"] = int(z)
    nonnegative = a.shape[0] - z
    if nonnegative:
        cones.append(clarabel.NonnegativeConeT(nonnegative))
        cone_dict["l"] = int(nonnegative)
    return p, q, sp.csc_matrix(a), b, cones, cone_dict


def _cone_data(cone_problem: dict, clarabel):
    a = sp.csc_matrix(cone_problem["A"])
    b = np.asarray(cone_problem["b"], dtype=float)
    q = np.asarray(cone_problem["q"], dtype=float)
    p = cone_problem.get("P")
    if p is None:
        p = sp.csc_matrix((a.shape[1], a.shape[1]))
    else:
        p = sp.csc_matrix(p)

    parsed = _parse_cones(dict(cone_problem["cone"]), a.shape[0], clarabel)
    if isinstance(parsed, SolverResult):
        return parsed
    cones, keep_rows, cone_dict = parsed
    if len(keep_rows) != a.shape[0]:
        a = a[keep_rows, :]
        b = b[keep_rows]
    if "s" in cone_dict:
        # Convert s-cone rows from canonical col-major lower PSD ordering to
        # the row-major lower ordering expected by Clarabel's PSDTriangleConeT.
        row_perm = cone_row_perm_canonical_to_row_major(cone_dict, a.shape[0])
        a = a[row_perm, :]
        b = b[row_perm]
    return p, q, sp.csc_matrix(a), b, cones, cone_dict


def _parse_cones(cone: dict, row_count: int, clarabel):
    cones = []
    keep_rows: list[int] = []
    cone_dict: dict = {}
    row = 0
    # Rows follow the fixed layout, not the order of the dictionary's keys.
    for name, value in cone_items_in_row_order(cone):
        if name in ("f", "z"):
            dim = int(value)
            if dim:
                cones.append(clarabel.ZeroConeT(dim))
                keep_rows.extend(range(row, row + dim))
                cone_dict["z"] = cone_dict.get("z", 0) + dim
            row += dim
            continue
        if name == "l":
            dim = int(value)
            if dim:
                cones.append(clarabel.NonnegativeConeT(dim))
                keep_rows.extend(range(row, row + dim))
                cone_dict["l"] = cone_dict.get("l", 0) + dim
            row += dim
            continue
        if name == "q":
            for dim in as_list(value):
                dim = int(dim)
                if dim:
                    cones.append(clarabel.SecondOrderConeT(dim))
                    keep_rows.extend(range(row, row + dim))
                    cone_dict.setdefault("q", []).append(dim)
                row += dim
            continue
        if name == "s":
            for dim in as_list(value):
                dim = int(dim)
                triangle_dim = dim * (dim + 1) // 2
                if dim:
                    cones.append(clarabel.PSDTriangleConeT(dim))
                    keep_rows.extend(range(row, row + triangle_dim))
                    cone_dict.setdefault("s", []).append(dim)
                row += triangle_dim
            continue
        if name == "ep":
            # Same (x, y, z), y exp(x / y) <= z convention: rows pass through.
            count = int(value)
            if count:
                cones.extend(clarabel.ExponentialConeT() for _ in range(count))
                keep_rows.extend(range(row, row + 3 * count))
                cone_dict["ep"] = cone_dict.get("ep", 0) + count
            row += 3 * count
            continue
        if name == "p":
            alphas = [float(alpha) for alpha in as_list(value)]
            dual = [alpha for alpha in alphas if not 0.0 < alpha < 1.0]
            if dual:
                # SCS encodes the dual power cone as a negative alpha;
                # Clarabel's PowerConeT is the primal cone only.
                return SolverResult(
                    status=status.SKIPPED_UNSUPPORTED,
                    info={
                        "reason": (
                            "Clarabel adapter supports power cones with 0 < alpha < 1 "
                            f"only (got {dual!r})"
                        )
                    },
                )
            cones.extend(clarabel.PowerConeT(alpha) for alpha in alphas)
            keep_rows.extend(range(row, row + 3 * len(alphas)))
            cone_dict.setdefault("p", []).extend(alphas)
            row += 3 * len(alphas)
            continue
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={"reason": f"Clarabel adapter does not support cone key {name!r}"},
        )

    if row != row_count:
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={
                "reason": "Clarabel cone dimensions do not match A rows",
                "cone_rows": row,
                "a_rows": row_count,
            },
        )
    return cones, keep_rows, cone_dict

