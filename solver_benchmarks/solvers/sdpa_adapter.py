"""SDPA adapter via sdpa-python.

Settings are sdpa-python's own parameters (``maxIteration``, ``epsilonStar``,
``epsilonDash``, ``lowerBound``, ``upperBound``, ``numThreads``, ...) plus the
adapter's ``optimality_tolerance``, ``result_file`` and ``print``. The harness
aliases are translated: ``max_iter`` -> ``maxIteration``, ``eps_abs`` ->
``epsilonStar``, ``eps_rel`` -> ``epsilonDash`` (``eps`` supplies both) and
``threads`` -> ``numThreads``. Any other name raises ``SolverConfigError``:
sdpa-python silently ignores names it does not know, and resets a value of the
wrong type to its default, so the adapter also refuses a numeric parameter
that is not a number and an ``epsilonStar`` / ``epsilonDash`` /
``optimality_tolerance`` that is not positive, or a ``maxIteration`` below 1.
SDPA has no time limit; one is recorded as ignored.

Defaults that differ from sdpa-python's:

* ``lowerBound`` / ``upperBound`` are -1e30 / 1e30 instead of -1e5 / 1e5.
  SDPA stops with ``pUNBD`` / ``dUNBD`` as soon as an objective crosses
  them, so with the library defaults any feasible problem whose optimum is
  beyond about 1e5 in magnitude (SDPLIB qpG51, DIMACS torusg3-15) was
  reported as unbounded or infeasible. Genuinely infeasible or unbounded
  problems are still detected (``pINF_dFEAS`` / ``pFEAS_dINF``).
* ``numThreads`` is 1 unless ``threads`` (or ``numThreads``) is given;
  sdpa-python otherwise uses every core, which oversubscribes parallel runs
  and makes its timings incomparable with single-threaded solvers.
  ``threads: 0`` leaves it to sdpa-python (every core).

Status: ``pdOPT`` / ``pdFEAS`` are ``optimal`` when sdpa-python's primal and
dual errors and relative duality gap are within ``optimality_tolerance``
(default ``max(1e-6, epsilonStar, epsilonDash)``) and ``optimal_inaccurate``
otherwise. sdpa-python reports a primal or dual error of ``None`` when its
eigenvalue computation does not converge (common for PSD blocks of order 40
and up); that error is then unknown and left out, not treated as a failure.
A run that stops at ``maxIteration`` without SDPA declaring a solution or a
certificate is ``max_iter_reached``.

sdpa-python refuses a problem with no constraint rows ("Inconsistent Size");
that refusal is recorded as ``solver_error``, the solver failing that problem.
"""

from __future__ import annotations

import numbers
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, ProblemData, as_list, cone_items_in_row_order
from solver_benchmarks.core.result import SolverResult

from .base import (
    DEFAULT_THREADS,
    IMPORT_ERRORS,
    SolverAdapter,
    SolverConfigError,
    coerce_int_setting,
    finite_point,
    mark_time_limit_ignored,
    module_available,
    pop_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    translate_tolerance_aliases,
)
from .cone_layout import psd_triangle_to_full


class SDPASolverAdapter(SolverAdapter):
    solver_name = "sdpa"
    supported_problem_kinds = {CONE}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("sdpap")

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import sdpap
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "sdpap", "Install with the sdpa extra to use SDPA", exc
            ) from exc

        cone_problem = problem.cone
        p = cone_problem.get("P")
        if p is not None and sp.csc_matrix(p).nnz:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": "SDPA supports linear cone objectives, not quadratic objectives"},
            )

        try:
            prepared = _prepare_sdpap_problem(cone_problem, sdpap)
        except ValueError as exc:
            return SolverResult(status=status.SKIPPED_UNSUPPORTED, info={"reason": str(exc)})

        settings = settings_with_defaults(self.settings)
        optimality_tolerance = settings.pop("optimality_tolerance", None)
        ignore_markers: dict = {}
        option = _sdpa_options(settings, artifacts_dir, info=ignore_markers)
        if optimality_tolerance is None:
            optimality_tolerance = _default_optimality_tolerance(option)
        else:
            optimality_tolerance = check_optimality_tolerance(optimality_tolerance)
        threads = option.get("numThreads", 0)
        start = time.perf_counter()
        try:
            result = sdpap.solve(prepared.a, prepared.b, prepared.c, prepared.k, prepared.j, option)
        except ValueError as exc:
            # sdpa-python's size check, which a problem without constraint
            # rows fails (its empty b counts as one row).
            if not str(exc).startswith("Inconsistent Size"):
                raise
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start,
                info={
                    "raw_status": "error",
                    "error": f"sdpa-python refused the problem: {exc}",
                    "threads": threads,
                    **ignore_markers,
                },
            )
        elapsed = time.perf_counter() - start
        if result is None:
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=elapsed,
                info={"raw_status": "no_result", "threads": threads, **ignore_markers},
            )
        x_raw, y_raw, sdpap_info, time_info, sdpa_info = result
        x = _dense(x_raw)
        y_sdpap = _dense(y_raw)
        y = prepared.dual_to_original(y_sdpap)
        mapped = _map_sdpa_status(sdpap_info, sdpa_info, option, float(optimality_tolerance))
        kkt_dict = _compute_kkt(cone_problem, mapped, x, y)
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, sdpap_info.get("primalObj")),
            iterations=_maybe_int(sdpa_info.get("iteration")),
            run_time_seconds=elapsed,
            solve_time_seconds=_maybe_float(time_info.get("sdpa")),
            info={
                "raw_status": sdpap_info.get("phasevalue"),
                "phasevalue": sdpap_info.get("phasevalue"),
                "primalObj": sdpap_info.get("primalObj"),
                "dualObj": sdpap_info.get("dualObj"),
                "dualityGap": sdpap_info.get("dualityGap"),
                "primalError": sdpap_info.get("primalError"),
                "dualError": sdpap_info.get("dualError"),
                "time": time_info,
                "sdpa": sdpa_info,
                "optimality_tolerance": float(optimality_tolerance),
                "numThreads": option.get("numThreads"),
                "threads": threads,
                "lowerBound": option.get("lowerBound"),
                "upperBound": option.get("upperBound"),
                **ignore_markers,
            },
            kkt=kkt_dict,
        )


def _compute_kkt(cone_problem: dict, mapped: str, x: np.ndarray, y: np.ndarray):
    """Solution residuals for every status that may carry a point
    (max-iter iterates included, so kkt-verify --promote treats SDPA like
    every other solver); certificate checks for infeasibility statuses."""
    a = sp.csc_matrix(cone_problem["A"])
    b = np.asarray(cone_problem["b"], dtype=float)
    cone = cone_problem["cone"]
    p_mat = sp.csc_matrix((a.shape[1], a.shape[1]))
    if mapped in status.ANY_FEASIBLE:
        if not finite_point(x, y):
            return None
        return kkt.cone_residuals(p_mat, cone_problem["q"], a, b, cone, x, y, b - a @ x)
    if mapped in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        if not finite_point(y):
            return None
        return kkt.cone_primal_infeasibility_cert(a, b, cone, y)
    if mapped in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        if not finite_point(x):
            return None
        return kkt.cone_dual_infeasibility_cert(p_mat, cone_problem["q"], a, cone, x)
    return None


@dataclass
class _PreparedSDPAProblem:
    a: sp.csc_matrix
    b: np.ndarray
    c: np.ndarray
    k: object
    j: object
    dual_blocks: list[tuple[int, int, int, int, sp.csc_matrix | None]]

    def dual_to_original(self, y_sdpap: np.ndarray) -> np.ndarray:
        if not self.dual_blocks:
            return np.array([], dtype=float)
        original_size = max(start + size for start, size, _, _, _ in self.dual_blocks)
        y = np.zeros(original_size)
        for orig_start, orig_size, sdpa_start, sdpa_size, transform in self.dual_blocks:
            block = y_sdpap[sdpa_start : sdpa_start + sdpa_size]
            if transform is not None:
                block = transform.T @ block
            y[orig_start : orig_start + orig_size] = np.asarray(block, dtype=float).reshape(-1)
        return y


def _prepare_sdpap_problem(cone_problem: dict, sdpap) -> _PreparedSDPAProblem:
    raw_a = -sp.csc_matrix(cone_problem["A"])
    raw_b = -np.asarray(cone_problem["b"], dtype=float)
    c = np.asarray(cone_problem["q"], dtype=float)
    cone = dict(cone_problem["cone"])
    sdpa_row = 0
    a_parts: list[sp.csc_matrix] = []
    b_parts: list[np.ndarray] = []
    dual_blocks: list[tuple[int, int, int, int, sp.csc_matrix | None]] = []
    unsupported = set(cone) - {"f", "z", "l", "q", "s"}
    if unsupported:
        raise ValueError(f"SDPA adapter does not support cone keys: {', '.join(sorted(unsupported))}")
    if any(int(dim) for dim in as_list(cone.get("q", ()))):
        # sdpa-python ends the Python process on any second-order cone: it
        # prints "block struct :: line 93 in sdpa_block.cpp" and exits with
        # status 0, even for a single cone of dimension 3 handed straight to
        # sdpap.solve (seen with 0.2.3). The problem is refused here, so the
        # row says why instead of the worker disappearing. The layout of the
        # second-order blocks below is kept for a release that solves them.
        raise ValueError(
            "SDPA adapter does not run problems with second-order cones: "
            "sdpa-python ends the process on them"
        )
    blocks = _cone_blocks(cone)
    row = sum(block[2] for block in blocks)
    if row != raw_a.shape[0]:
        raise ValueError(f"SDPA cone dimensions ({row}) do not match A rows ({raw_a.shape[0]})")
    by_kind: dict[str, list[tuple[int, int, int]]] = {"f": [], "l": [], "q": [], "s": []}
    for kind, start, dim, order in blocks:
        by_kind[kind].append((start, dim, order))

    for kind in ("f", "l", "q"):
        for start, dim, _ in by_kind[kind]:
            _append_identity_block(raw_a, raw_b, start, dim, sdpa_row, a_parts, b_parts, dual_blocks)
            sdpa_row += dim
    for start, dim, psd_order in by_kind["s"]:
        transform = psd_triangle_to_full(psd_order)
        a_parts.append(transform @ raw_a[start : start + dim, :])
        b_parts.append(transform @ raw_b[start : start + dim])
        dual_blocks.append((start, dim, sdpa_row, psd_order * psd_order, transform))
        sdpa_row += psd_order * psd_order

    j_f = sum(dim for _, dim, _ in by_kind["f"])
    j_l = sum(dim for _, dim, _ in by_kind["l"])
    j_q = tuple(dim for _, dim, _ in by_kind["q"])
    j_s = tuple(order for _, _, order in by_kind["s"])
    a = sp.vstack(a_parts, format="csc") if a_parts else sp.csc_matrix((0, raw_a.shape[1]))
    b = np.concatenate([np.asarray(part).reshape(-1) for part in b_parts]) if b_parts else np.array([])
    k = sdpap.SymCone(f=int(raw_a.shape[1]))
    j = sdpap.SymCone(f=j_f, l=j_l, q=j_q, s=j_s)
    return _PreparedSDPAProblem(a=a, b=b, c=c, k=k, j=j, dual_blocks=dual_blocks)


def _cone_blocks(cone: dict) -> list[tuple[str, int, int, int]]:
    blocks = []
    row = 0
    # Rows follow the fixed layout, not the order of the dictionary's keys.
    for key, value in cone_items_in_row_order(cone):
        if key in {"f", "z"}:
            dim = int(value)
            if dim:
                blocks.append(("f", row, dim, dim))
            row += dim
        elif key == "l":
            dim = int(value)
            if dim:
                blocks.append(("l", row, dim, dim))
            row += dim
        elif key == "q":
            for dim in as_list(value):
                dim = int(dim)
                if dim:
                    blocks.append(("q", row, dim, dim))
                row += dim
        elif key == "s":
            for order in as_list(value):
                order = int(order)
                dim = order * (order + 1) // 2
                if dim:
                    blocks.append(("s", row, dim, order))
                row += dim
    return blocks


def _append_identity_block(
    raw_a,
    raw_b,
    row: int,
    dim: int,
    sdpa_row: int,
    a_parts: list,
    b_parts: list,
    dual_blocks: list,
) -> None:
    a_parts.append(raw_a[row : row + dim, :])
    b_parts.append(raw_b[row : row + dim])
    dual_blocks.append((row, dim, sdpa_row, dim, None))


# sdpa-python's parameter names (sdpap.param). Anything else is ignored by
# the library, so the adapter rejects it.
_SDPAP_PARAMETERS = frozenset(
    {
        "maxIteration",
        "epsilonStar",
        "lambdaStar",
        "omegaStar",
        "lowerBound",
        "upperBound",
        "betaStar",
        "betaBar",
        "gammaStar",
        "epsilonDash",
        "mpfPrecision",
        "isSymmetric",
        "isDimacs",
        "xPrint",
        "yPrint",
        "sPrint",
        "infPrint",
        "print",
        "resultFile",
        "sdpaResult",
        "numThreads",
        "frvMethod",
        "convMethod",
        "domainMethod",
        "rangeMethod",
        "rho",
        "zeroPoint",
    }
)
# sdpa-python resets a float parameter given as an int to its default (with a
# printed warning), so these are coerced.
_SDPAP_FLOATS = (
    "epsilonStar",
    "epsilonDash",
    "lambdaStar",
    "omegaStar",
    "lowerBound",
    "upperBound",
    "betaStar",
    "betaBar",
    "gammaStar",
    "rho",
    "zeroPoint",
)
_SDPAP_INTS = ("maxIteration", "numThreads", "mpfPrecision")
# The values sdpa-python (sdpap.param) accepts for its method parameters; it
# replaces any other value by its default with only a printed note.
_SDPAP_CHOICES = {
    "frvMethod": ("split", "elimination"),
    "convMethod": ("LMI", "EQ"),
    "domainMethod": ("none", "clique", "basis"),
    "rangeMethod": ("none", "clique", "decomp"),
}
# Wide enough that SDPA's objective-bound stop (pUNBD / dUNBD) never fires
# on a feasible, bounded problem; sdpa-python's defaults are -1e5 / 1e5.
SDPA_LOWER_BOUND = -1.0e30
SDPA_UPPER_BOUND = 1.0e30


def _sdpa_options(
    settings: dict,
    artifacts_dir: Path,
    *,
    info: dict | None = None,
) -> dict:
    """Translate adapter settings into the dict consumed by sdpap.solve.

    The optional ``info`` dict gets a ``time_limit_ignored`` marker when a
    time limit was requested (SDPA has none) and ``settings_translated``
    for the aliases that supplied a parameter.
    """
    settings = dict(settings)
    options: dict = {}
    translated: dict = {}
    verbose = bool(settings.pop("verbose", False))
    options["print"] = settings.pop("print", "display" if verbose else "no")
    if settings.pop("result_file", False):
        options["resultFile"] = str(artifacts_dir / "sdpa_result.txt")
    time_limit = pop_time_limit(settings)
    threads = pop_threads(settings)
    max_iter = pop_max_iter(settings)
    translated.update(translate_tolerance_aliases(settings, ("epsilonStar",), ("epsilonDash",)))
    unknown = sorted(str(key) for key in settings if key not in _SDPAP_PARAMETERS)
    if unknown:
        raise SolverConfigError(
            f"Unrecognized SDPA settings {unknown!r}; expected sdpa-python parameters "
            f"{sorted(_SDPAP_PARAMETERS)!r}, optimality_tolerance, result_file, or the "
            "aliases max_iter, eps, eps_abs, eps_rel, threads, time_limit"
        )
    options.update(settings)

    def put(key: str, value) -> None:
        if value is not None and key not in options:
            options[key] = value
            translated[key] = value

    put("maxIteration", max_iter)
    if threads is not None and threads >= 1:
        put("numThreads", threads)
    if threads != 0:
        # One thread unless configured; ``threads: 0`` leaves sdpa-python's
        # own default (every core).
        options.setdefault("numThreads", DEFAULT_THREADS)
    options.setdefault("lowerBound", SDPA_LOWER_BOUND)
    options.setdefault("upperBound", SDPA_UPPER_BOUND)
    for key in _SDPAP_FLOATS:
        if key in options:
            options[key] = _coerce_number(key, options[key], float, translated)
    for key in _SDPAP_INTS:
        if key in options:
            options[key] = _coerce_number(key, options[key], int, translated)
    for key, choices in _SDPAP_CHOICES.items():
        if key in options and options[key] not in choices:
            raise SolverConfigError(
                f"Invalid SDPA setting {key!r}={options[key]!r}: must be one of {list(choices)}"
            )
    # A negative numThreads ends the process inside sdpa-python (0 works).
    if options.get("numThreads", 0) < 0:
        raise SolverConfigError(
            f"Invalid SDPA setting 'numThreads'={options['numThreads']!r}: must be >= 0"
        )
    for key, minimum in (("epsilonStar", 0.0), ("epsilonDash", 0.0), ("maxIteration", 0)):
        if key in options and not options[key] > minimum:
            raise SolverConfigError(
                f"Invalid SDPA setting {key!r}={options[key]!r}"
                f"{_alias_note(key, translated)}: must be greater than {minimum}"
            )
    for key in translated:
        translated[key] = options[key]
    if info is not None:
        mark_time_limit_ignored(info, time_limit)
        if translated:
            info["settings_translated"] = translated
    return options


def _coerce_number(key: str, value, kind, translated: dict):
    """``kind(value)`` for a numeric sdpa-python parameter, else SolverConfigError.

    Integers follow the shared rule (``coerce_int_setting``: a numpy integer
    or ``30.0`` is 30); a float parameter takes any real number or a numeric
    string.
    """
    if kind is int:
        try:
            return coerce_int_setting(key, value)
        except SolverConfigError as exc:
            raise SolverConfigError(f"Invalid SDPA setting: {exc}{_alias_note(key, translated)}") from exc
    ok = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not ok:
        try:
            ok = kind is float and isinstance(value, str) and np.isfinite(float(value))
        except ValueError:
            ok = False
    if not ok:
        raise SolverConfigError(
            f"Invalid SDPA setting {key!r}={value!r}{_alias_note(key, translated)}: "
            f"must be {'an integer' if kind is int else 'a number'}"
        )
    return float(value)


def _alias_note(key: str, translated: dict) -> str:
    return " (set from an alias)" if key in translated else ""


def check_optimality_tolerance(value) -> float:
    """``optimality_tolerance`` as a positive float, else SolverConfigError."""
    try:
        tolerance = float(value)
    except (TypeError, ValueError):
        tolerance = float("nan")
    if isinstance(value, bool) or not (np.isfinite(tolerance) and tolerance > 0):
        raise SolverConfigError(
            f"Invalid SDPA setting 'optimality_tolerance'={value!r}: must be a positive number"
        )
    return tolerance


def _default_optimality_tolerance(options: dict) -> float:
    """The accuracy SDPA was asked for, but never tighter than 1e-6."""
    requested = [
        float(options[key]) for key in ("epsilonStar", "epsilonDash") if key in options
    ]
    return max([1.0e-6, *requested])


def _map_sdpa_status(sdpap_info, sdpa_info, options: dict, tol: float) -> str:
    phase = str(sdpap_info.get("phasevalue", sdpa_info.get("phasevalue", "")))
    iterations = _maybe_int(sdpa_info.get("iteration"))
    at_iteration_cap = iterations is not None and iterations >= int(
        options.get("maxIteration", 100)
    )
    if phase in {"pdOPT", "pdFEAS"}:
        if phase == "pdFEAS" and at_iteration_cap:
            # Feasible but not converged when the iteration cap hit.
            return status.MAX_ITER_REACHED
        # sdpa-python reports None for an error it could not compute (its
        # ARPACK eigenvalue call failed to converge): unknown, so left out.
        measures = [
            abs(float(value))
            for value in (
                sdpap_info.get("primalError"),
                sdpap_info.get("dualError"),
                sdpap_info.get("dualityGap"),
            )
            if value is not None
        ]
        if all(np.isfinite(value) and value <= tol for value in measures):
            return status.OPTIMAL
        return status.OPTIMAL_INACCURATE
    # CLP-form (sdpapinfo) phasevalue convention. By LP duality, primal unbounded
    # (pUNBD) implies dual infeasible, and dual unbounded (dUNBD) implies primal
    # infeasible. sdpap.py flips SDPA's internal labels into this CLP convention
    # (see GET_PRIMAL_STATUS_FROM_DUAL in the sdpap source). pUNBD / dUNBD come
    # from SDPA's objective bounds; with the adapter's default bounds they only
    # fire beyond 1e30, and the certificate check is recorded either way.
    if phase in {"pINF_dFEAS", "dUNBD"}:
        return status.PRIMAL_INFEASIBLE
    if phase in {"pFEAS_dINF", "pUNBD"}:
        return status.DUAL_INFEASIBLE
    if phase == "pdINF":
        return status.PRIMAL_OR_DUAL_INFEASIBLE
    if at_iteration_cap:
        return status.MAX_ITER_REACHED
    return status.SOLVER_ERROR


def _dense(value) -> np.ndarray:
    if sp.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=float).reshape(-1)


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
