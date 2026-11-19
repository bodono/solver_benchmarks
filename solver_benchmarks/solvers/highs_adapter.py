"""HiGHS adapter.

``iterations`` is the count of the algorithm that finished the solve: the QP
solver's when it ran, else interior point's, else PDLP's, else simplex's (a QP
runs simplex for phase 1 first, and an interior-point LP solve may end with
simplex crossover iterations); every algorithm's count stays on ``info``.

An infeasibility claim carries HiGHS's ray (``getDualRay`` /
``getPrimalRay``), checked like every other adapter's certificate; the time
HiGHS takes to produce it counts in the run time when one is recorded (see
:func:`_highs_certificate`).
"""

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
    EPS_ALIASES,
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    ToleranceUnknown,
    coerce_int_setting,
    coerce_tolerance,
    finite_point,
    loosest_tolerance,
    module_available,
    pop_max_iter,
    pop_threads,
    pop_time_limit,
    refuse_settings,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    threads_or_default,
)
from .qp_split import free_row_mask, normalize_qp_bounds, qp_certificate_kkt

# HiGHS's feasibility and optimality tolerances (checked on highspy 1.15.1).
HIGHS_DEFAULT_TOLERANCES = {
    "primal_feasibility_tolerance": 1e-7,
    "dual_feasibility_tolerance": 1e-7,
    "primal_residual_tolerance": 1e-7,
    "dual_residual_tolerance": 1e-7,
    "optimality_tolerance": 1e-7,
}
# The algorithm-specific optimality tolerance each value of ``solver`` adds;
# simplex and the active-set QP solver have none of their own.
HIGHS_SOLVER_TOLERANCES = {
    "choose": {"ipm_optimality_tolerance": 1e-8},
    "ipm": {"ipm_optimality_tolerance": 1e-8},
    "ipx": {"ipm_optimality_tolerance": 1e-8},
    "hipo": {"ipm_optimality_tolerance": 1e-8},
    "pdlp": {"pdlp_optimality_tolerance": 1e-7},
    "hipdlp": {"pdlp_optimality_tolerance": 1e-7},
    "simplex": {},
    "qpasm": {},
}
# "If changed from its default value, this tolerance is used for all
# feasibility and optimality (KKT) measures" (HiGHS's option description).
HIGHS_DEFAULT_KKT_TOLERANCE = 1e-7


class HighsSolverAdapter(SolverAdapter):
    solver_name = "highs"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("highspy")

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        refuse_settings(settings, EPS_ALIASES, "HiGHS", "set its own tolerances instead")
        kkt_tolerance = settings.get("kkt_tolerance")
        if kkt_tolerance is not None and coerce_tolerance("kkt_tolerance", kkt_tolerance) != HIGHS_DEFAULT_KKT_TOLERANCE:
            return loosest_tolerance(settings, {"kkt_tolerance": HIGHS_DEFAULT_KKT_TOLERANCE})
        solver = str(settings.get("solver") or "choose").strip().lower()
        if solver not in HIGHS_SOLVER_TOLERANCES:
            raise ToleranceUnknown(f"HiGHS solver {solver!r} is not one whose tolerances this adapter knows")
        return loosest_tolerance(settings, {**HIGHS_DEFAULT_TOLERANCES, **HIGHS_SOLVER_TOLERANCES[solver]})

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import highspy
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "highspy", "Install with the highs extra to use HiGHS", exc
            ) from exc

        qp = problem.qp
        p = sp.csc_matrix(qp["P"])
        q = np.asarray(qp["q"], dtype=float)
        a = sp.csr_matrix(qp["A"])
        l, u = normalize_qp_bounds(qp["l"], qp["u"])
        n = int(qp.get("n", int(q.shape[0])))
        m = int(qp.get("m", a.shape[0]))

        solver = highspy.Highs()
        settings = settings_with_defaults(self.settings)
        threads, translated = _configure_highs(solver, settings)
        reset_scheduler = getattr(type(solver), "resetGlobalScheduler", None)
        if reset_scheduler is not None:
            # HiGHS keeps one thread pool per process, sized by the first
            # run; a later run asking for another size fails, and threads 0
            # would reuse whatever pool an earlier run left. The runner
            # starts a fresh worker per solve, so this only matters for
            # in-process use, and resetting a pool that does not exist yet
            # costs nothing.
            reset_scheduler(True)
        inf = solver.getInfinity()

        start = time.perf_counter()
        build = {}
        build["addCols"] = solver.addCols(
            n,
            q,
            np.full(n, -inf),
            np.full(n, inf),
            0,
            np.zeros(n + 1, dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=float),
        )
        row_lower = np.where(np.isneginf(l), -inf, l)
        row_upper = np.where(np.isposinf(u), inf, u)
        # Free rows can trigger heap corruption in HiPO 1.15.1.
        keep = ~free_row_mask(l, u)
        a_highs = a[keep]
        build["addRows"] = solver.addRows(
            a_highs.shape[0],
            row_lower[keep],
            row_upper[keep],
            int(a_highs.nnz),
            a_highs.indptr.astype(np.int32),
            a_highs.indices.astype(np.int32),
            a_highs.data.astype(float),
        )
        if p.nnz:
            lower = sp.tril(p, format="csc")
            build["passHessian"] = solver.passHessian(
                n,
                int(lower.nnz),
                int(highspy.HessianFormat.kTriangular),
                lower.indptr.astype(np.int32),
                lower.indices.astype(np.int32),
                lower.data.astype(float),
            )
        rejected = [name for name, result in build.items() if _is_error(result)]
        if rejected:
            # HiGHS refuses data it cannot take (a coefficient at or above
            # large_matrix_value, an infinite entry, a wrong-signed infinite
            # bound) by returning an error and leaving the model without it.
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start,
                info={
                    "raw_status": "error",
                    "error": f"HiGHS rejected the model in {', '.join(rejected)} (see its log)",
                    "threads": threads,
                    **({"settings_translated": translated} if translated else {}),
                },
            )
        raw_status = solver.run()
        elapsed = time.perf_counter() - start

        model_status = solver.getModelStatus()
        mapped = _map_highs_status(model_status, highspy)
        # Read before the certificate: getDualRay may re-solve the LP.
        info = solver.getInfo()
        solution = solver.getSolution()
        certificate_start = time.perf_counter()
        farkas, ray, unavailable = _highs_certificate(solver, mapped, n, keep)
        certificate_seconds = time.perf_counter() - certificate_start
        y = None
        if solution.dual_valid:
            y = np.zeros(m)
            y[keep] = -np.asarray(solution.row_dual, dtype=float)
        x = np.asarray(solution.col_value, dtype=float) if solution.value_valid else None
        kkt_dict = None
        if mapped in status.ANY_FEASIBLE:
            if finite_point(x, y):
                kkt_dict = kkt.qp_residuals(p, q, a, l, u, x, y)
        else:
            kkt_dict = qp_certificate_kkt(mapped, p, q, a, l, u, farkas=farkas, ray=ray)
        certificate_info: dict = {}
        if mapped in _RAY_GETTERS:
            certificate_info["certificate_seconds"] = certificate_seconds
            if kkt_dict is not None:
                # The claim is checked on this certificate, so the time HiGHS
                # took to produce it counts. An attempt that gave none is
                # left out: the claim is then unverified whatever it cost,
                # and counting it could turn a claim made in time into a
                # time_limit row.
                elapsed += certificate_seconds
            else:
                certificate_info["certificate_unavailable"] = unavailable
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, info.objective_function_value),
            iterations=_highs_iterations(info),
            run_time_seconds=elapsed,
            info={
                "raw_status": str(model_status),
                "api_status": str(raw_status),
                "simplex_iteration_count": getattr(info, "simplex_iteration_count", None),
                "ipm_iteration_count": getattr(info, "ipm_iteration_count", None),
                "qp_iteration_count": getattr(info, "qp_iteration_count", None),
                "pdlp_iteration_count": getattr(info, "pdlp_iteration_count", None),
                "primal_solution_status": getattr(info, "primal_solution_status", None),
                "dual_solution_status": getattr(info, "dual_solution_status", None),
                "objective_function_value": getattr(info, "objective_function_value", None),
                "version": solver.version(),
                "threads": threads,
                **certificate_info,
                **({"settings_translated": translated} if translated else {}),
            },
            kkt=kkt_dict,
        )


# The ray each infeasibility claim is certified by: a dual ray (a Farkas
# certificate over the rows) for an infeasible LP, a primal ray (an unbounded
# direction) for an unbounded one.
_RAY_GETTERS = {
    status.PRIMAL_INFEASIBLE: ("getDualRay", "dual ray"),
    status.DUAL_INFEASIBLE: ("getPrimalRay", "primal ray"),
}


def _highs_certificate(solver, mapped: str, n: int, keep):
    """HiGHS's certificate for an infeasibility claim, as ``(farkas, ray, unavailable)``.

    ``farkas`` is the dual ray over the original rows (zero on the free rows
    the adapter leaves out) in the harness's dual sign: HiGHS's dual ray has
    the sign of its row duals, which the adapter negates for solutions too
    (checked on highspy 1.15.1: ``x1 + x2 <= 1, x1 >= 1, x2 >= 1`` gives
    ``[-1, 1, 1]``). ``ray`` is the primal ray, which needs no conversion.
    ``unavailable`` says why there is none, else None.

    No setting is needed. HiGHS keeps the ray of the simplex solve that
    proved the claim; when that solve was of the presolved LP (the ray is not
    carried through postsolve), ``getDualRay`` / ``getPrimalRay`` solve the
    original LP again without presolve to find one, which can take as long
    as a solve (NETLIB gosh: about 0.01 s to the claim, 3 to 7 s to the
    ray). The caller records that time as ``info.certificate_seconds`` and
    counts it in the run time when a certificate is recorded; HiGHS's
    ``time_limit`` covers it, since it bounds all runs of one ``Highs``
    object. An unbounded QP gets no ray, and nor do NETLIB cplex2, klein1 and
    qual (the re-solve ends with status Unknown); with highspy 1.15.1 neither
    does a claim made by interior point or PDLP (``solver: ipm`` / ``pdlp``),
    whose re-solve uses the same algorithm, while 1.16.0 gives one. Those
    claims have no certificate to check. The claim is already made when the
    ray is asked for, so any failure to get one (an error HiGHS raises, a ray
    of the wrong size or not finite) leaves the claim without a certificate
    rather than losing the row.
    """
    if mapped not in _RAY_GETTERS:
        return None, None, None
    getter, kind = _RAY_GETTERS[mapped]
    try:
        _, has_ray, values = getattr(solver, getter)()
        if not has_ray:
            return None, None, f"HiGHS returned no {kind}"
        values = np.asarray(values, dtype=float).reshape(-1)
        size = n if mapped == status.DUAL_INFEASIBLE else int(np.count_nonzero(keep))
        if values.shape != (size,):
            return None, None, f"HiGHS's {kind} has {values.size} entries for {size}"
        if not np.all(np.isfinite(values)):
            return None, None, f"HiGHS's {kind} is not finite"
        if mapped == status.DUAL_INFEASIBLE:
            return None, values, None
        farkas = np.zeros(len(keep))
        farkas[keep] = -values
        return farkas, None, None
    except Exception as exc:
        # A highspy without the getter, or an error from the re-solve (a
        # pybind11 RuntimeError, say).
        return None, None, f"{getter} failed: {type(exc).__name__}: {exc}"


# HiGHS caps each algorithm separately; ``max_iter`` sets all of them.
_ITERATION_LIMIT_OPTIONS = (
    "simplex_iteration_limit",
    "ipm_iteration_limit",
    "qp_iteration_limit",
    "pdlp_iteration_limit",
)


def _configure_highs(solver, settings: dict) -> tuple[int, dict]:
    """Apply ``settings`` to a ``highspy.Highs`` instance.

    highspy reports an unknown option or a bad value by returning
    ``HighsStatus.kError`` rather than raising, so every return value is
    checked and a rejected setting raises SolverConfigError naming it.
    ``threads`` is one unless configured (0: HiGHS's own choice). Returns the
    thread count and the iteration limits ``max_iter`` set, for ``info``.
    """
    refuse_settings(
        settings,
        EPS_ALIASES,
        "HiGHS",
        "it has no absolute/relative tolerance pair; set its own tolerances, e.g. "
        "primal_feasibility_tolerance, dual_feasibility_tolerance, "
        "ipm_optimality_tolerance",
    )
    if "verbose" in settings:
        _set_option(solver, "output_flag", bool(settings.pop("verbose")))
    time_limit = pop_time_limit(settings)
    if time_limit is not None:
        _set_option(solver, "time_limit", float(time_limit))
    threads = threads_or_default(pop_threads(settings))
    _set_option(solver, "threads", threads)
    translated: dict = {}
    # The native limits follow the shared integer rule (highspy refuses an
    # integer-valued float such as YAML's 1e3 and parses a string).
    for option in _ITERATION_LIMIT_OPTIONS:
        if settings.get(option) is not None:
            settings[option] = coerce_int_setting(option, settings[option], minimum=0)
    max_iter = pop_max_iter(settings, minimum=0)
    if max_iter is not None:
        for option in _ITERATION_LIMIT_OPTIONS:
            if option in settings:
                continue  # a native limit given alongside wins
            try:
                result = solver.setOptionValue(option, max_iter)
            except Exception:
                # Older highspy builds lack some of these options; the
                # available ones still take effect.
                continue
            if not _is_error(result):
                translated[option] = max_iter
        if not translated and not any(option in settings for option in _ITERATION_LIMIT_OPTIONS):
            raise SolverConfigError(f"Invalid HiGHS setting 'max_iter'={max_iter!r}")
    for key, value in settings.items():
        _set_option(solver, key, value)
    return threads, translated


def _set_option(solver, key: str, value) -> None:
    hint = ""
    if isinstance(value, bool) and _is_string_option(solver, key):
        # YAML 1.1 reads unquoted on/off (and yes/no) as booleans, but
        # HiGHS's on/off options (presolve, run_crossover, parallel, ranging)
        # are strings that refuse a bool.
        value = "on" if value else "off"
        hint = " (an unquoted on/off in YAML is read as a boolean)"
    try:
        result = solver.setOptionValue(key, value)
    except TypeError as exc:  # a value of a type the binding cannot take
        raise SolverConfigError(f"Invalid HiGHS setting {key!r}={value!r}: {exc}") from exc
    if _is_error(result):
        raise SolverConfigError(
            f"Invalid HiGHS setting {key!r}={value!r}{hint}: unknown option or value "
            "rejected by HiGHS (see its log)"
        )


def _is_string_option(solver, key: str) -> bool:
    get_type = getattr(solver, "getOptionType", None)
    if get_type is None:
        return False
    try:
        result = get_type(key)
    except Exception:
        return False
    option_type = result[-1] if isinstance(result, tuple) else result
    return getattr(option_type, "name", None) == "kString"


def _is_error(result) -> bool:
    return getattr(result, "name", None) == "kError"


def _map_highs_status(model_status, highspy) -> str:
    model = highspy.HighsModelStatus
    mapping = {
        model.kOptimal: status.OPTIMAL,
        model.kInfeasible: status.PRIMAL_INFEASIBLE,
        model.kUnbounded: status.DUAL_INFEASIBLE,
        model.kUnboundedOrInfeasible: status.PRIMAL_OR_DUAL_INFEASIBLE,
        model.kIterationLimit: status.MAX_ITER_REACHED,
        model.kTimeLimit: status.TIME_LIMIT,
    }
    # Optional model statuses depending on highspy build. Use getattr so
    # the mapping works across versions.
    for attr_name, canonical in (
        ("kInterrupt", status.TIME_LIMIT),
        ("kModelEmpty", status.SOLVER_ERROR),
        ("kPresolveError", status.SOLVER_ERROR),
        ("kSolveError", status.SOLVER_ERROR),
        ("kPostsolveError", status.SOLVER_ERROR),
        ("kMemoryLimit", status.SOLVER_ERROR),
        ("kSolutionLimit", status.MAX_ITER_REACHED),
    ):
        attr = getattr(model, attr_name, None)
        if attr is not None:
            mapping[attr] = canonical
    return mapping.get(model_status, status.SOLVER_ERROR)


def _highs_iterations(info) -> int | None:
    """Iterations of the algorithm that finished the solve.

    The first algorithm with a positive count, in the order QP, interior
    point, PDLP, simplex: a QP's phase-1 simplex iterations and an LP's
    crossover iterations come before or after the algorithm that did the
    solve. HiGHS keeps -1 for algorithms that did not run (and for all of
    them when the solve errored before iterating); 0 when every algorithm
    that ran took no iteration (e.g. presolve solved the problem).
    """
    values = [
        getattr(info, "qp_iteration_count", None),
        getattr(info, "ipm_iteration_count", None),
        getattr(info, "pdlp_iteration_count", None),
        getattr(info, "simplex_iteration_count", None),
    ]
    counts = [int(value) for value in values if value is not None and value >= 0]
    positive = [count for count in counts if count > 0]
    if positive:
        return positive[0]
    return 0 if counts else None
