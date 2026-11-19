"""CVXOPT adapter.

CVXOPT is an interior-point solver for convex problems with linear,
second-order, and positive-semidefinite cone constraints, *and*
quadratic objectives. Problems with a linear objective (``P`` without
nonzeros: LPs, SOCPs, SDPs) go through ``cvxopt.solvers.conelp``::

    minimize    q' x
    subject to  G x + s = h,   s in K = R^l_+ × Q^q_1 × ... × S^s_1 × ...
                A x = b

which returns ``primal infeasible`` / ``dual infeasible`` with normalized
certificates (``h'z + b'y = -1`` or ``q'x = -1``). Its infeasibility test
divides a residual by minus the current objective, so at the starting point
(iteration 0, a least-squares point) it only compares that point's objective
with ``1 / feastol``: NETLIB agg, agg2, agg3 and grow7 (optima of 1e7 to
5e7 in magnitude) were "dual infeasible" at iteration 0. Such a claim is
not evidence, so the adapter discards it and solves the problem with
``coneqp``. A later claim can be false too on a badly scaled feasible
problem (NETLIB scagr25 at iteration 4, gfrd-pnc at 1, nesm at 17), and
its scaled residual can pass the certificate check, so the adapter also
solves such a problem with ``coneqp``, which cannot certify infeasibility:
the claim stands unless ``coneqp`` finds a solution, in which case that
solution is reported. Either way ``info["conelp_discarded_claim"]`` and
``info["coneqp_cross_check"]`` say what happened, and the time of both
calls is reported, so a genuinely infeasible problem costs two solves.
NETLIB shell stays a false claim (``coneqp`` fails on it, as before). With
second-order cones ``conelp`` would default to the dense ``qr`` KKT solver;
the adapter passes the sparse ``chol`` that ``coneqp`` uses (``qr`` took
DIMACS sched_50_50_scaled from 80 s to an error after 490 s) unless
``kktsolver`` is set. PSD-only problems keep ``qr``: with ``chol``, ``conelp``
fails on SDPLIB control1 and hinf9, which ``qr`` solves, though ``qr`` is
slower and stops at ``maxiters`` on hinf1/6/14 and qap6/7.

Problems with a quadratic objective go through ``cvxopt.solvers.coneqp``
(the same form plus ``(1/2) x' P x``), which only ever reports ``optimal``
or ``unknown`` and cannot certify infeasibility. A problem with no cone rows at all
(only equalities) also goes to ``coneqp``, which then solves the KKT
system ``[[P, A'], [A, 0]]`` once and reports ``optimal`` whatever the
residual. CVXOPT's default ``chol2`` KKT solver returns a wrong point there
when ``P`` is singular (Maros-Meszaros HS52, GENHS28), so the adapter uses
the ``ldl`` KKT solver when ``n + p`` is at most
``CVXOPT_DENSE_LDL_MAX_DIM`` (2000). ``ldl`` factors that KKT matrix as a
dense ``(n + p) x (n + p)`` array, which would need 7.3 GB for AUG2DC
(``n + p = 30200``), so larger systems keep ``chol2``; a wrong point from it
is caught by the residual check below (``solver_error``).

CVXOPT's ``unknown`` status covers both reaching ``maxiters`` and stopping
early on a numerical failure (e.g. a singular KKT matrix). It maps to
``MAX_ITER_REACHED`` only when the iteration count reached the effective
``maxiters``; an early stop maps to ``OPTIMAL_INACCURATE`` when the gap
and residuals are small and to ``SOLVER_ERROR`` otherwise.

CVXOPT's PSD-cone vec layout differs from the canonical layout used
elsewhere in this codebase: it uses BLAS unpacked 'L' storage —
column-major over the full ``n*n`` square matrix with the strict
upper triangle ignored. The canonical layout is column-major lower
with ``n*(n+1)/2`` entries and ``√2`` scaling on off-diagonals. We
convert via a sparse permutation/scaling matrix.

CVXOPT does not support exponential cones; CONE problems with the
``e``/``ep`` key fall to ``SKIPPED_UNSUPPORTED``.

Settings are CVXOPT's options (``maxiters``, ``abstol``, ``reltol``,
``feastol``, ``refinement``, ``kktreg``, ``show_progress``, ``debug``,
``use_correction``) plus ``kktsolver``, which is passed to the solver
call. The harness aliases ``max_iter`` / ``max_iters`` (-> ``maxiters``),
``eps`` / ``eps_abs`` (-> ``abstol`` and ``feastol``) and ``eps_rel``
(-> ``reltol``) are translated (recorded in ``info.settings_translated``);
any other name raises ``SolverConfigError``, since CVXOPT would silently
ignore it. CVXOPT checks its option values (and ``kktsolver``) in the same
call that solves, raising the ``ValueError`` it also raises on numerical
trouble, so the adapter checks them first and raises ``SolverConfigError``
for a bad value; a numerical ``ValueError`` / ``ArithmeticError`` from the
solve is ``solver_error`` with the message on ``info["error"]``. Options are
global state in CVXOPT (``cvxopt.solvers.options``); the
adapter snapshots and restores them around each solve so sequential
solves in the same process do not leak knobs into each other.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData, as_list
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

from .base import (
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    coerce_int_setting,
    finite_point,
    loosest_tolerance,
    mark_threads_ignored,
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
from .cone_layout import psd_triangle_to_full, qp_dual_in_native_order

# Largest equality-only KKT system (n + p rows) the adapter hands to CVXOPT's
# ``ldl`` KKT solver, which factors it as a dense matrix (32 MB at this size).
# Larger systems keep CVXOPT's default sparse ``chol2``.
CVXOPT_DENSE_LDL_MAX_DIM = 2000


# CVXOPT's documented defaults: absolute and relative gap, and feasibility.
# The gap tests are alternatives (either suffices), so all three bound what
# a solution is asked to meet.
CVXOPT_DEFAULT_TOLERANCES = {"abstol": 1e-7, "reltol": 1e-6, "feastol": 1e-7}


class CVXOPTSolverAdapter(SolverAdapter):
    solver_name = "cvxopt"
    supported_problem_kinds = {QP, CONE}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("cvxopt")

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        # The same preparation as solve(): what it takes off before the
        # options are resolved.
        settings = dict(settings or {})
        pop_time_limit(settings)
        pop_threads(settings)
        settings.pop("verbose", None)
        settings.pop("kktsolver", None)
        options, _ = _resolve_options(settings)
        return loosest_tolerance(options, CVXOPT_DEFAULT_TOLERANCES)

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import cvxopt
            import cvxopt.solvers as cs
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "cvxopt", "Install with the cvxopt extra to use CVXOPT", exc
            ) from exc

        settings = settings_with_defaults(self.settings)
        # CVXOPT does not expose a wall-clock time limit; record the
        # request as ignored. Threads are also recorded as ignored —
        # CVXOPT calls into LAPACK/BLAS, so the active thread count is
        # set by the underlying BLAS, not the adapter.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        verbose = bool(settings.pop("verbose", False))
        kktsolver = settings.pop("kktsolver", None)
        options, translated = _resolve_options(settings)

        if problem.kind == QP:
            data, dims, cone_dict = _qp_to_cvxopt(problem.qp, cvxopt)
        else:
            built = _cone_to_cvxopt(problem.cone, cvxopt)
            if isinstance(built, SolverResult):
                return built
            data, dims, cone_dict = built

        has_cone_rows = data["G"].size[0] > 0
        # conelp certifies infeasibility; coneqp cannot. conelp needs at
        # least one cone row ([G; A] must have full column rank).
        entry_point = "conelp" if has_cone_rows and not _has_nonzeros(data["P"]) else "coneqp"
        if kktsolver is None and not has_cone_rows:
            # coneqp solves the equality-constrained KKT system once; the
            # default chol2 factorization gets it wrong for singular P, and
            # ldl factors it densely, so only small systems switch.
            kkt_dim = int(data["P"].size[0]) + int(data["A"].size[0])
            if kkt_dim <= CVXOPT_DENSE_LDL_MAX_DIM:
                kktsolver = "ldl"
        if kktsolver is None and entry_point == "conelp" and dims["q"]:
            # conelp would default to the dense 'qr' KKT solver once there are
            # second-order cones; keep the sparse 'chol' coneqp used (DIMACS
            # sched_50_50_scaled: 80 s with chol, an error after 490 s with
            # qr). PSD-only problems keep 'qr', which conelp solves better
            # (SDPLIB control1 and hinf9 are solver_error with chol).
            kktsolver = "chol"
        _check_kktsolver(kktsolver, entry_point)

        def call(entry: str, kkt) -> dict:
            if entry == "conelp":
                return cs.conelp(
                    data["q"],
                    data["G"],
                    data["h"],
                    dims,
                    A=data["A"],
                    b=data["b"],
                    kktsolver=kkt,
                )
            return cs.coneqp(
                data["P"],
                data["q"],
                G=data["G"],
                h=data["h"],
                dims=dims,
                A=data["A"],
                b=data["b"],
                kktsolver=kkt,
            )

        # Save and restore global CVXOPT options so the adapter does
        # not leak settings across concurrent solves. ``options`` is a
        # plain dict; copying it before / restoring after is safe.
        raw: dict = {}
        solver_error: str | None = None
        discarded_claim: dict | None = None
        cross_check: dict | None = None
        with _cvxopt_options(cs, options, verbose=verbose):
            max_iters = int(cs.options.get("maxiters", 100))
            feastol = float(cs.options.get("feastol", 1e-7))
            start = time.perf_counter()
            # CVXOPT can throw ``ArithmeticError`` / ``ValueError`` on
            # numerical failures (e.g. an infeasible QP that pushes
            # coneqp into a domain-error regime, or a rank-deficient
            # KKT system). Catch those and surface them as SOLVER_ERROR
            # rather than letting the exception escape — matches how the
            # other adapters handle native solver crashes.
            try:
                raw = call(entry_point, kktsolver)
                if entry_point == "conelp" and _is_infeasibility_claim(raw):
                    claim = {
                        "status": raw.get("status"),
                        "iterations": _maybe_int(raw.get("iterations")),
                    }
                    coneqp_kkt = None if kktsolver == "qr" else kktsolver
                    if claim["iterations"] == 0:
                        # conelp's infeasibility test at iteration 0 only says
                        # the least-squares starting point has an objective
                        # beyond 1/feastol in magnitude: not evidence. Solve
                        # with coneqp (which has no 'qr' KKT solver).
                        discarded_claim = claim
                        entry_point = "coneqp"
                        raw = {}
                        raw = call(entry_point, coneqp_kkt)
                    else:
                        # A later claim can still be false on a badly scaled
                        # feasible problem (NETLIB scagr25, gfrd-pnc, nesm),
                        # and its scaled residual can pass the certificate
                        # check. Keep it unless coneqp solves the problem.
                        check, check_status = _coneqp_cross_check(
                            call, coneqp_kkt, max_iters=max_iters, feastol=feastol
                        )
                        cross_check = {"status": check_status, "raw_status": check.get("status")}
                        if check_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
                            discarded_claim = claim
                            entry_point = "coneqp"
                            raw = check
            except (ArithmeticError, ValueError) as exc:
                if isinstance(exc, ValueError) and _is_option_error(exc):
                    # A value the checks above let through but CVXOPT refuses.
                    raise SolverConfigError(f"CVXOPT rejected a setting value: {exc}") from exc
                solver_error = f"cvxopt.{entry_point} raised {type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - start

        info = _flatten_info(raw)
        info["raw_status"] = raw.get("status")
        info["entry_point"] = entry_point
        info["threads"] = None
        if discarded_claim is not None:
            info["conelp_discarded_claim"] = discarded_claim
        if cross_check is not None:
            info["coneqp_cross_check"] = cross_check
        if kktsolver is not None:
            info["kktsolver"] = kktsolver if isinstance(kktsolver, str) else repr(kktsolver)
        if translated:
            info["settings_translated"] = translated
        if solver_error is not None:
            info["raw_status"] = "error"
            info["error"] = solver_error
        mark_time_limit_ignored(info, time_limit)
        mark_threads_ignored(info, threads)

        if solver_error is not None:
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=elapsed,
                info=info,
            )

        mapped = _map_cvxopt_status(raw, max_iters=max_iters, feastol=feastol)
        # The objective is reported only for a claimed solution; limit
        # iterates and certificates carry none (mirrors SCS / ECOS).
        objective = reported_objective(mapped, _maybe_float(raw.get("primal objective")))

        x = _matrix_to_array(raw.get("x"))
        y = _matrix_to_array(raw.get("y"))
        z_dual = _matrix_to_array(raw.get("z"))
        s_slack = _matrix_to_array(raw.get("s"))

        kkt_dict = _compute_kkt(
            problem,
            mapped,
            x=x,
            y_eq=y,
            z_ineq=z_dual,
            s_slack=s_slack,
            cone_dict=cone_dict,
        )

        return SolverResult(
            status=mapped,
            objective_value=objective,
            iterations=_maybe_int(raw.get("iterations")),
            run_time_seconds=elapsed,
            info=info,
            kkt=kkt_dict,
        )


def _is_infeasibility_claim(raw: dict) -> bool:
    """Whether ``conelp`` claimed primal or dual infeasibility."""
    return str(raw.get("status", "")) in {"primal infeasible", "dual infeasible"}


def _coneqp_cross_check(call, kktsolver, *, max_iters: int, feastol: float) -> tuple[dict, str]:
    """Solve with ``coneqp`` to test a later ``conelp`` infeasibility claim.

    ``coneqp`` cannot certify infeasibility, so on a genuinely infeasible
    problem it stops without a solution (or raises); only a solution it
    finds overturns the claim. Its failures are returned as a status, never
    raised, so the claim stands when the check cannot be made.
    """
    try:
        check = call("coneqp", kktsolver)
    except (ArithmeticError, ValueError):
        return {}, status.SOLVER_ERROR
    return check, _map_cvxopt_status(check, max_iters=max_iters, feastol=feastol)


def _has_nonzeros(matrix) -> bool:
    """Whether a CVXOPT (sparse or dense) matrix has a nonzero entry."""
    values = np.asarray(getattr(matrix, "V", matrix), dtype=float)
    return bool(np.any(values != 0.0))


# ---------------------------------------------------------------------------
# Settings forwarding.
# ---------------------------------------------------------------------------


# Names CVXOPT's conelp / coneqp read from ``cvxopt.solvers.options``.
_CVXOPT_OPTIONS = frozenset(
    {
        "maxiters",
        "abstol",
        "reltol",
        "feastol",
        "refinement",
        "kktreg",
        "show_progress",
        "debug",
        "use_correction",
    }
)


def _resolve_options(settings: dict) -> tuple[dict, dict]:
    """Validate the settings and translate the harness aliases.

    Returns ``(options, translated)``: the CVXOPT options to apply and the
    entries the aliases inserted. ``max_iter`` / ``max_iters`` become
    ``maxiters``; ``eps_abs`` (or ``eps``) supplies ``abstol`` and
    ``feastol`` and ``eps_rel`` (or ``eps``) supplies ``reltol``. A native
    name always wins over an alias. Any other name raises
    ``SolverConfigError``: CVXOPT ignores keys it does not know, so a typo
    would otherwise run silently at the defaults. Values are checked against
    the ranges CVXOPT enforces (``_check_options``).
    """
    settings = dict(settings)
    max_iter = pop_max_iter(settings)
    max_iters = settings.pop("max_iters", None)
    if max_iters is not None:
        max_iters = coerce_int_setting("max_iters", max_iters, minimum=1)
    unknown = sorted(
        str(key) for key in settings if key not in _CVXOPT_OPTIONS and key not in _EPS_NAMES
    )
    if unknown:
        raise SolverConfigError(
            f"Unrecognized CVXOPT settings {unknown!r}; expected CVXOPT options "
            f"{sorted(_CVXOPT_OPTIONS)!r}, kktsolver, or the aliases max_iter, "
            "eps, eps_abs, eps_rel"
        )
    translated: dict = translate_tolerance_aliases(settings, ("abstol", "feastol"), ("reltol",))
    options = dict(settings)
    iteration_cap = max_iters if max_iters is not None else max_iter
    if iteration_cap is not None and options.get("maxiters") is None:
        options["maxiters"] = iteration_cap
        translated["maxiters"] = iteration_cap
    # CVXOPT wants Python ints; a numpy integer or 100.0 is the same count.
    for key, minimum in (("maxiters", 1), ("refinement", 0)):
        if options.get(key) is not None:
            options[key] = coerce_int_setting(key, options[key], minimum=minimum)
    _check_options(options, translated)
    return options, translated


_EPS_NAMES = frozenset({"eps", "eps_abs", "eps_rel"})


def _is_real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# The ranges conelp / coneqp enforce on their options (cvxopt.coneprog).
_OPTION_RANGES = {
    "maxiters": ("an integer >= 1", lambda v: _is_int(v) and v >= 1),
    "abstol": ("a real number", _is_real),
    "reltol": ("a real number", _is_real),
    "feastol": ("a positive real number", lambda v: _is_real(v) and v > 0),
    "refinement": ("an integer >= 0", lambda v: _is_int(v) and v >= 0),
    "kktreg": ("null or a real number >= 0", lambda v: v is None or (_is_real(v) and v >= 0)),
}


def _check_options(options: dict, translated: dict) -> None:
    """Raise SolverConfigError for an option value CVXOPT would refuse.

    CVXOPT raises the same ``ValueError`` for these as for numerical trouble
    during the solve, so they are checked before it.
    """
    for key, (description, test) in _OPTION_RANGES.items():
        if key in options and not test(options[key]):
            source = " (set from an alias)" if key in translated else ""
            raise SolverConfigError(
                f"Invalid CVXOPT setting {key!r}={options[key]!r}{source}: must be {description}"
            )
    abstol = options.get("abstol", _CVXOPT_DEFAULT_ABSTOL)
    reltol = options.get("reltol", _CVXOPT_DEFAULT_RELTOL)
    if abstol <= 0 and reltol <= 0:
        raise SolverConfigError(
            f"Invalid CVXOPT settings abstol={abstol!r}, reltol={reltol!r}: at least one "
            "must be positive"
        )


_CVXOPT_DEFAULT_ABSTOL = 1e-7
_CVXOPT_DEFAULT_RELTOL = 1e-6
# The KKT solvers each entry point accepts by name (coneqp has no 'qr').
_KKT_SOLVERS = {
    "conelp": ("ldl", "ldl2", "qr", "chol", "chol2"),
    "coneqp": ("ldl", "ldl2", "chol", "chol2"),
}


def _check_kktsolver(kktsolver, entry_point: str) -> None:
    """Raise SolverConfigError for a ``kktsolver`` CVXOPT would refuse."""
    if kktsolver is None or callable(kktsolver):
        return
    accepted = _KKT_SOLVERS[entry_point]
    if not isinstance(kktsolver, str) or kktsolver not in accepted:
        raise SolverConfigError(
            f"Invalid CVXOPT setting 'kktsolver'={kktsolver!r}: {entry_point} (used for this "
            f"problem) accepts {list(accepted)} or a function"
        )


def _is_option_error(exc: ValueError) -> bool:
    """Whether CVXOPT's ValueError is its refusal of an option or kktsolver."""
    text = str(exc)
    return (
        text.startswith("options[")
        or text.startswith("at least one of options")
        or "is not a valid value for kktsolver" in text
    )


@contextmanager
def _cvxopt_options(cs, settings: dict, *, verbose: bool):
    """Apply per-solve CVXOPT options under save/restore semantics.

    CVXOPT's solvers are configured via the global ``solvers.options``
    dict; mutating it without restoring would leak knobs across solves
    that share a process (and across pytest tests). The contract here
    is **per-process sequential**: the snapshot/restore protects
    against leaking to *later* solves in the same process. It does
    **not** make concurrent in-process solves safe — two threads
    racing through ``coneqp`` would still observe each other's
    in-flight option mutations. The benchmark runner uses
    subprocess-level parallelism, so this is sufficient for our use.
    Callers running CVXOPT from in-process threads should wrap solve
    calls in their own lock.
    """
    snapshot = dict(cs.options)
    try:
        cs.options["show_progress"] = bool(verbose)
        for key, value in settings.items():
            cs.options[key] = value
        yield
    finally:
        cs.options.clear()
        cs.options.update(snapshot)


# ---------------------------------------------------------------------------
# Status mapping.
# ---------------------------------------------------------------------------


_CVXOPT_INFEASIBILITY_ACCEPT_THRESHOLD = 1e-3
_CVXOPT_RELATIVE_GAP_ACCEPT_THRESHOLD = 1e-3


def _map_cvxopt_status(raw: dict, *, max_iters: int = 100, feastol: float = 1e-7) -> str:
    """Translate a CVXOPT result to a canonical status.

    ``conelp`` reports ``optimal``, ``primal infeasible``, ``dual
    infeasible`` or ``unknown``; ``coneqp`` only ``optimal`` or ``unknown``.

    - ``optimal`` is OPTIMAL when CVXOPT's own primal and dual infeasibility
      are within ``feastol`` (always true after an interior-point run). The
      one-shot KKT solve of a problem without cone rows returns ``optimal``
      unconditionally, so a residual above ``feastol`` there demotes it:
      OPTIMAL_INACCURATE when below the accept threshold, else SOLVER_ERROR.
    - ``unknown`` is MAX_ITER_REACHED when the iteration count reached
      ``maxiters``. An earlier stop (numerical trouble such as a singular
      KKT matrix) is OPTIMAL_INACCURATE when the relative gap and both
      infeasibilities are below the accept threshold, and SOLVER_ERROR
      otherwise. A large iterate residual is not evidence of
      infeasibility, so ``unknown`` never maps to a certificate status.
    """
    raw_status = str(raw.get("status", "")).lower()
    prim_inf = raw.get("primal infeasibility")
    dual_inf = raw.get("dual infeasibility")
    if raw_status == "optimal":
        reported = [
            _coerce_finite_float(v) for v in (prim_inf, dual_inf) if v is not None
        ]
        worst = max((float("inf") if v is None else v for v in reported), default=0.0)
        if worst <= feastol:
            return status.OPTIMAL
        if worst < _CVXOPT_INFEASIBILITY_ACCEPT_THRESHOLD:
            return status.OPTIMAL_INACCURATE
        return status.SOLVER_ERROR
    if raw_status == "primal infeasible":
        return status.PRIMAL_INFEASIBLE
    if raw_status == "dual infeasible":
        return status.DUAL_INFEASIBLE
    if raw_status == "unknown":
        iterations = _maybe_int(raw.get("iterations"))
        if iterations is not None and iterations >= int(max_iters):
            return status.MAX_ITER_REACHED
        if (
            _is_finite_below(raw.get("relative gap"), _CVXOPT_RELATIVE_GAP_ACCEPT_THRESHOLD)
            and _is_finite_below(prim_inf, _CVXOPT_INFEASIBILITY_ACCEPT_THRESHOLD)
            and _is_finite_below(dual_inf, _CVXOPT_INFEASIBILITY_ACCEPT_THRESHOLD)
        ):
            return status.OPTIMAL_INACCURATE
        return status.SOLVER_ERROR
    return status.SOLVER_ERROR


def _coerce_finite_float(value) -> float | None:
    """Return ``float(value)`` if finite, else None.

    Used by the status-mapping accept gates: a NaN or inf
    infeasibility metric is treated as "no signal" rather than
    silently passing the comparison.
    """
    if value is None:
        return None
    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if coerced != coerced or coerced in (float("inf"), float("-inf")):  # NaN check + inf
        return None
    return coerced


def _is_finite_below(value, threshold: float) -> bool:
    """True iff ``value`` is finite and ``< threshold``. None is
    treated as a failure (no evidence of acceptably small value)."""
    coerced = _coerce_finite_float(value)
    return coerced is not None and coerced < threshold


# ---------------------------------------------------------------------------
# QP / CONE -> CVXOPT data.
# ---------------------------------------------------------------------------


def _qp_to_cvxopt(qp: dict, cvxopt):
    """Build the CVXOPT data tuple for a QP via qp_to_nonnegative_cone.

    The transform produces ``a_cone x + s = b_cone`` with the first
    ``z`` rows in the zero cone (equalities) and the rest in the
    nonneg cone. CVXOPT wants equalities in (A, b) and inequalities
    in (G, h) with ``Gx + s = h, s in K``. Since ``s = b_cone - a x``,
    we have ``a x + s = b_cone`` with ``s >= 0``, which matches CVXOPT
    by setting ``G = a``, ``h = b_cone``.
    """
    a_cone, b_cone, z = qp_to_nonnegative_cone(qp)
    a_cone = sp.csc_matrix(a_cone)
    p = sp.csc_matrix(qp.get("P") if qp.get("P") is not None else (a_cone.shape[1], a_cone.shape[1]))
    q_vec = np.asarray(qp["q"], dtype=float)

    a_eq = a_cone[:z, :]
    b_eq = b_cone[:z]
    a_ineq = a_cone[z:, :]
    b_ineq = b_cone[z:]

    n_ineq = int(a_ineq.shape[0])
    dims = {"l": n_ineq, "q": [], "s": []}
    cone_dict: dict = {}
    if z:
        cone_dict["z"] = int(z)
    if n_ineq:
        cone_dict["l"] = n_ineq

    return (
        {
            "P": _scipy_to_cvxopt(p, cvxopt),
            "q": _np_to_cvxopt(q_vec, cvxopt),
            "G": _scipy_to_cvxopt(a_ineq, cvxopt) if n_ineq else _empty_matrix(0, p.shape[1], cvxopt),
            "h": _np_to_cvxopt(b_ineq, cvxopt) if n_ineq else _empty_matrix(0, 1, cvxopt),
            "A": _scipy_to_cvxopt(a_eq, cvxopt) if z else _empty_matrix(0, p.shape[1], cvxopt),
            "b": _np_to_cvxopt(b_eq, cvxopt) if z else _empty_matrix(0, 1, cvxopt),
        },
        dims,
        cone_dict,
    )


def _cone_to_cvxopt(cone_problem: dict, cvxopt):
    """Map a CONE-shape problem into CVXOPT's coneqp inputs.

    Cone keys handled: ``z`` (and legacy ``f``) → A,b; ``l`` → dims['l'];
    ``q`` → dims['q']; ``s`` → dims['s'] with the canonical
    PSD-triangle vec converted to CVXOPT's BLAS unpacked 'L' (column-
    major full matrix) layout via ``psd_triangle_to_full``. Other
    cone keys (e/ep for exponential, anything novel) yield a
    ``SKIPPED_UNSUPPORTED`` SolverResult.
    """
    a = sp.csc_matrix(cone_problem["A"])
    b = np.asarray(cone_problem["b"], dtype=float)
    q_vec = np.asarray(cone_problem["q"], dtype=float)
    p = cone_problem.get("P")
    p = (
        sp.csc_matrix((a.shape[1], a.shape[1]))
        if p is None
        else sp.csc_matrix(p)
    )
    cone = dict(cone_problem["cone"])

    z_count = int(cone.pop("z", 0)) + int(cone.pop("f", 0))
    l_count = int(cone.pop("l", 0))
    # A second-order cone of dimension 0 has no rows and CVXOPT refuses it;
    # an empty PSD cone is dropped the same way.
    q_list = [int(dim) for dim in as_list(cone.pop("q", [])) if int(dim)]
    s_list = [int(dim) for dim in as_list(cone.pop("s", [])) if int(dim)]
    bad_keys = sorted(cone.keys())
    if bad_keys:
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={
                "reason": (
                    f"CVXOPT does not support cone keys {bad_keys!r} (only "
                    "z/f, l, q, s are handled; exponential and power cones "
                    "would need a custom kkt solver)."
                )
            },
        )

    # Layout the rows (z first, then l, then q*, then s*) — matches
    # canonical cone problem convention.
    z_rows = a[:z_count, :]
    z_b = b[:z_count]
    cursor = z_count

    l_rows = a[cursor : cursor + l_count, :]
    l_b = b[cursor : cursor + l_count]
    cursor += l_count

    q_rows_list = []
    q_b_list = []
    for qd in q_list:
        q_rows_list.append(a[cursor : cursor + qd, :])
        q_b_list.append(b[cursor : cursor + qd])
        cursor += int(qd)

    # PSD cones: convert canonical ``n*(n+1)/2`` triangle rows into
    # CVXOPT's ``n*n`` full-matrix vec layout.
    s_rows_list = []
    s_b_list = []
    for sd in s_list:
        triangle_dim = sd * (sd + 1) // 2
        canonical_rows = a[cursor : cursor + triangle_dim, :]
        canonical_b = b[cursor : cursor + triangle_dim]
        full = psd_triangle_to_full(int(sd))
        s_rows_list.append(full @ canonical_rows)
        s_b_list.append(full @ canonical_b)
        cursor += triangle_dim

    # Stack inequality (G, h) blocks: l, q*, s*.
    g_blocks = []
    h_blocks = []
    if l_rows.shape[0]:
        g_blocks.append(l_rows)
        h_blocks.append(l_b)
    g_blocks.extend(q_rows_list)
    h_blocks.extend(q_b_list)
    g_blocks.extend(s_rows_list)
    h_blocks.extend(s_b_list)

    # CONE form is ``A x + s = b, s in K`` with K listed, but coneqp
    # expects ``Gx + s = h``. In our CONE schema rows correspond to
    # ``A x + s = b`` so ``G = A, h = b`` lines up directly.
    a_ineq = (
        sp.vstack(g_blocks, format="csc")
        if g_blocks
        else sp.csc_matrix((0, a.shape[1]))
    )
    b_ineq = np.concatenate(h_blocks) if h_blocks else np.array([], dtype=float)

    dims = {"l": int(l_count), "q": [int(d) for d in q_list], "s": [int(d) for d in s_list]}
    cone_dict: dict = {}
    if z_count:
        cone_dict["z"] = int(z_count)
    if l_count:
        cone_dict["l"] = int(l_count)
    if q_list:
        cone_dict["q"] = [int(d) for d in q_list]
    if s_list:
        cone_dict["s"] = [int(d) for d in s_list]

    return (
        {
            "P": _scipy_to_cvxopt(p, cvxopt),
            "q": _np_to_cvxopt(q_vec, cvxopt),
            "G": _scipy_to_cvxopt(a_ineq, cvxopt) if a_ineq.shape[0] else _empty_matrix(0, a.shape[1], cvxopt),
            "h": _np_to_cvxopt(b_ineq, cvxopt) if b_ineq.size else _empty_matrix(0, 1, cvxopt),
            "A": _scipy_to_cvxopt(z_rows, cvxopt) if z_count else _empty_matrix(0, a.shape[1], cvxopt),
            "b": _np_to_cvxopt(z_b, cvxopt) if z_count else _empty_matrix(0, 1, cvxopt),
        },
        dims,
        cone_dict,
    )


def _psd_full_to_triangle(dim: int) -> sp.csc_matrix:
    """Inverse of ``psd_triangle_to_full``: map BLAS unpacked 'L'
    (column-major full ``dim*dim``) back to the canonical
    ``dim*(dim+1)/2`` lower-triangle vec with ``√2`` off-diagonal
    scaling. Used to convert CVXOPT's PSD-cone duals and slacks back
    into the layout the canonical KKT helpers expect.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    sqrt2 = float(np.sqrt(2.0))
    triangle_idx = 0
    for j in range(dim):
        for i in range(j, dim):
            if i == j:
                rows.append(triangle_idx)
                cols.append(i + j * dim)
                data.append(1.0)
            else:
                # Canonical[k] = √2 * blas[i + j*dim] (the strictly
                # lower entry; the strict upper is mirrored / unused).
                rows.append(triangle_idx)
                cols.append(i + j * dim)
                data.append(sqrt2)
            triangle_idx += 1
    return sp.csc_matrix(
        (data, (rows, cols)),
        shape=(dim * (dim + 1) // 2, dim * dim),
    )


# ---------------------------------------------------------------------------
# CVXOPT <-> NumPy/SciPy conversions.
# ---------------------------------------------------------------------------


def _scipy_to_cvxopt(matrix: sp.spmatrix | sp.csc_matrix, cvxopt):
    """Convert a scipy sparse matrix to a CVXOPT sparse matrix."""
    coo = sp.coo_matrix(matrix)
    return cvxopt.spmatrix(
        coo.data.astype(float).tolist(),
        coo.row.astype(int).tolist(),
        coo.col.astype(int).tolist(),
        size=coo.shape,
        tc="d",
    )


def _np_to_cvxopt(vec: np.ndarray, cvxopt):
    arr = np.asarray(vec, dtype=float).reshape(-1, 1)
    return cvxopt.matrix(arr, tc="d")


def _empty_matrix(rows: int, cols: int, cvxopt):
    """Return an empty matrix in the right CVXOPT shape.

    CVXOPT requires the right-hand-side vectors ``h`` and ``b`` to be
    *dense* ``'d'`` matrices with one column even when empty; the
    coefficient matrices ``G`` and ``A`` accept either dense or
    sparse, so a sparse zero matrix is fine.
    """
    if cols == 1:
        return cvxopt.matrix(np.zeros((rows, 1)), tc="d")
    return cvxopt.spmatrix([], [], [], (rows, cols), tc="d")


def _matrix_to_array(value):
    if value is None:
        return None
    return np.asarray(value).ravel()


def _flatten_info(raw: dict) -> dict:
    """Pull the JSON-friendly scalar fields out of CVXOPT's result dict.

    The full result includes the cvxopt.matrix vectors (x, y, z, s);
    those are returned to the caller separately. Everything else is
    a scalar (status, gap, infeasibility metrics, iteration count).
    """
    keep = (
        "status",
        "gap",
        "relative gap",
        "primal objective",
        "dual objective",
        "primal infeasibility",
        "dual infeasibility",
        "primal slack",
        "dual slack",
        "iterations",
    )
    return {k: _maybe_float(raw.get(k)) if k != "status" else raw.get(k) for k in keep}


def _maybe_float(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _maybe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# KKT translation.
# ---------------------------------------------------------------------------


def _compute_kkt(problem, mapped_status, *, x, y_eq, z_ineq, s_slack, cone_dict):
    """Forward to the canonical KKT helpers, reassembling duals into
    the layout each helper expects.

    Solution residuals are computed for every status in ANY_FEASIBLE that
    carries a point (max-iter iterates included, so the KKT verification's
    promotion treats CVXOPT like every other solver). A conelp primal-infeasibility
    certificate comes with ``z``/``y`` but no ``x``.
    """
    if x is None and mapped_status not in {
        status.PRIMAL_INFEASIBLE,
        status.PRIMAL_INFEASIBLE_INACCURATE,
    }:
        return None
    if x is not None:
        x = np.asarray(x, dtype=float)

    if problem.kind == QP:
        qp = problem.qp
        n_eq = int(cone_dict.get("z", 0))
        if mapped_status in status.ANY_FEASIBLE:
            y_full = qp_dual_in_native_order(qp, y_eq, z_ineq, n_eq)
            if not finite_point(x, y_full):
                return None
            return kkt.qp_residuals(
                qp["P"] if qp.get("P") is not None else sp.csc_matrix((x.size, x.size)),
                qp["q"],
                qp["A"],
                qp["l"],
                qp["u"],
                x,
                y_full,
            )
        if mapped_status in {
            status.PRIMAL_INFEASIBLE,
            status.PRIMAL_INFEASIBLE_INACCURATE,
        }:
            y_full = qp_dual_in_native_order(qp, y_eq, z_ineq, n_eq)
            if not finite_point(y_full):
                return None
            return kkt.qp_primal_infeasibility_cert(qp["A"], qp["l"], qp["u"], y_full)
        if mapped_status in {
            status.DUAL_INFEASIBLE,
            status.DUAL_INFEASIBLE_INACCURATE,
        } and finite_point(x):
            return kkt.qp_dual_infeasibility_cert(
                qp["P"] if qp.get("P") is not None else sp.csc_matrix((x.size, x.size)),
                qp["q"],
                qp["A"],
                qp["l"],
                qp["u"],
                x,
            )
        return None

    cone_problem = problem.cone
    p = cone_problem.get("P")
    if p is None:
        p = sp.csc_matrix((sp.csc_matrix(cone_problem["A"]).shape[1],) * 2)
    if (
        mapped_status in status.ANY_FEASIBLE
        and z_ineq is not None
        and s_slack is not None
    ):
        y_combined = _cone_dual_combined(y_eq, z_ineq, cone_dict)
        s_combined = _cone_slack_combined(s_slack, cone_dict)
        if not finite_point(x, y_combined, s_combined):
            return None
        return kkt.cone_residuals(
            p,
            cone_problem["q"],
            cone_problem["A"],
            cone_problem["b"],
            cone_dict,
            x,
            y_combined,
            s_combined,
        )
    if mapped_status in {
        status.PRIMAL_INFEASIBLE,
        status.PRIMAL_INFEASIBLE_INACCURATE,
    } and (z_ineq is not None or y_eq is not None):
        y_combined = _cone_dual_combined(y_eq, z_ineq, cone_dict)
        if not finite_point(y_combined):
            return None
        return kkt.cone_primal_infeasibility_cert(
            cone_problem["A"], cone_problem["b"], cone_dict, y_combined
        )
    if mapped_status in {
        status.DUAL_INFEASIBLE,
        status.DUAL_INFEASIBLE_INACCURATE,
    } and finite_point(x):
        return kkt.cone_dual_infeasibility_cert(
            p, cone_problem["q"], cone_problem["A"], cone_dict, x
        )
    return None


def _cone_dual_combined(y_eq, z_ineq, cone_dict):
    """Combine equality and inequality duals into the canonical row
    layout. PSD-cone entries in ``z_ineq`` come back from CVXOPT in
    BLAS unpacked 'L' (n*n) layout; we convert each PSD block back to
    canonical (n*(n+1)/2 with √2 off-diagonal scaling) so the KKT
    helpers see the layout they expect.
    """
    parts: list[np.ndarray] = []
    if cone_dict.get("z"):
        parts.append(
            np.asarray(y_eq, dtype=float).ravel()
            if y_eq is not None
            else np.zeros(int(cone_dict["z"]))
        )
    if z_ineq is not None and len(z_ineq):
        parts.append(_blas_psd_to_canonical(np.asarray(z_ineq, dtype=float).ravel(), cone_dict))
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def _cone_slack_combined(s_slack, cone_dict):
    """Pad the ineq-block slack with zeros for the equality block, so
    the combined slack aligns with the canonical (z, l, q*, s*) row
    layout the KKT helpers expect. PSD-cone entries are converted
    back to canonical layout (mirrors ``_cone_dual_combined``).
    """
    n_eq = int(cone_dict.get("z", 0))
    s_canonical = _blas_psd_to_canonical(np.asarray(s_slack, dtype=float).ravel(), cone_dict)
    if n_eq:
        return np.concatenate([np.zeros(n_eq), s_canonical])
    return s_canonical


def _blas_psd_to_canonical(blas_vec: np.ndarray, cone_dict: dict) -> np.ndarray:
    """Convert an inequality-block vector from BLAS PSD layout to
    canonical layout in-place per cone block.

    Layout assumption: rows are ordered [l, q*, s*]. Only the s blocks
    are repacked from ``n*n`` (BLAS) to ``n*(n+1)/2`` (canonical). The
    ``l`` and ``q`` blocks pass through unchanged.
    """
    if not cone_dict.get("s"):
        return blas_vec
    out_parts: list[np.ndarray] = []
    cursor = 0
    l_count = int(cone_dict.get("l", 0))
    if l_count:
        out_parts.append(blas_vec[cursor : cursor + l_count])
        cursor += l_count
    for q_dim in cone_dict.get("q", []) or []:
        q_dim = int(q_dim)
        out_parts.append(blas_vec[cursor : cursor + q_dim])
        cursor += q_dim
    for s_dim in cone_dict.get("s", []) or []:
        s_dim = int(s_dim)
        full_size = s_dim * s_dim
        block = blas_vec[cursor : cursor + full_size]
        canonical = _psd_full_to_triangle(s_dim) @ block
        out_parts.append(canonical)
        cursor += full_size
    if not out_parts:
        return np.array([], dtype=float)
    return np.concatenate(out_parts)
