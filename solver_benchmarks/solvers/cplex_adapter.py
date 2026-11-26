"""IBM CPLEX adapter.

A license failure raised by the solve (the Community Edition that ``pip
install cplex`` ships with refuses larger problems inside ``solve()``, error
1016) is a broken environment, not CPLEX failing the problem: it raises
``SolverUnavailable``. Any other CPLEX error from the solve is
``solver_error``; other exceptions are left to raise. ``threads`` is one
unless configured (0: CPLEX's own choice). ``info["raw_status"]`` is CPLEX's
status string, ``info["raw_status_code"]`` its integer code.

An infeasibility claim on an LP carries CPLEX's certificate (its dual Farkas
proof or its unbounded ray), checked like every other adapter's; when the
solve that made the claim leaves none (or a vector that is clearly not one),
the LP is solved again for it, and that time counts in the run time when a
certificate is recorded (see :func:`_cplex_certificate`). Every further
solve of a run gets what is left of its time limit (CPLEX's ``timelimit``
bounds each ``solve()``). CPLEX has none for QPs. Not run against a real
CPLEX here (not installed on the development machine); the conversion
follows the CPLEX reference manual (CPXdualfarkas, CPXgetray) and is pinned
by the fake in ``tests/test_commercial_adapters_fake.py``.
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
    SolverUnavailable,
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
from .qp_split import normalize_qp_bounds, qp_certificate_kkt

# CPLEX's documented defaults (dotted parameter paths): simplex feasibility
# and optimality, and the barrier's convergence tolerance. Not checked here
# (CPLEX needs a license).
CPLEX_DEFAULT_TOLERANCES = {
    "simplex.tolerances.feasibility": 1e-6,
    "simplex.tolerances.optimality": 1e-6,
    "barrier.convergetol": 1e-8,
}


class CPLEXSolverAdapter(SolverAdapter):
    solver_name = "cplex"
    supported_problem_kinds = {QP}
    solver_modules = ("cplex",)

    @classmethod
    def is_available(cls) -> bool:
        return module_available("cplex")

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        refuse_settings(settings, EPS_ALIASES, "CPLEX", "set its own tolerances instead")
        return loosest_tolerance(settings, CPLEX_DEFAULT_TOLERANCES)

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import cplex
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "cplex", "Install with the cplex extra to use CPLEX", exc
            ) from exc

        settings = dict(self.settings)
        use_ranged_rows = settings.pop("use_ranged_rows", False)
        if not isinstance(use_ranged_rows, bool):
            raise SolverConfigError("use_ranged_rows must be a boolean")

        qp = problem.qp
        q = np.asarray(qp["q"], dtype=float)
        p = sp.coo_matrix(qp["P"])
        a = sp.csr_matrix(qp["A"])
        l, u = normalize_qp_bounds(qp["l"], qp["u"])
        n = int(qp.get("n", int(q.shape[0])))
        model = cplex.Cplex()
        # Always end() the model so the C-level state and license token
        # are released deterministically, even when an exception fires.
        try:
            model.objective.set_sense(model.objective.sense.minimize)
            translated, threads, time_limit = _configure_cplex(model, settings)
            infinity = cplex.infinity
            model.variables.add(
                obj=q.tolist(),
                lb=[-infinity] * n,
                ub=[infinity] * n,
                names=[f"x_{idx}" for idx in range(n)],
            )
            constraint_rows = _add_cplex_constraints(
                model, cplex, a, l, u, use_ranged_rows=use_ranged_rows
            )
            if p.nnz:
                model.objective.set_quadratic_coefficients(
                    [(int(i), int(j), float(v)) for i, j, v in zip(p.row, p.col, p.data, strict=True)]
                )

            start = time.perf_counter()
            try:
                model.solve()
                resolved = _resolve_infeasible_or_unbounded(model, settings, time_limit, start)
            except _cplex_error_type(cplex) as exc:
                if _is_license_error(exc):
                    raise SolverUnavailable(f"CPLEX license error: {exc}") from exc
                return SolverResult(
                    status=status.SOLVER_ERROR,
                    run_time_seconds=time.perf_counter() - start,
                    info={
                        "raw_status": "error",
                        "error": str(exc),
                        "use_ranged_rows": use_ranged_rows,
                        "threads": threads,
                    },
                )
            claimed = time.perf_counter()
            elapsed = claimed - start
            raw_status = model.solution.get_status()
            mapped = _map_cplex_status(raw_status, model)
            iteration_counts = _cplex_iterations(model)
            problem_type = model.problem_type[model.get_problem_type()]
            objective_value = None
            kkt_dict = None
            certificate_info: dict = {}
            if mapped in _CERTIFICATE_SOURCES:
                # Everything else about the solve is read first: getting the
                # certificate may solve the model again.
                kkt_dict, certificate_info = _cplex_certificate(
                    model,
                    mapped,
                    is_lp=problem_type == "LP",
                    time_limit=time_limit,
                    elapsed=elapsed,
                    record=lambda vector: _certificate_record(
                        mapped, vector, p, q, a, l, u, constraint_rows, n
                    ),
                )
                certificate_info["certificate_seconds"] = time.perf_counter() - claimed
                if kkt_dict is not None:
                    # The claim is checked on this certificate, so the time
                    # it took counts; an attempt that gave none does not
                    # (the claim is unverified whatever it cost).
                    elapsed += certificate_info["certificate_seconds"]
            elif mapped in status.ANY_FEASIBLE:
                # A returned iterate may exist even with an error or limit status.
                try:
                    x = np.asarray(model.solution.get_values(), dtype=float)
                    objective_value = model.solution.get_objective_value()
                    duals = np.asarray(model.solution.get_dual_values(), dtype=float)
                except cplex.exceptions.CplexSolverError:
                    pass  # No complete primal/dual point is available.
                else:
                    y = np.zeros(a.shape[0])
                    # CPLEX uses the opposite dual sign; sum any split rows.
                    np.add.at(y, constraint_rows, -duals)
                    if finite_point(x, y):
                        kkt_dict = kkt.qp_residuals(p, q, a, l, u, x, y)
            return SolverResult(
                status=mapped,
                kkt=kkt_dict,
                objective_value=reported_objective(mapped, objective_value),
                iterations=iteration_counts["iterations"],
                run_time_seconds=elapsed,
                info={
                    "raw_status": _status_string(model, raw_status),
                    "raw_status_code": raw_status,
                    "threads": threads,
                    "status_string": _status_string(model, raw_status),
                    "problem_type": problem_type,
                    "use_ranged_rows": use_ranged_rows,
                    "simplex_iterations": iteration_counts["simplex_iterations"],
                    "barrier_iterations": iteration_counts["barrier_iterations"],
                    **certificate_info,
                    **({"settings_translated": translated} if translated else {}),
                    **({"inf_or_unbd_resolved": True} if resolved else {}),
                },
            )
        finally:
            try:
                model.end()
            except Exception:
                pass


def _add_cplex_constraints(
    model, cplex, a, l, u, *, use_ranged_rows: bool = False
) -> list[int]:
    constraint_rows = []
    lin_expr = []
    senses = []
    rhs = []
    names = []
    range_values = []
    for row in range(a.shape[0]):
        start, end = a.indptr[row], a.indptr[row + 1]
        expr = cplex.SparsePair(
            ind=[int(col) for col in a.indices[start:end]],
            val=[float(val) for val in a.data[start:end]],
        )
        l_val = float(l[row])
        u_val = float(u[row])
        if l_val <= -1.0e20 and u_val >= 1.0e20:
            continue
        if abs(l_val - u_val) <= 1.0e-10:
            constraint_rows.append(row)
            lin_expr.append(expr)
            senses.append("E")
            rhs.append(u_val)
            names.append(f"c_{row}_eq")
            range_values.append(0.0)
            continue
        # A negative CPLEX range reverses the interval rather than making it
        # infeasible, so keep contradictory bounds as separate inequalities.
        if use_ranged_rows and -1.0e20 < l_val < u_val < 1.0e20:
            constraint_rows.append(row)
            lin_expr.append(expr)
            senses.append("R")
            rhs.append(l_val)
            names.append(f"c_{row}_range")
            range_values.append(u_val - l_val)
            continue
        if l_val > -1.0e20:
            constraint_rows.append(row)
            lin_expr.append(expr)
            senses.append("G")
            rhs.append(l_val)
            names.append(f"c_{row}_lb")
            range_values.append(0.0)
        if u_val < 1.0e20:
            constraint_rows.append(row)
            lin_expr.append(expr)
            senses.append("L")
            rhs.append(u_val)
            names.append(f"c_{row}_ub")
            range_values.append(0.0)
    if lin_expr:
        options = {"range_values": range_values} if use_ranged_rows else {}
        model.linear_constraints.add(
            lin_expr=lin_expr, senses=senses, rhs=rhs, names=names, **options
        )
    return constraint_rows


def _configure_cplex(model, settings: dict) -> tuple[dict, int, float | None]:
    """Apply ``settings`` (dotted CPLEX parameter paths) to ``model``.

    Returns the aliases it translated, the thread count and the time limit
    in seconds (CPLEX's own ``timelimit`` when given, which is applied after
    and so wins over the ``time_limit`` aliases; None if neither is set):
    ``max_iter`` caps both the barrier and the simplex iteration counts
    unless either path is given, and ``threads`` (CPLEX's own name too) is
    one unless configured. The ``eps`` aliases are refused (CPLEX's
    tolerances are not an absolute/relative pair), and a path CPLEX does not
    have raises SolverConfigError naming it.
    """
    settings = settings_with_defaults(settings)
    refuse_settings(
        settings,
        EPS_ALIASES,
        "CPLEX",
        "set its own tolerances instead (barrier.convergetol, "
        "simplex.tolerances.feasibility, simplex.tolerances.optimality)",
    )
    translated: dict = {}
    max_iter = pop_max_iter(settings, minimum=0)
    if max_iter is not None:
        for path in ("barrier.limits.iteration", "simplex.limits.iterations"):
            if path not in settings:
                settings[path] = max_iter
                translated[path] = max_iter
    verbose = bool(settings.pop("verbose", False))
    if not verbose:
        model.set_results_stream(None)
        model.set_log_stream(None)
        model.set_warning_stream(None)
        model.set_error_stream(None)
    time_limit = pop_time_limit(settings)
    if time_limit is not None:
        model.parameters.timelimit.set(float(time_limit))
    threads = threads_or_default(pop_threads(settings))
    model.parameters.threads.set(threads)
    for key, value in settings.items():
        _set_cplex_parameter(model, key, value)
    if settings.get("timelimit") is not None:
        time_limit = float(settings["timelimit"])
    return translated, threads, time_limit


def _set_cplex_parameter(model, key: str, value) -> None:
    node = model.parameters
    try:
        for part in str(key).split("."):
            node = getattr(node, part)
        setter = node.set
    except AttributeError:
        raise SolverConfigError(f"Unknown CPLEX parameter {key!r}") from None
    try:
        setter(value)
    except Exception as exc:
        raise SolverConfigError(f"Invalid CPLEX parameter {key!r}={value!r}: {exc}") from exc


def _cplex_error_type(cplex):
    """CPLEX's base exception (``CplexError``; ``CplexSolverError`` derives from it)."""
    exceptions = cplex.exceptions
    return getattr(exceptions, "CplexError", exceptions.CplexSolverError)


# CPLEX error codes for license failures: Community Edition size limits
# (CPXERR_RESTRICTED_VERSION) and no valid license (CPXERR_ILOG_LICENSE).
_LICENSE_CODES = frozenset({1016, 32201})


def _is_license_error(exc: Exception) -> bool:
    # CplexError's args are (message, environment, error code).
    args = tuple(exc.args)
    code = args[2] if len(args) >= 3 else None
    text = str(exc).lower()
    return code in _LICENSE_CODES or "community edition" in text or "licen" in text


def _status_string(model, raw_status) -> str:
    try:
        return str(model.solution.get_status_string(raw_status))
    except Exception:
        return str(raw_status)


def _map_cplex_status(raw_status: int, model) -> str:
    sol_status = model.solution.status
    mapping = {
        sol_status.optimal: status.OPTIMAL,
        sol_status.infeasible: status.PRIMAL_INFEASIBLE,
        sol_status.unbounded: status.DUAL_INFEASIBLE,
        sol_status.infeasible_or_unbounded: status.PRIMAL_OR_DUAL_INFEASIBLE,
        sol_status.abort_iteration_limit: status.MAX_ITER_REACHED,
        sol_status.abort_time_limit: status.TIME_LIMIT,
    }
    for name, canonical in [
        ("optimal_infeasible", status.OPTIMAL_INACCURATE),
        ("num_best", status.OPTIMAL_INACCURATE),
        ("abort_dettime_limit", status.TIME_LIMIT),
    ]:
        if hasattr(sol_status, name):
            mapping[getattr(sol_status, name)] = canonical
    return mapping.get(raw_status, status.SOLVER_ERROR)


def _resolve_infeasible_or_unbounded(model, settings: dict, time_limit: float | None, start: float) -> bool:
    """Re-solve without presolve reductions after infeasible_or_unbounded.

    CPLEX's presolve can prove only that a model has no optimum; turning its
    reductions off (preprocessing.reduce = 0) and re-solving gives the
    definite answer, so the row can carry an infeasibility or unboundedness
    status instead of primal_or_dual_infeasible, which never counts. The
    re-solve gets what is left of ``time_limit`` (none when nothing is
    left). Returns whether the model was re-solved (the elapsed time covers
    both solves).
    """
    status_codes = model.solution.status
    if model.solution.get_status() != status_codes.infeasible_or_unbounded:
        return False
    if settings.get("preprocessing.reduce") == 0:
        return False  # already off: the answer is final
    if not _limit_to_time_left(model, time_limit, time.perf_counter() - start):
        return False
    _set_cplex_parameter(model, "preprocessing.reduce", 0)
    model.solve()
    return True


def _limit_to_time_left(model, time_limit: float | None, elapsed: float) -> bool:
    """Give CPLEX's next solve of this run what is left of ``time_limit``.

    CPLEX's ``timelimit`` bounds each ``solve()`` call, so a second solve
    would otherwise get the whole limit again. Returns False, setting
    nothing, when no time is left.
    """
    if time_limit is None:
        return True
    remaining = float(time_limit) - elapsed
    if remaining <= 0.0:
        return False
    model.parameters.timelimit.set(remaining)
    return True


# CPLEX's certificate for each infeasibility claim and the simplex method
# (``lpmethod``) whose solve leaves one: the dual simplex a Farkas proof of
# infeasibility (CPXdualfarkas), the primal simplex an unbounded ray
# (CPXgetray).
_CERTIFICATE_SOURCES = {
    status.PRIMAL_INFEASIBLE: ("dual Farkas proof", 2),
    status.DUAL_INFEASIBLE: ("unbounded ray", 1),
}


def _read_cplex_certificate(model, mapped: str) -> np.ndarray:
    advanced = model.solution.advanced
    if mapped == status.PRIMAL_INFEASIBLE:
        # (y, proof): y over the rows the adapter added, proof = y'b - y'Az > 0.
        farkas, _proof = advanced.dual_farkas()
        return np.asarray(farkas, dtype=float).reshape(-1)
    return np.asarray(advanced.get_ray(), dtype=float).reshape(-1)


def _certificate_record(mapped: str, vector, p, q, a, l, u, constraint_rows, n: int):
    """The check of CPLEX's certificate ``vector``, as ``(record, unavailable)``.

    For ``primal_infeasible`` ``vector`` holds CPLEX's Farkas multipliers,
    one per CPLEX row: they are <= 0 on <= rows and >= 0 on >= rows
    (CPXdualfarkas), the opposite of the harness's sign, as for its duals,
    and a row split into two gets their sum, which certifies at least as
    strongly (its support term is no larger). For ``dual_infeasible`` it is
    the ray over the variables, taken as it is. A vector of the wrong size
    or not finite gives no record, and ``unavailable`` says why.
    """
    kind = _CERTIFICATE_SOURCES[mapped][0]
    size = len(constraint_rows) if mapped == status.PRIMAL_INFEASIBLE else n
    if vector.shape != (size,):
        return None, f"CPLEX's {kind} has {vector.size} entries for {size}"
    if not np.all(np.isfinite(vector)):
        return None, f"CPLEX's {kind} is not finite"
    if mapped == status.PRIMAL_INFEASIBLE:
        farkas = np.zeros(a.shape[0])
        np.add.at(farkas, constraint_rows, -vector)
        return qp_certificate_kkt(mapped, p, q, a, l, u, farkas=farkas), None
    return qp_certificate_kkt(mapped, p, q, a, l, u, ray=vector), None


def _cplex_certificate(
    model, mapped: str, *, is_lp: bool, time_limit: float | None, elapsed: float, record
) -> tuple[dict | None, dict]:
    """The certificate check of an infeasibility claim, and what to put on ``info``.

    Returns ``(kkt_record, info)``; ``record`` turns CPLEX's vector into the
    check (:func:`_certificate_record`). CPLEX gives a certificate for LPs
    only, and only from a simplex solve that proved the claim: not when
    presolve proved it or another method (barrier, the concurrent optimizer)
    found it, when asking for it raises. The solve's certificate is used
    when it is one (``kkt.valid`` not false; a vector that fails by more
    than ``kkt.CERT_TOL`` is clearly not a certificate). Otherwise the model
    is solved again with presolve off (``preprocessing.presolve`` 0) by the
    simplex method that leaves the certificate (dual simplex for a Farkas
    proof, primal simplex for a ray), and that solve's certificate is
    recorded as it is, for the verification to judge. The second solve
    costs up to a full solve of the LP without presolve, on infeasibility
    claims only; it runs within what is left of the time limit (the
    solver's ``time_limit`` / ``timelimit``; without one it has none, since
    ``run.timeout_seconds`` is not passed to the solver). ``info``
    says what happened: ``certificate_resolved`` when the model was solved
    again, ``certificate_unavailable`` (the reason) when there is no
    certificate. The claim is already made, so any error on the way leaves
    it without a certificate rather than losing the row.
    """
    kind, method = _CERTIFICATE_SOURCES[mapped]
    if not is_lp:
        return None, {"certificate_unavailable": f"CPLEX gives a {kind} for LPs only"}
    try:
        checked, _ = record(_read_cplex_certificate(model, mapped))
    except Exception:
        checked = None  # not from a simplex solve that proved the claim
    if checked is not None and checked["valid"] is not False:
        return checked, {}
    info: dict = {}
    try:
        if not _limit_to_time_left(model, time_limit, elapsed):
            info["certificate_unavailable"] = f"no time left to solve again for a {kind}"
            return None, info
        _set_cplex_parameter(model, "lpmethod", method)
        _set_cplex_parameter(model, "preprocessing.presolve", 0)
        info["certificate_resolved"] = True
        model.solve()
        again = model.solution.get_status()
        if _map_cplex_status(again, model) != mapped:
            info["certificate_unavailable"] = (
                f"solved again for a {kind}, CPLEX ended with {_status_string(model, again)}"
            )
            return None, info
        checked, unavailable = record(_read_cplex_certificate(model, mapped))
    except Exception as exc:
        info["certificate_unavailable"] = f"no {kind}: {exc}"
        return None, info
    if unavailable is not None:
        info["certificate_unavailable"] = unavailable
    return checked, info


def _cplex_iterations(model) -> dict:
    """Iteration counts; ``iterations`` is the count of the method that ran.

    ``get_num_iterations`` counts simplex (or crossover) iterations, which is
    0 for a QP solved by barrier without crossover, the default for
    continuous QPs; the barrier count is used whenever barrier ran.
    """
    progress = model.solution.progress

    def count(method) -> int | None:
        try:
            return int(method())
        except Exception:
            return None

    simplex = count(progress.get_num_iterations)
    barrier = count(getattr(progress, "get_num_barrier_iterations", lambda: None))
    return {
        "iterations": barrier if barrier else simplex,
        "simplex_iterations": simplex,
        "barrier_iterations": barrier,
    }
