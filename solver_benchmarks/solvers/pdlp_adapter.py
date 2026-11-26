"""PDLP adapter using OR-Tools directly.

PDLP is a first-order LP solver, so this adapter only solves linear problems.
QP-shaped inputs are accepted when ``P`` is structurally zero because the
benchmark suite represents LP datasets through the same QP dictionary format.
"""

# ortools' linear_solver subpackage exposes linear_solver_pb2 at
# runtime (an auto-generated protobuf module) but mypy's bundled type
# stubs don't include it. Disable the attr-defined check at the
# file level rather than scattering # type: ignore comments across
# every place we import the proto module.
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData, cone_row_count
from solver_benchmarks.core.result import SolverResult

from .base import (
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    SolverUnavailable,
    ToleranceUnknown,
    coerce_int_setting,
    coerce_time_limit,
    finite_point,
    loosest_tolerance,
    module_available,
    pop_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    translate_eps_alias,
)
from .qp_split import free_row_mask, normalize_qp_bounds, scatter_rows

# Protobuf messages must be smaller than 2 GiB, including the request envelope.
MAX_PROTOBUF_BYTES = (1 << 31) - 1


# PDLP's termination tolerances (TerminationCriteria, checked on OR-Tools
# 9.15): eps_abs / eps_rel set simple_optimality_criteria's absolute and
# relative tolerances, both 1e-6 by default.
PDLP_DEFAULT_TOLERANCES = {"eps_optimal_absolute": 1e-6, "eps_optimal_relative": 1e-6}
# Words of a parameters_text that can change an optimality tolerance; such a
# text is read with OR-Tools when it is installed, and without it the
# requested tolerance is unknown.
_PARAMETERS_TEXT_TOLERANCE_WORDS = ("eps_optimal", "optimality_criteria")


class PDLPSolverAdapter(SolverAdapter):
    solver_name = "pdlp"
    supported_problem_kinds = {QP, CONE}
    solver_modules = (
        "google.protobuf.text_format",
        "ortools.linear_solver.linear_solver_pb2",
        "ortools.pdlp.solve_log_pb2",
        "ortools.pdlp.solvers_pb2",
    )

    @classmethod
    def is_available(cls) -> bool:
        return module_available("ortools", _import_ortools)

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        pop_time_limit(settings)
        settings.pop("solver_time_limit_sec", None)
        translate_eps_alias(settings)
        parameters_text = settings.get("parameters_text")
        if parameters_text:
            try:
                from google.protobuf import text_format  # noqa: F401
                from ortools.pdlp import solvers_pb2  # noqa: F401
            except IMPORT_ERRORS as exc:
                if any(word in str(parameters_text) for word in _PARAMETERS_TEXT_TOLERANCE_WORDS):
                    raise ToleranceUnknown(
                        "PDLP parameters_text may set a tolerance and OR-Tools, needed to read it, "
                        f"cannot be imported ({exc})"
                    ) from exc
            else:
                return _requested_tolerance_from_parameters(_pdlp_parameters_from_settings(settings))
        given = {
            "eps_optimal_absolute": settings.get("eps_abs"),
            "eps_optimal_relative": settings.get("eps_rel"),
        }
        return loosest_tolerance(given, PDLP_DEFAULT_TOLERANCES)

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        _import_ortools()
        if problem.kind == QP:
            return self._solve_qp_lp(problem, artifacts_dir)
        if problem.kind == CONE:
            return self._solve_linear_cone(problem, artifacts_dir)
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={"reason": f"PDLP does not support problem kind {problem.kind!r}"},
        )

    def _solve_qp_lp(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        qp = problem.qp
        p = sp.csc_matrix(qp["P"])
        if p.nnz:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": "PDLP supports LPs only; this problem has nonzero P"},
            )
        # Count exactly the entries the builder will retain: COO duplicates may
        # merge on conversion, and fully unbounded QP rows are omitted entirely.
        a = qp["A"]
        retained_rows = ~free_row_mask(qp["l"], qp["u"])
        if (
            sp.issparse(a) and a.format in {"csr", "csc"}
            and a.has_canonical_format and np.all(retained_rows)
        ):
            # Reject common large inputs without even allocating a CSR copy.
            nnz = a.nnz
        else:
            a = sp.csr_matrix(a)
            nnz = int(np.sum(np.diff(a.indptr)[retained_rows]))
        oversized = _preflight_model_size(len(qp["q"]), int(np.count_nonzero(retained_rows)), nnz)
        if oversized is not None:
            return oversized
        model = _build_lp_model_from_qp(dict(qp, A=a))
        row_count = a.shape[0]
        retained = np.flatnonzero(retained_rows)

        def compute_kkt(mapped_status, x, y):
            # PDLP's dual has one entry per retained row; free rows get zero.
            return _qp_kkt(mapped_status, qp, x, scatter_rows(row_count, retained, y))

        return _solve_model(model, self.settings, artifacts_dir, compute_kkt)

    def _solve_linear_cone(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        cone_problem = problem.cone
        p = cone_problem.get("P")
        if p is not None and sp.csc_matrix(p).nnz:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": "PDLP supports LPs only; this problem has nonzero P"},
            )
        # A cone listed with no rows (q: [], ep: 0) constrains nothing.
        cone = {
            key: value
            for key, value in dict(cone_problem["cone"]).items()
            if key in {"f", "z", "l"} or cone_row_count({key: value}) != 0
        }
        unsupported = set(cone) - {"f", "z", "l"}
        if unsupported:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={
                    "reason": "PDLP supports linear zero/nonnegative cones only",
                    "unsupported_cones": sorted(unsupported),
                },
            )
        a = cone_problem["A"]
        if not (sp.issparse(a) and a.format in {"csr", "csc"} and a.has_canonical_format):
            a = sp.csr_matrix(a)
        if sum(int(cone.get(key, 0)) for key in ("z", "f", "l")) != a.shape[0]:
            raise ValueError("Linear cone dimensions do not match A rows")
        oversized = _preflight_model_size(len(cone_problem["q"]), a.shape[0], a.nnz)
        if oversized is not None:
            return oversized
        linear_problem = dict(cone_problem, A=a, cone=cone)
        model = _build_lp_model_from_linear_cone(linear_problem)

        def compute_kkt(mapped_status, x, y):
            return _cone_kkt(mapped_status, linear_problem, x, y)

        return _solve_model(model, self.settings, artifacts_dir, compute_kkt)


def _preflight_model_size(n: int, m: int, nnz: int) -> SolverResult | None:
    # The proto2 builders explicitly set three doubles per variable and two
    # per constraint, including zeros (8 bytes + a one-byte tag each). Packed
    # coefficient/index pairs cost at least 8 + 1 bytes per retained entry.
    # Names, repeated-field tags, lengths, and the request envelope only add
    # bytes, so exceeding this lower bound is sufficient to reject the model.
    model_bytes_lower_bound = 27 * int(n) + 18 * int(m) + 9 * int(nnz)
    if model_bytes_lower_bound > MAX_PROTOBUF_BYTES:
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={
                "reason": "PDLP model request exceeds the protobuf serialized size limit (2 GiB)",
                "protobuf_model_bytes_lower_bound": model_bytes_lower_bound,
                "protobuf_limit_bytes": MAX_PROTOBUF_BYTES,
            },
        )
    return None


def _import_ortools() -> None:
    try:
        # ortools first: protobuf ('google') comes with it, so without the
        # pdlp extra the failure names ortools and reads as not installed.
        import ortools
        from ortools.linear_solver import linear_solver_pb2  # noqa: F401
        from ortools.pdlp import solve_log_pb2, solvers_pb2  # noqa: F401

        _import_model_builder_helper()
        import google.protobuf  # noqa: F401
    except IMPORT_ERRORS as exc:
        raise SolverUnavailable(
            "Install with the pdlp extra to use PDLP; "
            f"OR-Tools could not be imported ({exc})"
        ) from exc
    if _version_tuple(getattr(ortools, "__version__", "0")) < (9, 3, 0):
        version_str = getattr(ortools, "__version__", "unknown")
        raise SolverUnavailable(
            f"OR-Tools {version_str} is too old for PDLP; expected >= 9.3.0"
        )


def _build_lp_model_from_qp(qp: dict):
    from ortools.linear_solver import linear_solver_pb2

    a = sp.csr_matrix(qp["A"], dtype=float)
    q = np.asarray(qp["q"], dtype=float)
    l, u = normalize_qp_bounds(qp["l"], qp["u"])

    model = linear_solver_pb2.MPModelProto()
    model.name = "solver_benchmarks_pdlp"
    model.objective_offset = 0.0
    _add_free_variables(model, q)

    lowers = l.tolist()
    uppers = u.tolist()
    for row_index in range(a.shape[0]):
        lower = lowers[row_index]
        upper = uppers[row_index]
        if lower == -np.inf and upper == np.inf:
            continue
        constraint = model.constraint.add()
        _set_row(constraint, a, row_index)
        # Always set BOTH bounds. MPConstraintProto's proto3 default is
        # 0 for lower_bound and upper_bound, so a one-sided constraint
        # ``Ax <= u`` whose lower_bound we leave unset would silently
        # become ``0 <= Ax <= u`` instead of ``-inf <= Ax <= u``.
        constraint.lower_bound = lower
        constraint.upper_bound = upper
    return model


# The model is built with one ``extend`` per row and no variable or constraint
# names: appending entries one at a time and naming every entity took seconds
# on large LPs (and names add about a fifth to the request PDLP is sent).


def _add_free_variables(model, objective: np.ndarray) -> None:
    for coefficient in objective.tolist():
        model.variable.add(
            objective_coefficient=coefficient, lower_bound=-np.inf, upper_bound=np.inf
        )


def _set_row(constraint, a: sp.csr_matrix, row_index: int) -> None:
    start = a.indptr[row_index]
    end = a.indptr[row_index + 1]
    constraint.var_index.extend(a.indices[start:end].tolist())
    constraint.coefficient.extend(a.data[start:end].tolist())


def _build_lp_model_from_linear_cone(cone_problem: dict):
    from ortools.linear_solver import linear_solver_pb2

    a = sp.csr_matrix(cone_problem["A"], dtype=float)
    b = np.asarray(cone_problem["b"], dtype=float)
    c = np.asarray(cone_problem["q"], dtype=float)
    cone = dict(cone_problem["cone"])
    zero_dim = int(cone.get("z", 0)) + int(cone.get("f", 0))
    nonnegative_dim = int(cone.get("l", 0))
    if zero_dim + nonnegative_dim != a.shape[0]:
        raise ValueError("Linear cone dimensions do not match A rows")

    model = linear_solver_pb2.MPModelProto()
    model.name = "solver_benchmarks_pdlp_cone"
    model.objective_offset = 0.0
    _add_free_variables(model, c)

    rhs = b.tolist()
    for row_index in range(a.shape[0]):
        constraint = model.constraint.add()
        _set_row(constraint, a, row_index)
        if row_index < zero_dim:
            constraint.lower_bound = rhs[row_index]
            constraint.upper_bound = rhs[row_index]
        else:
            # SCS-style data has A x + s = b, s >= 0, hence A x <= b.
            # Always set lower_bound explicitly so the proto3 default
            # of 0 cannot leak through and silently impose Ax >= 0.
            constraint.lower_bound = -np.inf
            constraint.upper_bound = rhs[row_index]
    return model


def _solve_model(model, settings: dict[str, Any], artifacts_dir: Path, compute_kkt=None) -> SolverResult:
    from google.protobuf import text_format
    from ortools.linear_solver import linear_solver_pb2
    from ortools.pdlp import solve_log_pb2
    model_builder_helper = _import_model_builder_helper()

    settings = settings_with_defaults(settings)
    verbose = bool(settings.pop("verbose"))
    # Cross-adapter aliases live in pop_time_limit; ortools also has a
    # legacy ``solver_time_limit_sec`` spelling we still accept here for
    # back-compat with pinned configs, validated the same way (a shared
    # alias given alongside wins).
    time_limit = pop_time_limit(settings)
    legacy_time_limit = settings.pop("solver_time_limit_sec", None)
    if time_limit is None and legacy_time_limit is not None:
        time_limit = coerce_time_limit("solver_time_limit_sec", legacy_time_limit)
    translated = translate_eps_alias(settings)
    save_response = settings.pop("save_response", None)
    if save_response is None:
        save_response = False
    elif not isinstance(save_response, bool):
        raise SolverConfigError(
            f"Setting 'save_response' must be true or false (got {save_response!r})"
        )

    request = linear_solver_pb2.MPModelRequest(
        enable_internal_solver_output=verbose,
        solver_type=linear_solver_pb2.MPModelRequest.PDLP_LINEAR_PROGRAMMING,
    )
    time_limit_ignored = False
    if time_limit is not None:
        if not _set_time_limit(request, float(time_limit)):
            time_limit_ignored = True

    given = sorted(settings)
    max_iter_alias = pop_max_iter(settings, minimum=0)
    if max_iter_alias is not None:
        settings["max_iter"] = max_iter_alias
    native_limit = settings.get("iteration_limit") is not None
    if native_limit:
        # PDLP itself refuses a negative limit, with its own message.
        settings["iteration_limit"] = coerce_int_setting("iteration_limit", settings["iteration_limit"])
    parameters = _pdlp_parameters_from_settings(settings)
    # Recorded only when it is the limit PDLP runs with (parameters_text,
    # merged last, can override it).
    if (
        max_iter_alias is not None
        and not native_limit
        and int(parameters.termination_criteria.iteration_limit) == max_iter_alias
    ):
        translated["iteration_limit"] = max_iter_alias
    if settings:
        raise SolverConfigError(f"Unsupported PDLP settings: {sorted(settings)}")

    request.solver_specific_parameters = text_format.MessageToString(parameters)

    # Check before CopyFrom: constructing MPModelRequest(model=model) duplicates
    # the entire model even when it cannot cross the protobuf API boundary.
    # ByteSize itself can allocate a serialization buffer on the upb backend;
    # the sparse-data preflight rejects obvious oversize models before building.
    model_bytes = model.ByteSize()
    model_field = request.DESCRIPTOR.fields_by_name["model"].number
    request_bytes = request.ByteSize() + _protobuf_field_size(model_field, model_bytes)
    if request_bytes > MAX_PROTOBUF_BYTES:
        return SolverResult(
            status=status.SKIPPED_UNSUPPORTED,
            info={
                "reason": "PDLP model request exceeds the protobuf serialized size limit (2 GiB)",
                "protobuf_model_bytes": model_bytes,
                "protobuf_request_bytes": request_bytes,
                "protobuf_limit_bytes": MAX_PROTOBUF_BYTES,
            },
        )
    request.model.CopyFrom(model)

    start = time.perf_counter()
    response = _solve_request(model_builder_helper, linear_solver_pb2, request)
    elapsed = time.perf_counter() - start

    solve_log = solve_log_pb2.SolveLog.FromString(response.solver_specific_info)
    x = np.asarray(response.variable_value, dtype=float) if len(response.variable_value) else None
    # OR-Tools returns the dual negated relative to our QP sign convention
    # (y_qp = λ_u − λ_l). Flip here so downstream KKT helpers see the right sign.
    y = -np.asarray(response.dual_value, dtype=float) if len(response.dual_value) else None
    primal_size = len(response.variable_value)
    dual_size = len(response.dual_value)
    (artifacts_dir / "pdlp_solve_log.textproto").write_text(
        text_format.MessageToString(solve_log)
    )
    if not save_response:
        # The solution vectors as text cost seconds and tens of MB per solve
        # on large LPs; the row keeps what the harness needs from them.
        for field in ("variable_value", "dual_value", "reduced_cost"):
            response.ClearField(field)
    (artifacts_dir / "pdlp_response.textproto").write_text(
        text_format.MessageToString(response)
    )

    invalid_parameter = getattr(
        solve_log_pb2.TerminationReason, "TERMINATION_REASON_INVALID_PARAMETER", None
    )
    if invalid_parameter is not None and solve_log.termination_reason == invalid_parameter:
        # PDLP validates its parameters only when asked to solve and answers
        # with a termination reason; a bad value is a configuration error.
        raise SolverConfigError(
            f"PDLP rejected a setting value in {given}: {solve_log.termination_string}"
        )
    mapped_status = _map_status(solve_log)
    kkt_dict = None
    if compute_kkt is not None and x is not None:
        kkt_dict = compute_kkt(mapped_status, x, y)

    termination_reason = solve_log_pb2.TerminationReason.Name(solve_log.termination_reason)
    info = {
        "raw_status": termination_reason,
        "termination_reason": termination_reason,
        "termination_string": solve_log.termination_string,
        "solver_time_sec": getattr(solve_log, "solve_time_sec", None),
        "primal_solution_size": primal_size,
        "dual_solution_size": dual_size,
        # PDLP's own default is one thread; ``num_threads`` (from ``threads``
        # or parameters_text) is the count it ran with.
        "threads": int(parameters.num_threads),
    }
    if time_limit_ignored and time_limit is not None:
        info["time_limit_ignored"] = True
        info["time_limit_seconds"] = float(time_limit)
    if translated:
        info["settings_translated"] = translated
    # For infeasibility certificates OR-Tools may still fill objective_value
    # with the certificate's residual; only a solution's objective is reported.
    return SolverResult(
        status=mapped_status,
        objective_value=reported_objective(mapped_status, response.objective_value),
        iterations=_extract_iterations(solve_log),
        run_time_seconds=elapsed,
        info=info,
        kkt=kkt_dict,
    )


def _pdlp_parameters_from_settings(settings: dict[str, Any]):
    """PDLP parameters for ``settings``, which are consumed in place."""
    from google.protobuf import text_format
    from ortools.pdlp import solvers_pb2

    parameters_text = settings.pop("parameters_text", None)
    # Keep the default path pure PDLP. In particular, Glop presolve can
    # short-circuit PDLP's infeasibility detection and return
    # primal_or_dual_infeasible without a Farkas certificate.
    use_glop = bool(settings.pop("use_glop", False))
    eps_abs = settings.pop("eps_abs", None)
    eps_rel = settings.pop("eps_rel", None)
    max_iter = settings.pop("max_iter", None)
    max_iter = settings.pop("iteration_limit", max_iter)
    threads = pop_threads(settings)

    parameters = solvers_pb2.PrimalDualHybridGradientParams()
    _apply_pure_pdlp_defaults(parameters)
    parameters.presolve_options.use_glop = use_glop
    try:
        if eps_abs is not None:
            criteria = parameters.termination_criteria.simple_optimality_criteria
            criteria.eps_optimal_absolute = float(eps_abs)
        if eps_rel is not None:
            criteria = parameters.termination_criteria.simple_optimality_criteria
            criteria.eps_optimal_relative = float(eps_rel)
        if max_iter is not None:
            parameters.termination_criteria.iteration_limit = int(max_iter)
        if threads:
            # PDLP parallelizes its matrix-vector products over num_threads
            # (default 1); 0 keeps that default, as for the other adapters.
            parameters.num_threads = int(threads)
        if parameters_text:
            text_format.Merge(str(parameters_text), parameters)
    except (TypeError, ValueError, text_format.ParseError) as exc:
        raise SolverConfigError(f"Invalid PDLP setting: {exc}") from exc
    return parameters


def _requested_tolerance_from_parameters(parameters) -> RequestedTolerance:
    """The tolerances a PrimalDualHybridGradientParams asks PDLP for."""
    criteria = parameters.termination_criteria
    which = criteria.WhichOneof("optimality_criteria")
    if which == "detailed_optimality_criteria":
        detailed = criteria.detailed_optimality_criteria
        names = [field.name for field in detailed.DESCRIPTOR.fields if field.name.startswith("eps_")]
        knobs = {name: float(getattr(detailed, name)) for name in names}
    elif which == "simple_optimality_criteria":
        simple = criteria.simple_optimality_criteria
        knobs = {name: float(getattr(simple, name)) for name in PDLP_DEFAULT_TOLERANCES}
    else:
        # Neither is set: PDLP uses the (deprecated) top-level fields.
        knobs = {name: float(getattr(criteria, name)) for name in PDLP_DEFAULT_TOLERANCES}
    return loosest_tolerance(knobs, knobs)


def _apply_pure_pdlp_defaults(parameters) -> None:
    """Disable PDLP's optional warm-start helpers (Glop presolve and
    feasibility-polishing) so the benchmark sees the unaided PDLP
    behavior. Each field is gated by ``hasattr`` because OR-Tools has
    renamed these flags between releases.
    """
    if hasattr(parameters, "presolve_options") and hasattr(
        parameters.presolve_options, "use_glop"
    ):
        parameters.presolve_options.use_glop = False
    for attr in (
        "use_feasibility_polishing",
        "apply_feasibility_polishing_after_limits_reached",
        "apply_feasibility_polishing_if_solver_is_interrupted",
        "use_diagonal_qp_trust_region_solver",
    ):
        if hasattr(parameters, attr):
            setattr(parameters, attr, False)


def _qp_kkt(mapped_status, qp, x, y):
    if mapped_status in status.ANY_FEASIBLE:
        if not finite_point(x, y):
            return None
        return kkt.qp_residuals(qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x, y)
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        if y is None:
            return None
        return kkt.qp_primal_infeasibility_cert(qp["A"], qp["l"], qp["u"], y)
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        return kkt.qp_dual_infeasibility_cert(qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x)
    return None


def _cone_kkt(mapped_status, cone_problem, x, y):
    a = sp.csc_matrix(cone_problem["A"])
    b = np.asarray(cone_problem["b"], dtype=float)
    c = np.asarray(cone_problem["q"], dtype=float)
    cone = dict(cone_problem["cone"])
    p = cone_problem.get("P")
    if p is None:
        p = sp.csc_matrix((a.shape[1], a.shape[1]))
    if mapped_status in status.ANY_FEASIBLE:
        if not finite_point(x, y):
            return None
        s = b - a @ x
        return kkt.cone_residuals(p, c, a, b, cone, x, y, s)
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        if y is None:
            return None
        return kkt.cone_primal_infeasibility_cert(a, b, cone, y)
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        return kkt.cone_dual_infeasibility_cert(p, c, a, cone, x)
    return None


def _import_model_builder_helper():
    try:
        from ortools.linear_solver.python import model_builder_helper

        return model_builder_helper
    except ModuleNotFoundError:
        from ortools.model_builder.python import model_builder_helper as helper

        return helper


def _set_time_limit(request, value: float) -> bool:
    """Set the solver time limit on an MPModelRequest, returning True
    on success. Returns False if no compatible field exists on this
    OR-Tools build (the caller can then mark the request as
    time_limit_ignored rather than crashing the solve).
    """
    fields = request.DESCRIPTOR.fields_by_name
    if "solver_time_limit_sec" in fields:
        request.solver_time_limit_sec = value
        return True
    if "solver_time_limit_seconds" in fields:
        request.solver_time_limit_seconds = value
        return True
    return False


def _protobuf_field_size(field_number: int, payload_bytes: int) -> int:
    """Size of a length-delimited protobuf field without serializing its payload."""
    tag = (field_number << 3) | 2
    return (tag.bit_length() + 6) // 7 + max(1, (payload_bytes.bit_length() + 6) // 7) + payload_bytes


def _solve_request(model_builder_helper, linear_solver_pb2, request):
    try:
        solver = model_builder_helper.ModelSolverHelper()
    except TypeError:
        solver = model_builder_helper.ModelSolverHelper("PDLP")
    if hasattr(solver, "Solve"):
        return solver.Solve(request)
    serialized = solver.solve_serialized_request(request.SerializeToString())
    return linear_solver_pb2.MPSolutionResponse.FromString(serialized)


def _map_status(solve_log) -> str:
    from ortools.pdlp import solve_log_pb2

    reason = solve_log.termination_reason
    termination = solve_log_pb2.TerminationReason
    if reason == termination.TERMINATION_REASON_OPTIMAL:
        return status.OPTIMAL
    if reason == termination.TERMINATION_REASON_PRIMAL_INFEASIBLE:
        return status.PRIMAL_INFEASIBLE
    if reason == termination.TERMINATION_REASON_DUAL_INFEASIBLE:
        return status.DUAL_INFEASIBLE
    if reason == termination.TERMINATION_REASON_PRIMAL_OR_DUAL_INFEASIBLE:
        return status.PRIMAL_OR_DUAL_INFEASIBLE
    if reason == termination.TERMINATION_REASON_TIME_LIMIT:
        return status.TIME_LIMIT
    if reason in {
        termination.TERMINATION_REASON_ITERATION_LIMIT,
        termination.TERMINATION_REASON_KKT_MATRIX_PASS_LIMIT,
    }:
        return status.MAX_ITER_REACHED
    # Reasons added in newer ortools versions; getattr keeps the
    # mapping forward-compatible without crashing on older builds.
    interrupted = getattr(termination, "TERMINATION_REASON_INTERRUPTED_BY_USER", None)
    numerical = getattr(termination, "TERMINATION_REASON_NUMERICAL_ERROR", None)
    invalid_problem = getattr(termination, "TERMINATION_REASON_INVALID_PROBLEM", None)
    invalid_param = getattr(termination, "TERMINATION_REASON_INVALID_PARAMETER", None)
    if interrupted is not None and reason == interrupted:
        return status.TIME_LIMIT
    if numerical is not None and reason == numerical:
        return status.SOLVER_ERROR
    if invalid_problem is not None and reason == invalid_problem:
        return status.SOLVER_ERROR
    if invalid_param is not None and reason == invalid_param:
        return status.SOLVER_ERROR
    return status.SOLVER_ERROR


def _extract_iterations(solve_log) -> int | None:
    iteration_count = getattr(solve_log, "iteration_count", None)
    if iteration_count is not None:
        return int(iteration_count)
    iteration_stats = getattr(solve_log, "iteration_stats", None)
    if iteration_stats:
        return len(iteration_stats)
    return None


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = []
    for piece in value.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)  # type: ignore[return-value]
