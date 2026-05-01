"""PDLP adapter using OR-Tools directly.

PDLP is a first-order LP solver, so this adapter only solves linear problems.
QP-shaped inputs are accepted when ``P`` is structurally zero because the
benchmark suite represents LP datasets through the same QP dictionary format.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import (
    SolverAdapter,
    SolverUnavailable,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)

INF_BOUND = 1.0e20


class PDLPSolverAdapter(SolverAdapter):
    solver_name = "pdlp"
    supported_problem_kinds = {QP, CONE}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import google.protobuf  # noqa: F401
            import ortools
            _import_model_builder_helper()
        except ModuleNotFoundError:
            return False
        return _version_tuple(getattr(ortools, "__version__", "0")) >= (9, 3, 0)

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
        model = _build_lp_model_from_qp(qp)

        def compute_kkt(mapped_status, x, y):
            return _qp_kkt(mapped_status, qp, x, y)

        return _solve_model(model, self.settings, artifacts_dir, compute_kkt)

    def _solve_linear_cone(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        cone_problem = problem.cone
        cone = dict(cone_problem["cone"])
        unsupported = set(cone) - {"f", "z", "l"}
        if unsupported:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={
                    "reason": "PDLP supports linear zero/nonnegative cones only",
                    "unsupported_cones": sorted(unsupported),
                },
            )
        model = _build_lp_model_from_linear_cone(cone_problem)

        def compute_kkt(mapped_status, x, y):
            return _cone_kkt(mapped_status, cone_problem, x, y)

        return _solve_model(model, self.settings, artifacts_dir, compute_kkt)


def _import_ortools() -> None:
    try:
        import google.protobuf  # noqa: F401
        import ortools
        from ortools.linear_solver import linear_solver_pb2  # noqa: F401
        from ortools.pdlp import solve_log_pb2, solvers_pb2  # noqa: F401
        _import_model_builder_helper()
    except ModuleNotFoundError as exc:
        raise SolverUnavailable("Install with the pdlp extra to use PDLP") from exc
    if _version_tuple(getattr(ortools, "__version__", "0")) < (9, 3, 0):
        raise SolverUnavailable(
            f"OR-Tools {ortools.__version__} is too old for PDLP; expected >= 9.3.0"
        )


def _build_lp_model_from_qp(qp: dict):
    from ortools.linear_solver import linear_solver_pb2

    a = sp.csr_matrix(qp["A"])
    q = np.asarray(qp["q"], dtype=float)
    l = np.asarray(qp["l"], dtype=float)
    u = np.asarray(qp["u"], dtype=float)

    model = linear_solver_pb2.MPModelProto()
    model.name = "solver_benchmarks_pdlp"
    model.objective_offset = 0.0

    for index, obj_coef in enumerate(q):
        variable = linear_solver_pb2.MPVariableProto(
            objective_coefficient=float(obj_coef),
            lower_bound=-np.inf,
            upper_bound=np.inf,
            name=f"x_{index}",
        )
        model.variable.append(variable)

    for row_index in range(a.shape[0]):
        lower = float(l[row_index])
        upper = float(u[row_index])
        if lower <= -INF_BOUND and upper >= INF_BOUND:
            continue
        constraint = linear_solver_pb2.MPConstraintProto(name=f"constraint_{row_index}")
        start = a.indptr[row_index]
        end = a.indptr[row_index + 1]
        for nz_index in range(start, end):
            constraint.var_index.append(int(a.indices[nz_index]))
            constraint.coefficient.append(float(a.data[nz_index]))
        # Always set BOTH bounds. MPConstraintProto's proto3 default is
        # 0 for lower_bound and upper_bound, so a one-sided constraint
        # ``Ax <= u`` whose lower_bound we leave unset would silently
        # become ``0 <= Ax <= u`` instead of ``-inf <= Ax <= u``.
        constraint.lower_bound = lower if lower > -INF_BOUND else -np.inf
        constraint.upper_bound = upper if upper < INF_BOUND else np.inf
        model.constraint.append(constraint)
    return model


def _build_lp_model_from_linear_cone(cone_problem: dict):
    from ortools.linear_solver import linear_solver_pb2

    a = sp.csr_matrix(cone_problem["A"])
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

    for index, obj_coef in enumerate(c):
        variable = linear_solver_pb2.MPVariableProto(
            objective_coefficient=float(obj_coef),
            lower_bound=-np.inf,
            upper_bound=np.inf,
            name=f"x_{index}",
        )
        model.variable.append(variable)

    for row_index in range(a.shape[0]):
        constraint = linear_solver_pb2.MPConstraintProto(name=f"constraint_{row_index}")
        start = a.indptr[row_index]
        end = a.indptr[row_index + 1]
        for nz_index in range(start, end):
            constraint.var_index.append(int(a.indices[nz_index]))
            constraint.coefficient.append(float(a.data[nz_index]))
        if row_index < zero_dim:
            constraint.lower_bound = float(b[row_index])
            constraint.upper_bound = float(b[row_index])
        else:
            # SCS-style data has A x + s = b, s >= 0, hence A x <= b.
            # Always set lower_bound explicitly so the proto3 default
            # of 0 cannot leak through and silently impose Ax >= 0.
            constraint.lower_bound = -np.inf
            constraint.upper_bound = float(b[row_index])
        model.constraint.append(constraint)
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
    # back-compat with pinned configs.
    time_limit = pop_time_limit(settings)
    if time_limit is None:
        time_limit = settings.pop("solver_time_limit_sec", None)
    threads = pop_threads(settings)

    request = linear_solver_pb2.MPModelRequest(
        model=model,
        enable_internal_solver_output=verbose,
        solver_type=linear_solver_pb2.MPModelRequest.PDLP_LINEAR_PROGRAMMING,
    )
    time_limit_ignored = False
    if time_limit is not None:
        if not _set_time_limit(request, float(time_limit)):
            time_limit_ignored = True

    parameters = _pdlp_parameters_from_settings(settings)
    if settings:
        raise ValueError(f"Unsupported PDLP settings: {sorted(settings)}")

    request.solver_specific_parameters = text_format.MessageToString(parameters)

    start = time.perf_counter()
    response = _solve_request(model_builder_helper, linear_solver_pb2, request)
    elapsed = time.perf_counter() - start

    solve_log = solve_log_pb2.SolveLog.FromString(response.solver_specific_info)
    (artifacts_dir / "pdlp_solve_log.textproto").write_text(
        text_format.MessageToString(solve_log)
    )
    (artifacts_dir / "pdlp_response.textproto").write_text(
        text_format.MessageToString(response)
    )

    mapped_status = _map_status(solve_log)
    x = np.asarray(response.variable_value, dtype=float) if len(response.variable_value) else None
    # OR-Tools returns the dual negated relative to our QP sign convention
    # (y_qp = λ_u − λ_l). Flip here so downstream KKT helpers see the right sign.
    y = -np.asarray(response.dual_value, dtype=float) if len(response.dual_value) else None
    kkt_dict = None
    if compute_kkt is not None and x is not None:
        try:
            kkt_dict = compute_kkt(mapped_status, x, y)
        except Exception:
            kkt_dict = None

    info = {
        "termination_reason": solve_log_pb2.TerminationReason.Name(
            solve_log.termination_reason
        ),
        "termination_string": solve_log.termination_string,
        "solver_time_sec": getattr(solve_log, "solve_time_sec", None),
        "primal_solution_size": len(response.variable_value),
        "dual_solution_size": len(response.dual_value),
    }
    if time_limit_ignored:
        info["time_limit_ignored"] = True
        info["time_limit_seconds"] = float(time_limit)
    if threads is not None:
        # PDLP is single-threaded by design; surface so callers can detect.
        info["threads_ignored"] = True
        info["threads_requested"] = int(threads)
    # Gate objective on a solution-bearing status; for infeasibility
    # certs OR-Tools may still populate objective_value with the
    # certificate's residual.
    objective_present = mapped_status in status.SOLUTION_PRESENT or (
        mapped_status == status.OPTIMAL_INACCURATE
    )
    return SolverResult(
        status=mapped_status,
        objective_value=float(response.objective_value) if objective_present else None,
        iterations=_extract_iterations(solve_log),
        run_time_seconds=elapsed,
        info=info,
        kkt=kkt_dict,
    )


def _pdlp_parameters_from_settings(settings: dict[str, Any]):
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

    parameters = solvers_pb2.PrimalDualHybridGradientParams()
    _apply_pure_pdlp_defaults(parameters)
    parameters.presolve_options.use_glop = use_glop
    if eps_abs is not None:
        criteria = parameters.termination_criteria.simple_optimality_criteria
        criteria.eps_optimal_absolute = float(eps_abs)
    if eps_rel is not None:
        criteria = parameters.termination_criteria.simple_optimality_criteria
        criteria.eps_optimal_relative = float(eps_rel)
    if max_iter is not None:
        parameters.termination_criteria.iteration_limit = int(max_iter)
    if parameters_text:
        text_format.Parse(str(parameters_text), parameters)
    return parameters


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
    if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
        if y is None:
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
    if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
        if y is None:
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
        from ortools.model_builder.python import model_builder_helper

        return model_builder_helper


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
