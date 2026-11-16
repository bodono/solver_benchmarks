"""QTQP adapter."""

from __future__ import annotations

import inspect
import json
import numbers
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.atomic import atomic_write_text
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult, to_jsonable
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

from .base import (
    SolverAdapter,
    SolverConfigError,
    SolverUnavailable,
    finite_point,
    mark_threads_ignored,
    mark_time_limit_ignored,
    normalize_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    translate_tolerance_aliases,
)


class QTQPSolverAdapter(SolverAdapter):
    solver_name = "qtqp"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import qtqp  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import qtqp
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install QTQP to use the QTQP adapter") from exc

        qp = problem.qp
        settings = settings_with_defaults(self.settings)
        # QTQP exposes neither a time-limit knob nor a thread-count
        # setting; record the configured values on info so callers can
        # detect they were ignored rather than silently dropping them.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        normalize_max_iter(settings)
        settings = _normalize_settings(settings, qtqp)
        sig = inspect.signature(qtqp.QTQP.solve)
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        solve_params = set(sig.parameters)
        translated = _translate_tolerances(settings, solve_params)
        if not accepts_any:
            unknown = sorted(k for k in settings if k not in solve_params)
            if unknown:
                raise SolverConfigError(
                    f"qtqp.solve() does not accept settings {unknown}; installed qtqp "
                    f"exposes {sorted(solve_params)}"
                )
        _check_setting_values(settings, translated)
        a, b, z = qp_to_nonnegative_cone(qp)
        p = sp.csc_matrix(qp["P"])
        c = np.asarray(qp["q"], dtype=float)

        # Per-iteration statistics (collect_stats) are diagnostics computed
        # inside the timed solve, 2-5 ms per solve on small problems; they are
        # collected only when the config asks, and the result's objective
        # comes from the returned point instead.
        solve_kwargs = dict(settings)
        start = time.perf_counter()
        try:
            solver = qtqp.QTQP(a=sp.csc_matrix(a), b=b, c=c, z=z, p=p)
            solution = solver.solve(**solve_kwargs)
        except ImportError as exc:
            # qtqp imports a linear-solver backend (qdldl, cholmod, ...) when
            # the solve starts; a missing one is a broken environment.
            raise SolverUnavailable(
                f"QTQP linear_solver backend could not be imported: {exc}"
            ) from exc
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            # qtqp raises ValueError for data it cannot take (non-finite or
            # non-symmetric P from the constructor; a non-convex P, from the
            # KKT assembly inside solve()) and for failed factorizations: the
            # solver failing on this problem. The setting values solve()
            # would refuse with a ValueError were checked above, so a bad
            # value still raises as a configuration error.
            error_info = {
                "raw_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "threads": None,
            }
            mark_time_limit_ignored(error_info, time_limit)
            mark_threads_ignored(error_info, threads)
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start,
                info=error_info,
            )
        elapsed = time.perf_counter() - start

        raw_status = getattr(solution.status, "value", str(solution.status))
        trace = list(getattr(solution, "stats", []) or [])
        _write_trace(artifacts_dir / "trace.jsonl", trace)
        # qtqp reports completed IPM steps on ``Solution.iterations`` (main
        # after google-deepmind/qtqp#131). The last trace row's ``iter`` is a
        # zero-based label, one below the completed count for any solve that
        # took a step, and is kept only as the fallback for older builds.
        iterations = _maybe_int(getattr(solution, "iterations", None))
        objective = _point_objective(getattr(solution, "x", None), p, c)
        if trace:
            # Imported only for a trace (collect_stats): workers run without pandas.
            import pandas as pd

            last = pd.DataFrame(trace).tail(1).iloc[0]
            if objective is None:
                objective = _maybe_float(last.get("pcost"))
            if iterations is None:
                iterations = _maybe_int(last.get("iter"))
            info = to_jsonable(last.to_dict())
        else:
            info = {}

        mapped = _map_qtqp_status(
            raw_status,
            hit_iteration_limit=_hit_iteration_limit(solver, iterations, solve_kwargs, sig),
        )
        cone_dict: dict = {}
        if z:
            cone_dict["z"] = int(z)
        if a.shape[0] - z:
            cone_dict["l"] = int(a.shape[0] - z)
        kkt_dict = _compute_kkt(mapped, solution, p, c, a, b, cone_dict)
        result_info = {"raw_status": raw_status, **info, "threads": None}
        mark_time_limit_ignored(result_info, time_limit)
        mark_threads_ignored(result_info, threads)
        if translated:
            result_info["settings_translated"] = translated
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, objective),
            iterations=iterations,
            run_time_seconds=elapsed,
            info=result_info,
            trace=[to_jsonable(row) for row in trace],
            kkt=kkt_dict,
        )


def _is_integer(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


# The ranges qtqp's solve() enforces, as (description, test) per setting. It
# checks them with asserts (gone under ``python -O``, and with no message) or
# with ValueErrors raised from the same call as its numerical failures, so the
# adapter checks them first and names the setting.
_SETTING_RANGES = {
    **{
        key: (">= 0", lambda v: _is_number(v) and v >= 0)
        for key in (
            "tol_feas", "tol_gap_abs", "tol_gap_rel", "tol_infeas_abs", "tol_infeas_rel",
            "min_static_regularization", "linear_solver_atol", "linear_solver_rtol",
            "atol", "rtol",
        )
    },
    "certificate_ktratio": (">= 1", lambda v: _is_number(v) and v >= 1),
    "step_size_scale": ("in (0, 1)", lambda v: _is_number(v) and 0 < v < 1),
    "max_iter": ("an integer >= 1", lambda v: _is_integer(v) and v >= 1),
    "max_iterative_refinement_steps": ("an integer >= 1", lambda v: _is_integer(v) and v >= 1),
    "max_centrality_correctors": ("an integer >= 0", lambda v: _is_integer(v) and v >= 0),
}


def _check_setting_values(settings: dict, translated: dict) -> None:
    """Raise SolverConfigError naming a setting whose value qtqp would refuse."""
    checks = dict(_SETTING_RANGES)
    # gmres_restart is used, and checked, only with GMRES refinement.
    strategy = settings.get("refinement_strategy")
    if getattr(strategy, "name", strategy) != "RICHARDSON":
        checks["gmres_restart"] = ("an integer >= 1", lambda v: _is_integer(v) and v >= 1)
    # warm_start_threshold is checked only when there is a warm start.
    if settings.get("warm_start") is not None:
        checks["warm_start_threshold"] = (
            "positive and finite",
            lambda v: _is_number(v) and 0 < v < np.inf,
        )
    for key, value in settings.items():
        if key in checks:
            description, test = checks[key]
            if not test(value):
                source = " (set from eps_abs / eps_rel / eps)" if key in translated else ""
                raise SolverConfigError(
                    f"Invalid QTQP setting {key!r}={value!r}{source}: must be {description}"
                )


def _point_objective(x, p, c) -> float | None:
    """``0.5 x'Px + c'x`` at the returned point, or None if it is not finite."""
    if not finite_point(x):
        return None
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ (p @ x) + c @ x)


def _map_qtqp_status(raw_status, *, hit_iteration_limit: bool = False) -> str:
    """Canonical status for a qtqp ``SolutionStatus`` value.

    qtqp returns ``almost_solved`` when it salvages its best iterate after
    running out of iterations or after a numerical breakdown. The first is a
    solve stopped by its iteration limit and is reported as such; the second
    is an inaccurate solution. qtqp also answers ``almost_solved`` when
    ``max_iter`` is exactly the number of iterations the solve needs, so that
    converged point is reported as ``max_iter_reached`` too (OSQP would say
    optimal there); ``bench kkt-verify --promote`` recovers it.
    """
    if str(raw_status) == "almost_solved" and hit_iteration_limit:
        return status.MAX_ITER_REACHED
    return {
        "solved": status.OPTIMAL,
        "almost_solved": status.OPTIMAL_INACCURATE,
        "infeasible": status.PRIMAL_INFEASIBLE,
        "unbounded": status.DUAL_INFEASIBLE,
        "hit_max_iter": status.MAX_ITER_REACHED,
        "unfinished": status.SOLVER_ERROR,
        "failed": status.SOLVER_ERROR,
    }.get(str(raw_status), status.SOLVER_ERROR)


def _hit_iteration_limit(solver, iterations, solve_kwargs, signature) -> bool:
    """Whether the solve used up its ``max_iter`` attempts.

    qtqp keeps the zero-based index of its last attempt on ``solver.it``; the
    completed-step count can be lower than ``max_iter`` when steps are
    retried, so it is only the fallback.
    """
    max_iter = solve_kwargs.get("max_iter")
    if max_iter is None:
        parameter = signature.parameters.get("max_iter")
        default = None if parameter is None else parameter.default
        max_iter = None if default is inspect.Parameter.empty else default
    if max_iter is None:
        return False
    last_attempt = getattr(solver, "it", None)
    if isinstance(last_attempt, int):
        return last_attempt + 1 >= int(max_iter)
    return iterations is not None and iterations >= int(max_iter)


def _compute_kkt(mapped_status, solution, p, c, a, b, cone_dict):
    x = getattr(solution, "x", None)
    y = getattr(solution, "y", None)
    s_slack = getattr(solution, "s", None)
    if x is None:
        return None
    if mapped_status in status.ANY_FEASIBLE:
        if not finite_point(x, y, s_slack):
            return None
        return kkt.cone_residuals(p, c, a, b, cone_dict, x, y, s_slack)
    if mapped_status in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}:
        if y is None:
            return None
        return kkt.cone_primal_infeasibility_cert(a, b, cone_dict, y)
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        return kkt.cone_dual_infeasibility_cert(p, c, a, cone_dict, x)
    return None


# qtqp 0.0.7 replaced ``atol``/``rtol`` with Clarabel-style termination
# settings. Accept both spellings plus the harness-wide ``eps``/``eps_abs``/
# ``eps_rel`` aliases and translate to whatever the installed build takes.
_NEW_TOL_KEYS = ("tol_feas", "tol_gap_abs", "tol_gap_rel")


def _translate_tolerances(settings: dict, solve_params: set[str]) -> dict:
    """Map tolerance aliases onto the installed ``solve()`` signature.

    Returns a dict describing what was translated (empty if nothing was).
    An alias supplies a setting only when the native name is absent, and only
    then is it recorded: the metadata must report what the solver actually
    received, so an explicit native value is never overwritten or misreported.
    """
    if "tol_feas" not in solve_params and "atol" not in solve_params:
        # unknown signature (e.g. **kwargs): pass names through unchanged
        return translate_tolerance_aliases(settings, ("tol_feas", "tol_gap_abs"), ("tol_gap_rel",))
    if "tol_feas" in solve_params:
        # legacy atol/rtol -> feasibility and gap tolerances; the eps aliases
        # win over them.
        atol = settings.pop("atol", None)
        rtol = settings.pop("rtol", None)
        translated = translate_tolerance_aliases(
            settings, ("tol_feas", "tol_gap_abs"), ("tol_gap_rel",)
        )
        legacy = {}
        if atol is not None:
            legacy["eps_abs"] = atol
        if rtol is not None:
            legacy["eps_rel"] = rtol
        if legacy:
            settings.update(legacy)
            translated.update(
                translate_tolerance_aliases(settings, ("tol_feas", "tol_gap_abs"), ("tol_gap_rel",))
            )
        return translated
    for key in _NEW_TOL_KEYS:
        settings.pop(key, None)
    return translate_tolerance_aliases(settings, ("atol",), ("rtol",))


def _enum_member(enum_type, value, key: str):
    """The member of ``enum_type`` named ``value`` (any case), else SolverConfigError."""
    members = {member.name.lower(): member for member in enum_type}
    member = members.get(str(value).strip().lower())
    if member is None:
        raise SolverConfigError(
            f"Unknown QTQP {key} {value!r}; expected one of: {', '.join(sorted(members))}"
        )
    return member


def _normalize_settings(settings: dict, qtqp_module):
    # QTQP's enum-valued knobs are named in YAML as plain strings; map any
    # that the installed build exposes.
    for key, enum_name in (
        ("linear_solver", "LinearSolver"),
        ("init_strategy", "InitStrategy"),
        ("equilibration_strategy", "EquilibrationStrategy"),
        ("refinement_strategy", "RefinementStrategy"),
    ):
        value = settings.get(key)
        if isinstance(value, str) and hasattr(qtqp_module, enum_name):
            settings[key] = _enum_member(getattr(qtqp_module, enum_name), value, key)
    return settings


def _write_trace(path: Path, trace: list[dict]) -> None:
    if not trace:
        path.unlink(missing_ok=True)
        return
    body = "".join(json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in trace)
    atomic_write_text(path, body)


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
