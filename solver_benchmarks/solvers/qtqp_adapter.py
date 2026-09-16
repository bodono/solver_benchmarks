"""QTQP adapter."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult, to_jsonable
from solver_benchmarks.core.storage import atomic_write_text
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

from .base import (
    SolverAdapter,
    SolverUnavailable,
    mark_threads_ignored,
    mark_time_limit_ignored,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
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
        settings = _normalize_settings(settings, qtqp)
        sig = inspect.signature(qtqp.QTQP.solve)
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        solve_params = set(sig.parameters)
        translated = _translate_tolerances(settings, solve_params)
        if not accepts_any:
            unknown = sorted(k for k in settings if k not in solve_params)
            if unknown:
                raise ValueError(f"qtqp.solve() does not accept settings {unknown}; installed qtqp exposes {sorted(solve_params)}")
        a, b, z = qp_to_nonnegative_cone(qp)
        p = sp.csc_matrix(qp["P"])
        c = np.asarray(qp["q"], dtype=float)

        start = time.perf_counter()
        solver = qtqp.QTQP(a=sp.csc_matrix(a), b=b, c=c, z=z, p=p)
        solve_kwargs = dict(settings)
        if "collect_stats" in inspect.signature(solver.solve).parameters:
            solve_kwargs["collect_stats"] = True
        solution = solver.solve(**solve_kwargs)
        elapsed = time.perf_counter() - start

        raw_status = getattr(solution.status, "value", str(solution.status))
        trace = list(getattr(solution, "stats", []) or [])
        _write_trace(artifacts_dir / "trace.jsonl", trace)
        stats = pd.DataFrame(trace) if trace else pd.DataFrame()
        # qtqp reports completed IPM steps on ``Solution.iterations`` (main
        # after google-deepmind/qtqp#131). The last trace row's ``iter`` is a
        # zero-based label, one below the completed count for any solve that
        # took a step, and is kept only as the fallback for older builds.
        iterations = _maybe_int(getattr(solution, "iterations", None))
        if not stats.empty:
            last = stats.tail(1).iloc[0]
            objective = _maybe_float(last.get("pcost"))
            if iterations is None:
                iterations = _maybe_int(last.get("iter"))
            info = to_jsonable(last.to_dict())
        else:
            objective = None
            info = {}

        mapped = _map_qtqp_status(raw_status)
        cone_dict: dict = {}
        if z:
            cone_dict["z"] = int(z)
        if a.shape[0] - z:
            cone_dict["l"] = int(a.shape[0] - z)
        kkt_dict = _compute_kkt(mapped, solution, p, c, a, b, cone_dict)
        result_info = {"raw_status": raw_status, **info}
        mark_time_limit_ignored(result_info, time_limit)
        mark_threads_ignored(result_info, threads)
        if translated:
            result_info["settings_translated"] = translated
        return SolverResult(
            status=mapped,
            objective_value=objective,
            iterations=iterations,
            run_time_seconds=elapsed,
            info=result_info,
            trace=[to_jsonable(row) for row in trace],
            kkt=kkt_dict,
        )


def _map_qtqp_status(raw_status) -> str:
    return {
        "solved": status.OPTIMAL,
        "almost_solved": status.OPTIMAL_INACCURATE,
        "infeasible": status.PRIMAL_INFEASIBLE,
        "unbounded": status.DUAL_INFEASIBLE,
        "hit_max_iter": status.MAX_ITER_REACHED,
        "unfinished": status.SOLVER_ERROR,
        "failed": status.SOLVER_ERROR,
    }.get(str(raw_status), status.SOLVER_ERROR)


def _compute_kkt(mapped_status, solution, p, c, a, b, cone_dict):
    x = getattr(solution, "x", None)
    y = getattr(solution, "y", None)
    s_slack = getattr(solution, "s", None)
    if x is None:
        return None
    if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
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


# qtqp 0.0.7 replaced ``atol``/``rtol`` with Clarabel-style termination
# settings. Accept both spellings plus the harness-wide ``eps``/``eps_abs``/
# ``eps_rel`` aliases and translate to whatever the installed build takes.
_LEGACY_TOL_KEYS = ("atol", "rtol")
_NEW_TOL_KEYS = ("tol_feas", "tol_gap_abs", "tol_gap_rel")


def _translate_tolerances(settings: dict, solve_params: set[str]) -> dict:
    """Map tolerance aliases onto the installed ``solve()`` signature.

    Returns a dict describing what was translated (empty if nothing was).
    """
    eps_abs = settings.pop("eps_abs", None)
    eps_rel = settings.pop("eps_rel", None)
    eps = settings.pop("eps", None)
    if eps is not None:
        eps_abs = eps if eps_abs is None else eps_abs
        eps_rel = eps if eps_rel is None else eps_rel
    translated: dict = {}
    if "tol_feas" not in solve_params and "atol" not in solve_params:
        # unknown signature (e.g. **kwargs): pass names through unchanged
        if eps_abs is not None:
            settings.setdefault("tol_feas", float(eps_abs))
            settings.setdefault("tol_gap_abs", float(eps_abs))
        if eps_rel is not None:
            settings.setdefault("tol_gap_rel", float(eps_rel))
        return translated
    new_api = "tol_feas" in solve_params
    if new_api:
        # legacy atol/rtol -> feasibility and gap tolerances
        atol = settings.pop("atol", None)
        rtol = settings.pop("rtol", None)
        abs_tol = eps_abs if eps_abs is not None else atol
        rel_tol = eps_rel if eps_rel is not None else rtol
        if abs_tol is not None:
            settings.setdefault("tol_feas", float(abs_tol))
            settings.setdefault("tol_gap_abs", float(abs_tol))
            translated["tol_feas"] = translated["tol_gap_abs"] = float(abs_tol)
        if rel_tol is not None:
            settings.setdefault("tol_gap_rel", float(rel_tol))
            translated["tol_gap_rel"] = float(rel_tol)
    else:
        for key in _NEW_TOL_KEYS:
            settings.pop(key, None)
        if eps_abs is not None:
            settings.setdefault("atol", float(eps_abs))
            translated["atol"] = float(eps_abs)
        if eps_rel is not None:
            settings.setdefault("rtol", float(eps_rel))
            translated["rtol"] = float(eps_rel)
    return translated


def _normalize_settings(settings: dict, qtqp_module):
    linear_solver = settings.get("linear_solver")
    if isinstance(linear_solver, str):
        lookup = {
            "qdldl": "QDLDL",
            "accelerate": "ACCELERATE",
            "cholmod": "CHOLMOD",
        }
        attr = lookup.get(linear_solver.lower(), linear_solver.upper())
        settings["linear_solver"] = getattr(qtqp_module.LinearSolver, attr)
    # QTQP's enum-valued knobs are named in YAML as plain strings; map any
    # that the installed build exposes.
    for key, enum_name in (
        ("init_strategy", "InitStrategy"),
        ("equilibration_strategy", "EquilibrationStrategy"),
        ("refinement_strategy", "RefinementStrategy"),
    ):
        value = settings.get(key)
        if isinstance(value, str) and hasattr(qtqp_module, enum_name):
            settings[key] = getattr(getattr(qtqp_module, enum_name), value.upper())
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
