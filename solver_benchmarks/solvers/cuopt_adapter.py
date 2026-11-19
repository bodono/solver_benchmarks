"""NVIDIA cuOpt adapter (GPU solver: PDLP for LPs, barrier for QPs).

cuOpt solves ``min c'x + (1/2) x'(Q + Q')x`` subject to
``cl <= A x <= cu`` and ``vl <= x <= vu`` on the GPU. The harness QP form
``min (1/2) x'P x + q'x, l <= A x <= u`` maps onto it with ``Q = P / 2``
and free variables, so the objective is identical.

Nothing is checked at run time. cuOpt's duals have the opposite sign to the
harness convention (checked numerically on an active-bound LP), hence the
default ``dual_sign: -1``; the setting overrides it if a release flips it,
which would otherwise show up as large ``dual_res_rel`` on every solve. The
``Q = P / 2`` objective mapping and the termination-status names were checked
by hand on a GPU with ``probe_cuopt`` in ``tools/bench_modal.py`` and are
pinned against a fake cuOpt in ``tests/test_cuopt_adapter.py``. Termination
names are matched against a static table across spellings (``_STATUS_MAP``);
a name it does not know becomes ``solver_error`` and the name itself is kept
on ``info["termination"]`` and ``info["raw_status"]``. The Python API has changed across releases, so
rerun the probe after upgrading cuOpt.

The adapter turns on PDLP's infeasibility detection (cuOpt's
``infeasibility_detection``, off by cuOpt's default) unless the settings
give it: off, PDLP never stops on an infeasible problem, which is right only
for a problem known in advance to be feasible, and a user cannot know that.
``info["infeasibility_detection"]`` is the value the solve ran with. cuOpt's
concurrent dual simplex and barrier detect infeasibility either way.

Infeasibility claims carry no certificate: cuOpt's API exposes none. Its
``Solution`` has no ray or Farkas getter, and on a PDLP infeasibility stop
it returns the last (or averaged) iterate, not the iterate difference its
infeasibility test is made on, which stays inside the solver; the dual
simplex's Farkas computation is switched off in its source (checked on
NVIDIA/cuopt main, 2026-11). A cuOpt infeasibility claim therefore cannot
be verified and counts as unsolved unless ``--allow-missing``.
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
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    coerce_tolerance,
    finite_point,
    loosest_tolerance,
    module_available,
    pop_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
)
from .qp_split import normalize_qp_bounds


def _import_cuopt():
    try:
        from cuopt import linear_programming  # noqa: F401
        from cuopt.linear_programming import data_model, solver, solver_settings  # noqa: F401
    except IMPORT_ERRORS as exc:
        raise solver_import_failed(
            "cuopt", "Install cuopt-cu12 (NVIDIA pip index) on a CUDA machine to use cuOpt", exc
        ) from exc
    return data_model, solver, solver_settings


def _termination_name(solution) -> str:
    """Best-effort readable termination status across cuOpt versions."""
    term = solution.get_termination_status()
    name = getattr(term, "name", None)
    if name:
        return str(name)
    try:
        reason = solution.get_termination_reason()
        if reason:
            return str(reason)
    except Exception:  # pragma: no cover
        pass
    return str(term)


_STATUS_MAP = {
    "Optimal": status.OPTIMAL,
    "OPTIMAL": status.OPTIMAL,
    "PrimalInfeasible": status.PRIMAL_INFEASIBLE,
    "PRIMALINFEASIBLE": status.PRIMAL_INFEASIBLE,
    "PRIMAL_INFEASIBLE": status.PRIMAL_INFEASIBLE,
    "DualInfeasible": status.DUAL_INFEASIBLE,
    "DUALINFEASIBLE": status.DUAL_INFEASIBLE,
    "DUAL_INFEASIBLE": status.DUAL_INFEASIBLE,
    "TimeLimit": status.TIME_LIMIT,
    "TIMELIMIT": status.TIME_LIMIT,
    "TIME_LIMIT": status.TIME_LIMIT,
    "IterationLimit": status.MAX_ITER_REACHED,
    "ITERATIONLIMIT": status.MAX_ITER_REACHED,
    "ITERATION_LIMIT": status.MAX_ITER_REACHED,
    "UnboundedOrInfeasible": status.PRIMAL_OR_DUAL_INFEASIBLE,
    "UNBOUNDEDORINFEASIBLE": status.PRIMAL_OR_DUAL_INFEASIBLE,
    "UNBOUNDED_OR_INFEASIBLE": status.PRIMAL_OR_DUAL_INFEASIBLE,
    "FeasibleFound": status.OPTIMAL_INACCURATE,
    "FEASIBLEFOUND": status.OPTIMAL_INACCURATE,
}


def _map_status(name: str) -> str:
    key = name.split(".")[-1]
    if key in _STATUS_MAP:
        return _STATUS_MAP[key]
    compact = key.replace("_", "").upper()
    for cand, mapped in _STATUS_MAP.items():
        if cand.replace("_", "").upper() == compact:
            return mapped
    return status.SOLVER_ERROR


# cuOpt's six optimality tolerances and their default (cuOpt documentation;
# not checked here: cuOpt needs an NVIDIA GPU, confirm on the cuOpt image).
CUOPT_TOLERANCE_NAMES = tuple(
    f"{scale}_{kind}_tolerance" for scale in ("absolute", "relative") for kind in ("primal", "dual", "gap")
)
CUOPT_DEFAULT_TOLERANCE = 1e-4


class CuOptSolverAdapter(SolverAdapter):
    solver_name = "cuopt"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("cuopt", _import_cuopt)

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        values: dict = {}
        # The order solve() applies them in: eps sets all six, then eps_abs /
        # eps_rel their three, then native settings in the order given
        # (optimality_tolerance sets all six).
        eps = settings.pop("eps", None)
        if eps is not None:
            values.update(dict.fromkeys(CUOPT_TOLERANCE_NAMES, coerce_tolerance("eps", eps)))
        for alias, scale in (("eps_abs", "absolute"), ("eps_rel", "relative")):
            value = settings.pop(alias, None)
            if value is not None:
                for name in CUOPT_TOLERANCE_NAMES:
                    if name.startswith(scale):
                        values[name] = coerce_tolerance(alias, value)
        for key, value in settings.items():
            if value is None:
                continue
            if key == "optimality_tolerance":
                values.update(dict.fromkeys(CUOPT_TOLERANCE_NAMES, coerce_tolerance(key, value)))
            elif key in CUOPT_TOLERANCE_NAMES:
                values[key] = coerce_tolerance(key, value)
        return loosest_tolerance(values, dict.fromkeys(CUOPT_TOLERANCE_NAMES, CUOPT_DEFAULT_TOLERANCE))

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        data_model, solver, solver_settings = _import_cuopt()
        qp = problem.qp
        settings = settings_with_defaults(self.settings)
        verbose = bool(settings.pop("verbose", False))
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        eps = settings.pop("eps", None)
        eps_abs = settings.pop("eps_abs", None)
        eps_rel = settings.pop("eps_rel", None)
        for alias, value in (("eps", eps), ("eps_abs", eps_abs), ("eps_rel", eps_rel)):
            if value is not None:
                coerce_tolerance(alias, value)  # refuses a boolean or a non-number
        # cuOpt reports duals with the opposite sign to the harness QP form
        # (checked numerically on an active-bound LP); override with dual_sign.
        dual_sign = float(settings.pop("dual_sign", -1.0))

        p = sp.csc_matrix(qp["P"])
        a = sp.csr_matrix(qp["A"])
        q = np.asarray(qp["q"], dtype=np.float64)
        l, u = normalize_qp_bounds(qp["l"], qp["u"])
        n = a.shape[1]

        model = data_model.DataModel()
        model.set_csr_constraint_matrix(
            a.data.astype(np.float64), a.indices.astype(np.int32), a.indptr.astype(np.int32)
        )
        model.set_constraint_lower_bounds(l)
        model.set_constraint_upper_bounds(u)
        model.set_objective_coefficients(q)
        model.set_variable_lower_bounds(np.full(n, -np.inf))
        model.set_variable_upper_bounds(np.full(n, np.inf))
        if p.nnz > 0:
            half = sp.csr_matrix(p * 0.5)  # cuOpt symmetrizes as Q + Q', so Q = P/2
            model.set_quadratic_objective_matrix(
                half.data.astype(np.float64), half.indices.astype(np.int32), half.indptr.astype(np.int32)
            )

        translated: dict = {}
        max_iter = pop_max_iter(settings, minimum=0)
        if max_iter is not None:
            if "iteration_limit" not in settings:
                settings["iteration_limit"] = max_iter
                translated["iteration_limit"] = max_iter
        # On unless configured: see the module docstring. A configured value
        # is cuOpt's to accept or refuse, like any other setting.
        settings.setdefault("infeasibility_detection", True)

        opts = solver_settings.SolverSettings()
        # ``eps`` sets all six of cuOpt's tolerances; ``eps_abs`` / ``eps_rel``
        # set the absolute and relative primal, dual and gap tolerances
        # independently (an explicit alias wins over ``eps``).
        try:
            if eps is not None:
                opts.set_optimality_tolerance(float(eps))
                translated["optimality_tolerance"] = float(eps)
            for alias, prefix in ((eps_abs, "absolute"), (eps_rel, "relative")):
                if alias is not None:
                    for kind in ("primal", "dual", "gap"):
                        opts.set_parameter(f"{prefix}_{kind}_tolerance", float(alias))
                        translated[f"{prefix}_{kind}_tolerance"] = float(alias)
        except (TypeError, ValueError) as exc:
            raise SolverConfigError(f"Invalid cuOpt tolerance setting: {exc}") from exc
        if time_limit is not None:
            opts.set_parameter("time_limit", float(time_limit))
        if threads is not None:
            opts.set_parameter("num_cpu_threads", int(threads))
        for key, value in settings.items():
            try:
                opts.set_parameter(key, value)
            except Exception as exc:
                raise SolverConfigError(f"Invalid cuOpt setting {key!r}={value!r}: {exc}") from exc
        if verbose:
            try:
                opts.set_parameter("log_to_console", True)
            except Exception:  # pragma: no cover
                pass

        start = time.perf_counter()
        solution = solver.Solve(model, opts)
        elapsed = time.perf_counter() - start

        term_name = _termination_name(solution)
        mapped = _map_status(term_name)
        info = {
            "raw_status": term_name,
            "termination": term_name,
            # GPU solver: ``threads`` sets only its CPU-side num_cpu_threads.
            "threads": threads,
            "infeasibility_detection": settings["infeasibility_detection"],
            "solve_time": _maybe_float(solution.get_solve_time()),
            "primal_objective": _maybe_float(solution.get_primal_objective()),
            "dual_objective": _maybe_float(solution.get_dual_objective()),
        }
        if translated:
            info["settings_translated"] = translated
        try:
            info["lp_stats"] = {k: _jsonable(v) for k, v in dict(solution.get_lp_stats()).items()}
        except Exception:  # pragma: no cover
            pass
        x = y = None
        kkt_dict = None
        if mapped in status.ANY_FEASIBLE:
            x = np.asarray(solution.get_primal_solution(), dtype=np.float64)
            y = np.asarray(solution.get_dual_solution(), dtype=np.float64)
            if x.size == n and y.size == a.shape[0] and finite_point(x, y):
                kkt_dict = kkt.qp_residuals(p, q, sp.csc_matrix(a), l, u, x, dual_sign * y)
        iterations = None
        stats = info.get("lp_stats") or {}
        for key in ("nb_iterations", "num_iterations", "iterations"):
            if key in stats:
                iterations = _maybe_int(stats[key])
                break
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, info["primal_objective"]),
            iterations=iterations,
            run_time_seconds=elapsed,
            solve_time_seconds=info["solve_time"],
            info=info,
            kkt=kkt_dict,
        )


def _maybe_float(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _maybe_int(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _jsonable(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_jsonable(x) for x in v]
    return v
