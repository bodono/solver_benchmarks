"""PIQP adapter."""

from __future__ import annotations

import time
from pathlib import Path

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import (
    IMPORT_ERRORS,
    SolverAdapter,
    SolverConfigError,
    mark_threads_ignored,
    mark_time_limit_ignored,
    module_available,
    normalize_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    threads_or_default,
    translate_eps_alias,
)
from .qp_split import (
    combine_qp_duals,
    dense_backend_requested,
    dual_from_lower_upper,
    split_qp_for_range_constraints,
    split_qp_kkt,
)


class PIQPSolverAdapter(SolverAdapter):
    solver_name = "piqp"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("piqp")

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import piqp
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "piqp", "Install with the piqp extra to use PIQP", exc
            ) from exc

        qp = problem.qp
        p, q, aeq, b, g, h_l, h_u, eq_idx, ineq_idx = split_qp_for_range_constraints(qp)
        settings = settings_with_defaults(self.settings)
        use_dense = dense_backend_requested(settings, "PIQP")
        # PIQP has no time limit; record a configured one as ignored. Its
        # thread count is ``num_threads`` (0 = PIQP's default), which
        # pop_threads also consumes, so it is put back under that name; one
        # thread unless configured, as for every adapter with a thread knob.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        normalize_max_iter(settings)
        translated = translate_eps_alias(settings)
        translated.update(_duality_gap_tolerances(settings))
        solver = piqp.DenseSolver() if use_dense else piqp.SparseSolver()
        threads_used = None
        if hasattr(solver.settings, "num_threads"):
            threads_used = threads_or_default(threads)
            settings["num_threads"] = threads_used
            threads = None
        settings.setdefault("compute_timings", True)
        _configure_piqp(solver, settings)

        start = time.perf_counter()
        if use_dense:
            solver.setup(
                p.toarray(),
                q,
                None if aeq is None else aeq.toarray(),
                b,
                None if g is None else g.toarray(),
                h_l,
                h_u,
                None,
                None,
            )
        else:
            solver.setup(p, q, aeq, b, g, h_l, h_u, None, None)
        raw_status = solver.solve()
        elapsed = time.perf_counter() - start
        if raw_status == piqp.PIQP_INVALID_SETTINGS:
            # PIQP checks its settings' ranges (eps_abs > 0, max_iter > 0,
            # ...) only when solving, and reports a failure as a status
            # without saying which one; it is a configuration error.
            raise SolverConfigError(
                "PIQP rejected a setting value (PIQP_INVALID_SETTINGS); check "
                f"the values of {sorted(settings)}"
            )

        result = solver.result
        mapped = _map_piqp_status(raw_status, piqp)
        ineq_dual = dual_from_lower_upper(
            getattr(result, "z_l", None),
            getattr(result, "z_u", None),
        )
        y = combine_qp_duals(
            int(qp["A"].shape[0]),
            eq_idx,
            getattr(result, "y", None),
            ineq_idx,
            ineq_dual,
        )
        kkt_dict = split_qp_kkt(mapped, qp, result.x, y)
        info = _info_dict(result.info)
        info["raw_status"] = info.get("status", str(raw_status))
        info["threads"] = threads_used
        mark_time_limit_ignored(info, time_limit)
        mark_threads_ignored(info, threads)
        if translated:
            info["settings_translated"] = translated
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, info.get("primal_obj")),
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_maybe_float(info.get("setup_time")),
            solve_time_seconds=_maybe_float(info.get("solve_time")),
            info=info,
            kkt=kkt_dict,
        )


def _configure_piqp(solver, settings: dict) -> None:
    for key, value in settings.items():
        if not hasattr(solver.settings, key):
            raise SolverConfigError(f"Invalid PIQP setting {key!r}")
        try:
            setattr(solver.settings, key, value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SolverConfigError(f"Invalid PIQP setting {key!r}={value!r}: {exc}") from exc


def _duality_gap_tolerances(settings: dict) -> dict:
    """Default PIQP's duality-gap tolerances to the configured ``eps_abs`` / ``eps_rel``.

    PIQP checks the duality gap by default against its own
    ``eps_duality_gap_abs`` 1e-8 and ``eps_duality_gap_rel`` 1e-9, whatever
    ``eps_abs`` / ``eps_rel`` say, so a run at a nominal 1e-4 would still
    close the gap to about 1e-8. As in the ProxQP adapter, a configured
    ``eps_abs`` supplies ``eps_duality_gap_abs`` and ``eps_rel``
    ``eps_duality_gap_rel`` (``eps_abs`` when ``eps_rel`` is 0) unless the
    gap tolerances are given; with no tolerance configured PIQP's defaults
    stay. Returns what was inserted, for ``info["settings_translated"]``.
    """
    eps_abs = settings.get("eps_abs")
    eps_rel = settings.get("eps_rel")
    translated: dict = {}

    def put(key: str, value) -> None:
        if value is not None and settings.get(key) is None:
            settings[key] = value
            translated[key] = value

    put("eps_duality_gap_abs", eps_abs)
    if eps_rel is not None and _is_zero(eps_rel):
        put("eps_duality_gap_rel", eps_abs)
    else:
        put("eps_duality_gap_rel", eps_rel)
    return translated


def _is_zero(value) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _map_piqp_status(raw_status, piqp) -> str:
    return {
        piqp.PIQP_SOLVED: status.OPTIMAL,
        piqp.PIQP_MAX_ITER_REACHED: status.MAX_ITER_REACHED,
        piqp.PIQP_PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE,
        piqp.PIQP_DUAL_INFEASIBLE: status.DUAL_INFEASIBLE,
        piqp.PIQP_NUMERICS: status.SOLVER_ERROR,
        piqp.PIQP_INVALID_SETTINGS: status.SOLVER_ERROR,
        piqp.PIQP_UNSOLVED: status.SOLVER_ERROR,
    }.get(raw_status, status.SOLVER_ERROR)


def _info_dict(info) -> dict:
    out = {}
    for key in [
        "status",
        "iter",
        "primal_obj",
        "dual_obj",
        "primal_res",
        "dual_res",
        "duality_gap",
        "duality_gap_rel",
        "run_time",
        "setup_time",
        "solve_time",
        "kkt_factor_time",
        "kkt_solve_time",
    ]:
        if hasattr(info, key):
            value = getattr(info, key)
            out[key] = str(value) if key == "status" else value
    return out


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
