"""ProxQP adapter via ProxSuite."""

from __future__ import annotations

import re
import time
from pathlib import Path

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import (
    SolverAdapter,
    SolverUnavailable,
    finite_point,
    mark_threads_ignored,
    mark_time_limit_ignored,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    translate_eps_alias,
)
from .qp_split import combine_qp_duals, split_qp_for_range_constraints


class ProxQPSolverAdapter(SolverAdapter):
    solver_name = "proxqp"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import proxsuite  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import proxsuite
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the proxqp extra to use ProxQP") from exc

        qp = problem.qp
        p, q, aeq, b, g, h_l, h_u, eq_idx, ineq_idx = split_qp_for_range_constraints(qp)
        settings = settings_with_defaults(self.settings)
        # Pop both keys unconditionally so an explicit dense=True does
        # not leak "backend" into the **settings forwarded to ProxQP.
        dense_flag = settings.pop("dense", False)
        backend_value = settings.pop("backend", "")
        use_dense = bool(dense_flag) or str(backend_value).lower() == "dense"
        # ProxQP has no native time-limit / threads knobs; record the
        # configured values on info so callers know they were ignored.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        translated = translate_eps_alias(settings)
        settings.setdefault("compute_timings", True)
        gap_settings = _duality_gap_settings(settings, proxsuite)
        backend = proxsuite.proxqp.dense if use_dense else proxsuite.proxqp.sparse
        _check_settings(settings, backend.solve)
        start = time.perf_counter()
        if use_dense:
            result = proxsuite.proxqp.dense.solve(
                p.toarray(),
                q,
                None if aeq is None else aeq.toarray(),
                b,
                None if g is None else g.toarray(),
                h_l,
                h_u,
                **settings,
            )
        else:
            result = proxsuite.proxqp.sparse.solve(
                p,
                q,
                aeq,
                b,
                g,
                h_l,
                h_u,
                **settings,
            )
        elapsed = time.perf_counter() - start

        mapped = _map_proxqp_status(result.info.status, proxsuite)
        y = combine_qp_duals(
            int(qp["A"].shape[0]),
            eq_idx,
            getattr(result, "y", None),
            ineq_idx,
            getattr(result, "z", None),
        )
        kkt_dict = None
        if mapped in status.ANY_FEASIBLE and finite_point(result.x, y):
            kkt_dict = kkt.qp_residuals(
                qp["P"],
                qp["q"],
                qp["A"],
                qp["l"],
                qp["u"],
                result.x,
                y,
            )
        info = _info_dict(result.info)
        mark_time_limit_ignored(info, time_limit)
        mark_threads_ignored(info, threads)
        info.update(gap_settings)
        if translated:
            info["settings_translated"] = translated
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, info.get("objValue")),
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_proxqp_seconds(info.get("setup_time")),
            solve_time_seconds=_proxqp_seconds(info.get("solve_time")),
            info=info,
            kkt=kkt_dict,
        )


def _duality_gap_settings(settings: dict, proxsuite) -> dict:
    """Turn on ProxQP's duality-gap stopping test unless the config says not to.

    ProxSuite's default convergence test checks only the primal and dual
    residuals (``check_duality_gap=False``), so it stops at points that are
    feasible and stationary but not complementary and calls them solved
    (min x s.t. 1 <= x <= 2 came back at x = 1.1; Maros-Meszaros QPTEST 28%
    off). The adapter enables the check and, unless given, sets its
    tolerances to the run's ``eps_abs`` / ``eps_rel`` (ProxQP's own defaults,
    1e-5 and 0, when those are not set), so a ProxQP "solved" means the gap
    is closed to the same accuracy as the residuals. Returns what was used,
    for ``info``.
    """
    check = bool(settings.setdefault("check_duality_gap", True))
    if not check:
        return {"check_duality_gap": False}
    defaults = proxsuite.proxqp.Settings()
    settings.setdefault(
        "eps_duality_gap_abs", float(settings.get("eps_abs", defaults.eps_abs))
    )
    settings.setdefault(
        "eps_duality_gap_rel", float(settings.get("eps_rel", defaults.eps_rel))
    )
    return {
        "check_duality_gap": True,
        "eps_duality_gap_abs": float(settings["eps_duality_gap_abs"]),
        "eps_duality_gap_rel": float(settings["eps_duality_gap_rel"]),
    }


# Positional problem data in ProxQP's ``solve`` signatures, not settings.
_DATA_ARGUMENTS = {"H", "g", "A", "b", "C", "l", "u", "l_box", "u_box", "x", "y", "z"}


def _check_settings(settings: dict, solve) -> None:
    """Raise ValueError naming any setting ProxQP's ``solve`` does not take.

    The nanobind binding would raise a TypeError that lists every passed
    keyword without saying which one is wrong. The accepted names are read
    from the binding's signature docstring (all overloads); if it cannot be
    parsed, the settings are passed through unchecked.
    """
    accepted: set[str] = set()
    for line in (getattr(solve, "__doc__", None) or "").splitlines():
        if line.startswith("solve(") and ") ->" in line:
            inner = line[len("solve(") : line.rindex(") ->")]
            accepted.update(re.findall(r"(?:^|,\s)(\w+):", inner))
    accepted -= _DATA_ARGUMENTS
    if not accepted:
        return
    unknown = sorted(key for key in settings if key not in accepted)
    if unknown:
        raise ValueError(
            f"Unrecognized ProxQP settings {unknown}; accepted: {sorted(accepted)}"
        )


def _map_proxqp_status(raw_status, proxsuite) -> str:
    proxqp = proxsuite.proxqp
    mapping = {
        proxqp.PROXQP_SOLVED: status.OPTIMAL,
        # Only with primal_infeasibility_solving. ProxSuite documents it as
        # "the closest (in L2 sense) feasible problem is solved", and its
        # sparse backend returns it for feasible problems too, with their
        # solution; it is not an infeasibility verdict. The KKT check of the
        # returned point tells the two cases apart for kkt-verify.
        proxqp.PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: status.OPTIMAL_INACCURATE,
        proxqp.PROXQP_MAX_ITER_REACHED: status.MAX_ITER_REACHED,
        proxqp.PROXQP_PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE,
        proxqp.PROXQP_DUAL_INFEASIBLE: status.DUAL_INFEASIBLE,
    }
    # Newer ProxSuite versions added inaccurate variants of the
    # infeasibility statuses. getattr keeps us forward-compatible.
    primal_inaccurate = getattr(proxqp, "PROXQP_PRIMAL_INFEASIBLE_INACCURATE", None)
    dual_inaccurate = getattr(proxqp, "PROXQP_DUAL_INFEASIBLE_INACCURATE", None)
    not_run = getattr(proxqp, "PROXQP_NOT_RUN", None)
    if primal_inaccurate is not None:
        mapping[primal_inaccurate] = status.PRIMAL_INFEASIBLE_INACCURATE
    if dual_inaccurate is not None:
        mapping[dual_inaccurate] = status.DUAL_INFEASIBLE_INACCURATE
    if not_run is not None:
        mapping[not_run] = status.SOLVER_ERROR
    return mapping.get(raw_status, status.SOLVER_ERROR)


def _info_dict(info) -> dict:
    out = {}
    for key in [
        "status",
        "iter",
        "iter_ext",
        "objValue",
        "pri_res",
        "dua_res",
        "duality_gap",
        "run_time",
        "setup_time",
        "solve_time",
        "rho",
        "mu_eq",
        "mu_in",
    ]:
        if hasattr(info, key):
            value = getattr(info, key)
            out[key] = str(value) if key == "status" else value
    return out


def _proxqp_seconds(value):
    return None if value is None else float(value) / 1.0e6


def _maybe_int(value):
    return None if value is None else int(value)
