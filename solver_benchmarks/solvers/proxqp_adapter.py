"""ProxQP adapter via ProxSuite."""

from __future__ import annotations

import time
from pathlib import Path

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import SolverAdapter, SolverUnavailable, settings_with_defaults
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
        use_dense = bool(settings.pop("dense", False) or settings.pop("backend", "") == "dense")
        settings.setdefault("compute_timings", True)
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
        if mapped in {status.OPTIMAL, status.OPTIMAL_INACCURATE} and y is not None:
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
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(info.get("objValue")),
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_proxqp_seconds(info.get("setup_time")),
            solve_time_seconds=_proxqp_seconds(info.get("solve_time")),
            info=info,
            kkt=kkt_dict,
        )


def _map_proxqp_status(raw_status, proxsuite) -> str:
    proxqp = proxsuite.proxqp
    return {
        proxqp.PROXQP_SOLVED: status.OPTIMAL,
        proxqp.PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: status.OPTIMAL_INACCURATE,
        proxqp.PROXQP_MAX_ITER_REACHED: status.MAX_ITER_REACHED,
        proxqp.PROXQP_PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE,
        proxqp.PROXQP_DUAL_INFEASIBLE: status.DUAL_INFEASIBLE,
    }.get(raw_status, status.SOLVER_ERROR)


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


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
