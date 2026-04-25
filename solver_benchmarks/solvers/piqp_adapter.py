"""PIQP adapter."""

from __future__ import annotations

from pathlib import Path
import time

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from .base import SolverAdapter, SolverUnavailable, settings_with_defaults
from .qp_split import combine_qp_duals, dual_from_lower_upper, split_qp_for_range_constraints


class PIQPSolverAdapter(SolverAdapter):
    solver_name = "piqp"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import piqp  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import piqp
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the piqp extra to use PIQP") from exc

        qp = problem.qp
        p, q, aeq, b, g, h_l, h_u, eq_idx, ineq_idx = split_qp_for_range_constraints(qp)
        settings = settings_with_defaults(self.settings)
        use_dense = bool(settings.pop("dense", False) or settings.pop("backend", "") == "dense")
        solver = piqp.DenseSolver() if use_dense else piqp.SparseSolver()
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
        kkt_dict = None
        if mapped == status.OPTIMAL and y is not None:
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
            objective_value=_maybe_float(info.get("primal_obj")),
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_maybe_float(info.get("setup_time")),
            solve_time_seconds=_maybe_float(info.get("solve_time")),
            info=info,
            kkt=kkt_dict,
        )


def _configure_piqp(solver, settings: dict) -> None:
    if "time_limit" in settings:
        settings.pop("time_limit")
    if "time_limit_sec" in settings:
        settings.pop("time_limit_sec")
    for key, value in settings.items():
        if not hasattr(solver.settings, key):
            raise ValueError(f"Invalid PIQP setting {key!r}")
        setattr(solver.settings, key, value)


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
