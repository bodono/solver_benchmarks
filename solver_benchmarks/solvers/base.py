"""Solver adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from solver_benchmarks.core.problem import ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.core import status as canonical


class SolverUnavailable(RuntimeError):
    pass


class SolverAdapter(ABC):
    solver_name: str
    supported_problem_kinds: set[str]

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = dict(settings or {})

    @classmethod
    def is_available(cls) -> bool:
        return True

    def supports(self, problem_kind: str) -> bool:
        return problem_kind in self.supported_problem_kinds

    @abstractmethod
    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        raise NotImplementedError


OLD_STATUS_MAP = {
    "optimal": canonical.OPTIMAL,
    "optimal inaccurate": canonical.OPTIMAL_INACCURATE,
    "optimal_inaccurate": canonical.OPTIMAL_INACCURATE,
    "primal infeasible": canonical.PRIMAL_INFEASIBLE,
    "primal_infeasible": canonical.PRIMAL_INFEASIBLE,
    "primal infeasible inaccurate": canonical.PRIMAL_INFEASIBLE_INACCURATE,
    "dual infeasible": canonical.DUAL_INFEASIBLE,
    "dual_infeasible": canonical.DUAL_INFEASIBLE,
    "dual infeasible inaccurate": canonical.DUAL_INFEASIBLE_INACCURATE,
    "primal or dual infeasible": canonical.PRIMAL_OR_DUAL_INFEASIBLE,
    "solver_error": canonical.SOLVER_ERROR,
    "max_iter_reached": canonical.MAX_ITER_REACHED,
    "time_limit": canonical.TIME_LIMIT,
}


def normalize_status(status: Any) -> str:
    key = str(status)
    return OLD_STATUS_MAP.get(key, OLD_STATUS_MAP.get(key.lower(), canonical.SOLVER_ERROR))


def qp_namespace(problem: ProblemData):
    return SimpleNamespace(qp_problem=problem.qp, prob_name=problem.name)
