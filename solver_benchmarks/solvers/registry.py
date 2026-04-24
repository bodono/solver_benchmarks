"""Solver registry."""

from __future__ import annotations

from .base import SolverAdapter
from .clarabel_adapter import ClarabelSolverAdapter
from .legacy_optional import GurobiSolverAdapter, MosekSolverAdapter
from .osqp_adapter import OSQPSolverAdapter
from .pdlp_adapter import PDLPSolverAdapter
from .qtqp_adapter import QTQPSolverAdapter
from .scs_adapter import SCSSolverAdapter


SOLVERS: dict[str, type[SolverAdapter]] = {
    "clarabel": ClarabelSolverAdapter,
    "gurobi": GurobiSolverAdapter,
    "mosek": MosekSolverAdapter,
    "osqp": OSQPSolverAdapter,
    "pdlp": PDLPSolverAdapter,
    "qtqp": QTQPSolverAdapter,
    "scs": SCSSolverAdapter,
}


def get_solver(name: str) -> type[SolverAdapter]:
    key = name.lower()
    try:
        return SOLVERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(SOLVERS))
        raise KeyError(f"Unknown solver {name!r}. Available: {available}") from exc


def list_solvers() -> list[str]:
    return sorted(SOLVERS)
