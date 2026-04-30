"""Solver registry."""

from __future__ import annotations

from .base import SolverAdapter
from .clarabel_adapter import ClarabelSolverAdapter
from .cplex_adapter import CPLEXSolverAdapter
from .gurobi_adapter import GurobiSolverAdapter
from .highs_adapter import HighsSolverAdapter
from .mosek_adapter import MosekSolverAdapter
from .osqp_adapter import OSQPSolverAdapter
from .pdlp_adapter import PDLPSolverAdapter
from .piqp_adapter import PIQPSolverAdapter
from .proxqp_adapter import ProxQPSolverAdapter
from .qtqp_adapter import QTQPSolverAdapter
from .scs_adapter import SCSSolverAdapter
from .sdpa_adapter import SDPASolverAdapter

SOLVERS: dict[str, type[SolverAdapter]] = {
    "clarabel": ClarabelSolverAdapter,
    "cplex": CPLEXSolverAdapter,
    "gurobi": GurobiSolverAdapter,
    "highs": HighsSolverAdapter,
    "mosek": MosekSolverAdapter,
    "osqp": OSQPSolverAdapter,
    "pdlp": PDLPSolverAdapter,
    "piqp": PIQPSolverAdapter,
    "proxqp": ProxQPSolverAdapter,
    "qtqp": QTQPSolverAdapter,
    "sdpa": SDPASolverAdapter,
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
