"""Solver registry.

An adapter module is imported when its solver is first looked up, so a
worker imports only the adapter it runs, and listing or validating solver
names imports none.
"""

from __future__ import annotations

from solver_benchmarks.core.lazy_registry import LazyRegistry

from .base import SolverAdapter


def _adapter(module: str, cls: str) -> tuple[str, str]:
    return f"{__package__}.{module}", cls


SOLVERS: LazyRegistry[type[SolverAdapter]] = LazyRegistry(
    {
        "clarabel": _adapter("clarabel_adapter", "ClarabelSolverAdapter"),
        "cplex": _adapter("cplex_adapter", "CPLEXSolverAdapter"),
        "cpo3": _adapter("cpo3_adapter", "Cpo3SolverAdapter"),
        "cvxopt": _adapter("cvxopt_adapter", "CVXOPTSolverAdapter"),
        "ecos": _adapter("ecos_adapter", "ECOSSolverAdapter"),
        "gurobi": _adapter("gurobi_adapter", "GurobiSolverAdapter"),
        "highs": _adapter("highs_adapter", "HighsSolverAdapter"),
        "mosek": _adapter("mosek_adapter", "MosekSolverAdapter"),
        "osqp": _adapter("osqp_adapter", "OSQPSolverAdapter"),
        "pdlp": _adapter("pdlp_adapter", "PDLPSolverAdapter"),
        "cuopt": _adapter("cuopt_adapter", "CuOptSolverAdapter"),
        "piqp": _adapter("piqp_adapter", "PIQPSolverAdapter"),
        "proxqp": _adapter("proxqp_adapter", "ProxQPSolverAdapter"),
        "qtqp": _adapter("qtqp_adapter", "QTQPSolverAdapter"),
        "sdpa": _adapter("sdpa_adapter", "SDPASolverAdapter"),
        "scs": _adapter("scs_adapter", "SCSSolverAdapter"),
    }
)


def get_solver(name: str) -> type[SolverAdapter]:
    key = name.lower()
    if key not in SOLVERS:
        available = ", ".join(sorted(SOLVERS))
        raise KeyError(f"Unknown solver {name!r}. Available: {available}")
    # An error importing the adapter (a KeyError included) is raised as it is.
    return SOLVERS[key]


def list_solvers() -> list[str]:
    return sorted(SOLVERS)
