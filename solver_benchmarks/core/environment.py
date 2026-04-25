"""Runtime environment metadata for benchmark results."""

from __future__ import annotations

from importlib import metadata
import platform
import sys


SOLVER_PACKAGES = {
    "clarabel": ("clarabel",),
    "cplex": ("cplex",),
    "gurobi": ("gurobipy",),
    "highs": ("highspy",),
    "mosek": ("Mosek", "mosek"),
    "osqp": ("osqp",),
    "pdlp": ("ortools",),
    "piqp": ("piqp",),
    "proxqp": ("proxsuite",),
    "qtqp": ("qtqp",),
    "scs": ("scs",),
    "sdpa": ("sdpa-python",),
}


def runtime_metadata(solver_name: str) -> dict:
    package_versions = {
        package: _package_version(package)
        for package in SOLVER_PACKAGES.get(solver_name, ())
    }
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "solver_package_versions": package_versions,
    }


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
