"""Runtime environment metadata for benchmark results."""

from __future__ import annotations

import platform
import sys
from importlib import metadata

from .system_info import _detect_cpu_model

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
    """Per-solve runtime metadata.

    Captures Python / OS / CPU-model fields and the installed version
    of the solver's backing package(s). System-level snapshots
    (memory, full CPU detail) are captured *once* per run in the
    manifest's ``system`` block — duplicating them here would bloat
    every results row without adding signal. The CPU model is the
    one cross-row field worth keeping, since a heterogeneous run
    (multiple environments / hosts in one matrix) benefits from
    per-row provenance.
    """
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
        "cpu_model": _detect_cpu_model(),
        "solver_package_versions": package_versions,
    }


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
