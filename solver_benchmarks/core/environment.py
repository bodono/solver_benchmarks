"""Runtime environment metadata for benchmark results."""

from __future__ import annotations

import os
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
    # ``python_executable`` is recorded as the basename only (e.g.
    # ``python3.12``) rather than the full path, since the full path
    # commonly contains a username (``/Users/<user>/...``,
    # ``/home/<user>/...``) that we don't want to publish into shared
    # report tables. The absolute path is captured under
    # ``manifest["system"]["python_executable_full"]`` only when
    # ``system_metadata(include_full_python_path=True)`` is requested.
    short_executable = os.path.basename(sys.executable) if sys.executable else None
    return {
        "python_executable": short_executable,
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
