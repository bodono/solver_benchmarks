"""Runtime environment metadata for benchmark results."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import Mapping

from .system_info import _detect_cpu_model, _package_version, usable_cpu_count

SOLVER_PACKAGES = {
    "clarabel": ("clarabel",),
    "cplex": ("cplex",),
    "cuopt": ("cuopt-cu12", "cuopt"),
    "cvxopt": ("cvxopt",),
    "ecos": ("ecos",),
    "gurobi": ("gurobipy",),
    "highs": ("highspy",),
    "mosek": ("Mosek", "mosek"),
    "osqp": ("osqp",),
    "pdlp": ("ortools",),
    "piqp": ("piqp",),
    "proxqp": ("proxsuite",),
    "cpo3": ("cpo3",),
    "qtqp": ("qtqp",),
    "scs": ("scs",),
    "sdpa": ("sdpa-python",),
}

# Environment variables that size the thread pools of the OpenMP runtime and
# of the BLAS / LAPACK libraries (OpenBLAS, MKL, Apple Accelerate) behind
# numpy, scipy and most solvers. The runner sets them for each worker (see
# runner.worker_thread_environ) and every row records their values.
THREAD_LIMIT_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def runtime_metadata(solver_name: str, environ: Mapping[str, str] | None = None) -> dict:
    """Per-solve runtime metadata.

    Captures Python / OS / CPU-model fields, the CPUs the solve may use,
    the BLAS / OpenMP thread variables of ``environ`` (default: this
    process's environment, which in a worker is what the runner gave it;
    ``None`` for a variable that is not set), and the installed versions
    of the solver's backing package(s) and of NumPy and SciPy. System-level snapshots
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
    # report tables.
    short_executable = os.path.basename(sys.executable) if sys.executable else None
    if environ is None:
        environ = os.environ
    return {
        "python_executable": short_executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": _detect_cpu_model(),
        "usable_cpu_count": usable_cpu_count(),
        "thread_env": {name: environ.get(name) for name in THREAD_LIMIT_ENV_VARS},
        "solver_package_versions": package_versions,
        # The BLAS/LAPACK behind most solvers ships with these, and in an
        # environment matrix each environment can have its own builds; the
        # manifest's system block only describes the driver's.
        "library_versions": {
            package: _package_version(package) for package in ("numpy", "scipy")
        },
    }
