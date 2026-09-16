"""Exercise the native HiGHS/OR-Tools libraries without pytest import-order effects."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

_MPS_SOLVE = r'''
import sys
from pathlib import Path

from solver_benchmarks.solvers.pdlp_adapter import PDLPSolverAdapter, _import_ortools

if sys.argv[1] == "highs_first":
    import highspy
    _import_ortools()
else:
    _import_ortools()
    import highspy

import numpy as np
import scipy.sparse as sp

from problem_classes.qpsreader import readMpsLp
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData

directory = Path(sys.argv[2])
mps = directory / "tiny.mps"
mps.write_text("""NAME          TINY
ROWS
 N  COST
 G  LOWER
 L  UPPER
COLUMNS
    X         COST      1
    X         LOWER     1
    X         UPPER     1
RHS
    RHS1      LOWER     1
    RHS1      UPPER     2
ENDATA
""")
a, q, lower, upper = readMpsLp(mps)
problem = ProblemData("smoke", "tiny_mps", QP, {
    "P": sp.csc_matrix((q.size, q.size)),
    "q": q, "A": a, "l": lower, "u": upper,
})
result = PDLPSolverAdapter({
    "eps_abs": 1e-8, "eps_rel": 1e-8, "time_limit_sec": 10,
}).solve(problem, directory)
assert result.status == status.OPTIMAL, result
assert np.isclose(result.objective_value, 1.0, atol=1e-6), result
assert result.kkt["primal_res_rel"] < 1e-6, result.kkt
assert result.kkt["dual_res_rel"] < 1e-6, result.kkt
'''


@pytest.mark.parametrize("import_order", ["highs_first", "ortools_first"])
def test_native_import_orders_and_mps_pdlp_solve(import_order, tmp_path, repo_root):
    # Check availability without importing either native extension into pytest.
    # A present but broken extension must fail the subprocess, not become a skip.
    for package in ("highspy", "ortools"):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"{package} is not installed")
    completed = subprocess.run(
        [sys.executable, "-c", _MPS_SOLVE, import_order, str(tmp_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
