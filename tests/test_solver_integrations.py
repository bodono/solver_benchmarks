from pathlib import Path

import pytest
from conftest import skip_or_fail_missing_solver

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.core import status
from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.solvers import get_solver

OPEN_SOURCE_SOLVERS = {
    "highs": {"verbose": False},
    "qtqp": {"verbose": False},
    "scs": {
        "verbose": False,
        "eps_abs": 1.0e-6,
        "eps_rel": 1.0e-6,
        "max_iters": 1000,
    },
    "clarabel": {"verbose": False},
    "cpo3": {"verbose": False},
    "cvxopt": {"verbose": False},
    "ecos": {"verbose": False},
    "osqp": {
        "verbose": False,
        "eps_abs": 1.0e-8,
        "eps_rel": 1.0e-8,
        "max_iter": 10000,
        "polish": True,
    },
    "pdlp": {"time_limit_sec": 10.0},
    "piqp": {"verbose": False, "eps_abs": 1.0e-8, "eps_rel": 1.0e-8, "max_iter": 10000},
    "proxqp": {
        "verbose": False,
        "eps_abs": 1.0e-8,
        "eps_rel": 1.0e-8,
        "max_iter": 10000,
    },
}

OPEN_SOURCE_CONE_SOLVERS = {
    "scs": {
        "verbose": False,
        "eps_abs": 1.0e-6,
        "eps_rel": 1.0e-6,
        "max_iters": 1000,
    },
    "clarabel": {"verbose": False},
    "cpo3": {"verbose": False},
    "cvxopt": {"verbose": False},
    "ecos": {"verbose": False},
    "sdpa": {
        "verbose": False,
        "max_iter": 50,
        "optimality_tolerance": 1.0e-5,
    },
}


def _require_solver(solver_name: str) -> None:
    """Skip this solver's case when it is not installed.

    Each solver skips on its own, so one missing or broken extra cannot hide
    the end-to-end coverage of the others. CI installs every extra and sets
    BENCH_REQUIRE_ALL_SOLVERS, which turns a missing solver into a failure:
    there an unavailable solver is a packaging or import regression.
    """
    if not get_solver(solver_name).is_available():
        skip_or_fail_missing_solver(solver_name)


def _solve_one(tmp_path: Path, repo_root: Path, dataset: str, problem: str, solver_name: str,
               settings: dict) -> dict:
    config = parse_run_config(
        {
            "run": {
                "dataset": dataset,
                "output_dir": str(tmp_path / "runs"),
                "include": [problem],
                "parallelism": 1,
                "timeout_seconds": 30,
            },
            "solvers": [{"id": solver_name, "solver": solver_name, "settings": settings}],
        }
    )
    store = run_benchmark(config, repo_root=repo_root)
    records = load_results(store.run_dir).to_dict("records")
    assert len(records) == 1, records
    assert records[0]["solver_id"] == solver_name
    assert records[0]["problem"] == problem
    return records[0]


@pytest.mark.parametrize("solver_name", sorted(OPEN_SOURCE_SOLVERS))
def test_open_source_solver_solves_synthetic_lp(solver_name: str, tmp_path: Path, repo_root: Path):
    _require_solver(solver_name)

    record = _solve_one(tmp_path, repo_root, "synthetic_qp", "one_variable_lp",
                        solver_name, OPEN_SOURCE_SOLVERS[solver_name])

    assert record["status"] == status.OPTIMAL, record
    assert record["objective_value"] == pytest.approx(1.0, abs=1.0e-4)


@pytest.mark.parametrize("solver_name", sorted(OPEN_SOURCE_CONE_SOLVERS))
def test_open_source_cone_solver_solves_synthetic_cone_lp(solver_name: str, tmp_path: Path, repo_root: Path):
    _require_solver(solver_name)

    record = _solve_one(tmp_path, repo_root, "synthetic_cone", "one_variable_cone_lp",
                        solver_name, OPEN_SOURCE_CONE_SOLVERS[solver_name])

    assert record["status"] == status.OPTIMAL, record
    assert record["objective_value"] == pytest.approx(1.0, abs=1.0e-4)
