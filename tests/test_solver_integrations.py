from pathlib import Path

import pytest

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.core import status
from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.solvers import get_solver


OPEN_SOURCE_SOLVERS = {
    "qtqp": {"verbose": False},
    "scs": {
        "verbose": False,
        "eps_abs": 1.0e-6,
        "eps_rel": 1.0e-6,
        "max_iters": 1000,
    },
    "clarabel": {"verbose": False},
    "osqp": {
        "verbose": False,
        "eps_abs": 1.0e-8,
        "eps_rel": 1.0e-8,
        "max_iter": 10000,
        "polish": True,
    },
    "pdlp": {"time_limit_sec": 10.0},
}


def test_all_open_source_solvers_solve_synthetic_lp(tmp_path: Path):
    missing = [
        solver_name
        for solver_name in OPEN_SOURCE_SOLVERS
        if not get_solver(solver_name).is_available()
    ]
    assert not missing, f"Missing solver extras: {', '.join(missing)}"

    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_lp"],
                "parallelism": 1,
                "timeout_seconds": 30,
            },
            "solvers": [
                {
                    "id": solver_name,
                    "solver": solver_name,
                    "settings": settings,
                }
                for solver_name, settings in OPEN_SOURCE_SOLVERS.items()
            ],
        }
    )

    store = run_benchmark(config, repo_root=Path.cwd())
    df = load_results(store.run_dir)

    assert set(df["solver_id"]) == set(OPEN_SOURCE_SOLVERS)
    assert set(df["problem"]) == {"one_variable_lp"}
    for record in df.to_dict("records"):
        assert record["status"] == status.OPTIMAL, record
        assert record["objective_value"] == pytest.approx(1.0, abs=1.0e-4)
