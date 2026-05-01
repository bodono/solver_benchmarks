"""Run one tiny benchmark problem for a named solver in CI."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.core import status
from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.runner import run_benchmark

SETTINGS = {
    "clarabel": {"verbose": False},
    "cvxopt": {
        "verbose": False,
        "abstol": 1.0e-9,
        "reltol": 1.0e-9,
        "feastol": 1.0e-9,
    },
    "ecos": {
        "verbose": False,
        "abstol": 1.0e-9,
        "reltol": 1.0e-9,
        "feastol": 1.0e-9,
    },
    "highs": {"verbose": False},
    "osqp": {"verbose": False, "eps_abs": 1.0e-8, "eps_rel": 1.0e-8, "max_iter": 10000},
    "pdlp": {"time_limit_sec": 10.0},
    "piqp": {"verbose": False, "eps_abs": 1.0e-8, "eps_rel": 1.0e-8, "max_iter": 10000},
    "proxqp": {"verbose": False, "eps_abs": 1.0e-8, "eps_rel": 1.0e-8, "max_iter": 10000},
    "qtqp": {"verbose": False},
    "scs": {"verbose": False, "eps_abs": 1.0e-6, "eps_rel": 1.0e-6, "max_iters": 1000},
    "sdpa": {"verbose": False, "max_iter": 50, "optimality_tolerance": 1.0e-5},
}

CONE_SOLVERS = {"sdpa"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solver", choices=sorted(SETTINGS))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    dataset = "synthetic_cone" if args.solver in CONE_SOLVERS else "synthetic_qp"
    problem = "one_variable_cone_lp" if args.solver in CONE_SOLVERS else "one_variable_lp"
    with tempfile.TemporaryDirectory(prefix=f"{args.solver}-smoke-") as tmp:
        config = parse_run_config(
            {
                "run": {
                    "dataset": dataset,
                    "output_dir": str(Path(tmp) / "runs"),
                    "include": [problem],
                    "parallelism": 1,
                    "timeout_seconds": 30,
                },
                "solvers": [
                    {
                        "id": f"{args.solver}_smoke",
                        "solver": args.solver,
                        "settings": SETTINGS[args.solver],
                    }
                ],
            }
        )
        store = run_benchmark(config, repo_root=args.repo_root)
        df = load_results(store.run_dir)
    if len(df) != 1:
        raise RuntimeError(f"Expected exactly one result, got {len(df)}")
    record = df.iloc[0].to_dict()
    if record["status"] != status.OPTIMAL:
        raise RuntimeError(f"{args.solver} smoke solve failed: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
