import json
from pathlib import Path

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.runner import run_benchmark


def test_runner_writes_results_logs_and_resumes(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
                "resume": True,
            },
            "solvers": [
                {
                    "id": "scs_smoke",
                    "solver": "scs",
                    "settings": {
                        "verbose": False,
                        "eps_abs": 1e-6,
                        "eps_rel": 1e-6,
                        "max_iters": 1000,
                    },
                }
            ],
        }
    )

    store = run_benchmark(config, repo_root=Path.cwd())
    results_path = store.results_jsonl_path
    rows = results_path.read_text().strip().splitlines()

    assert len(rows) == 1
    record = json.loads(rows[0])
    assert record["status"] == "optimal"
    artifact_dir = Path(record["artifact_dir"])
    assert (artifact_dir / "stdout.log").exists()
    assert (artifact_dir / "stderr.log").exists()
    assert (artifact_dir / "result.json").exists()
    assert store.results_parquet_path.exists()

    run_benchmark(config, run_dir=store.run_dir, repo_root=Path.cwd())
    assert len(results_path.read_text().strip().splitlines()) == 1

    df = load_results(store.run_dir)
    assert len(df) == 1
    assert df.loc[0, "solver_id"] == "scs_smoke"


def test_unsupported_combinations_skip_by_default(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "dimacs",
                "output_dir": str(tmp_path / "runs"),
                "include": ["qssp30"],
                "parallelism": 1,
            },
            "solvers": [{"id": "osqp_skip", "solver": "osqp", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=Path.cwd())
    df = load_results(store.run_dir)

    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped_unsupported"
    assert store.events_path.exists()


def test_pdlp_skips_cleanly_when_unavailable_or_non_lp(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
            },
            "solvers": [{"id": "pdlp_smoke", "solver": "pdlp", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=Path.cwd())
    df = load_results(store.run_dir)

    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped_unsupported"
