import json
import math
import sys
from pathlib import Path

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.core.config import parse_environment_run_config, parse_run_config
from solver_benchmarks.core.env_runner import run_environment_matrix
from solver_benchmarks.core.problem import CONE, QP, ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.core.storage import ResultStore
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.solvers import registry as solver_registry
from solver_benchmarks.solvers.base import SolverAdapter


def test_runner_writes_results_logs_and_resumes(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
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
    assert record["metadata"]["runtime"]["python_version"]
    assert "scs" in record["metadata"]["runtime"]["solver_package_versions"]

    run_benchmark(config, run_dir=store.run_dir, repo_root=Path.cwd())
    assert len(results_path.read_text().strip().splitlines()) == 1

    df = load_results(store.run_dir)
    assert len(df) == 1
    assert df.loc[0, "solver_id"] == "scs_smoke"


def test_result_store_normalizes_nonfinite_values_for_parquet(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "solver", "solver": "scs", "settings": {}}],
        }
    )
    store = ResultStore.create(config, run_dir=tmp_path / "run")

    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="p1",
            problem_kind=QP,
            solver_id="solver",
            solver="scs",
            status="optimal",
            objective_value=1.0,
            iterations=1,
            run_time_seconds=0.1,
        )
    )
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="p2",
            problem_kind=QP,
            solver_id="solver",
            solver="scs",
            status="max_iter_reached",
            objective_value=math.nan,
            iterations=None,
            run_time_seconds=0.2,
        )
    )

    records = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    df = load_results(store.run_dir)

    assert records[1]["objective_value"] is None
    assert store.results_parquet_path.exists()
    assert len(df) == 2
    assert df.loc[df["problem"] == "p2", "objective_value"].isna().all()


def test_parquet_rewrite_handles_legacy_string_nan(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "solver", "solver": "scs", "settings": {}}],
        }
    )
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    store.results_jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": store.run_id,
                        "dataset": "synthetic_qp",
                        "problem": "p1",
                        "problem_kind": QP,
                        "solver_id": "solver",
                        "solver": "scs",
                        "status": "optimal",
                        "objective_value": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "run_id": store.run_id,
                        "dataset": "synthetic_qp",
                        "problem": "p2",
                        "problem_kind": QP,
                        "solver_id": "solver",
                        "solver": "scs",
                        "status": "max_iter_reached",
                        "objective_value": "nan",
                    }
                ),
            ]
        )
        + "\n"
    )

    store.rewrite_parquet()
    df = load_results(store.run_dir)

    assert len(df) == 2
    assert df.loc[df["problem"] == "p2", "objective_value"].isna().all()


def test_unsupported_combinations_skip_by_default(monkeypatch, tmp_path: Path):
    class FakeConeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="fake_cone", name="cone_problem", kind=CONE)]

    class FakeQPSolver(SolverAdapter):
        solver_name = "fake_qp"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):
            raise AssertionError("Unsupported solver should not be invoked")

    monkeypatch.setitem(dataset_registry.DATASETS, "fake_cone", FakeConeDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "fake_qp", FakeQPSolver)

    config = parse_run_config(
        {
            "run": {
                "dataset": "fake_cone",
                "output_dir": str(tmp_path / "runs"),
                "include": ["cone_problem"],
                "parallelism": 1,
            },
            "solvers": [{"id": "fake_qp_skip", "solver": "fake_qp", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=Path.cwd())
    df = load_results(store.run_dir)

    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped_unsupported"
    assert df.loc[0, "problem"] == "cone_problem"
    assert store.events_path.exists()


def test_pdlp_skips_cleanly_when_unavailable_or_non_lp(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
                "parallelism": 1,
            },
            "solvers": [{"id": "pdlp_smoke", "solver": "pdlp", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=Path.cwd())
    df = load_results(store.run_dir)

    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped_unsupported"


def test_auto_prepare_data_invokes_dataset_prepare(monkeypatch, tmp_path: Path):
    called = {}

    class FakeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def prepare_data(self, problem_names=None, all_problems=False):
            called["problem_names"] = problem_names
            called["all_problems"] = all_problems

        def list_problems(self):
            return []

        def data_status(self):
            return type(
                "Status",
                (),
                {"message": "fake dataset has no problems"},
            )()

    monkeypatch.setitem(dataset_registry.DATASETS, "fake_empty", FakeDataset)
    config = parse_run_config(
        {
            "run": {
                "dataset": "fake_empty",
                "output_dir": str(tmp_path / "runs"),
                "include": ["needed"],
                "auto_prepare_data": True,
            },
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    run_benchmark(config, repo_root=Path.cwd())

    assert called == {"problem_names": ["needed"], "all_problems": False}


def test_runner_records_environment_metadata(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
                "parallelism": 1,
            },
            "solvers": [
                {
                    "id": "scs_env",
                    "solver": "scs",
                    "settings": {"verbose": False, "max_iters": 1000},
                }
            ],
        }
    )

    store = run_benchmark(
        config,
        repo_root=Path.cwd(),
        environment_id="scs_3_2",
        environment_metadata={"scs": "3.2.0"},
    )
    record = json.loads(store.results_jsonl_path.read_text().splitlines()[0])

    assert record["metadata"]["environment_id"] == "scs_3_2"
    assert record["metadata"]["environment_metadata"] == {"scs": "3.2.0"}


def test_environment_matrix_runs_current_python_and_preserves_manifest(tmp_path: Path):
    config = parse_environment_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
                "parallelism": 1,
            },
            "environments": [
                {
                    "id": "current_a",
                    "python": sys.executable,
                    "metadata": {"label": "current-a"},
                    "solvers": [
                        {
                            "id": "scs_current_a",
                            "solver": "scs",
                            "settings": {"verbose": False, "max_iters": 1000},
                        }
                    ],
                },
                {
                    "id": "current_b",
                    "python": sys.executable,
                    "metadata": {"label": "current-b"},
                    "solvers": [
                        {
                            "id": "scs_current_b",
                            "solver": "scs",
                            "settings": {"verbose": False, "max_iters": 1000},
                        }
                    ],
                },
            ],
        }
    )

    run_dir = run_environment_matrix(
        config,
        run_dir=tmp_path / "matrix",
        repo_root=Path.cwd(),
        stream_output=False,
    )
    df = load_results(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert set(df["solver_id"]) == {"scs_current_a", "scs_current_b"}
    assert set(df["metadata.environment_id"]) == {"current_a", "current_b"}
    assert set(df["status"]) == {"optimal"}
    assert {solver["id"] for solver in manifest["config"]["solvers"]} == {
        "scs_current_a",
        "scs_current_b",
    }
