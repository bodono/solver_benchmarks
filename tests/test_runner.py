import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.cli import main
from solver_benchmarks.core.config import parse_environment_run_config, parse_run_config
from solver_benchmarks.core.env_runner import run_environment_matrix
from solver_benchmarks.core.problem import CONE, QP, ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.core.storage import ResultStore
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.datasets.base import Dataset, filter_problem_specs_by_size
from solver_benchmarks.solvers import registry as solver_registry
from solver_benchmarks.solvers.base import SolverAdapter


def test_runner_writes_results_logs_and_resumes(tmp_path: Path, repo_root: Path):
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

    store = run_benchmark(config, repo_root=repo_root)
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

    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert len(results_path.read_text().strip().splitlines()) == 1

    df = load_results(store.run_dir)
    assert len(df) == 1
    assert df.loc[0, "solver_id"] == "scs_smoke"


def test_resume_reuses_legacy_rows_when_solve_signature_is_compatible(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    class ResumeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="resume_ds", name="p", kind=QP)]

    class ResumeSolver(SolverAdapter):
        solver_name = "resume_solver"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    calls: list[str] = []

    def fake_run_subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        calls.append(payload["solver"]["id"])
        artifact_dir = Path(payload["artifacts_dir"])
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
            artifact_dir=str(artifact_dir),
        ).to_record()
        (artifact_dir / "worker_result.json").write_text(json.dumps(record))
        stdout_path.write_text("")
        stderr_path.write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setitem(dataset_registry.DATASETS, "resume_ds", ResumeDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "resume_solver", ResumeSolver)
    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", fake_run_subprocess
    )
    config1 = parse_run_config(
        {
            "run": {"dataset": "resume_ds", "output_dir": str(tmp_path / "runs")},
            "solvers": [
                {
                    "id": "existing",
                    "solver": "resume_solver",
                    "settings": {"alpha": 1},
                }
            ],
        }
    )
    store = ResultStore.create(config1, run_dir=tmp_path / "run")
    # Simulate a pre-signature result row from an older release.
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="resume_ds",
            problem="p",
            problem_kind=QP,
            solver_id="existing",
            solver="resume_solver",
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
        )
    )
    config2 = parse_run_config(
        {
            "run": {
                "dataset": "resume_ds",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 2,
                "name": "expanded",
            },
            "solvers": [
                {
                    "id": "existing",
                    "solver": "resume_solver",
                    "settings": {"alpha": 1},
                },
                {
                    "id": "new",
                    "solver": "resume_solver",
                    "settings": {"alpha": 1},
                },
            ],
        }
    )

    run_benchmark(config2, run_dir=store.run_dir, repo_root=repo_root)

    rows = [
        json.loads(line)
        for line in store.results_jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    assert calls == ["new"]
    assert [row["solver_id"] for row in rows] == ["existing", "new"]
    new_row = rows[1]
    assert new_row["metadata"]["resume_signature"]


def test_resume_supersedes_legacy_rows_when_solver_settings_change(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    class ResumeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="resume_ds", name="p", kind=QP)]

    class ResumeSolver(SolverAdapter):
        solver_name = "resume_solver"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    calls: list[dict] = []

    def fake_run_subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        calls.append(payload["solver"])
        artifact_dir = Path(payload["artifacts_dir"])
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
            artifact_dir=str(artifact_dir),
        ).to_record()
        (artifact_dir / "worker_result.json").write_text(json.dumps(record))
        stdout_path.write_text("")
        stderr_path.write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setitem(dataset_registry.DATASETS, "resume_ds", ResumeDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "resume_solver", ResumeSolver)
    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", fake_run_subprocess
    )
    config1 = parse_run_config(
        {
            "run": {"dataset": "resume_ds", "output_dir": str(tmp_path / "runs")},
            "solvers": [
                {
                    "id": "existing",
                    "solver": "resume_solver",
                    "settings": {"alpha": 1},
                }
            ],
        }
    )
    store = ResultStore.create(config1, run_dir=tmp_path / "run")
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="resume_ds",
            problem="p",
            problem_kind=QP,
            solver_id="existing",
            solver="resume_solver",
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
        )
    )
    config2 = parse_run_config(
        {
            "run": {"dataset": "resume_ds", "output_dir": str(tmp_path / "runs")},
            "solvers": [
                {
                    "id": "existing",
                    "solver": "resume_solver",
                    "settings": {"alpha": 2},
                }
            ],
        }
    )

    run_benchmark(config2, run_dir=store.run_dir, repo_root=repo_root)

    rows = [
        json.loads(line)
        for line in store.results_jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    assert [call["id"] for call in calls] == ["existing"]
    assert calls[0]["settings"] == {"alpha": 2}
    # The row solved with alpha=1 no longer describes "existing": the new
    # result replaces it, and it is kept aside in results.superseded.jsonl.
    assert [row["solver_id"] for row in rows] == ["existing"]
    assert rows[0]["metadata"]["resume_signature"]
    superseded = [
        json.loads(line)
        for line in (store.run_dir / "results.superseded.jsonl").read_text().splitlines()
    ]
    assert len(superseded) == 1 and "resume_signature" not in superseded[0]["metadata"]


@pytest.mark.parametrize("parallelism", [1, 2])
@pytest.mark.parametrize("legacy", [False, True])
def test_resume_retries_worker_errors_without_duplicate_rows_or_lost_artifacts(
    monkeypatch, tmp_path: Path, repo_root: Path, parallelism: int, legacy: bool
):
    class RetrySolver(SolverAdapter):
        solver_name = "retry_solver"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    calls: dict[str, int] = {}

    def fake_run_subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        name = payload["problem"]
        calls[name] = calls.get(name, 0) + 1
        stdout_path.write_text(f"{name} attempt {calls[name]}\n")
        stderr_path.write_text("original diagnostic\n")
        if name == "one_variable_eq" and calls[name] < 3:
            return SimpleNamespace(returncode=1, stderr="", timed_out=False)
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=name,
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
            artifact_dir=payload["artifacts_dir"],
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, timed_out=False)

    monkeypatch.setitem(solver_registry.SOLVERS, "retry_solver", RetrySolver)
    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", fake_run_subprocess)
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "parallelism": parallelism},
            "solvers": [{"id": "retry", "solver": "retry_solver"}],
        }
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)

    def rows():
        return [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]

    original_rows = rows()
    error = next(row for row in original_rows if row["status"] == "worker_error")
    original_artifacts = {
        path: path.read_bytes() for path in Path(error["artifact_dir"]).iterdir() if path.is_file()
    }
    if legacy:
        for row in original_rows:
            row["metadata"].pop("resume_signature", None)
        store.results_jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in original_rows))

    # The second attempt also fails; it replaces the old error and remains retryable.
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    retried_rows = rows()
    assert len(retried_rows) == 2
    retry_error = next(row for row in retried_rows if row["status"] == "worker_error")
    retry_artifacts = Path(retry_error["artifact_dir"])
    assert retry_artifacts != Path(error["artifact_dir"])
    assert json.loads((retry_artifacts / "previous_results.jsonl").read_text())["status"] == "worker_error"
    failed_retry_files = {
        path: path.read_bytes() for path in retry_artifacts.iterdir() if path.is_file()
    }

    # The third attempt succeeds. Reports see one final result per problem.
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert calls == {"one_variable_eq": 3, "one_variable_lp": 1}
    assert len(rows()) == 2
    assert {row["status"] for row in rows()} == {"optimal"}
    results = load_results(store.run_dir)
    assert len(results) == 2 and (results["status"] == "optimal").all()
    for path, content in {**original_artifacts, **failed_retry_files}.items():
        assert path.read_bytes() == content

    # Once successful, both problems are skipped by future resumes.
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert calls == {"one_variable_eq": 3, "one_variable_lp": 1}
    assert len(rows()) == 2


@pytest.mark.parametrize("skip_reason", ["unavailable", "unsupported"])
def test_resume_planning_skip_preserves_pending_worker_error(
    monkeypatch, tmp_path: Path, repo_root: Path, skip_reason: str
):
    class RecoverableSolver(SolverAdapter):
        supported_problem_kinds = {QP}
        available = True

        @classmethod
        def is_available(cls):
            return cls.available

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    calls = []
    fail = True

    def subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        calls.append(payload["problem"])
        stdout_path.write_text("attempt diagnostic")
        stderr_path.write_text("original failure")
        if fail:
            return SimpleNamespace(returncode=1, stderr="", timed_out=False)
        record = ProblemResult(
            run_id=payload["run_id"], dataset=payload["dataset"], problem=payload["problem"],
            problem_kind=payload["problem_kind"], solver_id="recover", solver="recover",
            status="optimal", objective_value=0.0, iterations=1, run_time_seconds=0.01,
            artifact_dir=payload["artifacts_dir"],
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, timed_out=False)

    monkeypatch.setitem(solver_registry.SOLVERS, "recover", RecoverableSolver)
    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", subprocess)
    config = parse_run_config({"run": {"dataset": "synthetic_qp"},
                               "solvers": [{"id": "recover", "solver": "recover"}]})
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    original = store.results_jsonl_path.read_bytes()
    if skip_reason == "unavailable":
        RecoverableSolver.available = False
    else:
        RecoverableSolver.supported_problem_kinds = set()
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert store.results_jsonl_path.read_bytes() == original
    assert len(calls) == 2
    assert not list(store.run_dir.glob("problems/**/retry-*"))

    RecoverableSolver.available = True
    RecoverableSolver.supported_problem_kinds = {QP}
    fail = False
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    rows = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    assert len(calls) == 4 and len(rows) == 2
    assert {row["status"] for row in rows} == {"optimal"}
    assert {row["metadata"]["worker_error_retries"] for row in rows} == {1}


@pytest.mark.parametrize("retry_limit", [0, 1, 2])
def test_resume_caps_deterministic_worker_errors_and_can_extend_budget(
    monkeypatch, tmp_path: Path, repo_root: Path, retry_limit: int
):
    class BrokenSolver(SolverAdapter):
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    calls = []

    def subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        calls.append(payload["problem"])
        record = ProblemResult(
            run_id=payload["run_id"], dataset=payload["dataset"], problem=payload["problem"],
            problem_kind=payload["problem_kind"], solver_id="broken", solver="broken",
            status="worker_error", objective_value=None, iterations=None, run_time_seconds=None,
            error="deterministic adapter exception", artifact_dir=payload["artifacts_dir"],
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, timed_out=False)

    monkeypatch.setitem(solver_registry.SOLVERS, "broken", BrokenSolver)
    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", subprocess)
    raw = {"run": {"dataset": "synthetic_qp", "max_worker_error_retries": retry_limit},
           "solvers": [{"id": "broken", "solver": "broken"}]}
    config = parse_run_config(raw)
    for _ in range(retry_limit + 3):
        store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    assert len(calls) == 2 * (retry_limit + 1)
    rows = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    assert len(rows) == 2 and {row["status"] for row in rows} == {"worker_error"}
    assert {row["metadata"].get("worker_error_retries", 0) for row in rows} == {retry_limit}
    events = [json.loads(line) for line in store.events_path.read_text().splitlines()]
    cap_events = [event for event in events if "max_worker_error_retries" in event]
    assert len(cap_events) == 4  # Two problems on each of the two capped resumes.
    for event in cap_events:
        assert event["level"] == "warning"
        assert "increase run.max_worker_error_retries" in event["message"]
        assert event["dataset"] == "synthetic_qp"
        assert event["solver_id"] == "broken"
        assert event["problem"] in {row["problem"] for row in rows}
        assert event["worker_error_retries"] == retry_limit
        assert event["max_worker_error_retries"] == retry_limit

    raw["run"]["max_worker_error_retries"] = retry_limit + 1
    extended = parse_run_config(raw)
    assert extended.config_hash == config.config_hash
    run_benchmark(extended, run_dir=store.run_dir, repo_root=repo_root)
    assert len(calls) == 2 * (retry_limit + 2)
    extended_events = [json.loads(line) for line in store.events_path.read_text().splitlines()]
    assert [event for event in extended_events if "max_worker_error_retries" in event] == cap_events


def test_run_cli_uses_config_stem_name_and_copies_source_config(tmp_path: Path, repo_root: Path):
    config_path = tmp_path / "named_smoke_run.yaml"
    output_dir = tmp_path / "runs"
    config_text = f"""
run:
  dataset: synthetic_qp
  output_dir: {output_dir}
  include:
    - one_variable_eq
solvers:
  - id: scs_smoke
    solver: scs
    settings:
      verbose: false
      max_iters: 1000
"""
    config_path.write_text(config_text)

    result = CliRunner().invoke(
        main,
        ["run", str(config_path), "--repo-root", str(repo_root)],
    )

    assert result.exit_code == 0, result.output
    run_dir = Path(result.output.strip().splitlines()[-1])
    assert run_dir.parent == output_dir
    assert re.fullmatch(
        r"named_smoke_run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC",
        run_dir.name,
    )
    assert (run_dir / "run_config.yaml").read_text() == config_text
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["config"]["name"] == "named_smoke_run"
    assert manifest["config"]["config_hash"] not in run_dir.name
    assert "synthetic_qp" not in run_dir.name


def test_run_cli_can_disable_solver_streaming_without_changing_settings(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "verbose_override.yaml"
    config_path.write_text(
        """
run:
  dataset: synthetic_qp
  output_dir: runs
solvers:
  - id: scs_smoke
    solver: scs
    settings:
      verbose: true
  - id: qtqp_smoke
    solver: qtqp
    settings:
      max_iter: 10
"""
    )
    seen = {}

    def fake_run_benchmark(config, **kwargs):
        seen["settings"] = [solver.settings for solver in config.solvers]
        seen["stream_output"] = kwargs["stream_output"]
        seen["stream_solver_output"] = kwargs["stream_solver_output"]
        return SimpleNamespace(run_dir=tmp_path / "runs" / "fake_run")

    monkeypatch.setattr("solver_benchmarks.cli.run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        main,
        ["run", str(config_path), "--no-stream-output"],
    )

    assert result.exit_code == 0, result.output
    assert seen["settings"] == [
        {"verbose": True},
        {"max_iter": 10},
    ]
    assert seen["stream_output"] is True
    assert seen["stream_solver_output"] is False


def test_env_run_cli_can_disable_solver_streaming(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "env_stream_override.yaml"
    config_path.write_text(
        """
run:
  dataset: synthetic_qp
  output_dir: runs
environments:
  - id: current
    python: python
    solvers:
      - id: scs_current
        solver: scs
        settings:
          verbose: true
"""
    )
    seen = {}

    def fake_run_environment_matrix(config, **kwargs):
        seen["settings"] = [
            solver.settings
            for environment in config.environments
            for solver in environment.solvers
        ]
        seen["stream_output"] = kwargs["stream_output"]
        seen["stream_solver_output"] = kwargs["stream_solver_output"]
        return tmp_path / "runs" / "fake_env_run"

    monkeypatch.setattr(
        "solver_benchmarks.cli.run_environment_matrix",
        fake_run_environment_matrix,
    )

    result = CliRunner().invoke(
        main,
        ["env", "run", str(config_path), "--no-stream-output"],
    )

    assert result.exit_code == 0, result.output
    assert seen["settings"] == [{"verbose": True}]
    assert seen["stream_output"] is True
    assert seen["stream_solver_output"] is False


def test_run_benchmark_resolves_relative_output_dir_under_repo_root(
    monkeypatch, tmp_path: Path
):
    class EmptyDataset(Dataset):
        dataset_id = "empty_for_output_dir"

        def list_problems(self):
            return []

        def load_problem(self, name):
            raise KeyError(name)

    monkeypatch.setitem(
        dataset_registry.DATASETS, "empty_for_output_dir", EmptyDataset
    )
    config = parse_run_config(
        {
            "run": {"dataset": "empty_for_output_dir", "output_dir": "results"},
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=tmp_path)

    assert store.run_dir.parent == tmp_path / "results"


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
    # write_result only appends to jsonl; materialize the parquet
    # explicitly here as run_benchmark does at end of run.
    store.write_parquet()

    records = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    df = load_results(store.run_dir)

    assert records[1]["objective_value"] is None
    assert store.results_parquet_path.exists()
    assert len(df) == 2
    assert df.loc[df["problem"] == "p2", "objective_value"].isna().all()


def test_write_parquet_handles_legacy_string_nan(tmp_path: Path):
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

    store.write_parquet()
    df = load_results(store.run_dir)

    assert len(df) == 2
    assert df.loc[df["problem"] == "p2", "objective_value"].isna().all()


def test_unsupported_combinations_skip_by_default(monkeypatch, tmp_path: Path, repo_root: Path):
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

    store = run_benchmark(config, repo_root=repo_root)
    df = load_results(store.run_dir)

    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped_unsupported"
    assert df.loc[0, "problem"] == "cone_problem"
    assert store.events_path.exists()


def test_skip_only_runs_write_parquet_so_load_results_sees_every_skip(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """When every problem is skipped during planning, the runner must
    still call store.write_parquet() on the early-return path so that
    load_results() (which prefers parquet) sees every skip row, not
    just whatever happens to be in jsonl at that moment."""

    class FakeConeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            # Two problems so the test actually exercises >1 skip row.
            return [
                ProblemSpec(dataset_id="fake_cone", name="cone_a", kind=CONE),
                ProblemSpec(dataset_id="fake_cone", name="cone_b", kind=CONE),
            ]

    class FakeQPSolver(SolverAdapter):
        solver_name = "fake_qp"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover - skip path
            raise AssertionError("Unsupported solver should not be invoked")

    monkeypatch.setitem(dataset_registry.DATASETS, "fake_cone", FakeConeDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "fake_qp", FakeQPSolver)
    config = parse_run_config(
        {
            "run": {
                "dataset": "fake_cone",
                "output_dir": str(tmp_path / "runs"),
                "include": ["cone_a", "cone_b"],
                "parallelism": 1,
            },
            "solvers": [{"id": "fake_qp_skip", "solver": "fake_qp", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=repo_root)

    # Two rows in the JSONL — one per skip.
    jsonl_lines = [
        line for line in store.results_jsonl_path.read_text().splitlines() if line.strip()
    ]
    assert len(jsonl_lines) == 2

    # And — most importantly — load_results() (which prefers parquet)
    # must see both rows.
    df = load_results(store.run_dir)
    assert len(df) == 2
    assert set(df["problem"]) == {"cone_a", "cone_b"}
    assert set(df["status"]) == {"skipped_unsupported"}


def test_run_benchmark_unlinks_stale_parquet_at_start_and_writes_at_end(
    tmp_path: Path, repo_root: Path
):
    """Pin the success-sentinel contract: a leftover parquet from a
    prior aborted attempt must be removed at the start of run_benchmark
    so it cannot be mistaken for the completion of *this* run, and the
    parquet must reappear at the end with the new run's content."""
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
                    "id": "scs_sentinel",
                    "solver": "scs",
                    "settings": {"verbose": False, "max_iters": 1000},
                }
            ],
        }
    )

    run_dir = tmp_path / "runs" / "fixed_run_id"
    run_dir.mkdir(parents=True)
    stale_parquet = run_dir / "results.parquet"
    stale_parquet.write_bytes(b"definitely-not-a-parquet-file")
    stale_mtime_ns = stale_parquet.stat().st_mtime_ns

    store = run_benchmark(config, run_dir=run_dir, repo_root=repo_root)

    assert store.results_parquet_path == stale_parquet
    assert stale_parquet.exists()
    # The byte payload was overwritten, not patched: the old contents
    # would have failed pd.read_parquet, and the mtime advanced.
    assert stale_parquet.stat().st_mtime_ns > stale_mtime_ns
    df = load_results(store.run_dir)
    assert len(df) == 1
    assert df.loc[0, "problem"] == "one_variable_eq"


def test_pdlp_skips_cleanly_when_unavailable_or_non_lp(tmp_path: Path, repo_root: Path):
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

    store = run_benchmark(config, repo_root=repo_root)
    df = load_results(store.run_dir)

    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped_unsupported"


def test_pdlp_defaults_do_not_enable_glop_or_polishing():
    pytest.importorskip("ortools.pdlp.solvers_pb2")
    from solver_benchmarks.solvers.pdlp_adapter import _pdlp_parameters_from_settings

    parameters = _pdlp_parameters_from_settings({})

    assert not parameters.presolve_options.use_glop
    assert not parameters.use_feasibility_polishing
    assert not parameters.apply_feasibility_polishing_after_limits_reached
    assert not parameters.apply_feasibility_polishing_if_solver_is_interrupted
    assert not parameters.use_diagonal_qp_trust_region_solver


def test_pdlp_glop_presolve_requires_explicit_opt_in():
    pytest.importorskip("ortools.pdlp.solvers_pb2")
    from solver_benchmarks.solvers.pdlp_adapter import _pdlp_parameters_from_settings

    parameters = _pdlp_parameters_from_settings({"use_glop": True})

    assert parameters.presolve_options.use_glop


def testfilter_problem_specs_by_size_drops_oversized_paths_only(tmp_path: Path):
    small = tmp_path / "small.bin"
    large = tmp_path / "large.bin"
    small.write_bytes(b"x" * 10)
    large.write_bytes(b"x" * 1_500_001)

    specs = [
        ProblemSpec(dataset_id="d", name="small", kind=QP, path=small),
        ProblemSpec(dataset_id="d", name="large", kind=QP, path=large),
        ProblemSpec(dataset_id="d", name="synth", kind=QP, path=None),
        ProblemSpec(dataset_id="d", name="missing", kind=QP, path=tmp_path / "nope.bin"),
    ]

    assert [spec.name for spec in filter_problem_specs_by_size(specs, None)] == [
        "small",
        "large",
        "synth",
        "missing",
    ]
    assert [spec.name for spec in filter_problem_specs_by_size(specs, 1.0)] == [
        "small",
        "synth",
        "missing",
    ]


def testfilter_problem_specs_by_size_prefers_metadata_size_over_path_stat(tmp_path: Path):
    """Datasets that pack many problems into one shared file (e.g. SDPLIB
    tar members) must be able to advertise per-member sizes via
    metadata["size_bytes"]; the filter must use those instead of the
    shared path's size on disk."""
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"x" * 5_000_000)  # 5 MB on disk

    specs = [
        ProblemSpec(
            dataset_id="d",
            name="member_small",
            kind=QP,
            path=archive,
            metadata={"size_bytes": 10},
        ),
        ProblemSpec(
            dataset_id="d",
            name="member_large",
            kind=QP,
            path=archive,
            metadata={"size_bytes": 1_500_001},
        ),
    ]

    assert [spec.name for spec in filter_problem_specs_by_size(specs, 1.0)] == ["member_small"]


def test_pdlp_linear_cone_accepts_free_zero_cone_key():
    pytest.importorskip("ortools.linear_solver.linear_solver_pb2")

    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.solvers.pdlp_adapter import _build_lp_model_from_linear_cone

    cone_problem = {
        "A": sp.csr_matrix([[1.0], [-1.0]]),
        "b": np.array([1.0, -1.0]),
        "q": np.array([1.0]),
        "cone": {"f": 1, "l": 1},
    }

    model = _build_lp_model_from_linear_cone(cone_problem)

    assert len(model.constraint) == 2
    # Equality row: lower == upper == b.
    assert model.constraint[0].lower_bound == pytest.approx(1.0)
    assert model.constraint[0].upper_bound == pytest.approx(1.0)
    # Nonneg row: Ax <= b. lower_bound MUST be set explicitly because
    # MPConstraintProto's proto3 default is 0 — leaving it unset would
    # silently turn the half-line into 0 <= Ax <= b. -inf is the
    # canonical "no lower bound" marker.
    import math
    assert math.isinf(model.constraint[1].lower_bound) and model.constraint[1].lower_bound < 0
    assert model.constraint[1].upper_bound == pytest.approx(-1.0)


def test_auto_prepare_data_invokes_dataset_prepare(monkeypatch, tmp_path: Path, repo_root: Path):
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

    run_benchmark(config, repo_root=repo_root)

    assert called == {"problem_names": ["needed"], "all_problems": False}


def test_auto_prepare_data_honors_subset_all(monkeypatch, tmp_path: Path, repo_root: Path):
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
                {"available": True, "message": "fake dataset has no problems"},
            )()

    monkeypatch.setitem(dataset_registry.DATASETS, "fake_all", FakeDataset)
    config = parse_run_config(
        {
            "run": {
                "dataset": "fake_all",
                "dataset_options": {"subset": "all"},
                "output_dir": str(tmp_path / "runs"),
                "auto_prepare_data": True,
            },
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    run_benchmark(config, repo_root=repo_root)

    assert called == {"problem_names": None, "all_problems": True}


def test_run_cli_missing_downloadable_data_reports_exact_commands(
    monkeypatch,
    tmp_path: Path,
):
    class MissingDownloadDataset(Dataset):
        dataset_id = "remote_fake"
        description = "Missing remote fixture."
        automatic_download = True

        @property
        def data_dir(self):
            return self.problem_classes_dir / "remote_fake_data"

        def list_problems(self):
            return []

        def load_problem(self, name):
            raise AssertionError("No problem should be loaded")

    monkeypatch.setitem(dataset_registry.DATASETS, "remote_fake", MissingDownloadDataset)
    config_path = tmp_path / "remote_fake.json"
    config_path.write_text(
        json.dumps(
            {
                "run": {
                    "dataset": "remote_fake",
                    "output_dir": str(tmp_path / "runs"),
                    "include": ["needed"],
                },
                "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
            }
        )
    )

    result = CliRunner().invoke(
        main,
        ["run", str(config_path), "--repo-root", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert (
        f"bench data prepare remote_fake --repo-root {tmp_path} --problem needed"
        in result.output
    )
    assert (
        f"bench run {config_path} --prepare-data --repo-root {tmp_path}"
        in result.output
    )


def test_data_status_reports_exact_prepare_command(monkeypatch, tmp_path: Path):
    class MissingDownloadDataset(Dataset):
        dataset_id = "remote_status_fake"
        description = "Missing remote fixture."
        automatic_download = True

        @property
        def data_dir(self):
            return self.problem_classes_dir / "remote_status_fake_data"

        def list_problems(self):
            return []

        def load_problem(self, name):
            raise AssertionError("No problem should be loaded")

    monkeypatch.setitem(
        dataset_registry.DATASETS, "remote_status_fake", MissingDownloadDataset
    )

    result = CliRunner().invoke(
        main,
        ["data", "status", "remote_status_fake", "--repo-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "missing" in result.output
    assert f"bench data prepare remote_status_fake --repo-root {tmp_path}" in result.output


def test_runner_records_environment_metadata(tmp_path: Path, repo_root: Path):
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
        repo_root=repo_root,
        environment_id="scs_3_2",
        environment_metadata={"scs": "3.2.0"},
    )
    record = json.loads(store.results_jsonl_path.read_text().splitlines()[0])

    assert record["metadata"]["environment_id"] == "scs_3_2"
    assert record["metadata"]["environment_metadata"] == {"scs": "3.2.0"}


def test_runner_iterates_over_multiple_datasets(monkeypatch, tmp_path: Path, repo_root: Path):
    class _FakeDatasetA:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="ds_a", name="prob_one", kind=QP)]

    class _FakeDatasetB:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [
                ProblemSpec(dataset_id="ds_b", name="prob_one", kind=QP),
                ProblemSpec(dataset_id="ds_b", name="prob_two", kind=QP),
            ]

    class _StubSolver(SolverAdapter):
        solver_name = "stub"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):
            raise AssertionError("Stub solver should be replaced before solve")

    monkeypatch.setitem(dataset_registry.DATASETS, "ds_a", _FakeDatasetA)
    monkeypatch.setitem(dataset_registry.DATASETS, "ds_b", _FakeDatasetB)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", _StubSolver)

    def _fake_run_subprocess(
        cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output
    ):
        from types import SimpleNamespace

        payload_path = Path(cmd[-1])
        payload = json.loads(payload_path.read_text())
        artifact_dir = Path(payload["artifacts_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
            artifact_dir=str(artifact_dir),
        ).to_record()
        (artifact_dir / "worker_result.json").write_text(json.dumps(record))
        stdout_path.write_text("")
        stderr_path.write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", _fake_run_subprocess
    )

    config = parse_run_config(
        {
            "run": {
                "datasets": ["ds_a", "ds_b"],
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
            },
            "solvers": [{"id": "stub_solver", "solver": "stub", "settings": {}}],
        }
    )
    store = run_benchmark(config, repo_root=repo_root)
    df = load_results(store.run_dir)

    # Three problems total across the two datasets.
    assert len(df) == 3
    assert set(df["dataset"]) == {"ds_a", "ds_b"}
    # Same problem name in different datasets is preserved as two rows.
    a_rows = df[df["dataset"] == "ds_a"]["problem"].tolist()
    b_rows = sorted(df[df["dataset"] == "ds_b"]["problem"].tolist())
    assert a_rows == ["prob_one"]
    assert b_rows == ["prob_one", "prob_two"]


def test_runner_resume_keys_are_dataset_aware(monkeypatch, tmp_path: Path):
    """Resume must not conflate identically-named problems across datasets."""

    class _DatasetA:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="ds_a", name="shared", kind=QP)]

    class _DatasetB:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="ds_b", name="shared", kind=QP)]

    class _StubSolver(SolverAdapter):
        solver_name = "stub"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    monkeypatch.setitem(dataset_registry.DATASETS, "ds_a", _DatasetA)
    monkeypatch.setitem(dataset_registry.DATASETS, "ds_b", _DatasetB)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", _StubSolver)

    config = parse_run_config(
        {
            "run": {
                "datasets": ["ds_a", "ds_b"],
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
                "resume": True,
            },
            "solvers": [{"id": "stub_solver", "solver": "stub", "settings": {}}],
        }
    )

    # Pre-populate one dataset's row so resume only skips that dataset's task.
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="ds_a",
            problem="shared",
            problem_kind=QP,
            solver_id="stub_solver",
            solver="stub",
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.0,
        )
    )

    completed = store.completed_keys()
    assert ("ds_a", "shared", "stub_solver") in completed
    assert ("ds_b", "shared", "stub_solver") not in completed


def test_runner_distinguishes_repeats_of_one_adapter_via_explicit_ids(monkeypatch, tmp_path: Path, repo_root: Path):
    """Same registry name listed twice with distinct ids must produce two
    independent dataset slots: separate artifact directories, separate
    resume keys, and distinct ``dataset`` values on each row."""

    seen_options: list[dict] = []

    class _ConfigurableDataset:
        def __init__(self, repo_root=None, **options):
            seen_options.append(dict(options))
            self._variant = options.get("variant", "default")

        def list_problems(self):
            return [ProblemSpec(dataset_id=self._variant, name="shared", kind=QP)]

    class _StubSolver(SolverAdapter):
        solver_name = "stub"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    monkeypatch.setitem(dataset_registry.DATASETS, "configurable", _ConfigurableDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", _StubSolver)

    def _fake_run_subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        from types import SimpleNamespace

        payload_path = Path(cmd[-1])
        payload = json.loads(payload_path.read_text())
        artifact_dir = Path(payload["artifacts_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
            artifact_dir=str(artifact_dir),
        ).to_record()
        (artifact_dir / "worker_result.json").write_text(json.dumps(record))
        stdout_path.write_text("")
        stderr_path.write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", _fake_run_subprocess
    )

    config = parse_run_config(
        {
            "run": {
                "datasets": [
                    {
                        "id": "feasible",
                        "name": "configurable",
                        "dataset_options": {"variant": "feasible"},
                    },
                    {
                        "id": "infeasible",
                        "name": "configurable",
                        "dataset_options": {"variant": "infeasible"},
                    },
                ],
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
            },
            "solvers": [{"id": "stub_solver", "solver": "stub", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=repo_root)
    df = load_results(store.run_dir)

    # Both variants ran the adapter with their own options.
    assert {opts.get("variant") for opts in seen_options} == {"feasible", "infeasible"}
    # Two rows, one per dataset id, even though both share the same problem
    # name and the same registry adapter.
    assert sorted(df["dataset"].tolist()) == ["feasible", "infeasible"]
    assert set(df["problem"]) == {"shared"}
    # Artifact directories are slugged on the dataset id so they don't collide.
    assert (store.run_dir / "problems" / "feasible" / "shared" / "stub_solver").exists()
    assert (store.run_dir / "problems" / "infeasible" / "shared" / "stub_solver").exists()
    # Resume tracks (id, problem, solver_id), so both entries are independent.
    completed = store.completed_keys()
    assert ("feasible", "shared", "stub_solver") in completed
    assert ("infeasible", "shared", "stub_solver") in completed


def test_worker_loads_dataset_via_registry_name_not_entry_id(monkeypatch, tmp_path: Path, repo_root: Path):
    """When an entry uses an explicit ``id`` distinct from its registry
    ``name``, the worker must look the adapter up by ``dataset_name``.
    Looking up by the entry id (which is not in the registry) would make
    every solve fail with ``worker_error``."""

    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.core import status as status_module
    from solver_benchmarks.core.problem import ProblemData
    from solver_benchmarks.core.result import SolverResult
    from solver_benchmarks.solvers.base import SolverAdapter
    from solver_benchmarks.worker import run_payload

    seen_dataset_init = []

    class _ConfigurableDataset:
        def __init__(self, repo_root=None, **options):
            seen_dataset_init.append(options)

        def list_problems(self):
            return [ProblemSpec(dataset_id="configurable", name="tiny", kind=QP)]

        def load_problem(self, name):
            qp = {
                "P": sp.csc_matrix(np.array([[1.0]])),
                "q": np.array([0.0]),
                "r": 0.0,
                "A": sp.csc_matrix(np.array([[1.0]])),
                "l": np.array([0.0]),
                "u": np.array([0.0]),
                "n": 1,
                "m": 1,
                "obj_type": "min",
            }
            return ProblemData("configurable", name, QP, qp)

    class _StubSolver(SolverAdapter):
        solver_name = "stub"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):
            return SolverResult(
                status=status_module.OPTIMAL,
                objective_value=0.0,
                iterations=1,
                run_time_seconds=0.001,
                info={},
            )

    monkeypatch.setitem(dataset_registry.DATASETS, "configurable", _ConfigurableDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", _StubSolver)

    artifacts_dir = tmp_path / "artifacts"
    payload = {
        "run_id": "run",
        "dataset": "feasible",          # entry id — NOT a registry key
        "dataset_name": "configurable", # registry key for the adapter
        "dataset_options": {"variant": "feasible"},
        "problem": "tiny",
        "problem_kind": QP,
        "solver": {"id": "stub_solver", "solver": "stub", "settings": {}},
        "artifacts_dir": str(artifacts_dir),
        "repo_root": str(repo_root),
    }

    result = run_payload(payload)
    assert result.status == "optimal", result.error
    # Result identity preserves the entry id, while the lookup used the name.
    assert result.dataset == "feasible"
    assert seen_dataset_init == [{"variant": "feasible"}]


def test_environment_matrix_runs_current_python_and_preserves_manifest(tmp_path: Path, repo_root: Path):
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
        repo_root=repo_root,
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


def test_check_schema_version_accepts_int_and_none():
    from solver_benchmarks.core.result import PROBLEM_RESULT_SCHEMA_VERSION
    from solver_benchmarks.core.runner import _check_schema_version

    assert _check_schema_version(None) is None
    assert _check_schema_version(PROBLEM_RESULT_SCHEMA_VERSION) is None
    assert _check_schema_version(0) is None


def test_check_schema_version_rejects_future_version():
    from solver_benchmarks.core.result import PROBLEM_RESULT_SCHEMA_VERSION
    from solver_benchmarks.core.runner import _check_schema_version

    err = _check_schema_version(PROBLEM_RESULT_SCHEMA_VERSION + 100)
    assert err is not None
    assert "newer than the runner" in err


def test_check_schema_version_rejects_non_integer_payloads():
    """Pre-fix this comparison crashed the parent runner with
    ``TypeError: '>' not supported between instances of 'str' and
    'int'`` when a hand-edited worker_result.json had
    ``"schema_version": "2"``. Validate the type explicitly so
    malformed payloads become a worker_error row instead.
    """
    from solver_benchmarks.core.runner import _check_schema_version

    for bad in ("2", 1.5, True, False, [1], {"v": 1}):
        err = _check_schema_version(bad)
        assert err is not None, f"expected error for {bad!r}"
        assert "not an integer" in err


def test_load_worker_result_records_malformed_schema_version_as_worker_error(
    tmp_path: Path,
):
    """End-to-end check that a payload with a string ``schema_version``
    is recorded as a worker_error row rather than crashing the parent
    runner."""
    from solver_benchmarks.core.config import DatasetConfig, SolverConfig
    from solver_benchmarks.core.runner import _load_worker_result
    from solver_benchmarks.core.storage import ResultStore

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    payload = {
        "run_id": "r",
        "dataset": "d",
        "problem": "p",
        "problem_kind": QP,
        "solver_id": "s",
        "solver": "scs",
        "status": "optimal",
        "schema_version": "2",  # malformed: string instead of int
    }
    worker_path = artifacts_dir / "worker_result.json"
    worker_path.write_text(json.dumps(payload))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    store = ResultStore(run_id="r", run_dir=run_dir)
    dataset_config = DatasetConfig(name="synthetic_qp", id="d")
    problem = ProblemSpec(dataset_id="d", name="p", kind=QP)
    solver_config = SolverConfig(id="s", solver="scs")

    result = _load_worker_result(
        worker_path,
        store=store,
        dataset_config=dataset_config,
        problem=problem,
        solver_config=solver_config,
        artifacts_dir=artifacts_dir,
        environment_id=None,
        environment_metadata=None,
    )
    assert result.status == "worker_error"
    assert result.error is not None
    assert "schema_version" in result.error
    assert "not an integer" in result.error


def _fake_worker(statuses=None, calls=None):
    """A ``_run_subprocess`` stand-in that records the problem it was asked to
    solve and writes a worker result with ``statuses[problem]`` (default optimal)."""
    statuses = statuses or {}

    def run(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        if calls is not None:
            calls.append(payload["problem"])
        status = statuses.get(payload["problem"], "optimal")
        record = ProblemResult(
            run_id=payload["run_id"], dataset=payload["dataset"], problem=payload["problem"],
            problem_kind=payload["problem_kind"], solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"], status=status,
            objective_value=0.0 if status == "optimal" else None, iterations=1,
            run_time_seconds=0.01, artifact_dir=payload["artifacts_dir"],
            error="cone 's' not supported." if status == "skipped_unsupported" else None,
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    return run


def _events(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _synthetic_config(tmp_path: Path, **run):
    return parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "output_dir": str(tmp_path / "runs"), **run},
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )


# --- one writer per run directory --------------------------------------------


def _hold_lock(run_dir: Path, pid: str = "424242"):
    import fcntl
    import os

    run_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(run_dir / ".bench.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.ftruncate(fd, 0)
    os.write(fd, f"{pid}\n".encode())
    return fd


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl locks")
def test_a_run_directory_in_use_is_refused(monkeypatch, tmp_path: Path, repo_root: Path):
    """Two `bench run` on one directory would solve everything twice into the
    same artifact paths, and a results.jsonl rewrite in one would drop the
    other's rows."""
    import os

    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    run_dir = tmp_path / "run"
    fd = _hold_lock(run_dir)
    try:
        with pytest.raises(RuntimeError, match=r"in use by another bench run \(pid 424242\)") as raised:
            run_benchmark(_synthetic_config(tmp_path), run_dir=run_dir, repo_root=repo_root)
        assert str(run_dir.resolve()) in str(raised.value)
        assert calls == []
        assert not (run_dir / "results.jsonl").exists()

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text("run: {dataset: synthetic_qp}\nsolvers:\n  - {id: osqp, solver: osqp}\n")
        result = CliRunner().invoke(main, ["run", str(config_path), "--run-dir", str(run_dir)])
        assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
        assert "in use by another bench run" in result.output
    finally:
        os.close(fd)
    store = run_benchmark(_synthetic_config(tmp_path), run_dir=run_dir, repo_root=repo_root)
    assert sorted(calls) == ["one_variable_eq", "one_variable_lp"]
    assert store.results_parquet_path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl locks")
def test_the_run_directory_is_locked_only_while_the_run_lasts(monkeypatch, tmp_path: Path, repo_root: Path):
    import fcntl
    import os

    from solver_benchmarks.analysis.derive import kkt_verify, merge_runs
    from solver_benchmarks.core import runner

    seen = []
    fake = _fake_worker()

    def worker(cmd, **kwargs):
        lock = Path(json.loads(Path(cmd[-1]).read_text())["artifacts_dir"]).parents[3] / ".bench.lock"
        fd = os.open(lock, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen.append("unlocked")
        except BlockingIOError:
            seen.append(lock.read_text().strip())
        finally:
            os.close(fd)
        return fake(cmd, **kwargs)

    monkeypatch.setattr(runner, "_run_subprocess", worker)
    for run_dir in (tmp_path / "named", None):
        seen.clear()
        store = run_benchmark(_synthetic_config(tmp_path), run_dir=run_dir, repo_root=repo_root)
        assert seen == [str(os.getpid())] * 2
        os.close(_hold_lock(store.run_dir))  # released at the end of the run
    kkt_verify(store.run_dir, tmp_path / "verified", tol=1e-6)
    merge_runs([store.run_dir], tmp_path / "merged")
    assert not (tmp_path / "verified" / ".bench.lock").exists()
    assert not (tmp_path / "merged" / ".bench.lock").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl locks")
def test_a_filesystem_without_locks_runs_with_a_warning(monkeypatch, tmp_path: Path, repo_root: Path):
    import errno

    from solver_benchmarks.core import runner

    def no_locks(fd, operation):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker())
    monkeypatch.setattr("fcntl.flock", no_locks)
    store = run_benchmark(_synthetic_config(tmp_path), run_dir=tmp_path / "run", repo_root=repo_root)
    assert store.results_parquet_path.exists()
    warnings = [e for e in _events(store.run_dir) if e["level"] == "warning" and "lock" in e["message"]]
    assert len(warnings) == 1


# --- include / exclude names that match no problem ----------------------------


def test_an_include_name_that_matches_no_problem_is_an_error(monkeypatch, tmp_path: Path, repo_root: Path):
    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    for include in (["one_variable_eq", "One_Variable_LP"], ["one_variable_EQ"]):
        with pytest.raises(RuntimeError, match="no problem named") as raised:
            run_benchmark(_synthetic_config(tmp_path, include=include), run_dir=tmp_path / "run", repo_root=repo_root)
        message = str(raised.value)
        typo = include[-1]
        assert repr(typo) in message
        assert repr(typo.lower()) in message  # the suggestion
        assert calls == []
    assert not (tmp_path / "run" / "results.jsonl").exists()

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "run: {dataset: synthetic_qp, include: [one_variable_eq, one_variabel_lp]}\n"
        "solvers:\n  - {id: osqp, solver: osqp}\n"
    )
    result = CliRunner().invoke(main, ["run", str(config_path), "--run-dir", str(tmp_path / "cli")])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "did you mean 'one_variable_lp'" in result.output


def test_include_names_are_matched_per_dataset_entry_and_across_datasets(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """A run-level include is shared by every dataset without its own: each
    name must match in at least one of them. An entry's own include must
    match in that dataset."""
    from solver_benchmarks.core import runner
    from solver_benchmarks.datasets.synthetic import SyntheticConeDataset

    cone_problem = SyntheticConeDataset().list_problems()[0].name
    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))

    def config(datasets, **run):
        return parse_run_config(
            {
                "run": {"datasets": datasets, "output_dir": str(tmp_path / "runs"), **run},
                "solvers": [{"id": "scs", "solver": "scs"}],
            }
        )

    run_benchmark(
        config(["synthetic_qp", "synthetic_cone"], include=["one_variable_eq", cone_problem]),
        run_dir=tmp_path / "ok", repo_root=repo_root,
    )
    assert sorted(calls) == sorted(["one_variable_eq", cone_problem])
    with pytest.raises(RuntimeError, match="no_such_problem"):
        run_benchmark(
            config(["synthetic_qp", "synthetic_cone"], include=["one_variable_eq", "no_such_problem"]),
            run_dir=tmp_path / "typo", repo_root=repo_root,
        )
    with pytest.raises(RuntimeError, match=f"synthetic_qp.*{cone_problem}"):
        run_benchmark(
            config([{"name": "synthetic_qp", "include": [cone_problem]}, "synthetic_cone"]),
            run_dir=tmp_path / "entry", repo_root=repo_root,
        )


class _SizedDataset(Dataset):
    dataset_id = "sized_fixture"
    description = "Two problems, one of 5 MB."

    def list_problems(self):
        return [
            ProblemSpec(dataset_id="sized_fixture", name="small", kind=QP, metadata={"size_bytes": 1000}),
            ProblemSpec(dataset_id="sized_fixture", name="big", kind=QP, metadata={"size_bytes": 5_000_000}),
        ]

    def load_problem(self, name):  # pragma: no cover - the worker is stubbed
        raise AssertionError


def test_an_include_name_removed_by_max_size_mb_is_not_a_typo(monkeypatch, tmp_path: Path, repo_root: Path, capsys):
    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    monkeypatch.setitem(dataset_registry.DATASETS, "sized_fixture", _SizedDataset)
    config = parse_run_config(
        {
            "run": {"dataset": "sized_fixture", "dataset_options": {"max_size_mb": 1},
                    "include": ["small", "big"], "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root, stream_output=True)
    assert calls == ["small"]
    warnings = [e["message"] for e in _events(store.run_dir) if e["level"] == "warning"]
    assert any("'big'" in message and "max_size_mb" in message for message in warnings), warnings
    assert "max_size_mb" in capsys.readouterr().err


class _DownloadableDataset(Dataset):
    """Lists only the problems downloaded so far, like every remote dataset."""

    dataset_id = "downloadable_fixture"
    description = "One local problem out of many remote ones."
    automatic_download = True

    @property
    def data_dir(self):
        return self.problem_classes_dir / "downloadable_fixture"

    def list_problems(self):
        return [ProblemSpec(dataset_id="downloadable_fixture", name="local", kind=QP)]

    def load_problem(self, name):  # pragma: no cover - the worker is stubbed
        raise AssertionError


def test_an_include_name_a_downloadable_dataset_does_not_list_is_missing_data(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """A downloadable dataset lists what is local: a missing name may simply
    not be downloaded yet, so it cannot be called a typo. It is requested
    data that is missing, which stops the run with the command to prepare it."""
    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    monkeypatch.setitem(dataset_registry.DATASETS, "downloadable_fixture", _DownloadableDataset)
    config = parse_run_config(
        {
            "run": {"dataset": "downloadable_fixture", "include": ["local", "remote_one"],
                    "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )
    with pytest.raises(RuntimeError) as raised:
        run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root, prepare_data_command="RERUN")
    message = str(raised.value)
    assert "does not have the requested problem 'remote_one' locally" in message
    assert "bench data prepare downloadable_fixture" in message and "--problem remote_one" in message
    assert "RERUN" in message
    assert calls == []
    assert not (tmp_path / "run" / "results.jsonl").exists()


def test_an_exclude_name_that_matches_no_problem_is_a_warning(monkeypatch, tmp_path: Path, repo_root: Path, capsys):
    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    store = run_benchmark(
        _synthetic_config(tmp_path, exclude=["One_Variable_LP"]),
        run_dir=tmp_path / "run", repo_root=repo_root, stream_output=True,
    )
    assert sorted(calls) == ["one_variable_eq", "one_variable_lp"]
    warnings = [e["message"] for e in _events(store.run_dir) if e["level"] == "warning"]
    assert any("'One_Variable_LP'" in message and "exclude" in message for message in warnings), warnings
    assert "'One_Variable_LP'" in capsys.readouterr().err


# --- run.fail_on_unsupported ---------------------------------------------------


@pytest.mark.parametrize("parallelism", [1, 2])
def test_fail_on_unsupported_fails_on_a_skip_the_adapter_reports(
    monkeypatch, tmp_path: Path, repo_root: Path, parallelism: int
):
    """ECOS, CVXOPT, SDPA, PDLP, Clarabel and cpo3 report problems they cannot
    take (a cone type) from inside the worker."""
    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(
        runner, "_run_subprocess", _fake_worker({"one_variable_eq": "skipped_unsupported"}, calls=calls)
    )
    with pytest.raises(RuntimeError, match="fail_on_unsupported") as raised:
        run_benchmark(
            _synthetic_config(tmp_path, fail_on_unsupported=True, parallelism=parallelism),
            run_dir=tmp_path / "run", repo_root=repo_root,
        )
    assert "cone 's' not supported" in str(raised.value)
    assert ".." not in str(raised.value)
    rows = [json.loads(line) for line in (tmp_path / "run" / "results.jsonl").read_text().splitlines()]
    assert ("one_variable_eq", "skipped_unsupported") in [(r["problem"], r["status"]) for r in rows]
    assert not (tmp_path / "run" / "results.parquet").exists()


def test_fail_on_unsupported_fails_for_a_solver_that_is_not_installed(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    from solver_benchmarks.core import runner

    class Missing(SolverAdapter):
        solver_name = "missing_fixture"
        supported_problem_kinds = {QP}

        @classmethod
        def is_available(cls):
            return False

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    monkeypatch.setitem(solver_registry.SOLVERS, "missing_fixture", Missing)
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker())
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "fail_on_unsupported": True, "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "osqp", "solver": "osqp"}, {"id": "missing", "solver": "missing_fixture"}],
        }
    )
    with pytest.raises(RuntimeError, match="missing_fixture.*unavailable"):
        run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    assert not (tmp_path / "run" / "results.jsonl").exists()


# --- worker crash messages ------------------------------------------------------


@pytest.mark.parametrize(
    "returncode, stderr, expected",
    [
        (-9, "", ["SIGKILL", "possibly out of memory"]),
        (-6, "loading\n\nFatal: out of memory allocating 12.4 GB\n\n", ["SIGABRT", "Fatal: out of memory allocating 12.4 GB"]),
        (3, "Traceback (most recent call last):\n  ...\nImportError: libfoo.so\n", ["code 3", "ImportError: libfoo.so"]),
        (0, "", ["without writing worker_result.json"]),
    ],
)
def test_a_crashed_worker_row_says_why(monkeypatch, tmp_path: Path, repo_root: Path, returncode, stderr, expected):
    from solver_benchmarks.core import runner

    def crash(cmd, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout="", stderr=stderr, timed_out=False)

    monkeypatch.setattr(runner, "_run_subprocess", crash)
    store = run_benchmark(
        _synthetic_config(tmp_path, include=["one_variable_eq"]), run_dir=tmp_path / "run", repo_root=repo_root
    )
    (row,) = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    assert row["status"] == "worker_error"
    for text in expected:
        assert text in row["error"]


def test_a_crashed_worker_message_keeps_only_a_bounded_stderr_tail():
    from solver_benchmarks.core.runner import _worker_exit_message

    stderr = "".join(f"line {i} " + "x" * 300 + "\n" for i in range(50))
    message = _worker_exit_message(1, stderr)
    assert "line 49" in message and "line 40" not in message
    assert len(message) < 800


def test_fail_on_unsupported_fails_again_on_a_recorded_skip(monkeypatch, tmp_path: Path, repo_root: Path):
    """A resumed run counts a recorded skip as complete; under the flag it
    must fail every run, not only the one that recorded the skip, also when
    the flag is turned on over skip rows recorded without it."""
    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(
        runner, "_run_subprocess", _fake_worker({"one_variable_eq": "skipped_unsupported"}, calls=calls)
    )
    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError, match="fail_on_unsupported"):
        run_benchmark(_synthetic_config(tmp_path, fail_on_unsupported=True), run_dir=run_dir, repo_root=repo_root)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="one_variable_eq.*skipped_unsupported.*cone 's' not supported"):
            run_benchmark(_synthetic_config(tmp_path, fail_on_unsupported=True), run_dir=run_dir, repo_root=repo_root)
        assert not (run_dir / "results.parquet").exists()

    other = tmp_path / "other"
    run_benchmark(_synthetic_config(tmp_path), run_dir=other, repo_root=repo_root)
    assert (other / "results.parquet").exists()
    with pytest.raises(RuntimeError, match="one_variable_eq.*skipped_unsupported"):
        run_benchmark(_synthetic_config(tmp_path, fail_on_unsupported=True), run_dir=other, repo_root=repo_root)


def _fake_sdplib(root: Path, names) -> Path:
    import io
    import tarfile

    folder = root / "sdplib_data"
    folder.mkdir(parents=True)
    with tarfile.open(folder / "sdplib.tar", "w") as archive:
        for name in names:
            info = tarfile.TarInfo(f"sdplib/{name}.jld2")
            info.size = 4
            archive.addfile(info, io.BytesIO(b"fake"))
    return root


def test_a_mistyped_include_name_is_an_error_on_a_dataset_that_knows_all_its_names(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """SDPLIB lists every member of its archive and Kennington knows its
    bundled set: a name outside the set is a typo, not a missing download.
    A run-level include shared with such a dataset keeps another dataset's
    typo an error too."""
    from solver_benchmarks.core import runner

    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker())
    data_root = _fake_sdplib(tmp_path / "data", ["truss1", "truss2", "hinf1"])

    def config(datasets, **run):
        return parse_run_config(
            {
                "run": {"datasets": datasets, "dataset_options": {"data_root": str(data_root)},
                        "output_dir": str(tmp_path / "runs"), **run},
                "solvers": [{"id": "scs", "solver": "scs"}],
            }
        )

    with pytest.raises(RuntimeError, match=r"'Truss2' \(did you mean 'truss2'\?\)"):
        run_benchmark(config(["sdplib"], include=["truss1", "Truss2"]), run_dir=tmp_path / "a", repo_root=repo_root)
    with pytest.raises(RuntimeError, match="one_variable_EQ"):
        run_benchmark(
            config(["synthetic_qp", "sdplib"], include=["one_variable_EQ", "truss1"]),
            run_dir=tmp_path / "b", repo_root=repo_root,
        )

    kennington = data_root / "kennington"
    kennington.mkdir()
    import shutil

    shutil.copy(repo_root / "problem_classes" / "kennington" / "ken-07.mps.gz", kennington)
    with pytest.raises(RuntimeError, match=r"'KEN-11' \(did you mean 'ken-11'\?\)"):
        run_benchmark(config(["kennington"], include=["ken-07", "KEN-11"]), run_dir=tmp_path / "c", repo_root=repo_root)


def test_a_known_name_that_is_not_local_yet_is_an_error_with_a_command_that_applies(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    import shutil

    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    data_root = tmp_path / "data"
    (data_root / "kennington").mkdir(parents=True)
    shutil.copy(repo_root / "problem_classes" / "kennington" / "ken-07.mps.gz", data_root / "kennington")
    config = parse_run_config(
        {
            "run": {"dataset": "kennington", "dataset_options": {"data_root": str(data_root)},
                    "include": ["ken-07", "ken-11"], "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    with pytest.raises(RuntimeError) as raised:
        run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    message = str(raised.value)
    assert "'ken-11'" in message
    assert "bench data prepare kennington" in message and "--problem ken-11" in message
    assert "DATASET" not in message
    assert calls == []

    # Preparing first (Kennington copies from the checkout) solves both.
    prepared = parse_run_config(
        {
            "run": {"dataset": "kennington", "dataset_options": {"data_root": str(data_root)},
                    "include": ["ken-07", "ken-11"], "output_dir": str(tmp_path / "runs"),
                    "auto_prepare_data": True},
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    store = run_benchmark(prepared, run_dir=tmp_path / "run", repo_root=repo_root)
    assert sorted(calls) == ["ken-07", "ken-11"]
    assert not [e for e in _events(store.run_dir) if e["level"] == "warning"]


# --- requested problems that are not on disk ------------------------------------


def _qplib_config(tmp_path: Path, **run):
    return parse_run_config(
        {
            "run": {
                "dataset": "qplib",
                "dataset_options": {"subset": [8790, 8515, 8495], "data_root": str(tmp_path / "data")},
                "output_dir": str(tmp_path / "runs"),
                **run,
            },
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )


def _qplib_files(tmp_path: Path, *ids: str) -> Path:
    folder = tmp_path / "data" / "qplib_data"
    folder.mkdir(parents=True, exist_ok=True)
    for problem_id in ids:
        (folder / f"QPLIB_{problem_id}.qplib").write_text("")
    return folder


def test_subset_problems_a_downloadable_dataset_does_not_have_stop_the_run(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """One of three requested QPLIB problems is on disk: the run used to solve
    that one and say nothing about the other two."""
    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    _qplib_files(tmp_path, "8790")

    with pytest.raises(RuntimeError) as raised:
        run_benchmark(_qplib_config(tmp_path), run_dir=tmp_path / "run", repo_root=repo_root,
                      prepare_data_command="bench run cfg.yaml --prepare-data")
    message = str(raised.value)
    assert "does not have the requested problems '8495', '8515' locally" in message
    assert "bench data prepare qplib" in message
    assert "--problem 8495 --problem 8515" in message
    assert "bench run cfg.yaml --prepare-data" in message
    assert calls == []


def test_subset_problems_are_prepared_first_with_auto_prepare(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    from solver_benchmarks.core import runner
    from solver_benchmarks.datasets import qplib as qplib_module

    calls: list[str] = []
    downloads: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    folder = _qplib_files(tmp_path, "8790")

    def fake_download(name, target_folder):
        downloads.append(name)
        (target_folder / f"QPLIB_{name}.qplib").write_text("")

    monkeypatch.setattr(qplib_module, "download_qplib_problem", fake_download)
    store = run_benchmark(_qplib_config(tmp_path, auto_prepare_data=True), run_dir=tmp_path / "run",
                          repo_root=repo_root)

    assert sorted(downloads) == ["8495", "8515"]
    assert sorted(calls) == ["8495", "8515", "8790"]
    assert sorted(path.name for path in folder.iterdir()) == [
        "QPLIB_8495.qplib", "QPLIB_8515.qplib", "QPLIB_8790.qplib",
    ]
    assert not [e for e in _events(store.run_dir) if e["level"] == "warning"]


def test_subset_problems_a_manual_dataset_does_not_have_are_a_warning(
    monkeypatch, tmp_path: Path, repo_root: Path, capsys
):
    """CUTEst exports cannot be downloaded: the run solves what is there,
    says which requested problems are missing and how they are made, and the
    completion report plans the same problems."""
    import numpy as np

    from solver_benchmarks.analysis.tables import completion_summary, missing_results
    from solver_benchmarks.core import runner

    calls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    folder = tmp_path / "data" / "cutest_qp_data"
    folder.mkdir(parents=True)
    np.savez(folder / "HS35.npz", P=np.eye(1), q=np.ones(1))
    config = parse_run_config(
        {
            "run": {
                "dataset": "cutest_qp",
                "dataset_options": {"subset": "HS35,QAFIRO,HS35MOD", "data_root": str(tmp_path / "data")},
                "output_dir": str(tmp_path / "runs"),
            },
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root, stream_output=True)

    assert calls == ["HS35"]
    (warning,) = [e for e in _events(store.run_dir) if e["level"] == "warning"]
    assert warning["problems"] == ["HS35MOD", "QAFIRO"]
    assert "python scripts/prepare_cutest_qp.py" in warning["message"]
    assert "[bench] warning: Dataset 'cutest_qp' does not have 'HS35MOD', 'QAFIRO'" in capsys.readouterr().err
    assert completion_summary(store.run_dir, repo_root=repo_root)["expected"].sum() == 1
    assert missing_results(store.run_dir, repo_root=repo_root).empty


def test_a_run_level_include_only_downloadable_datasets_may_have_is_missing_data(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    from solver_benchmarks.core import runner

    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker())
    monkeypatch.setitem(dataset_registry.DATASETS, "downloadable_fixture", _DownloadableDataset)
    config = parse_run_config(
        {
            "run": {"datasets": ["downloadable_fixture", "synthetic_qp"],
                    "include": ["local", "one_variable_eq", "remote_one"],
                    "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )
    with pytest.raises(RuntimeError, match="'downloadable_fixture'.*requested problem 'remote_one'"):
        run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)


class _KnownNamesDataset(Dataset):
    """Knows its full set of names and selects them with a subset."""

    dataset_id = "known_names_fixture"
    description = "Two possible problems, one local."
    automatic_download = True
    option_names = frozenset({"subset"})

    def known_problem_names(self):
        return {"alpha", "beta"}

    def requested_problem_names(self):
        return set(self.options.get("subset") or ())

    def list_problems(self):
        return [ProblemSpec(dataset_id="known_names_fixture", name="alpha", kind=QP)]

    def load_problem(self, name):  # pragma: no cover - the worker is stubbed
        raise AssertionError


def test_a_subset_name_the_dataset_cannot_have_is_a_typo(monkeypatch, tmp_path: Path, repo_root: Path):
    from solver_benchmarks.core import runner

    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker())
    monkeypatch.setitem(dataset_registry.DATASETS, "known_names_fixture", _KnownNamesDataset)
    config = parse_run_config(
        {
            "run": {"dataset": "known_names_fixture", "dataset_options": {"subset": ["alpha", "Beta"]},
                    "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )
    with pytest.raises(RuntimeError, match=r"no problem named 'Beta' \(did you mean 'beta'\?\) \(its subset option\)"):
        run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)


@pytest.mark.parametrize("auto_prepare", [False, True])
@pytest.mark.parametrize(
    "filters", [{"include": ["8790"]}, {"exclude": ["8515", "8495"]}], ids=["include", "exclude"]
)
def test_subset_problems_the_filters_leave_out_are_not_missing(
    monkeypatch, tmp_path: Path, repo_root: Path, auto_prepare, filters
):
    """A name subset narrowed by include / exclude to what is on disk runs,
    and preparation fetches nothing the run will not solve."""
    from solver_benchmarks.core import runner
    from solver_benchmarks.datasets import qplib as qplib_module

    calls: list[str] = []
    downloads: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))
    monkeypatch.setattr(qplib_module, "download_qplib_problem", lambda name, folder: downloads.append(name))
    _qplib_files(tmp_path, "8790")

    store = run_benchmark(_qplib_config(tmp_path, auto_prepare_data=auto_prepare, **filters),
                          run_dir=tmp_path / "run", repo_root=repo_root)

    assert calls == ["8790"]
    assert downloads == []
    assert not [e for e in _events(store.run_dir) if "locally" in e["message"]]


def test_a_missing_subset_problem_the_run_would_solve_still_stops_it(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    from solver_benchmarks.core import runner

    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker())
    _qplib_files(tmp_path, "8790")
    with pytest.raises(RuntimeError) as raised:
        run_benchmark(_qplib_config(tmp_path, include=["8790", "8515"]), run_dir=tmp_path / "run",
                      repo_root=repo_root)
    headline = str(raised.value).splitlines()[0]
    assert "does not have the requested problem '8515' locally" in headline
    assert "8495" not in headline


def test_an_include_name_outside_the_subset_is_not_missing_data(monkeypatch, tmp_path: Path, repo_root: Path):
    from solver_benchmarks.core import runner

    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker())
    _qplib_files(tmp_path, "8790", "8515")
    config = parse_run_config(
        {
            "run": {"dataset": "qplib",
                    "dataset_options": {"subset": [8790], "data_root": str(tmp_path / "data")},
                    "include": ["8790", "8515"], "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )
    with pytest.raises(RuntimeError) as raised:
        run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    message = str(raised.value)
    assert "no problem named '8515'" in message
    assert "outside dataset_options.subset: add 8515 to it" in message


def _libsvm_config(tmp_path: Path, **run):
    options = {"data_root": str(tmp_path / "data"), **run.pop("dataset_options", {})}
    return parse_run_config(
        {
            "run": {"dataset": "libsvm_qp", "dataset_options": options,
                    "output_dir": str(tmp_path / "runs"), **run},
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )


@pytest.mark.parametrize("on_disk", [False, True])
@pytest.mark.parametrize("auto_prepare", [False, True])
def test_a_libsvm_include_outside_the_default_six_says_how_to_select_it(
    monkeypatch, tmp_path: Path, repo_root: Path, on_disk, auto_prepare
):
    """Without ``subset`` libsvm_qp lists its six defaults: ``svm_dual_a1a``
    cannot be listed whatever is on disk, and the error says so. Following
    its advice (``subset: a1a``) then runs the problem."""
    from solver_benchmarks.core import runner
    from solver_benchmarks.datasets import libsvm_qp as libsvm_module

    calls: list[str] = []
    urls: list[str] = []
    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker(calls=calls))

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"+1 1:0.5\n-1 1:1.5\n"

    def fake_urlopen(url, timeout=None):
        urls.append(url)
        return _Response()

    monkeypatch.setattr(libsvm_module.urllib.request, "urlopen", fake_urlopen)
    folder = tmp_path / "data" / "libsvm_data"
    folder.mkdir(parents=True)
    (folder / "heart.libsvm").write_text("+1 1:0.5\n-1 1:1.5\n")
    if on_disk:
        (folder / "a1a.libsvm").write_text("+1 1:0.5\n-1 1:1.5\n")

    with pytest.raises(RuntimeError) as raised:
        run_benchmark(
            _libsvm_config(tmp_path, include=["svm_dual_a1a"], auto_prepare_data=auto_prepare),
            run_dir=tmp_path / "run", repo_root=repo_root,
        )
    message = str(raised.value)
    assert "no problem named 'svm_dual_a1a'" in message
    assert "not in the dataset's default subset" in message and "subset: a1a" in message

    store = run_benchmark(
        _libsvm_config(tmp_path, include=["svm_dual_a1a"], auto_prepare_data=True,
                       dataset_options={"subset": "a1a"}),
        run_dir=tmp_path / "run2", repo_root=repo_root,
    )
    assert calls == ["svm_dual_a1a"]
    assert urls == ([] if on_disk else [f"{libsvm_module.LIBSVM_BASE_URL}/a1a"])
    assert not [e for e in _events(store.run_dir) if e["level"] == "warning"]


def test_the_missing_data_error_keeps_did_you_mean(monkeypatch, tmp_path: Path, repo_root: Path):
    from solver_benchmarks.core import runner

    monkeypatch.setattr(runner, "_run_subprocess", _fake_worker())
    _qplib_files(tmp_path, "8790")
    config = parse_run_config(
        {
            "run": {"datasets": ["qplib", "netlib"],
                    "dataset_options": {"data_root": str(tmp_path / "data")},
                    "include": ["8790", "afiro", "afirox"], "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "osqp", "solver": "osqp"}],
        }
    )
    import shutil

    netlib = tmp_path / "data" / "netlib_data" / "feasible"
    netlib.mkdir(parents=True)
    shutil.copy(repo_root / "problem_classes" / "netlib_data" / "feasible" / "afiro.mps.gz", netlib)
    with pytest.raises(RuntimeError) as raised:
        run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    message = str(raised.value)
    assert "'afirox' (did you mean 'afiro'?) locally: not downloaded yet, or misspelled" in message
