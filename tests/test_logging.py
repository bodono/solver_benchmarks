import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.config import (
    DatasetConfig,
    RunConfig,
    SolverConfig,
    parse_run_config,
)
from solver_benchmarks.core.problem import CONE, QP, ProblemData, ProblemSpec
from solver_benchmarks.core.result import ProblemResult, SolverResult
from solver_benchmarks.core.runner import _run_one, _run_subprocess, run_benchmark
from solver_benchmarks.core.storage import ResultStore
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.solvers import registry as solver_registry
from solver_benchmarks.solvers.base import SolverAdapter
from solver_benchmarks.worker import run_payload


def test_warning_events_are_structured_for_unsupported_combinations(monkeypatch, tmp_path: Path):
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
    events = [
        json.loads(line)
        for line in store.events_path.read_text().splitlines()
        if line.strip()
    ]
    event = next(item for item in events if item["level"] == "warning")

    assert event["level"] == "warning"
    assert event["solver_id"] == "fake_qp_skip"
    assert event["problem"] == "cone_problem"
    assert event["problem_kind"] == "cone"
    assert "does not support" in event["message"]


def test_run_benchmark_logs_aggregate_progress(monkeypatch, tmp_path: Path, capsys):
    class FakeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [
                ProblemSpec(dataset_id="fake_dataset", name="p1", kind=QP),
                ProblemSpec(dataset_id="fake_dataset", name="p2", kind=QP),
            ]

    class FakeSolver(SolverAdapter):
        solver_name = "fake_solver"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError("Subprocess is stubbed in this test")

    def fake_run_subprocess(
        cmd,
        *,
        cwd,
        timeout,
        stdout_path,
        stderr_path,
        stream_output,
    ):
        payload = json.loads(Path(cmd[-1]).read_text())
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

    monkeypatch.setitem(dataset_registry.DATASETS, "fake_dataset", FakeDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "fake_solver", FakeSolver)
    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess",
        fake_run_subprocess,
    )
    config = parse_run_config(
        {
            "run": {
                "dataset": "fake_dataset",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
            },
            "solvers": [
                {"id": "fake_solver", "solver": "fake_solver", "settings": {}}
            ],
        }
    )

    store = run_benchmark(config, repo_root=Path.cwd(), stream_output=True)
    captured = capsys.readouterr()

    assert "[bench] planned 2 solves" in captured.err
    assert "0 already complete" in captured.err
    assert "2 queued" in captured.err
    assert "progress 1/2 (50.00%)" in captured.err
    assert "progress 2/2 (100.00%)" in captured.err
    assert "queued_done 2/2" in captured.err
    assert "last fake_dataset/p2 fake_solver: optimal" in captured.err

    events = [
        json.loads(line)
        for line in store.events_path.read_text().splitlines()
        if line.strip()
    ]
    assert [event["message"] for event in events] == [
        "benchmark_plan",
        "benchmark_progress",
        "benchmark_complete",
    ]
    assert events[0]["total_expected"] == 2
    assert events[0]["queued"] == 2
    assert events[-1]["completed_total"] == 2
    assert events[-1]["completed_this_run"] == 2
    assert events[-1]["last_problem"] == "p2"


def test_run_one_captures_subprocess_stdout_and_stderr(monkeypatch, tmp_path: Path):
    dataset_config = DatasetConfig(name="synthetic_qp")
    config = RunConfig(
        datasets=[dataset_config],
        output_dir=tmp_path / "runs",
        solvers=[SolverConfig(id="fake_solver", solver="scs")],
        timeout_seconds=10,
    )
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    problem = ProblemSpec(dataset_id="synthetic_qp", name="fake_problem", kind=QP)
    solver = SolverConfig(id="fake_solver", solver="scs")

    def fake_run_subprocess(
        cmd,
        *,
        cwd,
        timeout,
        stdout_path,
        stderr_path,
        stream_output,
    ):
        payload_path = Path(cmd[-1])
        payload = json.loads(payload_path.read_text())
        artifact_dir = Path(payload["artifacts_dir"])
        result = ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="fake_problem",
            problem_kind=QP,
            solver_id="fake_solver",
            solver="scs",
            status="optimal",
            objective_value=1.0,
            iterations=3,
            run_time_seconds=0.1,
            artifact_dir=str(artifact_dir),
        )
        (artifact_dir / "worker_result.json").write_text(json.dumps(result.to_record()))
        stdout_path.write_text("solver stdout\n")
        stderr_path.write_text("solver stderr\n")
        return SimpleNamespace(
            returncode=0,
            stdout="solver stdout\n",
            stderr="solver stderr\n",
            timed_out=False,
        )

    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", fake_run_subprocess)

    result = _run_one(store, config, Path.cwd(), dataset_config, problem, solver)
    artifact_dir = Path(result.artifact_dir)

    assert result.status == "optimal"
    assert (artifact_dir / "stdout.log").read_text() == "solver stdout\n"
    assert (artifact_dir / "stderr.log").read_text() == "solver stderr\n"


def test_run_subprocess_streams_while_writing_logs(tmp_path: Path, capsys):
    result = _run_subprocess(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "print('solver stdout'); "
                "print('solver stderr', file=sys.stderr)"
            ),
        ],
        cwd=Path.cwd(),
        timeout=10,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stream_output=True,
    )

    captured = capsys.readouterr()

    assert result.returncode == 0
    assert result.stdout == "solver stdout\n"
    assert result.stderr == "solver stderr\n"
    assert (tmp_path / "stdout.log").read_text() == "solver stdout\n"
    assert (tmp_path / "stderr.log").read_text() == "solver stderr\n"
    assert "solver stdout" in captured.out
    assert "solver stderr" in captured.err


def test_worker_writes_solver_trace(monkeypatch, tmp_path: Path):
    class FakeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def load_problem(self, name):
            qp = {
                "P": sp.csc_matrix([[1.0]]),
                "q": np.array([0.0]),
                "r": 0.0,
                "A": sp.csc_matrix([[1.0]]),
                "l": np.array([1.0]),
                "u": np.array([1.0]),
                "n": 1,
                "m": 1,
                "obj_type": "min",
            }
            return ProblemData("fake_dataset", name, QP, qp)

    class FakeSolver:
        def __init__(self, settings):
            self.settings = settings

        def solve(self, problem, artifacts_dir):
            return SolverResult(
                status="optimal",
                objective_value=0.5,
                iterations=2,
                run_time_seconds=0.01,
                info={"custom": "value"},
                trace=[{"iter": 1, "residual": 1.0}, {"iter": 2, "residual": 0.1}],
            )

    monkeypatch.setattr("solver_benchmarks.worker.get_dataset", lambda name: FakeDataset)
    monkeypatch.setattr("solver_benchmarks.worker.get_solver", lambda name: FakeSolver)

    payload = {
        "run_id": "run",
        "dataset": "fake_dataset",
        "dataset_options": {},
        "problem": "fake_problem",
        "problem_kind": QP,
        "solver": {"id": "fake_solver", "solver": "fake_solver", "settings": {}},
        "artifacts_dir": str(tmp_path / "artifacts"),
        "repo_root": str(Path.cwd()),
    }

    result = run_payload(payload)
    trace_path = Path(payload["artifacts_dir"]) / "trace.jsonl"
    trace_lines = [json.loads(line) for line in trace_path.read_text().splitlines()]

    assert result.status == "optimal"
    assert result.info == {"custom": "value"}
    assert trace_lines == [{"iter": 1, "residual": 1.0}, {"iter": 2, "residual": 0.1}]
