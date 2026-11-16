"""Pin the ThreadPoolExecutor branch of run_benchmark.

Every existing runner test uses parallelism=1, so the parallel branch
was unexercised. This test stubs out the worker subprocess with a
sleep-and-write fake adapter and runs many tasks across multiple
worker threads, then asserts that:

- All results land in results.jsonl with exactly one record each.
- No two records share the same (problem, solver_id).
- The on-disk JSONL has no torn / interleaved bytes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.problem import QP, ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.solvers import registry as solver_registry
from solver_benchmarks.solvers.base import SolverAdapter

PROBLEM_COUNT = 8


class _ManyProblemDataset:
    def __init__(self, repo_root=None, **options):
        pass

    def list_problems(self):
        return [
            ProblemSpec(dataset_id="parallel_fixture", name=f"p{i:03d}", kind=QP)
            for i in range(PROBLEM_COUNT)
        ]


class _StubSolverAdapter(SolverAdapter):
    solver_name = "parallel_stub"
    supported_problem_kinds = {QP}

    def solve(self, problem, artifacts_dir):  # pragma: no cover - subprocess stub
        raise AssertionError("Subprocess is stubbed in this test")


def _fake_run_subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, env=None):
    """Pretend to invoke the worker by reading the payload and writing
    a deterministic ProblemResult-shaped JSON next to it."""
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
        run_time_seconds=0.001,
        artifact_dir=str(artifact_dir),
    ).to_record()
    (artifact_dir / "worker_result.json").write_text(json.dumps(record))
    stdout_path.write_text("")
    stderr_path.write_text("")
    return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)


@pytest.mark.parametrize("parallelism", [2, 4])
def test_parallel_run_writes_one_record_per_solve_no_torn_lines(
    monkeypatch, tmp_path: Path, repo_root: Path, parallelism: int
):
    monkeypatch.setitem(
        dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset
    )
    monkeypatch.setitem(
        solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter
    )
    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", _fake_run_subprocess
    )

    config = parse_run_config(
        {
            "run": {
                "dataset": "parallel_fixture",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": parallelism,
            },
            "solvers": [
                {"id": "parallel_stub", "solver": "parallel_stub", "settings": {}}
            ],
        }
    )
    store = run_benchmark(config, repo_root=repo_root)

    text = store.results_jsonl_path.read_text()
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == PROBLEM_COUNT

    records = []
    for line in lines:
        # Every line must be parseable JSON; a torn append from
        # concurrent writes would surface here as a JSONDecodeError.
        records.append(json.loads(line))

    keys = {(r["dataset"], r["problem"], r["solver_id"]) for r in records}
    assert len(keys) == PROBLEM_COUNT  # no duplicates and no losses
    assert all(r["status"] == "optimal" for r in records)


def test_parallel_run_writes_parquet_at_end(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """write_result only appends to jsonl during the run; the runner
    must call store.write_parquet() at the end so the parquet reflects
    the complete jsonl."""
    monkeypatch.setitem(
        dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset
    )
    monkeypatch.setitem(
        solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter
    )
    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", _fake_run_subprocess
    )

    config = parse_run_config(
        {
            "run": {
                "dataset": "parallel_fixture",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 4,
            },
            "solvers": [
                {"id": "parallel_stub", "solver": "parallel_stub", "settings": {}}
            ],
        }
    )
    store = run_benchmark(config, repo_root=repo_root)

    import pandas as pd

    df = pd.read_parquet(store.results_parquet_path)
    assert len(df) == PROBLEM_COUNT


def _parallel_config(tmp_path: Path, parallelism: int = 2):
    return parse_run_config(
        {
            "run": {
                "dataset": "parallel_fixture",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": parallelism,
            },
            "solvers": [{"id": "parallel_stub", "solver": "parallel_stub", "settings": {}}],
        }
    )


@pytest.mark.parametrize("parallelism", [1, 2])
def test_a_failing_task_becomes_a_worker_error_row(
    monkeypatch, tmp_path: Path, repo_root: Path, parallelism: int
):
    """An OSError launching one worker (EMFILE, ENOMEM, ...) must not abort
    the run and throw away every other result."""
    monkeypatch.setitem(dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter)

    def flaky(cmd, **kwargs):
        if json.loads(Path(cmd[-1]).read_text())["problem"] == "p003":
            raise OSError(24, "Too many open files")
        return _fake_run_subprocess(cmd, **kwargs)

    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", flaky)
    store = run_benchmark(_parallel_config(tmp_path, parallelism), repo_root=repo_root)

    rows = {
        row["problem"]: row
        for row in map(json.loads, store.results_jsonl_path.read_text().splitlines())
    }
    assert len(rows) == PROBLEM_COUNT
    assert rows["p003"]["status"] == "worker_error"
    assert "Too many open files" in rows["p003"]["error"]
    assert rows["p003"]["metadata"]["resume_signature"]
    assert {row["status"] for name, row in rows.items() if name != "p003"} == {"optimal"}
    assert store.results_parquet_path.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only: relies on process groups")
def test_interrupting_a_parallel_run_stops_it_and_keeps_finished_results(
    monkeypatch, tmp_path: Path, repo_root: Path, capsys
):
    """Ctrl+C reaches only the runner (workers run in their own sessions).
    Queued solves must not start, running workers must be killed rather than
    recorded (nor reported as worker errors), and results that had already
    finished must be written."""
    import subprocess
    import sys
    import threading
    import time
    import uuid

    from solver_benchmarks.core import runner

    monkeypatch.setitem(dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter)
    started: list[str] = []
    lock = threading.Lock()
    # Names this test's sleepers, not those of a concurrent run of it.
    marker = f"bench-interrupt-test-{uuid.uuid4().hex}"

    def worker(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, env=None):
        problem = json.loads(Path(cmd[-1]).read_text())["problem"]
        with lock:
            started.append(problem)
        if problem == "p000":
            return _fake_run_subprocess(
                cmd, cwd=cwd, timeout=timeout, stdout_path=stdout_path,
                stderr_path=stderr_path, stream_output=stream_output,
            )
        # A long solve, killed only if the runner kills its process group.
        return real_run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)", marker],
            cwd=cwd, timeout=None, stdout_path=stdout_path, stderr_path=stderr_path,
            stream_output=False,
        )

    real_run_subprocess = runner._run_subprocess
    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", worker)
    real_record = runner._ProgressReporter.record_result

    def record_then_interrupt(self, result):
        real_record(self, result)
        # Let the second worker start its subprocess, then press Ctrl+C.
        deadline = time.monotonic() + 10
        while len(started) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.5)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner._ProgressReporter, "record_result", record_then_interrupt)
    started_at = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(
            _parallel_config(tmp_path), run_dir=tmp_path / "run", repo_root=repo_root, stream_output=True
        )
    assert time.monotonic() - started_at < 20
    rows = [json.loads(line) for line in (tmp_path / "run" / "results.jsonl").read_text().splitlines()]
    assert [row["problem"] for row in rows] == ["p000"]
    assert len(started) <= 3
    assert "worker error" not in capsys.readouterr().err
    leftover = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    assert leftover.stdout.strip() == ""


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only: relies on process groups")
def test_sigterm_to_bench_run_kills_workers_and_keeps_finished_results(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """SIGTERM (Slurm scancel, kill, CI timeouts) reaches only the runner, as
    Ctrl+C does. Under ``bench run``'s handler it must unwind the same way:
    running workers killed, finished results kept, exit status 128 + 15."""
    import signal
    import subprocess
    import sys
    import threading
    import time
    import uuid

    from solver_benchmarks.cli import terminate_like_interrupt
    from solver_benchmarks.core import runner

    monkeypatch.setitem(dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter)
    started: list[str] = []
    lock = threading.Lock()
    marker = f"bench-sigterm-test-{uuid.uuid4().hex}"

    def worker(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, env=None):
        problem = json.loads(Path(cmd[-1]).read_text())["problem"]
        with lock:
            started.append(problem)
        if problem == "p000":
            return _fake_run_subprocess(
                cmd, cwd=cwd, timeout=timeout, stdout_path=stdout_path,
                stderr_path=stderr_path, stream_output=stream_output,
            )
        return real_run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)", marker],
            cwd=cwd, timeout=None, stdout_path=stdout_path, stderr_path=stderr_path,
            stream_output=False,
        )

    real_run_subprocess = runner._run_subprocess
    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", worker)
    real_record = runner._ProgressReporter.record_result

    def record_then_terminate(self, result):
        real_record(self, result)
        deadline = time.monotonic() + 10
        while len(started) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(5)  # the handler raises in this (the main) thread first

    monkeypatch.setattr(runner._ProgressReporter, "record_result", record_then_terminate)
    previous = signal.getsignal(signal.SIGTERM)
    started_at = time.monotonic()
    with pytest.raises(SystemExit) as raised:
        with terminate_like_interrupt():
            run_benchmark(_parallel_config(tmp_path), run_dir=tmp_path / "run", repo_root=repo_root)
    assert raised.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is previous
    assert time.monotonic() - started_at < 20
    rows = [json.loads(line) for line in (tmp_path / "run" / "results.jsonl").read_text().splitlines()]
    assert [row["problem"] for row in rows] == ["p000"]
    leftover = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    assert leftover.stdout.strip() == ""
