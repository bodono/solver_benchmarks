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

import pytest

from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.problem import QP, ProblemSpec
from solver_benchmarks.core.runner import _run_subprocess as _real_run_subprocess
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


@pytest.mark.parametrize("parallelism", [2, 4])
def test_parallel_run_writes_one_record_per_solve_no_torn_lines(
    monkeypatch, fake_worker, tmp_path: Path, repo_root: Path, parallelism: int
):
    monkeypatch.setitem(
        dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset
    )
    monkeypatch.setitem(
        solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter
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
    monkeypatch, fake_worker, tmp_path: Path, repo_root: Path
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
    monkeypatch, fake_worker, tmp_path: Path, repo_root: Path, parallelism: int
):
    """An OSError launching one worker (EMFILE, ENOMEM, ...) must not abort
    the run and throw away every other result."""
    monkeypatch.setitem(dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter)

    def flaky(cmd, **kwargs):
        if json.loads(Path(cmd[-1]).read_text())["problem"] == "p003":
            raise OSError(24, "Too many open files")
        return fake_worker(cmd, **kwargs)

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
    monkeypatch, fake_worker, tmp_path: Path, repo_root: Path, capsys
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

    def worker(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, **_kwargs):
        problem = json.loads(Path(cmd[-1]).read_text())["problem"]
        with lock:
            started.append(problem)
        if problem == "p000":
            return fake_worker(cmd, stdout_path=stdout_path, stderr_path=stderr_path)
        # A long solve, killed only if the runner kills its process group.
        return real_run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)", marker],
            cwd=cwd, timeout=None, stdout_path=stdout_path, stderr_path=stderr_path,
            stream_output=False,
        )

    real_run_subprocess = _real_run_subprocess
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
    monkeypatch, fake_worker, tmp_path: Path, repo_root: Path
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

    def worker(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, **_kwargs):
        problem = json.loads(Path(cmd[-1]).read_text())["problem"]
        with lock:
            started.append(problem)
        if problem == "p000":
            return fake_worker(cmd, stdout_path=stdout_path, stderr_path=stderr_path)
        return real_run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)", marker],
            cwd=cwd, timeout=None, stdout_path=stdout_path, stderr_path=stderr_path,
            stream_output=False,
        )

    real_run_subprocess = _real_run_subprocess
    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", worker)
    real_record = runner._ProgressReporter.record_result

    def record_then_terminate(self, result):
        real_record(self, result)
        deadline = time.monotonic() + 10
        while len(started) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.5)
        # Without the handler, SIGTERM would end the whole pytest session.
        assert signal.getsignal(signal.SIGTERM) is not previous
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


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only: relies on process groups")
def test_an_interrupt_writes_results_that_finished_but_were_not_yet_written(
    monkeypatch, fake_worker, tmp_path: Path, repo_root: Path
):
    """The interrupt comes while p000 is being recorded, after p001 and p002
    had finished but before the loop took them: p001 is written all the
    same, and p002, a solve whose settings the solver rejected, is not."""
    import subprocess
    import sys
    import threading
    import time
    import uuid

    from solver_benchmarks.core import runner

    monkeypatch.setitem(dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter)
    marker = f"bench-unconsumed-test-{uuid.uuid4().hex}"
    instant = {"p000", "p001", "p002"}
    fake_worker.outcomes["p002"] = {
        "status": "worker_error", "objective_value": None,
        "error": "SolverConfigError: Unrecognized settings ['max_itr']",
        "metadata": {"error_kind": "config"},
    }

    def worker(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, **_kwargs):
        if json.loads(Path(cmd[-1]).read_text())["problem"] in instant:
            return fake_worker(cmd, stdout_path=stdout_path, stderr_path=stderr_path)
        return _real_run_subprocess(
            [sys.executable, "-c", "import time; time.sleep(60)", marker],
            cwd=cwd, timeout=None, stdout_path=stdout_path, stderr_path=stderr_path,
            stream_output=False,
        )

    monkeypatch.setattr(runner, "_run_subprocess", worker)
    finished: set[str] = set()
    all_finished = threading.Event()
    first_recorded = threading.Event()
    real_run_task = runner._run_task

    def run_task(store, config, repo_root, dataset_config, problem, *args, **kwargs):
        if problem.name in instant - {"p000"}:
            # Finish only once the loop has taken p000, so these stay unconsumed.
            assert first_recorded.wait(timeout=10)
        result = real_run_task(store, config, repo_root, dataset_config, problem, *args, **kwargs)
        if problem.name in instant:
            finished.add(problem.name)
            if finished == instant:
                all_finished.set()
        return result

    monkeypatch.setattr(runner, "_run_task", run_task)
    real_record = runner._ProgressReporter.record_result
    recorded: list[str] = []

    def record_then_interrupt(self, result):
        real_record(self, result)
        recorded.append(result.problem)
        first_recorded.set()
        assert all_finished.wait(timeout=10)
        time.sleep(0.2)  # the pool thread marks the future done
        raise KeyboardInterrupt

    monkeypatch.setattr(runner._ProgressReporter, "record_result", record_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(_parallel_config(tmp_path, 3), run_dir=tmp_path / "run", repo_root=repo_root)

    rows = [json.loads(line) for line in (tmp_path / "run" / "results.jsonl").read_text().splitlines()]
    assert recorded == ["p000"]
    assert sorted(row["problem"] for row in rows) == ["p000", "p001"]
    leftover = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    assert leftover.stdout.strip() == ""


def test_an_interrupt_while_a_result_is_written_still_writes_it(
    monkeypatch, fake_worker, tmp_path: Path, repo_root: Path
):
    """A Ctrl+C that lands inside ``write_result``, before the row is
    appended, must not lose the result: the stopping path writes it."""
    from solver_benchmarks.core.storage import ResultStore

    monkeypatch.setitem(dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter)
    real_write = ResultStore.write_result
    interrupted: list[str] = []

    def write_result(self, result):
        if not interrupted:
            interrupted.append(result.problem)
            raise KeyboardInterrupt
        real_write(self, result)

    monkeypatch.setattr(ResultStore, "write_result", write_result)
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(_parallel_config(tmp_path, 2), run_dir=tmp_path / "run", repo_root=repo_root)

    rows = [json.loads(line) for line in (tmp_path / "run" / "results.jsonl").read_text().splitlines()]
    problems = [row["problem"] for row in rows]
    assert interrupted[0] in problems
    assert len(problems) == len(set(problems))


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only: relies on process groups")
def test_a_worker_started_after_its_pool_was_aborted_is_killed():
    """A pool thread can launch its worker just after the main thread aborted
    the pool; ``add`` kills it at once. Workers of other pools are left alone."""
    import signal
    import subprocess
    import sys
    import threading

    from solver_benchmarks.core.runner import _WorkerProcesses

    workers = _WorkerProcesses()
    workers.abort("pool-x")
    processes: dict[str, subprocess.Popen] = {}

    def start(name):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        processes[name] = process
        workers.add(process)

    for name in ("pool-x_0", "pool-y_0"):
        thread = threading.Thread(target=start, args=(name,), name=name)
        thread.start()
        thread.join()
    try:
        assert processes["pool-x_0"].wait(timeout=10) == -signal.SIGKILL
        assert processes["pool-y_0"].poll() is None
    finally:
        for process in processes.values():
            process.kill()
            process.wait()
