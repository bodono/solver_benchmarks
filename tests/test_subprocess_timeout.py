"""Pin the subprocess timeout / cleanup contract used by the runner.

The previous implementation never had a test for the timeout path,
which is the entire reason the audit-driven hardening (process
group, bounded waits, drain on teardown) was necessary.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from solver_benchmarks.core.runner import _run_subprocess


def test_run_subprocess_timeout_returns_timed_out_and_writes_logs(tmp_path: Path):
    """A child that hangs past the timeout is killed cleanly and the
    log files reflect the partial output, with timed_out=True."""
    script = (
        "import sys, time;"
        "sys.stdout.write('starting\\n'); sys.stdout.flush();"
        "time.sleep(60)"
    )
    result = _run_subprocess(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        timeout=1.0,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stream_output=False,
    )
    assert result.timed_out is True
    # SIGTERM/SIGKILL produce a non-zero / negative returncode; the
    # important contract is that we don't hang.
    assert result.returncode != 0
    # Log file must contain at least the line emitted before the sleep.
    stdout_log = (tmp_path / "stdout.log").read_text()
    assert "starting" in stdout_log


def _locked_helper_script(tmp_path: Path, *, parent_body: str, helper_setup: str = "") -> Path:
    """A worker that forks a helper holding an exclusive flock until it dies.

    The kernel releases a process's locks when it exits, before the process
    is reaped, so acquiring the lock proves the helper is dead. Probing with
    kill(pid, 0) does not: it succeeds on an unreaped zombie, and an orphan
    waits for PID 1 to reap it (on a fixed two-second cycle in some
    sandboxes, never in a container without an init process).
    """
    ready = tmp_path / "helper.ready"
    lock = tmp_path / "helper.lock"
    script = tmp_path / "worker.py"
    script.write_text(
        "import fcntl, os, pathlib, signal, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        f"    fd = os.open({str(lock)!r}, os.O_CREAT | os.O_RDWR)\n"
        "    fcntl.flock(fd, fcntl.LOCK_EX)\n"
        + "".join(f"    {line}\n" for line in helper_setup.splitlines())
        + f"    pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        f"while not pathlib.Path({str(ready)!r}).exists():\n"
        "    time.sleep(0.01)\n"
        + parent_body
    )
    return script


def _assert_helper_dies(tmp_path: Path, *, within: float = 10.0) -> None:
    import fcntl

    ready = tmp_path / "helper.ready"
    assert ready.exists(), "helper never started; the fixture misfired"
    fd = os.open(tmp_path / "helper.lock", os.O_RDWR)
    try:
        deadline = time.monotonic() + within
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() > deadline:
                    os.kill(int(ready.read_text()), 9)
                    pytest.fail("helper process survived _run_subprocess")
                time.sleep(0.02)
    finally:
        os.close(fd)


# Below the 10 s the log readers' joins stall for while a helper holds the
# pipes, with room for a loaded host; the flock check proves the kill.
_STALL_BOUND_SECONDS = 8.0


def _run(script: Path, tmp_path: Path, *, timeout: float | None, stream_output=False):
    return _run_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout=timeout,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stream_output=stream_output,
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only: relies on process groups")
@pytest.mark.parametrize("helper_setup", ["", "signal.signal(signal.SIGTERM, signal.SIG_IGN)"],
                         ids=["plain", "ignores-sigterm"])
def test_run_subprocess_timeout_kills_grandchildren(tmp_path: Path, helper_setup: str):
    """If the worker forks helpers, killing the direct child must not leave
    them running, even when a helper ignores SIGTERM."""
    script = _locked_helper_script(
        tmp_path,
        helper_setup=helper_setup,
        parent_body="sys.stdout.write('parent_started\\n'); sys.stdout.flush()\ntime.sleep(60)\n",
    )
    # Generous: the timeout only has to cover interpreter start and the fork.
    result = _run(script, tmp_path, timeout=5.0)
    assert result.timed_out is True
    _assert_helper_dies(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only: relies on process groups")
def test_run_subprocess_sweeps_helpers_after_a_normal_exit(tmp_path: Path):
    """A helper still holding the output pipes must not outlive the worker
    or stall the log readers."""
    script = _locked_helper_script(tmp_path, parent_body="print('done')\n")
    started = time.monotonic()
    result = _run(script, tmp_path, timeout=30.0)
    assert result.returncode == 0 and result.timed_out is False
    assert time.monotonic() - started < _STALL_BOUND_SECONDS
    _assert_helper_dies(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only: relies on process groups")
def test_run_subprocess_kills_the_worker_when_interrupted(tmp_path: Path, monkeypatch):
    """Workers run in their own session, so a terminal Ctrl+C reaches only
    the runner. The runner must kill the worker before re-raising."""
    script = _locked_helper_script(tmp_path, parent_body="time.sleep(60)\n")
    real_wait = subprocess.Popen.wait
    interrupted = []

    def wait(self, timeout=None):
        if not interrupted:
            interrupted.append(True)
            while not (tmp_path / "helper.ready").exists():
                time.sleep(0.01)
            raise KeyboardInterrupt
        return real_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run(script, tmp_path, timeout=120.0)
    assert time.monotonic() - started < _STALL_BOUND_SECONDS
    _assert_helper_dies(tmp_path)


def test_run_subprocess_keeps_reading_after_undecodable_bytes(tmp_path: Path):
    script = tmp_path / "latin1.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'iter 1  r\\xe9sidu 1e-3\\n'); sys.stdout.flush()\n"
        "for i in range(2, 200):\n"
        "    print(f'iter {i}')\n"
    )
    result = _run(script, tmp_path, timeout=30.0)
    assert result.returncode == 0
    log = (tmp_path / "stdout.log").read_text()
    assert "r\\xe9sidu" in log
    assert "iter 199" in log


def test_run_subprocess_keeps_draining_when_the_terminal_fails(tmp_path: Path, monkeypatch):
    class ClosedTerminal:
        def write(self, text):
            raise OSError(32, "Broken pipe")

        def flush(self):
            raise OSError(32, "Broken pipe")

    monkeypatch.setattr(sys, "stdout", ClosedTerminal())
    script = tmp_path / "chatty.py"
    script.write_text("for i in range(5000):\n    print(f'iter {i}')\n")
    result = _run(script, tmp_path, timeout=30.0, stream_output=True)
    assert result.returncode == 0
    assert "iter 4999" in (tmp_path / "stdout.log").read_text()


def test_run_subprocess_output_reaches_the_log_before_a_timeout_kill(tmp_path: Path, monkeypatch):
    """Python block-buffers stdout on a pipe, and SIGTERM drops the buffer."""
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    script = tmp_path / "slow.py"
    script.write_text(
        "import time\nfor i in range(100000):\n    print(f'iter {i}')\n    time.sleep(0.01)\n"
    )
    result = _run(script, tmp_path, timeout=2.0)
    assert result.timed_out is True
    assert "iter 1\n" in (tmp_path / "stdout.log").read_text()


def test_run_subprocess_returns_only_a_bounded_tail(tmp_path: Path):
    script = tmp_path / "verbose.py"
    script.write_text("for i in range(20000):\n    print(f'iter {i}')\n")
    result = _run(script, tmp_path, timeout=30.0)
    assert result.returncode == 0
    assert len((tmp_path / "stdout.log").read_text().splitlines()) == 20000
    tail = result.stdout.splitlines()
    assert 0 < len(tail) <= 1000 and tail[-1] == "iter 19999"


class _OneProblemDataset:
    def __init__(self, repo_root=None, **options):
        pass

    def list_problems(self):
        from solver_benchmarks.core.problem import QP, ProblemSpec

        return [ProblemSpec(dataset_id="timeout_fixture", name="slow", kind=QP)]


def _timeout_run(monkeypatch, tmp_path: Path, repo_root: Path, *, timeout, worker):
    """Run one stubbed solve; ``worker(timeout)`` returns (namespace, record fields)."""
    import json

    from solver_benchmarks.core.config import parse_run_config
    from solver_benchmarks.core.result import ProblemResult
    from solver_benchmarks.core.runner import run_benchmark
    from solver_benchmarks.datasets import registry as dataset_registry
    from solver_benchmarks.solvers import registry as solver_registry
    from solver_benchmarks.solvers.base import SolverAdapter

    class Stub(SolverAdapter):
        supported_problem_kinds = {"qp"}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    seen = {}

    def fake(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        seen["timeout"] = timeout
        seen["settings"] = payload["solver"]["settings"]
        completed, fields = worker(timeout)
        if fields is not None:
            record = ProblemResult(
                run_id=payload["run_id"], dataset=payload["dataset"], problem="slow",
                problem_kind="qp", solver_id="stub", solver="stub", objective_value=0.0,
                iterations=1, artifact_dir=payload["artifacts_dir"], **fields,
            ).to_record()
            (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return completed

    from types import SimpleNamespace

    monkeypatch.setitem(dataset_registry.DATASETS, "timeout_fixture", _OneProblemDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", Stub)
    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", fake)
    config = parse_run_config(
        {
            "run": {"dataset": "timeout_fixture", "timeout_seconds": timeout},
            "solvers": [{"id": "stub", "solver": "stub", "settings": {"eps": 1e-6}}],
        }
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    rows = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    assert len(rows) == 1
    return rows[0], seen, SimpleNamespace


@pytest.mark.parametrize(
    "limit, kill_after",
    [(1.0, 6.0), (10.0, 20.0), (300.0, 330.0)],
)
def test_worker_is_killed_soon_after_the_time_limit(
    monkeypatch, tmp_path: Path, repo_root: Path, limit: float, kill_after: float
):
    """The kill deadline adds a grace for start-up and loading equal to the
    limit, between 5 and 30 s; a fixed 30 s let a 1 s limit run for 31 s."""
    from types import SimpleNamespace

    def worker(timeout):
        return SimpleNamespace(returncode=-15, timed_out=True), None

    row, seen, _ = _timeout_run(monkeypatch, tmp_path, repo_root, timeout=limit, worker=worker)
    assert seen["timeout"] == pytest.approx(kill_after)
    # The limit is enforced by the runner, not passed to the solver.
    assert seen["settings"] == {"eps": 1e-6}
    assert row["status"] == "time_limit"
    assert row["run_time_seconds"] == pytest.approx(limit)
    assert f"timeout_seconds={limit:g}" in row["error"]


@pytest.mark.parametrize(
    "limit, reported_status, run_time, expected",
    [
        (1.0, "optimal", 23.46, "time_limit"),
        (1.0, "optimal_inaccurate", 7.99, "time_limit"),
        (1.0, "primal_infeasible", 2.0, "time_limit"),
        (1.0, "optimal", 0.5, "optimal"),
        (1.0, "max_iter_reached", 23.0, "max_iter_reached"),
        (None, "optimal", 23.46, "optimal"),
        (0.0, "optimal", 23.46, "optimal"),
    ],
)
def test_success_after_the_time_limit_is_recorded_as_time_limit(
    monkeypatch, tmp_path: Path, repo_root: Path, limit, reported_status, run_time, expected
):
    from types import SimpleNamespace

    def worker(timeout):
        return (
            SimpleNamespace(returncode=0, timed_out=False),
            {"status": reported_status, "run_time_seconds": run_time},
        )

    row, _, _ = _timeout_run(monkeypatch, tmp_path, repo_root, timeout=limit, worker=worker)
    assert row["status"] == expected
    assert row["run_time_seconds"] == run_time
    if expected == "time_limit":
        assert row["info"]["status_before_timeout"] == reported_status
        assert "timeout_seconds=1" in row["error"]
    else:
        assert "status_before_timeout" not in row["info"]
