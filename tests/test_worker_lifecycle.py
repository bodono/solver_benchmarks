"""The worker process: what it imports at start-up, and that it never
outlives the runner that started it."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env() -> dict[str, str]:
    path = os.environ.get("PYTHONPATH")
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT) + (os.pathsep + path if path else "")}


def test_worker_start_up_does_not_import_pandas():
    """Every solve starts a worker inside its time budget; pandas is not
    needed to solve and costs a large share of the start-up."""
    code = (
        "import sys, solver_benchmarks.worker\n"
        "from solver_benchmarks.solvers import registry\n"
        "assert registry.SOLVERS\n"
        "assert 'pandas' not in sys.modules, 'pandas imported at worker start-up'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=_env(), capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


# A stand-in for `bench run`: registers a solver that sleeps, then runs one
# solve per problem (two problems) with the given parallelism under the CLI's
# signal handling. Each solve's worker holds an exclusive flock on its own
# file while it sleeps, so acquiring the lock proves the worker died.
_RUNNER = textwrap.dedent(
    """
    import fcntl, os, sys, time
    from pathlib import Path
    from solver_benchmarks.cli import terminate_like_interrupt
    from solver_benchmarks.core.config import parse_run_config
    from solver_benchmarks.core.runner import run_benchmark

    run_dir, parallelism = Path(sys.argv[1]), int(sys.argv[2])
    config = parse_run_config({
        "run": {"dataset": "synthetic_qp", "parallelism": parallelism, "timeout_seconds": 120},
        "solvers": [{"id": "sleepy", "solver": "sleepy", "settings": {"lock_dir": str(run_dir.parent)}}],
    })
    with terminate_like_interrupt():
        run_benchmark(config, run_dir=run_dir)
    """
)

_SITECUSTOMIZE = textwrap.dedent(
    """
    import os
    if os.environ.get("BENCH_TEST_SLEEPY"):
        import fcntl, time
        from pathlib import Path
        from solver_benchmarks.core.result import SolverResult
        from solver_benchmarks.solvers import registry
        from solver_benchmarks.solvers.base import SolverAdapter

        class Sleepy(SolverAdapter):
            solver_name = "sleepy"
            supported_problem_kinds = {"qp", "cone"}

            def solve(self, problem, artifacts_dir):
                lock_dir = Path(self.settings["lock_dir"])
                fd = os.open(lock_dir / f"{problem.name}.lock", os.O_CREAT | os.O_RDWR)
                fcntl.flock(fd, fcntl.LOCK_EX)
                (lock_dir / f"{problem.name}.ready").write_text(str(os.getpid()))
                time.sleep(120)
                return SolverResult(status="optimal", run_time_seconds=120.0)

        registry.SOLVERS["sleepy"] = Sleepy
    """
)

_PROBLEMS = ("one_variable_eq", "one_variable_lp")


def _start_runner(tmp_path: Path, *, parallelism: int, ignore_sighup: bool = False):
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    script = tmp_path / "runner.py"
    script.write_text(_RUNNER)
    env = _env()
    env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + str(site)
    env["BENCH_TEST_SLEEPY"] = "1"

    def preexec():
        # Set explicitly: pytest itself may run under nohup.
        signal.signal(signal.SIGHUP, signal.SIG_IGN if ignore_sighup else signal.SIG_DFL)

    process = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path / "run"), str(parallelism)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=preexec,
    )
    expected = _PROBLEMS[:parallelism]
    deadline = time.monotonic() + 60
    while not all((tmp_path / f"{name}.ready").exists() for name in expected):
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            pytest.fail("the workers never started")
        time.sleep(0.05)
    return process, expected


def _worker_is_dead(tmp_path: Path, name: str, *, within: float) -> bool:
    import fcntl

    fd = os.open(tmp_path / f"{name}.lock", os.O_RDWR)
    try:
        deadline = time.monotonic() + within
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() > deadline:
                    return False
                time.sleep(0.05)
    finally:
        os.close(fd)


def _kill_leftovers(tmp_path: Path, names) -> None:
    for name in names:
        try:
            os.kill(int((tmp_path / f"{name}.ready").read_text()), signal.SIGKILL)
        except (OSError, ValueError):
            pass


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
@pytest.mark.parametrize("parallelism", [1, 2], ids=["serial", "parallel"])
def test_sighup_to_the_runner_kills_its_workers(tmp_path: Path, parallelism: int):
    """Closing the terminal or losing the ssh session sends SIGHUP; the
    workers run in their own sessions, so only the runner gets it."""
    process, workers = _start_runner(tmp_path, parallelism=parallelism)
    try:
        process.send_signal(signal.SIGHUP)
        assert process.wait(timeout=30) == 128 + signal.SIGHUP
        for name in workers:
            assert _worker_is_dead(tmp_path, name, within=10), f"worker for {name} survived SIGHUP"
    finally:
        process.kill()
        _kill_leftovers(tmp_path, workers)


@pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="no SIGHUP on this platform")
def test_an_ignored_sighup_stays_ignored(tmp_path: Path):
    """`nohup bench run ...` must keep running when the terminal closes."""
    process, workers = _start_runner(tmp_path, parallelism=1, ignore_sighup=True)
    try:
        process.send_signal(signal.SIGHUP)
        time.sleep(1.0)
        assert process.poll() is None
        assert not _worker_is_dead(tmp_path, workers[0], within=0.5)
    finally:
        process.kill()
        process.wait(timeout=30)
        _kill_leftovers(tmp_path, workers)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parent-death signal is Linux-only")
@pytest.mark.parametrize("parallelism", [1, 2], ids=["serial", "parallel"])
def test_workers_die_with_a_runner_killed_hard(tmp_path: Path, parallelism: int):
    """SIGKILL (the OOM killer, kill -9, a scheduler's escalation) cannot be
    handled by the runner; its workers must not keep solving unsupervised."""
    process, workers = _start_runner(tmp_path, parallelism=parallelism)
    try:
        process.kill()
        process.wait(timeout=30)
        for name in workers:
            assert _worker_is_dead(tmp_path, name, within=10), f"worker for {name} survived SIGKILL"
    finally:
        _kill_leftovers(tmp_path, workers)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parent-death signal is Linux-only")
def test_worker_dies_with_its_runner_while_in_a_long_c_call(tmp_path: Path):
    """The worker must die even while the solver holds the GIL in C code,
    where no Python thread can run."""
    ready = tmp_path / "c.ready"
    lock = tmp_path / "c.lock"
    # The "runner" starts the worker module's guard in a child and is then
    # killed; the child is blocked in a C call (a huge regex backtrack holds
    # the GIL) after taking the lock.
    child = textwrap.dedent(
        f"""
        import fcntl, os, re, sys
        from solver_benchmarks import worker
        worker._die_with_runner(int(sys.argv[1]))
        fd = os.open({str(lock)!r}, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        open({str(ready)!r}, "w").write(str(os.getpid()))
        re.match(r"(a+)+$", "a" * 64 + "b")
        """
    )
    parent = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {child!r}, str(os.getpid())], start_new_session=True)
        time.sleep(600)
        """
    )
    process = subprocess.Popen([sys.executable, "-c", parent], cwd=REPO_ROOT, env=_env())
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert time.monotonic() < deadline, "child never started"
            time.sleep(0.05)
        process.kill()
        process.wait(timeout=30)
        assert _worker_is_dead(tmp_path, "c", within=10), "worker survived its runner"
    finally:
        process.kill()
        _kill_leftovers(tmp_path, ["c"])


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_the_polling_fallback_kills_a_worker_whose_runner_died(tmp_path: Path):
    """Where the kernel offers no parent-death signal, a daemon thread
    notices the runner is gone (while the solver lets Python run)."""
    ready = tmp_path / "p.ready"
    lock = tmp_path / "p.lock"
    child = textwrap.dedent(
        f"""
        import fcntl, os, sys, time
        from solver_benchmarks import worker
        from solver_benchmarks.core import parent_watch
        parent_watch._set_parent_death_signal = lambda signum: False
        worker._die_with_runner(int(sys.argv[1]))
        fd = os.open({str(lock)!r}, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        open({str(ready)!r}, "w").write(str(os.getpid()))
        time.sleep(600)
        """
    )
    parent = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", {child!r}, str(os.getpid())], start_new_session=True)
        time.sleep(600)
        """
    )
    process = subprocess.Popen([sys.executable, "-c", parent], cwd=REPO_ROOT, env=_env())
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert time.monotonic() < deadline, "child never started"
            time.sleep(0.05)
        process.kill()
        process.wait(timeout=30)
        assert _worker_is_dead(tmp_path, "p", within=10), "worker survived its runner"
    finally:
        process.kill()
        _kill_leftovers(tmp_path, ["p"])


def test_worker_guard_is_a_no_op_without_a_runner(monkeypatch):
    """Running the worker module by hand (no runner pid in the environment)
    installs nothing; the runner sets the variable for its workers."""
    from solver_benchmarks import worker
    from solver_benchmarks.core import runner

    assert worker._RUNNER_PID_ENV == runner.RUNNER_PID_ENV
    monkeypatch.delenv(worker._RUNNER_PID_ENV, raising=False)
    assert worker._runner_pid() is None
    assert worker._die_with_runner(None) is None
    monkeypatch.setenv(worker._RUNNER_PID_ENV, "1234")
    assert worker._runner_pid() == 1234


def test_a_qtqp_solve_without_collect_stats_does_not_import_pandas(tmp_path: Path):
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from solver_benchmarks.datasets import get_dataset\n"
        "from solver_benchmarks.solvers import get_solver\n"
        "problem = get_dataset('synthetic_qp')().load_problem('one_variable_eq')\n"
        "result = get_solver('qtqp')({}).solve(problem, Path(sys.argv[1]))\n"
        "assert result.status == 'optimal', result.status\n"
        "assert 'pandas' not in sys.modules, 'pandas imported by a QTQP solve'\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=REPO_ROOT, env=_env(), capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


@pytest.mark.skipif(os.name != "posix", reason="POSIX launcher script")
def test_a_worker_started_through_a_launcher_that_does_not_exec_solves_normally(
    monkeypatch, tmp_path: Path
):
    """When sys.executable is a launcher that runs the interpreter as its
    child (a wrapper script, a Windows venv's python.exe), the worker's
    parent is not the runner; that alone must not stop it."""
    import json

    from solver_benchmarks.core.config import parse_run_config
    from solver_benchmarks.core.runner import run_benchmark

    launcher = tmp_path / "python-launcher"
    launcher.write_text(f'#!/bin/sh\n"{sys.executable}" "$@"\n')
    launcher.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(launcher))
    config = parse_run_config(
        {"run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"]},
         "solvers": [{"id": "osqp", "solver": "osqp"}]}
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=REPO_ROOT)
    (row,) = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    assert row["status"] == "optimal", row["error"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX processes")
def test_a_worker_whose_runner_is_gone_says_why_and_exits(tmp_path: Path):
    from solver_benchmarks.core.runner import RUNNER_PID_ENV

    env = {**_env(), RUNNER_PID_ENV: str(_dead_pid())}
    completed = subprocess.run(
        [sys.executable, "-m", "solver_benchmarks.worker", "--payload", str(tmp_path / "unused.json")],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode != 0
    assert "is gone" in completed.stderr and env[RUNNER_PID_ENV] in completed.stderr
    assert not (tmp_path / "worker_result.json").exists()


def test_nothing_is_installed_on_windows(monkeypatch):
    """os.kill(pid, 0) terminates the process on Windows, and a venv's
    python.exe is a launcher, so the guard stays off there."""
    from solver_benchmarks.core import parent_watch

    def no_kill(*args):
        raise AssertionError("os.kill called")

    monkeypatch.setattr(parent_watch, "_windows", lambda: True)
    monkeypatch.setattr(parent_watch.os, "kill", no_kill)
    gone = []
    assert parent_watch.watch_parent(_dead_pid(), signum=9, on_gone=lambda: gone.append(1)) is None
    assert gone == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parent-death signal is Linux-only")
def test_a_matrix_child_dies_with_a_bench_env_run_killed_hard(tmp_path: Path):
    """A `bench env run` killed outright must not leave the current
    environment's `bench run` and its workers solving."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    config = tmp_path / "cfg.yaml"
    config.write_text(
        "run: {dataset: synthetic_qp, parallelism: 2, timeout_seconds: 120}\n"
        f"solvers: [{{id: sleepy, solver: sleepy, settings: {{lock_dir: '{tmp_path}'}}}}]\n"
    )
    env = _env()
    env["PYTHONPATH"] = env["PYTHONPATH"] + os.pathsep + str(site)
    env["BENCH_TEST_SLEEPY"] = "1"
    child = [sys.executable, "-m", "solver_benchmarks.cli", "run", str(config),
             "--run-dir", str(tmp_path / "run"), "--no-stream-output"]
    matrix = textwrap.dedent(
        f"""
        from pathlib import Path
        from solver_benchmarks.core import env_runner
        env_runner._run_child({child!r}, cwd=Path({str(REPO_ROOT)!r}))
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", matrix], cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_DFL),
    )
    try:
        deadline = time.monotonic() + 60
        while not all((tmp_path / f"{name}.ready").exists() for name in _PROBLEMS):
            if process.poll() is not None or time.monotonic() > deadline:
                process.kill()
                pytest.fail("the workers never started")
            time.sleep(0.05)
        process.kill()
        process.wait(timeout=30)
        for name in _PROBLEMS:
            assert _worker_is_dead(tmp_path, name, within=20), f"worker for {name} survived"
        lock = tmp_path / "run" / ".bench.lock"
        import fcntl

        fd = os.open(lock, os.O_RDWR)
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    assert time.monotonic() < deadline, "the matrix child kept running"
                    time.sleep(0.05)
        finally:
            os.close(fd)
    finally:
        process.kill()
        _kill_leftovers(tmp_path, _PROBLEMS)
