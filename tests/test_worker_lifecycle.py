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
    # A runner killed outright cannot remove its workers' temporary directories.
    env["TMPDIR"] = str(tmp_path)

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
    # A runner killed outright cannot remove its workers' temporary directories.
    env["TMPDIR"] = str(tmp_path)
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


# What the worker's environment holds: the BLAS / OpenMP thread caps, the
# harness it imports and its temporary directory. ``envprobe`` reports them
# from inside the worker; a sitecustomize on PYTHONPATH registers it there.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

_ENVPROBE = textwrap.dedent(
    """
    import json, os, tempfile
    import solver_benchmarks
    from solver_benchmarks.core.result import SolverResult
    from solver_benchmarks.solvers import registry
    from solver_benchmarks.solvers.base import SolverAdapter

    VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

    class EnvProbe(SolverAdapter):
        solver_name = "envprobe"
        supported_problem_kinds = {"qp", "cone"}

        def solve(self, problem, artifacts_dir):
            return SolverResult(
                status="optimal", objective_value=0.0, iterations=1, run_time_seconds=0.0,
                info={
                    "env": {name: os.environ.get(name) for name in VARS},
                    "harness": os.path.dirname(solver_benchmarks.__file__),
                    "json": json.__file__,
                    "tempdir": tempfile.gettempdir(),
                },
            )

    registry.SOLVERS["envprobe"] = EnvProbe
    """
)


@pytest.fixture
def envprobe(tmp_path: Path, monkeypatch):
    """Register ``envprobe`` here and in every worker this test starts."""
    import importlib.util

    site = tmp_path / "probe_site"
    site.mkdir()
    (site / "bench_envprobe.py").write_text(_ENVPROBE)
    (site / "sitecustomize.py").write_text(
        "import os\nif os.environ.get('BENCH_TEST_ENVPROBE'):\n    import bench_envprobe\n"
    )
    from solver_benchmarks.solvers import registry

    # Loading the module registers envprobe; undone at teardown.
    monkeypatch.setitem(registry.SOLVERS, "envprobe", None)
    spec = importlib.util.spec_from_file_location("bench_envprobe_parent", site / "bench_envprobe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(registry.SOLVERS, "envprobe", module.EnvProbe)
    path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(site) + (os.pathsep + path if path else ""))
    monkeypatch.setenv("BENCH_TEST_ENVPROBE", "1")
    for name in _THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    return site


def _rows(store_or_dir) -> list[dict]:
    import json

    path = getattr(store_or_dir, "results_jsonl_path", None) or Path(store_or_dir) / "results.jsonl"
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _caps(value):
    return {name: value for name in _THREAD_VARS}


_THREAD_CASES = [
    ({}, {}, _caps("1")),
    ({"threads": 2}, {}, _caps("2")),
    ({"threads": 0}, {}, _caps(None)),
    ({"threads": 2}, {"OMP_NUM_THREADS": "8"}, {**_caps(None), "OMP_NUM_THREADS": "8"}),
    ({}, {"OPENBLAS_NUM_THREADS": "3"}, {**_caps(None), "OPENBLAS_NUM_THREADS": "3"}),
    ({"threads": 2}, {"OMP_NUM_THREADS": ""}, _caps("2")),
]


@pytest.mark.parametrize(
    "settings, exported, expected",
    _THREAD_CASES,
    ids=["default", "threads-2", "threads-0", "exported-omp", "exported-openblas", "exported-empty"],
)
def test_the_runner_hands_the_worker_thread_caps(
    monkeypatch, tmp_path: Path, settings, exported, expected
):
    """The BLAS / OpenMP pools of a worker follow the solver's `threads`
    (one by default, none set for 0); if the user exported any of the four
    variables, the runner sets none of them. The row records the values the
    worker ran with, also when the runner writes the row itself (here: a
    timeout)."""
    from types import SimpleNamespace

    from solver_benchmarks.core.config import parse_run_config
    from solver_benchmarks.core.runner import run_benchmark

    for name in _THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in exported.items():
        monkeypatch.setenv(name, value)
    handed = []

    def fake(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, env=None):
        handed.append(dict(env or {}))
        return SimpleNamespace(returncode=-15, timed_out=True, stderr="")

    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", fake)
    config = parse_run_config(
        {"run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"], "timeout_seconds": 5},
         "solvers": [{"id": "s", "solver": "scs", "settings": settings}]}
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=REPO_ROOT)
    user_managed = any(exported.values())
    added = {} if user_managed else {name: value for name, value in expected.items() if value is not None}
    assert handed == [added]
    (row,) = _rows(store)
    assert row["status"] == "time_limit"
    assert row["metadata"]["runtime"]["thread_env"] == expected


def test_a_worker_runs_with_the_thread_caps(envprobe, tmp_path: Path):
    from solver_benchmarks.core.config import parse_run_config
    from solver_benchmarks.core.runner import run_benchmark

    cases = {"default": {}, "two": {"threads": 2}, "solver_default": {"threads": 0}}
    config = parse_run_config(
        {"run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"]},
         "solvers": [{"id": key, "solver": "envprobe", "settings": settings}
                     for key, settings in cases.items()]}
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=REPO_ROOT)
    rows = {row["solver_id"]: row for row in _rows(store)}
    expected = {"default": _caps("1"), "two": _caps("2"), "solver_default": _caps(None)}
    for key, caps in expected.items():
        assert rows[key]["status"] == "optimal", rows[key]["error"]
        assert rows[key]["info"]["env"] == caps
        assert rows[key]["metadata"]["runtime"]["thread_env"] == caps


_PLANTED = ["harness-copy", "stdlib-name"]


def _planted_repo_root(tmp_path: Path, planted: str) -> tuple[Path, Path]:
    """A data root holding another `solver_benchmarks`, or a stray json.py."""
    root = tmp_path / "data_root"
    marker = tmp_path / "planted_code_ran"
    (root / "problem_classes").mkdir(parents=True)
    if planted == "harness-copy":
        (root / "solver_benchmarks").mkdir()
        (root / "solver_benchmarks" / "__init__.py").write_text("")
        for name in ("worker.py", "cli.py"):
            (root / "solver_benchmarks" / name).write_text(
                f"import pathlib, sys\npathlib.Path({str(marker)!r}).write_text('{name}')\nsys.exit(3)\n"
            )
        return root, marker
    (root / "json.py").write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).write_text('json.py')\n"
        "raise ImportError('planted json.py')\n"
    )
    return root, marker


# These two use a real solver: envprobe's sitecustomize would import the
# harness and json before the worker's own imports.
@pytest.mark.parametrize("planted", _PLANTED)
def test_a_worker_does_not_import_code_from_the_data_root(tmp_path: Path, planted: str):
    """--repo-root is where the data is; a harness copy or a module named like
    a standard one there must not be what the workers run."""
    from solver_benchmarks.core.config import parse_run_config
    from solver_benchmarks.core.runner import run_benchmark

    root, marker = _planted_repo_root(tmp_path, planted)
    config = parse_run_config(
        {"run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"]},
         "solvers": [{"id": "osqp", "solver": "osqp"}]}
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=root)
    (row,) = _rows(store)
    assert row["status"] == "optimal", row["error"]
    assert not marker.exists(), marker.read_text()


@pytest.mark.parametrize("planted", _PLANTED)
def test_a_matrix_environment_does_not_import_code_from_the_data_root(tmp_path: Path, planted: str):
    from solver_benchmarks.core.config import parse_environment_run_config
    from solver_benchmarks.core.env_runner import run_environment_matrix

    root, marker = _planted_repo_root(tmp_path, planted)
    config = parse_environment_run_config(
        {"run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"],
                 "output_dir": str(tmp_path / "runs")},
         "environments": [{"id": "here", "python": sys.executable,
                           "solvers": [{"id": "osqp", "solver": "osqp"}]}]}
    )
    run_dir = run_environment_matrix(
        config, run_dir=tmp_path / "matrix", repo_root=root, stream_output=False
    )
    (row,) = _rows(run_dir)
    assert row["status"] == "optimal", row["error"]
    assert not marker.exists(), marker.read_text()


# A runner (and a `bench env run`) that imports the harness from a directory
# it put on sys.path itself: neither PYTHONPATH, the working directory nor the
# installed package points there, so the children find that code only through
# the import roots the parent hands them.
_ELSEWHERE_PARENT = textwrap.dedent(
    """
    import json, os, sys
    harness, probe_site, root, out = sys.argv[1:5]
    sys.path.insert(0, harness)
    sys.path.insert(0, probe_site)
    import bench_envprobe  # registers envprobe in this process
    import solver_benchmarks
    from solver_benchmarks.core.config import parse_environment_run_config, parse_run_config
    from solver_benchmarks.core.env_runner import run_environment_matrix
    from solver_benchmarks.core.runner import run_benchmark

    assert solver_benchmarks.__file__.startswith(harness), solver_benchmarks.__file__
    os.environ["PYTHONPATH"] = probe_site  # for the children's sitecustomize only
    os.environ["BENCH_TEST_ENVPROBE"] = "1"
    run = {"dataset": "synthetic_qp", "include": ["one_variable_eq"]}
    solvers = [{"id": "probe", "solver": "envprobe"}]
    run_benchmark(parse_run_config({"run": run, "solvers": solvers}),
                  run_dir=os.path.join(out, "run"), repo_root=root)
    run_environment_matrix(
        parse_environment_run_config({"run": run, "environments": [
            {"id": "here", "python": sys.executable, "solvers": solvers}]}),
        run_dir=os.path.join(out, "matrix"), repo_root=root, stream_output=False)
    """
)


@pytest.mark.skipif(os.name != "posix", reason="symlinked harness copy")
def test_children_run_the_harness_the_parent_imported(envprobe, tmp_path: Path):
    harness = tmp_path / "harness_elsewhere"
    harness.mkdir()
    for package in ("solver_benchmarks", "problem_classes"):
        (harness / package).symlink_to(REPO_ROOT / package, target_is_directory=True)
    root = tmp_path / "data_root"
    root.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.pop("BENCH_TEST_ENVPROBE", None)
    completed = subprocess.run(
        [sys.executable, "-c", _ELSEWHERE_PARENT, str(harness), str(envprobe), str(root), str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    for run_dir in ("run", "matrix"):
        (row,) = _rows(tmp_path / run_dir)
        assert row["status"] == "optimal", row["error"]
        assert row["info"]["harness"] == str(harness / "solver_benchmarks"), run_dir


def test_the_harness_pythonpath_keeps_the_inherited_one_and_skips_site_packages(monkeypatch):
    """An inherited PYTHONPATH (bench_modal, a sitecustomize) stays, after the
    harness; a site-packages directory is not put ahead of the standard
    library, nor handed to a `bench env run` environment."""
    import site

    from solver_benchmarks.core import runner

    roots = {str(REPO_ROOT)}
    runner._harness_import_roots.cache_clear()
    try:
        assert set(runner.harness_pythonpath().split(os.pathsep)) == roots
        assert runner.harness_pythonpath("/a" + os.pathsep + "/b").split(os.pathsep)[-2:] == ["/a", "/b"]
        runner._harness_import_roots.cache_clear()
        monkeypatch.setattr(site, "getsitepackages", lambda: [str(REPO_ROOT)])
        assert runner.harness_pythonpath() == ""
        assert runner.harness_pythonpath("/a") == "/a"
    finally:
        runner._harness_import_roots.cache_clear()


@pytest.mark.parametrize(
    "inherited",
    ["{sep}{extra}", "{extra}{sep}", "{extra}{sep}{sep}rel"],
    ids=["leading-empty", "trailing-empty", "relative"],
)
def test_the_inherited_pythonpath_is_handed_on_without_the_working_directory(
    monkeypatch, tmp_path: Path, inherited: str
):
    """An empty PYTHONPATH entry means the working directory, which is the
    data root in a worker; a relative one is resolved against the runner's
    working directory, as the runner's own sys.path did."""
    from solver_benchmarks.core import runner

    monkeypatch.chdir(tmp_path)
    extra = str(tmp_path / "extra")
    value = inherited.format(sep=os.pathsep, extra=extra)
    entries = runner.harness_pythonpath(value).split(os.pathsep)
    roots = list(runner._harness_import_roots())
    expected = [extra, str(tmp_path / "rel")] if "rel" in inherited else [extra]
    assert entries == roots + expected


def test_an_empty_pythonpath_entry_does_not_put_the_data_root_on_the_worker_path(
    monkeypatch, tmp_path: Path
):
    """`export PYTHONPATH=$PYTHONPATH:/dir` with PYTHONPATH unset gives a
    leading empty entry; a json.py in the data root must still not run."""
    from solver_benchmarks.core.config import parse_run_config
    from solver_benchmarks.core.runner import run_benchmark

    root, marker = _planted_repo_root(tmp_path, "stdlib-name")
    path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep + str(tmp_path / "extra") + (os.pathsep + path if path else "")
    )
    config = parse_run_config(
        {"run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"]},
         "solvers": [{"id": "osqp", "solver": "osqp"}]}
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=root)
    (row,) = _rows(store)
    assert row["status"] == "optimal", row["error"]
    assert not marker.exists(), marker.read_text()


_NO_HANDOVER = "[bench env] no harness directory is handed to the environments"


def test_bench_env_run_says_when_it_hands_over_no_harness(monkeypatch, tmp_path: Path, capsys):
    """Imported from site-packages, `bench env run` hands no directory to the
    environments; each runs the solver_benchmarks installed in it, and the
    run says so once."""
    from solver_benchmarks.core import env_runner, runner
    from solver_benchmarks.core.config import parse_environment_run_config

    monkeypatch.setattr(runner, "_harness_import_roots", lambda: ())
    children = []
    monkeypatch.setattr(env_runner, "_run_child", lambda cmd, cwd: children.append(cmd))
    config = parse_environment_run_config(
        {"run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"],
                 "output_dir": str(tmp_path / "runs")},
         "environments": [{"id": env_id, "python": sys.executable,
                           "solvers": [{"id": f"osqp_{env_id}", "solver": "osqp"}]}
                          for env_id in ("a", "b")]}
    )
    env_runner.run_environment_matrix(
        config, run_dir=tmp_path / "matrix", repo_root=REPO_ROOT, stream_output=False
    )
    assert len(children) == 2
    err = capsys.readouterr().err
    assert err.count(_NO_HANDOVER) == 1, err
    assert "each environment runs the solver_benchmarks installed in it" in err


def test_bench_env_run_is_quiet_about_a_handed_over_harness(monkeypatch, tmp_path: Path, capsys):
    from solver_benchmarks.core import env_runner
    from solver_benchmarks.core.config import parse_environment_run_config

    monkeypatch.setattr(env_runner, "_run_child", lambda cmd, cwd: None)
    config = parse_environment_run_config(
        {"run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"],
                 "output_dir": str(tmp_path / "runs")},
         "environments": [{"id": "here", "python": sys.executable,
                           "solvers": [{"id": "osqp", "solver": "osqp"}]}]}
    )
    env_runner.run_environment_matrix(
        config, run_dir=tmp_path / "matrix", repo_root=REPO_ROOT, stream_output=False
    )
    assert _NO_HANDOVER not in capsys.readouterr().err


_SHOW_PYTHONPATH = "import os, sys; open(sys.argv[1], 'w').write(repr(os.environ.get('PYTHONPATH')))"


@pytest.mark.parametrize("separators", [1, 2], ids=["sep", "sep-sep"])
def test_a_separator_only_pythonpath_is_not_handed_to_a_worker(
    monkeypatch, tmp_path: Path, separators: int
):
    """With no harness directory to hand over (site-packages install), an
    inherited PYTHONPATH of only separators must not reach the worker: its
    empty entries mean the working directory, the data root."""
    from solver_benchmarks.core import runner

    monkeypatch.setattr(runner, "_harness_import_roots", lambda: ())
    monkeypatch.setenv("PYTHONPATH", os.pathsep * separators)
    seen = tmp_path / "seen"
    result = runner._run_subprocess(
        [sys.executable, "-c", _SHOW_PYTHONPATH, str(seen)],
        cwd=tmp_path,
        timeout=60.0,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stream_output=False,
    )
    assert result.returncode == 0
    assert seen.read_text() == "None"


@pytest.mark.parametrize("separators", [1, 2], ids=["sep", "sep-sep"])
def test_a_separator_only_pythonpath_is_not_handed_to_a_matrix_child(
    monkeypatch, tmp_path: Path, separators: int
):
    from solver_benchmarks.core import env_runner, runner

    monkeypatch.setattr(runner, "_harness_import_roots", lambda: ())
    monkeypatch.setenv("PYTHONPATH", os.pathsep * separators)
    seen = tmp_path / "seen"
    env_runner._run_child([sys.executable, "-c", _SHOW_PYTHONPATH, str(seen)], cwd=tmp_path)
    assert seen.read_text() == "None"
