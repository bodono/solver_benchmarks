"""Run benchmark configs across externally managed Python environments."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from solver_benchmarks.core.config import (
    EnvironmentConfig,
    EnvironmentRunConfig,
    resolve_data_roots,
    resolve_output_dir,
)
from solver_benchmarks.core.runner import (
    harness_import_roots,
    harness_pythonpath,
    refuse_to_rerun_into,
)
from solver_benchmarks.core.storage import ResultStore, make_run_id
from solver_benchmarks.datasets.base import default_repo_root


def run_environment_matrix(
    config: EnvironmentRunConfig,
    *,
    run_dir: Path | None = None,
    repo_root: Path | None = None,
    stream_output: bool = True,
    stream_solver_output: bool | None = None,
    source_config_path: Path | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve() if repo_root else default_repo_root()
    if stream_solver_output is None:
        stream_solver_output = stream_output
    combined_config = replace(
        config.run,
        solvers=[
            solver
            for environment in config.environments
            for solver in environment.solvers
        ],
    )
    combined_config = resolve_output_dir(combined_config, repo_root)
    combined_config = replace(
        combined_config, datasets=resolve_data_roots(combined_config.datasets, repo_root)
    )
    if run_dir is None:
        env_name = (
            f"{combined_config.name}_env"
            if combined_config.name is not None
            else "environment_run"
        )
        named_config = replace(combined_config, name=env_name)
        run_dir = combined_config.output_dir / make_run_id(named_config)
    run_dir = Path(run_dir).resolve()
    if not combined_config.resume:
        refuse_to_rerun_into(run_dir, combined_config)
    run_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore.create(combined_config, run_dir=run_dir)
    if source_config_path is not None:
        store.copy_source_config(source_config_path, name="environment_config")

    if not harness_import_roots():
        print(
            "[bench env] no harness directory is handed to the environments "
            "(solver_benchmarks is not imported from a checkout or a PYTHONPATH "
            "directory): each environment runs the solver_benchmarks installed in it",
            file=sys.stderr,
        )
    failures: list[tuple[str, str]] = []
    for environment in config.environments:
        try:
            _install_environment(
                environment,
                repo_root=repo_root,
                stream_output=stream_output,
            )
        except (subprocess.CalledProcessError, OSError, KeyError, IndexError, ValueError) as exc:
            # OSError: the python or installer does not exist or cannot run.
            # KeyError/IndexError/ValueError: the command has stray braces.
            error = _describe_failure(exc)
            message = f"environment {environment.id} install failed: {error}"
            store.append_event(
                "error",
                "environment_install_failed",
                environment_id=environment.id,
                returncode=getattr(exc, "returncode", None),
                error=error,
            )
            print(f"[bench env] {message}", file=sys.stderr)
            failures.append((environment.id, "install"))
            continue
        child_config = replace(combined_config, solvers=environment.solvers)
        child_path = run_dir / f"{environment.id}_config.json"
        child_path.write_text(
            json.dumps(child_config.to_manifest(), indent=2, default=str)
        )
        # The environment's interpreter and installed solver packages run
        # this harness's code: -P keeps the data root (the working
        # directory) off sys.path and _run_child puts the harness first on
        # PYTHONPATH.
        cmd = [
            environment.python,
            "-P",
            "-m",
            "solver_benchmarks.cli",
            "run",
            str(child_path),
            "--run-dir",
            str(run_dir),
            "--repo-root",
            str(repo_root),
            "--environment-id",
            environment.id,
            "--environment-metadata",
            json.dumps(environment.metadata, sort_keys=True),
        ]
        if not stream_solver_output:
            cmd.append("--no-stream-output")
        if stream_output:
            print(f"[bench env] running {environment.id}: {' '.join(cmd)}", file=sys.stderr)
        try:
            _run_child(cmd, cwd=repo_root)
        except (subprocess.CalledProcessError, OSError) as exc:
            # OSError: a mistyped `python:` or a venv that does not exist yet.
            error = _describe_failure(exc)
            message = f"environment {environment.id} run failed: {error}"
            store.append_event(
                "error",
                "environment_run_failed",
                environment_id=environment.id,
                returncode=getattr(exc, "returncode", None),
                error=error,
            )
            print(f"[bench env] {message}", file=sys.stderr)
            failures.append((environment.id, "run"))
        finally:
            store.write_manifest(combined_config)
    if failures:
        # Surface any per-environment failures as a single error after
        # the matrix has finished, so the partial results are still
        # written to disk and inspectable.
        labels = ", ".join(f"{env_id} ({phase})" for env_id, phase in failures)
        raise RuntimeError(f"environment matrix had failures: {labels}")
    return run_dir


# How long a child `bench run` gets to kill its workers after a SIGTERM
# before it is killed itself.
_CHILD_SHUTDOWN_SECONDS = 10.0
# Tells a child `bench run` which matrix runner started it, so that it stops
# when that runner dies (see cli.run_cmd).
MATRIX_RUNNER_PID_ENV = "SOLVER_BENCHMARKS_MATRIX_RUNNER_PID"


def _run_child(cmd: list[str], *, cwd: Path) -> None:
    """Run one environment's ``bench run`` like ``subprocess.run(check=True)``.

    If this process is interrupted (Ctrl+C, or the SIGTERM handler's
    SystemExit) while the child runs, the child is asked to stop with
    SIGTERM and given time to kill its workers, which run in their own
    sessions; ``subprocess.run`` would SIGKILL it at once and leave them
    solving with no time limit. It is killed only if it does not exit in
    time. The child also gets this process's pid in its environment, so that
    it stops the same way if this process is killed outright, and the
    directories this process imports the harness from first on its
    PYTHONPATH (see runner.harness_pythonpath).
    """
    env = {**os.environ, MATRIX_RUNNER_PID_ENV: str(os.getpid())}
    pythonpath = harness_pythonpath(os.environ.get("PYTHONPATH"))
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    else:
        # Separators only would put the data root (the working directory)
        # on the child's sys.path.
        env.pop("PYTHONPATH", None)
    process = subprocess.Popen(cmd, cwd=cwd, env=env)
    try:
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=_CHILD_SHUTDOWN_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd)


def _describe_failure(exc: BaseException) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        return f"{' '.join(map(str, exc.cmd))} -> exit {exc.returncode}"
    return f"{type(exc).__name__}: {exc}"


def _install_environment(
    environment: EnvironmentConfig,
    *,
    repo_root: Path,
    stream_output: bool,
) -> None:
    for command in environment.install:
        formatted = command.format(python=environment.python, repo_root=repo_root)
        cmd = shlex.split(formatted)
        if stream_output:
            print(f"[bench env] install {environment.id}: {formatted}", file=sys.stderr)
        subprocess.run(cmd, cwd=repo_root, check=True)
