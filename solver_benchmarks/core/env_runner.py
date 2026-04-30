"""Run benchmark configs across externally managed Python environments."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from solver_benchmarks.core.config import (
    EnvironmentConfig,
    EnvironmentRunConfig,
    resolve_output_dir,
)
from solver_benchmarks.core.storage import ResultStore, make_run_id


def run_environment_matrix(
    config: EnvironmentRunConfig,
    *,
    run_dir: Path | None = None,
    repo_root: Path | None = None,
    stream_output: bool = True,
    stream_solver_output: bool | None = None,
    source_config_path: Path | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
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
    if run_dir is None:
        env_name = (
            f"{combined_config.name}_env"
            if combined_config.name is not None
            else "environment_run"
        )
        named_config = replace(combined_config, name=env_name)
        run_dir = combined_config.output_dir / make_run_id(named_config)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore.create(combined_config, run_dir=run_dir)
    if source_config_path is not None:
        store.copy_source_config(source_config_path, name="environment_config")

    failures: list[tuple[str, str]] = []
    for environment in config.environments:
        try:
            _install_environment(
                environment,
                repo_root=repo_root,
                stream_output=stream_output,
            )
        except subprocess.CalledProcessError as exc:
            message = (
                f"environment {environment.id} install failed: "
                f"{' '.join(map(str, exc.cmd))} -> exit {exc.returncode}"
            )
            store.append_event(
                "error",
                "environment_install_failed",
                environment_id=environment.id,
                returncode=exc.returncode,
            )
            print(f"[bench env] {message}", file=sys.stderr)
            failures.append((environment.id, "install"))
            continue
        child_config = replace(combined_config, solvers=environment.solvers)
        child_path = run_dir / f"{environment.id}_config.json"
        child_path.write_text(
            json.dumps(child_config.to_manifest(), indent=2, default=str)
        )
        cmd = [
            environment.python,
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
            subprocess.run(cmd, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            message = (
                f"environment {environment.id} run failed: exit {exc.returncode}"
            )
            store.append_event(
                "error",
                "environment_run_failed",
                environment_id=environment.id,
                returncode=exc.returncode,
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
