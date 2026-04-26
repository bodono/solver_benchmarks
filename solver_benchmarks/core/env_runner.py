"""Run benchmark configs across externally managed Python environments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import shlex
import subprocess
import sys

from solver_benchmarks.core.config import EnvironmentConfig, EnvironmentRunConfig
from solver_benchmarks.core.storage import ResultStore, make_run_id


def run_environment_matrix(
    config: EnvironmentRunConfig,
    *,
    run_dir: Path | None = None,
    repo_root: Path | None = None,
    stream_output: bool = True,
    source_config_path: Path | None = None,
) -> Path:
    repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    combined_config = replace(
        config.run,
        solvers=[
            solver
            for environment in config.environments
            for solver in environment.solvers
        ],
    )
    if run_dir is None:
        env_name = (
            f"{combined_config.name}_env"
            if combined_config.name is not None
            else "environment_run"
        )
        named_config = replace(combined_config, name=env_name)
        run_dir = combined_config.output_dir / make_run_id(named_config)
        if not run_dir.is_absolute():
            run_dir = repo_root / run_dir
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    store = ResultStore.create(combined_config, run_dir=run_dir)
    if source_config_path is not None:
        store.copy_source_config(source_config_path, name="environment_config")

    for environment in config.environments:
        _install_environment(
            environment,
            repo_root=repo_root,
            stream_output=stream_output,
        )
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
        if stream_output:
            print(f"[bench env] running {environment.id}: {' '.join(cmd)}", file=sys.stderr)
        try:
            subprocess.run(cmd, cwd=repo_root, check=True)
        finally:
            store.write_manifest(combined_config)
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
