"""Generic benchmark runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import json
import subprocess
import sys

from solver_benchmarks.core import status
from solver_benchmarks.core.config import RunConfig, SolverConfig
from solver_benchmarks.core.problem import ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.storage import ResultStore, atomic_write_text
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.solvers import get_solver


def run_benchmark(
    config: RunConfig,
    *,
    run_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> ResultStore:
    repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    store = ResultStore.create(config, run_dir=run_dir)
    dataset_cls = get_dataset(config.dataset)
    dataset = dataset_cls(repo_root=repo_root, **config.dataset_options)
    if config.auto_prepare_data and hasattr(dataset, "prepare_data"):
        dataset.prepare_data(problem_names=config.include or None)
    problems = _filter_problems(dataset.list_problems(), config.include, config.exclude)
    if not problems:
        data_status = dataset.data_status() if hasattr(dataset, "data_status") else None
        message = (
            data_status.message
            if data_status is not None
            else f"Dataset {config.dataset!r} produced no problems."
        )
        store.append_event("warning", message, dataset=config.dataset)
    completed = store.completed_keys() if config.resume else set()

    tasks: list[tuple[ProblemSpec, SolverConfig]] = []
    for solver_config in config.solvers:
        solver_cls = get_solver(solver_config.solver)
        if not solver_cls.is_available():
            message = f"Solver {solver_config.solver!r} is unavailable; skipping {solver_config.id!r}"
            store.append_event("warning", message, solver_id=solver_config.id)
            for problem in problems:
                if _already_done(problem, solver_config, completed):
                    continue
                _write_skip(store, config, problem, solver_config, message)
            continue
        solver = solver_cls(solver_config.settings)
        for problem in problems:
            if _already_done(problem, solver_config, completed):
                continue
            if not solver.supports(problem.kind):
                message = (
                    f"Solver {solver_config.solver!r} does not support "
                    f"{problem.kind!r} problems; skipping {problem.name!r}"
                )
                if config.fail_on_unsupported:
                    raise RuntimeError(message)
                store.append_event(
                    "warning",
                    message,
                    solver_id=solver_config.id,
                    problem=problem.name,
                    problem_kind=problem.kind,
                )
                _write_skip(store, config, problem, solver_config, message)
                continue
            tasks.append((problem, solver_config))

    if not tasks:
        return store

    parallelism = max(1, int(config.parallelism))
    if parallelism == 1:
        for problem, solver_config in tasks:
            store.write_result(_run_one(store, config, repo_root, problem, solver_config))
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [
                executor.submit(_run_one, store, config, repo_root, problem, solver_config)
                for problem, solver_config in tasks
            ]
            for future in as_completed(futures):
                store.write_result(future.result())
    return store


def _run_one(
    store: ResultStore,
    config: RunConfig,
    repo_root: Path,
    problem: ProblemSpec,
    solver_config: SolverConfig,
) -> ProblemResult:
    artifacts_dir = store.problem_solver_dir(problem.name, solver_config.id)
    payload = {
        "run_id": store.run_id,
        "dataset": config.dataset,
        "dataset_options": config.dataset_options,
        "problem": problem.name,
        "problem_kind": problem.kind,
        "solver": {
            "id": solver_config.id,
            "solver": solver_config.solver,
            "settings": solver_config.settings,
            "timeout_seconds": solver_config.timeout_seconds,
        },
        "artifacts_dir": str(artifacts_dir),
        "repo_root": str(repo_root),
    }
    payload_path = artifacts_dir / "payload.json"
    atomic_write_text(payload_path, json.dumps(payload, indent=2, default=str))
    timeout = solver_config.timeout_seconds or config.timeout_seconds
    cmd = [sys.executable, "-m", "solver_benchmarks.worker", "--payload", str(payload_path)]
    try:
        completed = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        atomic_write_text(artifacts_dir / "stdout.log", exc.stdout or "")
        atomic_write_text(artifacts_dir / "stderr.log", exc.stderr or "")
        return ProblemResult(
            run_id=store.run_id,
            dataset=config.dataset,
            problem=problem.name,
            problem_kind=problem.kind,
            solver_id=solver_config.id,
            solver=solver_config.solver,
            status=status.TIME_LIMIT,
            objective_value=None,
            iterations=None,
            run_time_seconds=float(timeout) if timeout else None,
            error=f"Subprocess exceeded timeout_seconds={timeout}",
            artifact_dir=str(artifacts_dir),
            metadata=dict(problem.metadata),
        )

    atomic_write_text(artifacts_dir / "stdout.log", completed.stdout)
    atomic_write_text(artifacts_dir / "stderr.log", completed.stderr)
    worker_result_path = artifacts_dir / "worker_result.json"
    if completed.returncode == 0 and worker_result_path.exists():
        record = json.loads(worker_result_path.read_text())
        return ProblemResult(**record)
    return ProblemResult(
        run_id=store.run_id,
        dataset=config.dataset,
        problem=problem.name,
        problem_kind=problem.kind,
        solver_id=solver_config.id,
        solver=solver_config.solver,
        status=status.WORKER_ERROR,
        objective_value=None,
        iterations=None,
        run_time_seconds=None,
        error=f"Worker exited with code {completed.returncode}",
        artifact_dir=str(artifacts_dir),
        metadata=dict(problem.metadata),
    )


def _filter_problems(
    problems: Iterable[ProblemSpec], include: list[str], exclude: list[str]
) -> list[ProblemSpec]:
    include_set = set(include)
    exclude_set = set(exclude)
    selected = []
    for problem in problems:
        if include_set and problem.name not in include_set:
            continue
        if problem.name in exclude_set:
            continue
        selected.append(problem)
    return selected


def _already_done(problem: ProblemSpec, solver_config: SolverConfig, completed) -> bool:
    return (problem.name, solver_config.id) in completed


def _write_skip(
    store: ResultStore,
    config: RunConfig,
    problem: ProblemSpec,
    solver_config: SolverConfig,
    message: str,
) -> None:
    artifacts_dir = store.problem_solver_dir(problem.name, solver_config.id)
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset=config.dataset,
            problem=problem.name,
            problem_kind=problem.kind,
            solver_id=solver_config.id,
            solver=solver_config.solver,
            status=status.SKIPPED_UNSUPPORTED,
            objective_value=None,
            iterations=None,
            run_time_seconds=None,
            error=message,
            artifact_dir=str(artifacts_dir),
            metadata=dict(problem.metadata),
        )
    )
