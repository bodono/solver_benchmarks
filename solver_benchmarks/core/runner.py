"""Generic benchmark runner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import json
import math
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

from solver_benchmarks.core import status
from solver_benchmarks.core.config import (
    DatasetConfig,
    RunConfig,
    SolverConfig,
    resolve_output_dir,
)
from solver_benchmarks.core.data_prepare import (
    data_prepare_command,
    data_prepare_selection,
)
from solver_benchmarks.core.environment import runtime_metadata
from solver_benchmarks.core.problem import ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.storage import ResultStore, atomic_write_text
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.datasets.base import filter_problem_specs_by_size
from solver_benchmarks.solvers import get_solver


# Grace period added on top of the configured timeout before the subprocess
# is force-killed. This lets solvers that honor their own internal time
# limit return a partial result (status, iterations, residuals) instead of
# being killed mid-return.
SUBPROCESS_TIMEOUT_GRACE_SECONDS = 30.0


def run_benchmark(
    config: RunConfig,
    *,
    run_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    stream_output: bool = False,
    stream_solver_output: bool | None = None,
    environment_id: str | None = None,
    environment_metadata: dict | None = None,
    prepare_data_command: str | None = None,
    source_config_path: str | Path | None = None,
) -> ResultStore:
    repo_root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
    config = resolve_output_dir(config, repo_root)
    if stream_solver_output is None:
        stream_solver_output = stream_output
    store = ResultStore.create(config, run_dir=run_dir)
    if source_config_path is not None:
        store.copy_source_config(source_config_path)
    completed = store.completed_keys() if config.resume else set()

    tasks: list[tuple[DatasetConfig, ProblemSpec, SolverConfig]] = []
    already_complete = 0
    skipped_during_planning = 0
    for dataset_config in config.datasets:
        dataset_cls = get_dataset(dataset_config.name)
        dataset = dataset_cls(repo_root=repo_root, **dataset_config.dataset_options)
        include, exclude = config.effective_filters(dataset_config)
        if config.auto_prepare_data and hasattr(dataset, "prepare_data"):
            problem_names, all_problems = data_prepare_selection(config, dataset_config)
            dataset.prepare_data(
                problem_names=problem_names,
                all_problems=all_problems,
            )
        problems = _filter_problems(dataset.list_problems(), include, exclude)
        problems = filter_problem_specs_by_size(
            problems, dataset_config.dataset_options.get("max_size_mb")
        )
        if not problems:
            data_status = (
                dataset.data_status() if hasattr(dataset, "data_status") else None
            )
            message = (
                data_status.message
                if data_status is not None
                else f"Dataset {dataset_config.id!r} produced no problems."
            )
            if data_status is not None and not getattr(data_status, "available", True):
                problem_names, all_problems = data_prepare_selection(
                    config, dataset_config
                )
                message = _missing_data_run_message(
                    dataset_config,
                    message,
                    problem_names=problem_names,
                    all_problems=all_problems,
                    repo_root=repo_root,
                    automatic_download=bool(
                        getattr(dataset, "automatic_download", False)
                    ),
                    prepare_data_command=prepare_data_command,
                )
                raise RuntimeError(message)
            store.append_event(
                "warning", message, dataset=dataset_config.id
            )
            continue

        for solver_config in config.solvers:
            solver_cls = get_solver(solver_config.solver)
            if not solver_cls.is_available():
                message = (
                    f"Solver {solver_config.solver!r} is unavailable; skipping "
                    f"{solver_config.id!r}"
                )
                store.append_event(
                    "warning",
                    message,
                    solver_id=solver_config.id,
                    dataset=dataset_config.id,
                )
                for problem in problems:
                    if _already_done(dataset_config, problem, solver_config, completed):
                        already_complete += 1
                        continue
                    _write_skip(
                        store,
                        config,
                        dataset_config,
                        problem,
                        solver_config,
                        message,
                        environment_id=environment_id,
                        environment_metadata=environment_metadata,
                    )
                    skipped_during_planning += 1
                continue
            solver = solver_cls(solver_config.settings)
            for problem in problems:
                if _already_done(dataset_config, problem, solver_config, completed):
                    already_complete += 1
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
                        dataset=dataset_config.id,
                        problem=problem.name,
                        problem_kind=problem.kind,
                    )
                    _write_skip(
                        store,
                        config,
                        dataset_config,
                        problem,
                        solver_config,
                        message,
                        environment_id=environment_id,
                        environment_metadata=environment_metadata,
                    )
                    skipped_during_planning += 1
                    continue
                tasks.append((dataset_config, problem, solver_config))

    parallelism = max(1, int(config.parallelism))
    progress = _ProgressReporter(
        store=store,
        stream_output=stream_output,
        total_expected=already_complete + skipped_during_planning + len(tasks),
        already_complete=already_complete,
        skipped_during_planning=skipped_during_planning,
        queued=len(tasks),
        parallelism=parallelism,
    )
    progress.emit_plan()
    if not tasks:
        progress.emit_final()
        return store

    if parallelism == 1:
        for dataset_config, problem, solver_config in tasks:
            result = _run_one(
                store,
                config,
                repo_root,
                dataset_config,
                problem,
                solver_config,
                stream_output=stream_output,
                stream_solver_output=stream_solver_output,
                environment_id=environment_id,
                environment_metadata=environment_metadata,
            )
            store.write_result(result)
            progress.record_result(result)
    else:
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    store,
                    config,
                    repo_root,
                    dataset_config,
                    problem,
                    solver_config,
                    stream_output,
                    environment_id,
                    environment_metadata,
                    stream_solver_output,
                )
                for dataset_config, problem, solver_config in tasks
            ]
            for future in as_completed(futures):
                result = future.result()
                store.write_result(result)
                progress.record_result(result)
    progress.emit_final()
    return store


class _ProgressReporter:
    def __init__(
        self,
        *,
        store: ResultStore,
        stream_output: bool,
        total_expected: int,
        already_complete: int,
        skipped_during_planning: int,
        queued: int,
        parallelism: int,
    ) -> None:
        self.store = store
        self.stream_output = stream_output
        self.total_expected = int(total_expected)
        self.already_complete = int(already_complete)
        self.skipped_during_planning = int(skipped_during_planning)
        self.queued = int(queued)
        self.parallelism = int(parallelism)
        self.completed_this_run = 0
        self.started_at = time.monotonic()
        self.last_result: ProblemResult | None = None
        self._final_emitted = False

    def emit_plan(self) -> None:
        fields = {
            "total_expected": self.total_expected,
            "already_complete": self.already_complete,
            "skipped_during_planning": self.skipped_during_planning,
            "queued": self.queued,
            "parallelism": self.parallelism,
        }
        self.store.append_event("info", "benchmark_plan", **fields)
        _emit_progress(
            self.stream_output,
            (
                f"planned {self.total_expected} solves: "
                f"{self.already_complete} already complete, "
                f"{self.skipped_during_planning} skipped during planning, "
                f"{self.queued} queued, parallelism={self.parallelism}"
            ),
        )

    def record_result(self, result: ProblemResult) -> None:
        self.completed_this_run += 1
        self.last_result = result
        self._emit_progress_snapshot("benchmark_progress")

    def emit_final(self) -> None:
        if self._final_emitted:
            return
        self._final_emitted = True
        self._emit_progress_snapshot(
            "benchmark_complete",
            print_terminal=self.completed_this_run == 0,
        )

    def _emit_progress_snapshot(
        self, message: str, *, print_terminal: bool = True
    ) -> None:
        now = time.monotonic()
        elapsed = max(0.0, now - self.started_at)
        completed_total = self._completed_total()
        percent = (
            100.0 * completed_total / self.total_expected
            if self.total_expected
            else 100.0
        )
        rate = self.completed_this_run / elapsed if elapsed > 0.0 else None
        remaining = max(0, self.queued - self.completed_this_run)
        eta = remaining / rate if rate and rate > 0.0 else None
        fields = {
            "completed_total": completed_total,
            "total_expected": self.total_expected,
            "already_complete": self.already_complete,
            "skipped_during_planning": self.skipped_during_planning,
            "completed_this_run": self.completed_this_run,
            "queued": self.queued,
            "elapsed_seconds": elapsed,
            "rate_solves_per_second": rate,
            "eta_seconds": eta,
            "parallelism": self.parallelism,
        }
        if self.last_result is not None:
            fields.update(
                {
                    "last_dataset": self.last_result.dataset,
                    "last_problem": self.last_result.problem,
                    "last_solver_id": self.last_result.solver_id,
                    "last_status": self.last_result.status,
                    "last_run_time_seconds": self.last_result.run_time_seconds,
                    "last_setup_time_seconds": self.last_result.setup_time_seconds,
                    "last_solve_time_seconds": self.last_result.solve_time_seconds,
                }
            )
        self.store.append_event("info", message, **fields)

        last = ""
        if self.last_result is not None:
            last = (
                " | last "
                f"{self.last_result.dataset}/{self.last_result.problem} "
                f"{self.last_result.solver_id}: {self.last_result.status} "
                f"in {_format_short_seconds(self.last_result.run_time_seconds)}"
            )
        rate_text = f"{rate:.2f} solves/s" if rate is not None else "unknown"
        eta_text = _format_duration(eta) if eta is not None else "unknown"
        if print_terminal:
            _emit_progress(
                self.stream_output,
                (
                    f"progress {completed_total}/{self.total_expected} "
                    f"({percent:.2f}%) | queued_done "
                    f"{self.completed_this_run}/{self.queued} | elapsed "
                    f"{_format_duration(elapsed)} | rate {rate_text} | eta {eta_text}"
                    f"{last}"
                ),
            )

    def _completed_total(self) -> int:
        return (
            self.already_complete
            + self.skipped_during_planning
            + self.completed_this_run
        )


def _run_one(
    store: ResultStore,
    config: RunConfig,
    repo_root: Path,
    dataset_config: DatasetConfig,
    problem: ProblemSpec,
    solver_config: SolverConfig,
    stream_output: bool = False,
    environment_id: str | None = None,
    environment_metadata: dict | None = None,
    stream_solver_output: bool | None = None,
) -> ProblemResult:
    if stream_solver_output is None:
        stream_solver_output = stream_output
    artifacts_dir = store.problem_solver_dir(
        dataset_config.id, problem.name, solver_config.id
    )
    payload = {
        "run_id": store.run_id,
        "dataset": dataset_config.id,
        "dataset_name": dataset_config.name,
        "dataset_options": dataset_config.dataset_options,
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
        "environment_id": environment_id,
        "environment_metadata": environment_metadata or {},
    }
    payload_path = artifacts_dir / "payload.json"
    atomic_write_text(payload_path, json.dumps(payload, indent=2, default=str))
    configured_timeout = solver_config.timeout_seconds or config.timeout_seconds
    subprocess_timeout = (
        float(configured_timeout) + SUBPROCESS_TIMEOUT_GRACE_SECONDS
        if configured_timeout
        else None
    )
    cmd = [sys.executable, "-m", "solver_benchmarks.worker", "--payload", str(payload_path)]
    _emit_progress(stream_output, f"starting {problem.name} with {solver_config.id}")
    completed = _run_subprocess(
        cmd,
        cwd=repo_root,
        timeout=subprocess_timeout,
        stdout_path=artifacts_dir / "stdout.log",
        stderr_path=artifacts_dir / "stderr.log",
        stream_output=stream_solver_output,
    )
    if completed.timed_out:
        _emit_progress(stream_output, f"timeout {problem.name} with {solver_config.id}")
        return ProblemResult(
            run_id=store.run_id,
            dataset=dataset_config.id,
            problem=problem.name,
            problem_kind=problem.kind,
            solver_id=solver_config.id,
            solver=solver_config.solver,
            status=status.TIME_LIMIT,
            objective_value=None,
            iterations=None,
            run_time_seconds=float(subprocess_timeout) if subprocess_timeout else None,
            error=f"Subprocess exceeded timeout_seconds={subprocess_timeout}",
            artifact_dir=str(artifacts_dir),
            metadata=_metadata_with_environment(
                problem,
                solver_config,
                environment_id=environment_id,
                environment_metadata=environment_metadata,
            ),
        )

    worker_result_path = artifacts_dir / "worker_result.json"
    if completed.returncode == 0 and worker_result_path.exists():
        record = json.loads(worker_result_path.read_text())
        result = ProblemResult(**record)
        _emit_progress(
            stream_output,
            f"finished {problem.name} with {solver_config.id}: {result.status}",
        )
        return result
    _emit_progress(
        stream_output,
        f"worker error {problem.name} with {solver_config.id}: exit {completed.returncode}",
    )
    return ProblemResult(
        run_id=store.run_id,
        dataset=dataset_config.id,
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
        metadata=_metadata_with_environment(
            problem,
            solver_config,
            environment_id=environment_id,
            environment_metadata=environment_metadata,
        ),
    )


def _emit_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[bench] {message}", file=sys.stderr, flush=True)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours < 24:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"


def _format_short_seconds(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return f"{seconds:.3f}s"
    if seconds < 10.0:
        return f"{seconds:.2f}s"
    if seconds < 100.0:
        return f"{seconds:.1f}s"
    return _format_duration(seconds)


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: float | None,
    stdout_path: Path,
    stderr_path: Path,
    stream_output: bool,
):
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    with stdout_path.open("w") as stdout_log, stderr_path.open("w") as stderr_log:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        threads = [
            threading.Thread(
                target=_tee_stream,
                args=(process.stdout, stdout_log, sys.stdout, stdout_chunks, stream_output),
                daemon=True,
            ),
            threading.Thread(
                target=_tee_stream,
                args=(process.stderr, stderr_log, sys.stderr, stderr_chunks, stream_output),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
        for thread in threads:
            thread.join()
    return SimpleNamespace(
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        timed_out=timed_out,
    )


def _tee_stream(source, log_file, sink, chunks: list[str], stream_output: bool) -> None:
    for line in source:
        chunks.append(line)
        log_file.write(line)
        log_file.flush()
        if stream_output:
            sink.write(line)
            sink.flush()


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


def _filter_by_size(
    problems: Iterable[ProblemSpec], max_size_mb: float | None
) -> list[ProblemSpec]:
    return filter_problem_specs_by_size(list(problems), max_size_mb)


def _missing_data_run_message(
    dataset_config: DatasetConfig,
    status_message: str,
    *,
    problem_names: list[str] | None,
    all_problems: bool,
    repo_root: Path,
    automatic_download: bool,
    prepare_data_command: str | None,
) -> str:
    lines = [
        f"Dataset {dataset_config.id!r} ({dataset_config.name!r}) has no local data.",
        status_message,
    ]
    if automatic_download:
        lines.extend(
            [
                "Prepare the requested dataset data with:",
                "  "
                + data_prepare_command(
                    dataset_config,
                    problem_names=problem_names,
                    all_problems=all_problems,
                    repo_root=repo_root,
                ),
            ]
        )
        if prepare_data_command:
            lines.extend(
                [
                    "Or rerun this benchmark with data preparation enabled:",
                    f"  {prepare_data_command}",
                ]
            )
    else:
        lines.append("This dataset does not have an automatic download command.")
    return "\n".join(lines)


def _already_done(
    dataset_config: DatasetConfig,
    problem: ProblemSpec,
    solver_config: SolverConfig,
    completed,
) -> bool:
    return (dataset_config.id, problem.name, solver_config.id) in completed


def _write_skip(
    store: ResultStore,
    config: RunConfig,
    dataset_config: DatasetConfig,
    problem: ProblemSpec,
    solver_config: SolverConfig,
    message: str,
    *,
    environment_id: str | None = None,
    environment_metadata: dict | None = None,
) -> None:
    artifacts_dir = store.problem_solver_dir(
        dataset_config.id, problem.name, solver_config.id
    )
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset=dataset_config.id,
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
            metadata=_metadata_with_environment(
                problem,
                solver_config,
                environment_id=environment_id,
                environment_metadata=environment_metadata,
            ),
        )
    )


def _metadata_with_environment(
    problem: ProblemSpec,
    solver_config: SolverConfig,
    *,
    environment_id: str | None,
    environment_metadata: dict | None,
) -> dict:
    metadata = dict(problem.metadata)
    metadata.update(
        {
            "environment_id": environment_id,
            "environment_metadata": environment_metadata or {},
            "runtime": runtime_metadata(solver_config.solver),
        }
    )
    return metadata
