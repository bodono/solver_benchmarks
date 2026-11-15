"""Generic benchmark runner."""

from __future__ import annotations

import dataclasses
import itertools
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from solver_benchmarks.core import status
from solver_benchmarks.core.config import (
    DatasetConfig,
    RunConfig,
    SolverConfig,
    resolve_data_roots,
    resolve_output_dir,
    solve_signature,
)
from solver_benchmarks.core.data_prepare import (
    data_prepare_command,
    data_prepare_selection,
)
from solver_benchmarks.core.environment import runtime_metadata
from solver_benchmarks.core.problem import ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.storage import (
    SKIP_SOLVER_UNAVAILABLE,
    ResultStore,
    atomic_write_text,
)
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.datasets.base import default_repo_root, filter_problem_specs_by_size
from solver_benchmarks.solvers import get_solver

# Bounds of the grace added on top of the configured timeout before the
# worker is force-killed (see _subprocess_timeout). The grace covers worker
# start-up and problem loading, and lets solvers that honor their own
# internal time limit return a partial result (status, iterations,
# residuals) instead of being killed mid-return.
SUBPROCESS_TIMEOUT_GRACE_SECONDS = 30.0
SUBPROCESS_TIMEOUT_MIN_GRACE_SECONDS = 5.0
# Lines of each worker stream kept in memory; the logs hold everything.
_OUTPUT_TAIL_LINES = 200

logger = logging.getLogger(__name__)
_POOL_IDS = itertools.count(1)


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
    repo_root = Path(repo_root).resolve() if repo_root else default_repo_root()
    config = resolve_output_dir(config, repo_root)
    # Workers start in repo_root; a data_root relative to this process's
    # directory must name the same place there.
    config = dataclasses.replace(config, datasets=resolve_data_roots(config.datasets, Path.cwd()))
    if stream_solver_output is None:
        stream_solver_output = stream_output
    if run_dir is not None and not config.resume:
        refuse_to_rerun_into(run_dir, config)
    previous_manifest = (
        ResultStore.read_manifest(run_dir)
        if run_dir is not None and config.resume
        else None
    )
    available_solver_ids = {
        solver_config.id
        for solver_config in config.solvers
        if get_solver(solver_config.solver).is_available()
    }
    store = ResultStore.create(config, run_dir=run_dir)
    # results.parquet is a success sentinel: it exists iff a run finished
    # all queued tasks without raising. Clear any leftover parquet from
    # a prior aborted attempt at the same run_dir (resume case) so that
    # its eventual reappearance unambiguously means "this run completed".
    store.results_parquet_path.unlink(missing_ok=True)
    if source_config_path is not None:
        store.copy_source_config(source_config_path)
    completed = (
        store.completed_keys(
            config=config,
            previous_manifest=previous_manifest,
            available_solver_ids=available_solver_ids,
        )
        if config.resume
        else set()
    )
    retired = store.retire_superseded_rows(completed)
    if retired:
        store.append_event(
            "info",
            "Moved results solved under other settings to results.superseded.jsonl",
            superseded_rows=retired,
        )

    tasks: list[tuple[DatasetConfig, ProblemSpec, SolverConfig]] = []
    already_complete = 0
    skipped_during_planning = 0
    for dataset_config in config.datasets:
        dataset_cls = get_dataset(dataset_config.name)
        dataset = dataset_cls(repo_root=repo_root, **dataset_config.dataset_options)
        include, exclude = config.effective_filters(dataset_config)
        if config.auto_prepare_data and hasattr(dataset, "prepare_data"):
            _prepare_data(dataset, config, dataset_config)
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
            if solver_config.id not in available_solver_ids:
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
                        skip_reason=SKIP_SOLVER_UNAVAILABLE,
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
        # No work to do (e.g. every problem skipped during planning, or
        # nothing left after resume) still counts as a successful run.
        # Materialize the parquet so its existence reflects completion.
        _write_parquet(store)
        progress.emit_final()
        return store

    def run_task(
        dataset_config: DatasetConfig, problem: ProblemSpec, solver_config: SolverConfig
    ) -> ProblemResult:
        return _run_task(
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

    if parallelism == 1:
        for task in tasks:
            result = run_task(*task)
            store.write_result(result)
            progress.record_result(result)
    else:
        _run_tasks_in_parallel(tasks, run_task, store, progress, parallelism)
    # All queued tasks completed successfully — materialize the parquet
    # exactly once. results.jsonl is the canonical live record during
    # the run; results.parquet is the end-of-run derived artifact whose
    # presence signals "this run completed".
    _write_parquet(store)
    progress.emit_final()
    return store


def _write_parquet(store: ResultStore) -> None:
    """Write results.parquet; a failure is logged, not raised.

    The parquet is derived from results.jsonl, which already holds every
    result: failing the finished run over it would lose nothing but report
    the run as failed (and fail an environment matrix's child run).
    """
    try:
        store.write_parquet()
    except Exception as exc:  # noqa: BLE001 - logged; results.jsonl is the record.
        logger.warning("Could not write %s: %s", store.results_parquet_path, exc)
        store.append_event(
            "error",
            "Could not write results.parquet; results.jsonl holds every result",
            error=f"{type(exc).__name__}: {exc}",
        )


def _prepare_data(dataset: Any, config: RunConfig, dataset_config: DatasetConfig) -> None:
    """Prepare the requested data that is not present yet.

    Problems already listed locally are not requested again: a dataset's
    problem names are not always its download names (libsvm_qp lists
    ``svm_dual_<name>``), so asking for a bundled problem could request a
    URL that does not exist. A failure is raised as RuntimeError naming the
    dataset, which ``bench run`` reports without a traceback.
    """
    problem_names, all_problems = data_prepare_selection(config, dataset_config)
    if problem_names and not all_problems:
        listed = {problem.name for problem in dataset.list_problems()}
        problem_names = [name for name in problem_names if name not in listed]
        if not problem_names:
            return
    try:
        dataset.prepare_data(problem_names=problem_names, all_problems=all_problems)
    except Exception as exc:
        raise RuntimeError(
            f"Preparing data for dataset {dataset_config.id!r} failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _run_tasks_in_parallel(
    tasks: list[tuple[DatasetConfig, ProblemSpec, SolverConfig]],
    run_task: Callable[[DatasetConfig, ProblemSpec, SolverConfig], ProblemResult],
    store: ResultStore,
    progress: _ProgressReporter,
    parallelism: int,
) -> None:
    """Run ``tasks`` on a thread pool and write each result as it lands.

    On any exception in this thread, KeyboardInterrupt included, solves that
    have not started are cancelled, results that had already finished are
    written, and the workers still running are killed (their results are not
    recorded) before the exception propagates. Workers run in their own
    sessions, so a terminal Ctrl+C reaches only this process.
    """
    prefix = f"bench-pool-{next(_POOL_IDS)}"
    executor = ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix=prefix)
    futures = [
        executor.submit(run_task, dataset_config, problem, solver_config)
        for dataset_config, problem, solver_config in tasks
    ]
    written: set[Future] = set()
    try:
        for future in as_completed(futures):
            result = future.result()
            written.add(future)
            store.write_result(result)
            progress.record_result(result)
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        for future in futures:
            if future in written or not future.done() or future.cancelled():
                continue
            if future.exception() is not None:
                continue
            try:
                store.write_result(future.result())
            except Exception:  # noqa: BLE001 - keep the original exception.
                logger.exception("Could not write a finished result while stopping")
        _WORKERS.abort(prefix)
        raise
    finally:
        executor.shutdown(wait=True)
        _WORKERS.release(prefix)


def _run_task(
    store: ResultStore,
    config: RunConfig,
    repo_root: Path,
    dataset_config: DatasetConfig,
    problem: ProblemSpec,
    solver_config: SolverConfig,
    **kwargs: Any,
) -> ProblemResult:
    """``_run_one``, with a failure to run the worker recorded as its result.

    An OSError launching one worker (EMFILE, ENOMEM, ENOSPC writing its
    payload) would otherwise abort the whole run. The row is a worker error,
    so a later resume retries it.
    """
    try:
        return _run_one(store, config, repo_root, dataset_config, problem, solver_config, **kwargs)
    except Exception as exc:  # noqa: BLE001 - recorded on the result row.
        logger.exception(
            "Could not run %s/%s with %s", dataset_config.id, problem.name, solver_config.id
        )
        store.append_event(
            "error",
            "Could not run the worker",
            dataset=dataset_config.id,
            problem=problem.name,
            solver_id=solver_config.id,
            error=f"{type(exc).__name__}: {exc}",
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
            error=f"Could not run the worker: {type(exc).__name__}: {exc}",
            metadata=_metadata_with_environment(
                problem,
                solver_config,
                environment_id=kwargs.get("environment_id"),
                environment_metadata=kwargs.get("environment_metadata"),
                resume_signature=solve_signature(config, dataset_config, solver_config),
            ),
        )


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
                f"planned {self.total_expected} total solves: "
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
        fields: dict[str, Any] = {
            "completed_total": completed_total,
            "total_expected": self.total_expected,
            "already_complete": self.already_complete,
            "skipped_during_planning": self.skipped_during_planning,
            "completed_this_run": self.completed_this_run,
            "queued": self.queued,
            "remaining_queued": remaining,
            "elapsed_seconds": elapsed,
            "rate_solves_per_second": rate,
            "eta_remaining_seconds": eta,
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
                    f"({percent:.2f}%) | this_run "
                    f"{self.completed_this_run}/{self.queued} | elapsed "
                    f"{_format_duration(elapsed)} | rate {rate_text} "
                    f"| eta_remaining {eta_text}"
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
    resume_signature = solve_signature(config, dataset_config, solver_config)
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
        "resume_signature": resume_signature,
        "memory_limit_mb": config.worker_memory_limit_mb,
    }
    payload_path = artifacts_dir / "payload.json"
    atomic_write_text(payload_path, json.dumps(payload, indent=2, default=str))
    # Distinguish "no timeout configured" (None) from "explicitly disabled"
    # (0). A solver-level None means "fall back to the run-level setting";
    # any other numeric value (including 0, which means no limit) takes
    # precedence over the global default.
    if solver_config.timeout_seconds is not None:
        configured_timeout = solver_config.timeout_seconds
    else:
        configured_timeout = config.timeout_seconds
    time_limit = (
        float(configured_timeout)
        if configured_timeout is not None and configured_timeout > 0
        else None
    )
    subprocess_timeout = _subprocess_timeout(time_limit)
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
            run_time_seconds=time_limit,
            error=(
                f"Worker killed after {subprocess_timeout:g} s: timeout_seconds="
                f"{time_limit:g} plus {subprocess_timeout - time_limit:g} s grace "
                "for start-up and loading"
                if subprocess_timeout is not None and time_limit is not None
                else "Worker killed after the subprocess timeout"
            ),
            artifact_dir=str(artifacts_dir),
            metadata=_metadata_with_environment(
                problem,
                solver_config,
                environment_id=environment_id,
                environment_metadata=environment_metadata,
                resume_signature=resume_signature,
            ),
        )

    worker_result_path = artifacts_dir / "worker_result.json"
    if completed.returncode == 0 and worker_result_path.exists():
        result = _load_worker_result(
            worker_result_path,
            store=store,
            dataset_config=dataset_config,
            problem=problem,
            solver_config=solver_config,
            artifacts_dir=artifacts_dir,
            environment_id=environment_id,
            environment_metadata=environment_metadata,
            resume_signature=resume_signature,
        )
        _apply_time_limit(result, time_limit)
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
            resume_signature=resume_signature,
        ),
    )


# Statuses that claim a conclusion. Reached after the time limit, they are
# recorded as time_limit: the limit is the budget every solver gets.
_CONCLUSIVE_STATUSES = frozenset(
    {status.OPTIMAL, status.OPTIMAL_INACCURATE, *status.ANY_INFEASIBLE}
)


def _subprocess_timeout(time_limit: float | None) -> float | None:
    """Seconds after which the worker's process group is killed.

    The clock starts with the worker process, so it also covers interpreter
    start, imports and problem loading; a solver honouring its own time
    limit also needs a moment to return. The grace for that equals the
    limit, at least SUBPROCESS_TIMEOUT_MIN_GRACE_SECONDS and at most
    SUBPROCESS_TIMEOUT_GRACE_SECONDS. A fixed 30 s grace let a solve with a
    1 s limit run for 31 s.
    """
    if time_limit is None:
        return None
    grace = min(
        SUBPROCESS_TIMEOUT_GRACE_SECONDS,
        max(SUBPROCESS_TIMEOUT_MIN_GRACE_SECONDS, time_limit),
    )
    return time_limit + grace


def _apply_time_limit(result: ProblemResult, time_limit: float | None) -> None:
    """Record a conclusion reached after ``time_limit`` as time_limit.

    The solver's own status stays in ``info["status_before_timeout"]``.
    """
    if (
        time_limit is None
        or result.status not in _CONCLUSIVE_STATUSES
        or result.run_time_seconds is None
        or not result.run_time_seconds > time_limit
    ):
        return
    result.info = {**(result.info or {}), "status_before_timeout": result.status}
    result.status = status.TIME_LIMIT
    if result.error is None:
        result.error = (
            f"Solve took {result.run_time_seconds:.3g} s, more than "
            f"timeout_seconds={time_limit:g}"
        )


def _check_schema_version(record_version: object) -> str | None:
    """Return an error string if ``record_version`` is invalid or too
    new for the current runner; ``None`` means the version is fine.

    Pre-fix the comparison ``record_version > PROBLEM_RESULT_SCHEMA_VERSION``
    raised ``TypeError`` for non-int values like ``"2"`` (a hand-edited
    payload) and escaped ``_load_worker_result``, crashing the parent
    runner instead of being recorded as a worker error. Validate the
    type explicitly here so every malformed payload becomes a worker
    error row.
    """
    from solver_benchmarks.core.result import PROBLEM_RESULT_SCHEMA_VERSION

    if record_version is None:
        return None
    # ``bool`` is a subclass of ``int``; reject it explicitly so a
    # ``schema_version: true`` payload is not silently treated as ``1``.
    if isinstance(record_version, bool) or not isinstance(record_version, int):
        return (
            f"worker_result.json schema_version is not an integer "
            f"(got {record_version!r}); the worker payload is malformed."
        )
    if record_version > PROBLEM_RESULT_SCHEMA_VERSION:
        return (
            f"worker_result.json schema_version={record_version} is "
            f"newer than the runner's "
            f"({PROBLEM_RESULT_SCHEMA_VERSION}); upgrade the runner."
        )
    return None


def _load_worker_result(
    path: Path,
    *,
    store: ResultStore,
    dataset_config: DatasetConfig,
    problem: ProblemSpec,
    solver_config: SolverConfig,
    artifacts_dir: Path,
    environment_id: str | None,
    environment_metadata: dict | None,
    resume_signature: str | None = None,
) -> ProblemResult:
    """Parse a worker_result.json into a ProblemResult.

    Tolerates partial writes (truncated JSON), schema drift (unknown
    keys are dropped, missing keys fall back to defaults), and unparseable
    payloads, surfacing them as WORKER_ERROR rather than crashing the
    whole benchmark.
    """
    error: str | None = None
    record: dict | None = None
    try:
        text = path.read_text()
    except OSError as exc:
        error = f"Could not read worker_result.json: {exc}"
    else:
        try:
            record = json.loads(text)
        except json.JSONDecodeError as exc:
            error = f"Could not parse worker_result.json: {exc}"
    if record is not None:
        known = {field.name for field in dataclasses.fields(ProblemResult)}
        unknown = set(record) - known
        if unknown:
            record = {key: value for key, value in record.items() if key in known}
        # If the worker is from a future incompatible release the runner
        # would otherwise read a version field higher than its own; we
        # reject explicitly so the failure is obvious in the events log
        # rather than surfacing as a silently-mismatched record.
        record_version = record.get("schema_version")
        version_check = _check_schema_version(record_version)
        if version_check is not None:
            error = version_check
            record = None
        else:
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            if resume_signature is not None:
                metadata = {**metadata, "resume_signature": resume_signature}
            record["metadata"] = metadata
            try:
                return ProblemResult(**record)
            except TypeError as exc:
                error = f"Could not construct ProblemResult: {exc}"
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
        error=error,
        artifact_dir=str(artifacts_dir),
        metadata=_metadata_with_environment(
            problem,
            solver_config,
            environment_id=environment_id,
            environment_metadata=environment_metadata,
            resume_signature=resume_signature,
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


# Time budget after SIGTERM before the whole process group is SIGKILLed.
# A solver that handles SIGTERM can finish writing its output; the worker's
# Python output is unbuffered (PYTHONUNBUFFERED), but output a C library
# still holds in its stdio buffer is lost on SIGTERM.
_KILL_GRACE_SECONDS = 5.0
_FINAL_WAIT_SECONDS = 5.0


class _WorkerProcesses:
    """Live worker processes, so an interrupted parallel run can kill them.

    Workers run in their own sessions, so a terminal Ctrl+C reaches only the
    runner, whose main thread then kills the workers its pool threads
    started. A pool's threads are recognised by their name prefix; a worker
    a pool thread starts after its pool was aborted is killed at once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._live: dict[subprocess.Popen, str] = {}
        self._aborted: set[str] = set()

    def add(self, process: subprocess.Popen) -> None:
        name = threading.current_thread().name
        with self._lock:
            self._live[process] = name
            aborted = any(name.startswith(f"{prefix}_") for prefix in self._aborted)
        if aborted:
            _kill_process_group(process)

    def discard(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._live.pop(process, None)

    def abort(self, prefix: str) -> None:
        with self._lock:
            self._aborted.add(prefix)
            victims = [
                process
                for process, name in self._live.items()
                if name.startswith(f"{prefix}_")
            ]
        for process in victims:
            _kill_process_group(process)

    def release(self, prefix: str) -> None:
        with self._lock:
            self._aborted.discard(prefix)


_WORKERS = _WorkerProcesses()


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: float | None,
    stdout_path: Path,
    stderr_path: Path,
    stream_output: bool,
):
    # Only a bounded tail stays in memory: verbose solvers can print hundreds
    # of MB, and the logs on disk already hold everything.
    stdout_tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
    stderr_tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
    # Line-buffered open so the on-disk log reflects in-progress output if
    # the worker is killed mid-line (prior bufsize=1 referred only to the
    # child pipes).
    stdout_log = stdout_path.open("w", buffering=1, encoding="utf-8")
    stderr_log = stderr_path.open("w", buffering=1, encoding="utf-8")
    threads: list[threading.Thread] = []
    process: subprocess.Popen | None = None
    try:
        # Run the worker in its own session/process group so we can kill
        # any helper processes the solver may have spawned.
        popen_kwargs: dict = {
            "cwd": cwd,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            # Solver output is not always UTF-8 (a Latin-1 problem name, a C
            # library's log). A strict decode would kill the log reader and
            # close the pipe, and the worker would then die of EPIPE.
            "encoding": "utf-8",
            "errors": "backslashreplace",
            "bufsize": 1,
            # Python's stdout is block-buffered on a pipe, and a timeout kill
            # drops the buffer: the log of a timed-out solve would lose the
            # iterations that explain the timeout.
            "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
        }
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        process = subprocess.Popen(cmd, **popen_kwargs)
        _WORKERS.add(process)
        assert process.stdout is not None
        assert process.stderr is not None
        threads = [
            threading.Thread(
                target=_tee_stream,
                args=(process.stdout, stdout_log, sys.stdout, stdout_tail, stream_output),
                daemon=True,
            ),
            threading.Thread(
                target=_tee_stream,
                args=(process.stderr, stderr_log, sys.stderr, stderr_tail, stream_output),
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
            returncode = _terminate_process(process)
        except BaseException:
            # KeyboardInterrupt or any other error while waiting: the worker
            # would otherwise keep solving, unsupervised and with no timeout.
            _kill_process_group(process)
            raise
        else:
            # Helpers the worker left behind would hold the pipes open, stall
            # the log readers and compete for CPU with later solves.
            _kill_process_group(process)
    finally:
        if process is not None:
            _WORKERS.discard(process)
        # Always drain pipes (so we don't lose the final partial line)
        # and close the log files. Bound the join so a wedged tee thread
        # doesn't hang the runner.
        for thread in threads:
            thread.join(timeout=_FINAL_WAIT_SECONDS)
        try:
            stdout_log.close()
        except OSError:
            pass
        try:
            stderr_log.close()
        except OSError:
            pass
    return SimpleNamespace(
        returncode=returncode,
        stdout="".join(stdout_tail),
        stderr="".join(stderr_tail),
        timed_out=timed_out,
    )


def _kill_process_group(process: subprocess.Popen) -> None:
    """SIGKILL the worker's process group, leader and helpers alike.

    The group is the worker's own (``start_new_session``), so its id is the
    worker's pid. Linux does not reuse a pid while it is still the id of a
    live process group, so this is safe after the worker was reaped; it
    raises ProcessLookupError once the group is empty.
    """
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return
    try:
        process.kill()
    except OSError:
        pass


def _terminate_process(process: subprocess.Popen) -> int:
    """Kill the process group and return the exit code.

    SIGTERM first, so a solver that handles it can finish its output, then
    SIGKILL to the whole group once the worker exits or the grace runs out:
    a helper that ignores SIGTERM must not outlive the timeout. Always
    returns a returncode (using -SIGKILL as a last resort) rather than
    blocking forever on ``wait()``.
    """
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            returncode = process.wait(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            returncode = None
        _kill_process_group(process)
        if returncode is not None:
            return returncode
    process.kill()
    try:
        return process.wait(timeout=_FINAL_WAIT_SECONDS)
    except subprocess.TimeoutExpired:
        return -signal.SIGKILL


def _tee_stream(source, log_file, sink, tail: deque[str], stream_output: bool) -> None:
    """Copy ``source`` to the log (and the terminal) until it closes.

    Keeps draining when the log or the terminal fails: closing the pipe
    early would kill the worker with EPIPE and lose a result it had
    already written.
    """
    log_ok = True
    sink_ok = stream_output
    try:
        for line in source:
            tail.append(line)
            if log_ok:
                try:
                    log_file.write(line)
                except (OSError, ValueError):
                    # Disk full, or the log was closed by the parent during
                    # teardown.
                    log_ok = False
            if sink_ok:
                try:
                    sink.write(line)
                    sink.flush()
                except (OSError, ValueError):
                    sink_ok = False
    except (OSError, ValueError):
        pass
    finally:
        try:
            source.close()
        except (OSError, ValueError):
            pass


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


def refuse_to_rerun_into(run_dir: str | Path, config: RunConfig) -> None:
    """Raise if ``run_dir`` already holds results for solves ``config`` plans.

    With ``resume: false`` every planned solve runs again, and its rows would
    be appended next to the earlier ones for the same solves. Rows of other
    dataset/solver pairs do not conflict: an environment matrix runs each
    environment's solvers into one shared directory.
    """
    results_path = Path(run_dir) / "results.jsonl"
    if not results_path.exists():
        return
    planned = {
        (str(dataset.id), solver.id) for dataset in config.datasets for solver in config.solvers
    }
    with results_path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            if (str(record.get("dataset", "")), str(record.get("solver_id"))) in planned:
                raise RuntimeError(
                    f"Run directory {Path(run_dir).resolve()} already has results for "
                    "this config and run.resume is false, so every solve would run "
                    "again next to its earlier result. Set `resume: true` to resume "
                    "the run, or use a new --run-dir."
                )


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
    skip_reason: str | None = None,
) -> None:
    # An unavailable solver (or unsupported kind on this host) is not a new
    # solve attempt. Preserve the failed row and its retry budget so another
    # host can still recover it; do not create retry artifacts for a skip.
    if store.has_pending_retry(dataset_config.id, problem.name, solver_config.id):
        return
    metadata = _metadata_with_environment(
        problem,
        solver_config,
        environment_id=environment_id,
        environment_metadata=environment_metadata,
        resume_signature=solve_signature(config, dataset_config, solver_config),
    )
    if skip_reason is not None:
        metadata["skip_reason"] = skip_reason
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
            metadata=metadata,
        )
    )


def _metadata_with_environment(
    problem: ProblemSpec,
    solver_config: SolverConfig,
    *,
    environment_id: str | None,
    environment_metadata: dict | None,
    resume_signature: str | None = None,
) -> dict:
    metadata = dict(problem.metadata)
    metadata.update(
        {
            "environment_id": environment_id,
            "environment_metadata": environment_metadata or {},
            "runtime": runtime_metadata(solver_config.solver),
        }
    )
    if resume_signature is not None:
        metadata["resume_signature"] = resume_signature
    return metadata
