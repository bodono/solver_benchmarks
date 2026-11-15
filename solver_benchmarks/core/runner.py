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
    close_names,
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
from solver_benchmarks.datasets.base import (
    Dataset,
    default_repo_root,
    filter_problem_specs_by_size,
)
from solver_benchmarks.solvers import get_solver

# Bounds of the grace added on top of the configured timeout before the
# worker is force-killed (see _subprocess_timeout). The grace covers worker
# start-up and problem loading, and lets solvers that honor their own
# internal time limit return a partial result (status, iterations,
# residuals) instead of being killed mid-return.
SUBPROCESS_TIMEOUT_GRACE_SECONDS = 30.0
SUBPROCESS_TIMEOUT_MIN_GRACE_SECONDS = 5.0
# Lines of the worker's stderr kept in memory; the logs hold everything.
_OUTPUT_TAIL_LINES = 200
# How much of that tail a worker error row quotes.
_ERROR_TAIL_LINES = 5
_ERROR_TAIL_CHARS = 500
# Tells a worker which process started it, so that it dies with it.
RUNNER_PID_ENV = "SOLVER_BENCHMARKS_RUNNER_PID"
# Held (flock) by the one process writing to a run directory.
RUN_DIR_LOCK_NAME = ".bench.lock"

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
    """Run every solve ``config`` plans into a run directory; return its store.

    A run directory has one writer at a time: the run holds an exclusive
    lock on ``<run_dir>/.bench.lock`` from start to finish, and a second run
    on the same directory fails at once (see :class:`_RunDirLock`).
    """
    lock = _RunDirLock()
    try:
        return _run_benchmark(
            config,
            lock=lock,
            run_dir=run_dir,
            repo_root=repo_root,
            stream_output=stream_output,
            stream_solver_output=stream_solver_output,
            environment_id=environment_id,
            environment_metadata=environment_metadata,
            prepare_data_command=prepare_data_command,
            source_config_path=source_config_path,
        )
    finally:
        lock.release()


def _run_benchmark(
    config: RunConfig,
    *,
    lock: _RunDirLock,
    run_dir: str | Path | None,
    repo_root: str | Path | None,
    stream_output: bool,
    stream_solver_output: bool | None,
    environment_id: str | None,
    environment_metadata: dict | None,
    prepare_data_command: str | None,
    source_config_path: str | Path | None,
) -> ResultStore:
    repo_root = Path(repo_root).resolve() if repo_root else default_repo_root()
    config = resolve_output_dir(config, repo_root)
    # Like output_dir, a relative data_root names a directory under repo_root,
    # where the workers start; made absolute, it names the same place when
    # this process lists the problems from another directory.
    config = dataclasses.replace(config, datasets=resolve_data_roots(config.datasets, repo_root))
    if stream_solver_output is None:
        stream_solver_output = stream_output
    if run_dir is not None:
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        lock.acquire(run_dir)
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
    if config.fail_on_unsupported:
        for solver_config in config.solvers:
            if solver_config.id not in available_solver_ids:
                raise RuntimeError(
                    f"Solver {solver_config.solver!r} is unavailable (not installed) and "
                    f"run.fail_on_unsupported is true, so {solver_config.id!r} is not "
                    "skipped: install it or remove it from the config."
                )
    listings = _list_datasets(config, repo_root)
    filter_warnings = _check_filter_names(config, listings, repo_root=repo_root)
    store = ResultStore.create(config, run_dir=run_dir)
    if run_dir is None:
        lock.acquire(store.run_dir)
    if lock.warning is not None:
        store.append_event("warning", lock.warning)
        _emit_progress(stream_output, f"warning: {lock.warning}")
    for message, fields in filter_warnings:
        store.append_event("warning", message, **fields)
        _emit_progress(stream_output, f"warning: {message}")
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

    recorded_skips = (
        _recorded_skips(store, completed) if config.fail_on_unsupported else {}
    )
    tasks: list[tuple[DatasetConfig, ProblemSpec, SolverConfig]] = []
    already_complete = 0
    skipped_during_planning = 0
    for dataset_config, dataset, listed in listings:
        include, exclude = config.effective_filters(dataset_config)
        selected = _filter_problems(listed, include, exclude)
        max_size_mb = dataset_config.dataset_options.get("max_size_mb")
        problems = filter_problem_specs_by_size(selected, max_size_mb)
        too_large = sorted(
            ({problem.name for problem in selected} - {problem.name for problem in problems})
            & set(include)
        )
        if too_large:
            message = (
                f"Dataset {dataset_config.id!r}: {_quoted(too_large)} in include "
                f"{'is' if len(too_large) == 1 else 'are'} larger than "
                f"max_size_mb={max_size_mb} and not solved."
            )
            store.append_event("warning", message, dataset=dataset_config.id, problems=too_large)
            _emit_progress(stream_output, f"warning: {message}")
        if not problems:
            data_status = (
                dataset.data_status() if hasattr(dataset, "data_status") else None
            )
            if listed:
                message = (
                    f"Dataset {dataset_config.id!r} has no problems left after the "
                    f"include / exclude / max_size_mb filters ({len(listed)} listed)."
                )
            elif data_status is not None:
                message = data_status.message
            else:
                message = f"Dataset {dataset_config.id!r} produced no problems."
            if data_status is not None and not getattr(data_status, "available", True):
                problem_names, all_problems = data_prepare_selection(
                    config, dataset_config
                )
                message = _missing_data_run_message(
                    dataset_config,
                    data_status.message,
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
                    key = (dataset_config.id, problem.name, solver_config.id)
                    if key in recorded_skips:
                        raise RuntimeError(
                            f"{solver_config.id!r} has {dataset_config.id}/{problem.name} recorded "
                            f"as skipped_unsupported ({str(recorded_skips[key]).rstrip('.')}); "
                            "run.fail_on_unsupported is true, so the run stops."
                        )
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
        _retire_replaced_rows(store)
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

    try:
        if parallelism == 1:
            for task in tasks:
                result = run_task(*task)
                store.write_result(result)
                progress.record_result(result)
                _check_supported(config, result)
        else:
            _run_tasks_in_parallel(
                tasks, run_task, store, progress, parallelism,
                check_result=lambda result: _check_supported(config, result),
            )
    finally:
        # Also after an interruption, so the results written so far stand
        # alone in results.jsonl.
        _retire_replaced_rows(store)
    # All queued tasks completed successfully — materialize the parquet
    # exactly once. results.jsonl is the canonical live record during
    # the run; results.parquet is the end-of-run derived artifact whose
    # presence signals "this run completed".
    _write_parquet(store)
    progress.emit_final()
    return store


class _RunDirLock:
    """An exclusive ``flock`` on ``<run_dir>/.bench.lock`` for one run.

    Two runs writing one directory would solve everything twice into the
    same artifact paths, and a rewrite of results.jsonl in one would drop
    rows the other appended meanwhile. The holder's pid is written into the
    file for the error message of a second run. The kernel releases the lock
    when the holder exits, however it dies, so nothing stale is left. Where
    locks are not available (no ``fcntl``, or a file system that refuses
    them, as some network mounts do) the run goes on unlocked and
    ``warning`` says so.
    """

    def __init__(self) -> None:
        self._fd: int | None = None
        self.warning: str | None = None

    def acquire(self, run_dir: Path) -> None:
        try:
            import fcntl
        except ImportError:
            self.warning = (
                f"File locks are not available here, so nothing stops another bench run "
                f"from writing to {run_dir} at the same time."
            )
            return
        path = Path(run_dir) / RUN_DIR_LOCK_NAME
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        except OSError as exc:
            self.warning = f"Could not lock {run_dir} ({exc}); running without a lock."
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            try:
                holder = os.read(fd, 64).decode(errors="replace").strip()
            except OSError:
                holder = ""
            os.close(fd)
            raise RuntimeError(
                f"Run directory {run_dir} is in use by another bench run"
                + (f" (pid {holder})" if holder else "")
                + "; wait for it to finish, or use another --run-dir."
            ) from None
        except OSError as exc:
            os.close(fd)
            self.warning = f"Could not lock {run_dir} ({exc}); running without a lock."
            return
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:
            pass
        self._fd = fd

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


def _list_datasets(
    config: RunConfig, repo_root: Path
) -> list[tuple[DatasetConfig, Any, list[ProblemSpec]]]:
    """Build each dataset, prepare its data if asked to, and list its problems."""
    listings = []
    for dataset_config in config.datasets:
        dataset_cls = get_dataset(dataset_config.name)
        dataset = dataset_cls(repo_root=repo_root, **dataset_config.dataset_options)
        if config.auto_prepare_data and hasattr(dataset, "prepare_data"):
            _prepare_data(dataset, config, dataset_config)
        listings.append((dataset_config, dataset, list(dataset.list_problems())))
    return listings


def _may_have_problem(dataset: Any, listed: list[ProblemSpec], name: str) -> bool:
    """Whether ``dataset`` may have a problem ``name`` it does not list.

    A dataset that knows every name it can list (``known_problem_names``)
    answers from that set. Otherwise, datasets that download or prepare their
    data list only what is present, so they cannot tell a misspelled name
    from one not downloaded yet; nor can a dataset with no data listed at all.
    """
    known = dataset.known_problem_names() if hasattr(dataset, "known_problem_names") else None
    if known is not None:
        return name in known
    if not listed:
        return True
    if getattr(dataset, "automatic_download", False) or getattr(dataset, "prepare_command", None):
        return True
    prepare = getattr(type(dataset), "prepare_data", None)
    return prepare is not None and prepare is not Dataset.prepare_data


def _check_filter_names(
    config: RunConfig,
    listings: list[tuple[DatasetConfig, Any, list[ProblemSpec]]],
    *,
    repo_root: Path,
) -> list[tuple[str, dict[str, Any]]]:
    """Raise for ``include`` names that match no problem; warn about the rest.

    A dataset entry's own ``include`` must match in that dataset, and the
    run-level ``include`` in at least one of the datasets it applies to. A
    name a dataset may have without listing it (see ``_may_have_problem``)
    gets a warning instead; a dataset with no data at all gets the
    missing-data error later. ``exclude`` names that match nothing are
    warnings. Returns the warnings as ``(message, event fields)``.
    """
    warnings: list[tuple[str, dict[str, Any]]] = []
    run_include = list(config.include)
    run_include_datasets: list[tuple[DatasetConfig, Any, list[ProblemSpec]]] = []
    run_exclude_known: set[str] = set()
    any_listed = False
    for dataset_config, dataset, listed in listings:
        known = {problem.name for problem in listed}
        any_listed = any_listed or bool(listed)
        if dataset_config.include:
            unmatched = [name for name in dataset_config.include if name not in known]
            suggestions = known | _known_names(dataset)
            typos = [name for name in unmatched if not _may_have_problem(dataset, listed, name)]
            if typos:
                raise RuntimeError(
                    f"Dataset {dataset_config.id!r} has no problem named "
                    f"{_described(typos, suggestions)} (its include list). "
                    f"`bench list problems {dataset_config.name}` lists its problems."
                )
            if unmatched and listed:
                warnings.append(
                    (
                        f"Dataset {dataset_config.id!r} does not have "
                        f"{_described(unmatched, suggestions)} locally: not prepared yet, "
                        "or misspelled."
                        + _prepare_advice(dataset_config, dataset, unmatched, repo_root),
                        {"dataset": dataset_config.id, "problems": unmatched},
                    )
                )
        elif run_include:
            run_include_datasets.append((dataset_config, dataset, listed))
        unmatched_exclude = [name for name in dataset_config.exclude if name not in known]
        if unmatched_exclude and listed:
            warnings.append(
                (
                    f"Dataset {dataset_config.id!r} has no problem named "
                    f"{_described(unmatched_exclude, known)} (its exclude list); "
                    "nothing was excluded for it.",
                    {"dataset": dataset_config.id, "problems": unmatched_exclude},
                )
            )
        run_exclude_known |= known
    if run_include_datasets:
        known = {problem.name for _, _, listed in run_include_datasets for problem in listed}
        suggestions = known.union(*(_known_names(dataset) for _, dataset, _ in run_include_datasets))
        unmatched = [name for name in run_include if name not in known]
        ids = [str(dataset_config.id) for dataset_config, _, _ in run_include_datasets]
        where = f"dataset{'s' if len(ids) > 1 else ''} {_quoted(ids)}"
        typos = [
            name
            for name in unmatched
            if not any(_may_have_problem(dataset, listed, name) for _, dataset, listed in run_include_datasets)
        ]
        if typos:
            raise RuntimeError(
                f"run.include: no problem named {_described(typos, suggestions)} in {where}. "
                "`bench list problems DATASET` lists a dataset's problems."
            )
        if unmatched and known:
            advice = "".join(
                _prepare_advice(dataset_config, dataset, maybe, repo_root)
                for dataset_config, dataset, listed in run_include_datasets
                if (maybe := [name for name in unmatched if _may_have_problem(dataset, listed, name)])
            )
            warnings.append(
                (
                    f"run.include: {_described(unmatched, suggestions)} not found locally in "
                    f"{where}: not prepared yet, or misspelled.{advice}",
                    {"problems": unmatched},
                )
            )
    unmatched = [name for name in config.exclude if name not in run_exclude_known]
    if unmatched and any_listed:
        warnings.append(
            (
                f"run.exclude: no problem named {_described(unmatched, run_exclude_known)} "
                "in any dataset; nothing was excluded for it.",
                {"problems": unmatched},
            )
        )
    return warnings


def _known_names(dataset: Any) -> set[str]:
    known = dataset.known_problem_names() if hasattr(dataset, "known_problem_names") else None
    return set(known or ())


def _prepare_advice(
    dataset_config: DatasetConfig,
    dataset: Any,
    names: list[str],
    repo_root: Path,
) -> str:
    """How to prepare ``names`` for this dataset, or nothing if it has no way."""
    if getattr(dataset, "automatic_download", False):
        command = data_prepare_command(dataset_config, problem_names=names, repo_root=repo_root)
        return f" Prepare them with `{command}`, or rerun with --prepare-data."
    prepare_command = getattr(dataset, "prepare_command", None)
    if prepare_command:
        return f" Dataset {dataset_config.id!r} is prepared with `{prepare_command}`."
    return ""


def _described(names: list[str], known: set[str]) -> str:
    parts = []
    for name in names:
        close = close_names(name, known)
        parts.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    return ", ".join(parts)


def _quoted(names: Iterable[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _recorded_skips(
    store: ResultStore, completed: set[tuple[str, str, str]]
) -> dict[tuple[str, str, str], str | None]:
    """The completed solves whose latest row is skipped_unsupported, with its reason.

    A resumed run counts such a row as complete; under
    ``fail_on_unsupported`` it must fail the run like the skip did.
    """
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    path = store.results_jsonl_path
    if not path.exists():
        return {}
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            key = (str(record.get("dataset")), str(record.get("problem")), str(record.get("solver_id")))
            if key in completed:
                latest[key] = record
    return {
        key: record.get("error")
        for key, record in latest.items()
        if record.get("status") == status.SKIPPED_UNSUPPORTED
    }


def _check_supported(config: RunConfig, result: ProblemResult) -> None:
    """Fail the run on a skip the adapter reported, under ``fail_on_unsupported``.

    The row is written first, so it is not lost.
    """
    if config.fail_on_unsupported and result.status == status.SKIPPED_UNSUPPORTED:
        raise RuntimeError(
            f"{result.solver_id!r} skipped {result.dataset}/{result.problem}: "
            f"{str(result.error).rstrip('.')}; run.fail_on_unsupported is true, so the run stops."
        )


def _retire_replaced_rows(store: ResultStore) -> None:
    """Move the superseded rows replaced during the run out of results.jsonl.

    A failure is logged, not raised: the rows stay next to their
    replacements, and the next resume moves them.
    """
    try:
        retired = store.retire_replaced_rows()
    except Exception as exc:  # noqa: BLE001 - logged; the next resume retries.
        logger.warning("Could not move superseded rows out of %s: %s",
                       store.results_jsonl_path, exc)
        store.append_event(
            "error",
            "Could not move superseded rows to results.superseded.jsonl; "
            "the next resume moves them",
            error=f"{type(exc).__name__}: {exc}",
        )
        return
    if retired:
        store.append_event(
            "info",
            "Moved results solved under other settings to results.superseded.jsonl",
            superseded_rows=retired,
        )


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
    *,
    check_result: Callable[[ProblemResult], None] | None = None,
) -> None:
    """Run ``tasks`` on a thread pool and write each result as it lands.

    ``check_result`` sees each result once it is written and may raise to
    stop the run. On any exception in this thread, KeyboardInterrupt
    included, solves that have not started are cancelled, results that had
    already finished are written, and the workers still running are killed
    (their results are not recorded) before the exception propagates. Workers run in their own
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
            if check_result is not None:
                check_result(result)
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
    if not _WORKERS.aborting():
        # A worker the runner killed itself while stopping is not an error
        # of its own, and its result is not recorded.
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
        error=_worker_exit_message(completed.returncode, completed.stderr),
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


def _worker_exit_message(returncode: int | None, stderr: str | None) -> str:
    """Why a worker ended without a result: its exit status and stderr tail.

    A negative code is the signal that killed it; SIGKILL usually comes from
    the out-of-memory killer. The last few non-empty stderr lines usually
    name the reason (an abort message, an import error).
    """
    sigkill = getattr(signal, "SIGKILL", None)
    if returncode is not None and returncode < 0:
        try:
            name = signal.Signals(-returncode).name
        except ValueError:
            name = f"signal {-returncode}"
        message = f"Worker killed by {name}"
        if sigkill is not None and -returncode == sigkill:
            message += " (possibly out of memory)"
    elif returncode == 0:
        message = "Worker exited with code 0 without writing worker_result.json"
    else:
        message = f"Worker exited with code {returncode}"
    lines = [line.rstrip() for line in (stderr or "").splitlines() if line.strip()]
    tail = "\n".join(lines[-_ERROR_TAIL_LINES:])
    if len(tail) > _ERROR_TAIL_CHARS:
        tail = "..." + tail[-_ERROR_TAIL_CHARS:]
    if tail:
        message += f"; stderr ends with:\n{tail}"
    return message


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
    """Mark a result returned after ``time_limit``; a conclusion becomes time_limit.

    Every such result gets ``info["exceeded_time_limit"]``, whatever its
    status, so that ``bench kkt-verify --promote`` does not count its point
    as solved. A conclusive status is recorded as time_limit, with the
    solver's own status in ``info["status_before_timeout"]``; it loses its
    objective (reported only for optimal and optimal_inaccurate rows), which
    is kept in ``info["objective_before_timeout"]``. Other statuses are kept.
    """
    if (
        time_limit is None
        or result.run_time_seconds is None
        or not result.run_time_seconds > time_limit
    ):
        return
    info = {**(result.info or {}), "exceeded_time_limit": True}
    if result.status in _CONCLUSIVE_STATUSES:
        info["status_before_timeout"] = result.status
        info["objective_before_timeout"] = result.objective_value
        result.status = status.TIME_LIMIT
        result.objective_value = None
        if result.error is None:
            result.error = (
                f"Solve took {result.run_time_seconds:.3g} s, more than "
                f"timeout_seconds={time_limit:g}"
            )
    result.info = info


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

    def aborting(self) -> bool:
        """Whether the calling thread belongs to a pool being aborted."""
        name = threading.current_thread().name
        with self._lock:
            return any(name.startswith(f"{prefix}_") for prefix in self._aborted)


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
    # Only a bounded tail of stderr stays in memory, to say why a worker
    # crashed: verbose solvers can print hundreds of MB, and the logs on disk
    # already hold everything.
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
            # The worker dies with this process (see worker._die_with_runner).
            "env": {**os.environ, "PYTHONUNBUFFERED": "1", RUNNER_PID_ENV: str(os.getpid())},
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
                args=(process.stdout, stdout_log, sys.stdout, None, stream_output),
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
        else:
            # Helpers the worker left behind would hold the pipes open, stall
            # the log readers and compete for CPU with later solves.
            _kill_process_group(process)
    except BaseException:
        # KeyboardInterrupt or any other error once the worker started: it
        # would otherwise keep solving, unsupervised and with no timeout.
        if process is not None:
            _kill_process_group(process)
        raise
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


def _tee_stream(source, log_file, sink, tail: deque[str] | None, stream_output: bool) -> None:
    """Copy ``source`` to the log (and the terminal, and ``tail``) until it closes.

    Keeps draining when the log or the terminal fails: closing the pipe
    early would kill the worker with EPIPE and lose a result it had
    already written.
    """
    log_ok = True
    sink_ok = stream_output
    try:
        for line in source:
            if tail is not None:
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
