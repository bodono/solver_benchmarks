"""Generic benchmark runner."""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import itertools
import json
import logging
import math
import os
import shutil
import signal
import site
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
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
from solver_benchmarks.core.environment import THREAD_LIMIT_ENV_VARS, runtime_metadata
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
from solver_benchmarks.solvers.base import (
    CONFIG_ERROR_KIND,
    SolverConfigError,
    pop_threads,
    threads_or_default,
)

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
    on the same directory fails at once (see :class:`_RunDirLock`). When
    planning fails (missing data, a solver that may not be skipped), a run
    directory this call created is removed again, so no directory holding
    only a manifest is left behind; one that existed before is kept.
    """
    lock = _RunDirLock()
    new_dir = _NewRunDir()
    try:
        return _run_benchmark(
            config,
            lock=lock,
            new_dir=new_dir,
            run_dir=run_dir,
            repo_root=repo_root,
            stream_output=stream_output,
            stream_solver_output=stream_solver_output,
            environment_id=environment_id,
            environment_metadata=environment_metadata,
            prepare_data_command=prepare_data_command,
            source_config_path=source_config_path,
        )
    except BaseException:  # Ctrl+C in planning too
        new_dir.remove_if_unplanned()
        raise
    finally:
        lock.release()


@dataclasses.dataclass
class _NewRunDir:
    """The run directory a call created, until its solves are planned, and
    the directories above it the call created for it (innermost first)."""

    path: Path | None = None
    parents: list[Path] = dataclasses.field(default_factory=list)
    planned: bool = False

    def remove_if_unplanned(self) -> None:
        if self.path is None or self.planned:
            return
        shutil.rmtree(self.path, ignore_errors=True)
        for parent in self.parents:
            try:
                parent.rmdir()  # only while empty
            except OSError:
                break


def _missing_dirs(path: Path) -> list[Path]:
    """``path`` and its ancestors that do not exist yet, innermost first."""
    missing = []
    for candidate in (path, *path.parents):
        if candidate.exists():
            break
        missing.append(candidate)
    return missing


def _run_benchmark(
    config: RunConfig,
    *,
    lock: _RunDirLock,
    new_dir: _NewRunDir,
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
        missing = _missing_dirs(run_dir)
        try:
            run_dir.mkdir(parents=True)
            created = True
        except FileExistsError:
            created = False
        lock.acquire(run_dir)
        if created:
            new_dir.path = run_dir
            new_dir.parents = missing[1:]
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
                    f"Solver {solver_config.solver!r} is unavailable (not installed, or installed "
                    "but it could not be loaded) and "
                    f"run.fail_on_unsupported is true, so {solver_config.id!r} is not "
                    "skipped: install it or remove it from the config."
                )
    listings = _list_datasets(config, repo_root)
    filter_warnings = _check_filter_names(
        config, listings, repo_root=repo_root, prepare_data_command=prepare_data_command
    )
    missing_output_dirs = _missing_dirs(Path(config.output_dir)) if run_dir is None else []
    store = ResultStore.create(config, run_dir=run_dir)
    if run_dir is None:
        new_dir.path = store.run_dir
        new_dir.parents = missing_output_dirs
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
    new_dir.planned = True

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

    rejected = _RejectedSolvers(store, stream_output, environment_id=environment_id)

    def run_task(
        dataset_config: DatasetConfig, problem: ProblemSpec, solver_config: SolverConfig
    ) -> ProblemResult | None:
        if rejected.rejects(solver_config.id):
            return None  # its settings were rejected: the solve is not run
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
                if result is None or rejected.discard(result):
                    continue
                store.write_result(result)
                progress.record_result(result)
                _check_supported(config, result)
        else:
            _run_tasks_in_parallel(
                tasks, run_task, store, progress, parallelism,
                check_result=lambda result: _check_supported(config, result),
                discard_result=rejected.discard,
            )
    finally:
        # Also after an interruption, so the results written so far stand
        # alone in results.jsonl.
        _retire_replaced_rows(store)
    # A solver whose settings were rejected left solves undone: the run did not
    # complete, so there is no parquet sentinel and the run fails.
    rejected.raise_if_any()
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
    answers from that set, and one whose options select problems by name
    (``requested_problem_names``) cannot list a name outside them. Otherwise,
    datasets that download or prepare their data list only what is present,
    so they cannot tell a misspelled name from one not downloaded yet; nor
    can a dataset with no data listed at all.
    """
    known = dataset.known_problem_names() if hasattr(dataset, "known_problem_names") else None
    if known is not None:
        return name in known
    requested = _requested_names(dataset)
    if requested is not None and name not in requested:
        return False
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
    prepare_data_command: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Check the problems a config asks for by name against the listings.

    The names asked for are a dataset entry's ``include``, the run-level
    ``include`` (which must match in at least one of the datasets it applies
    to) and the names the dataset's ``subset`` option asks for
    (``Dataset.requested_problem_names``). A name a dataset cannot have (see
    ``_may_have_problem``) raises, with suggestions. A name it may have but
    does not list is missing data: for a dataset that downloads its data
    automatically that is an error naming the problems and how to prepare
    them; otherwise a warning naming the dataset's prepare command. A dataset
    with nothing listed at all gets the missing-data error later. ``exclude``
    names that match nothing are warnings. Returns the warnings as
    ``(message, event fields)``.
    """
    warnings: list[tuple[str, dict[str, Any]]] = []
    # Per dataset id: the requested names it may have but does not list, and
    # per name the names to suggest for it (none for a subset name: the
    # subset asks for it by that name).
    missing: dict[str, tuple[DatasetConfig, Any, list[str], dict[str, set[str]]]] = {}

    def add_missing(
        dataset_config: DatasetConfig, dataset: Any, names: list[str], suggestions: set[str]
    ) -> None:
        entry = missing.setdefault(str(dataset_config.id), (dataset_config, dataset, [], {}))
        entry[2].extend(name for name in names if name not in entry[2])
        for name in names:
            entry[3].setdefault(name, set()).update(suggestions)

    run_include = list(config.include)
    run_include_datasets: list[tuple[DatasetConfig, Any, list[ProblemSpec]]] = []
    run_exclude_known: set[str] = set()
    any_listed = False
    for dataset_config, dataset, listed in listings:
        known = {problem.name for problem in listed}
        any_listed = any_listed or bool(listed)
        suggestions = known | _known_names(dataset)
        requested = _requested_names(dataset)
        include, exclude = config.effective_filters(dataset_config)
        # Only the subset names the run would solve: those include and
        # exclude keep, as preparation prepares.
        unmatched_subset = sorted(
            name
            for name in set(requested or ()) - known
            if (not include or name in include) and name not in exclude
        )
        typos = [name for name in unmatched_subset if not _may_have_problem(dataset, listed, name)]
        if typos:
            raise RuntimeError(
                f"Dataset {dataset_config.id!r} has no problem named "
                f"{_described(typos, suggestions)} (its subset option). "
                f"`bench list problems {dataset_config.name}` lists its problems."
            )
        if unmatched_subset and listed:
            add_missing(dataset_config, dataset, unmatched_subset, set())
        if dataset_config.include:
            unmatched = [name for name in dataset_config.include if name not in known]
            typos = [name for name in unmatched if not _may_have_problem(dataset, listed, name)]
            if typos:
                raise RuntimeError(
                    f"Dataset {dataset_config.id!r} has no problem named "
                    f"{_described(typos, suggestions)} (its include list). "
                    f"`bench list problems {dataset_config.name}` lists its problems."
                    + _outside_subset_hint(dataset_config, dataset, typos)
                )
            if unmatched and listed:
                add_missing(dataset_config, dataset, unmatched, suggestions)
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
                + "".join(
                    _outside_subset_hint(dataset_config, dataset, typos)
                    for dataset_config, dataset, _ in run_include_datasets
                )
            )
        if unmatched and known:
            ambiguous = []
            for name in unmatched:
                holders = [
                    (dataset_config, dataset)
                    for dataset_config, dataset, listed in run_include_datasets
                    if _may_have_problem(dataset, listed, name)
                ]
                # Missing data only where every dataset that may have the
                # name can download it; otherwise it may need a manual step.
                if all(getattr(dataset, "automatic_download", False) for _, dataset in holders):
                    for dataset_config, dataset in holders:
                        add_missing(dataset_config, dataset, [name], suggestions)
                else:
                    ambiguous.append(name)
            if ambiguous:
                advice = "".join(
                    _prepare_advice(dataset_config, dataset, maybe, repo_root)
                    for dataset_config, dataset, listed in run_include_datasets
                    if (maybe := [name for name in ambiguous if _may_have_problem(dataset, listed, name)])
                )
                warnings.append(
                    (
                        f"run.include: {_described(ambiguous, suggestions)} not found locally in "
                        f"{where}: not prepared yet, or misspelled.{advice}",
                        {"problems": ambiguous},
                    )
                )
    errors = []
    for dataset_config, dataset, names, suggested in missing.values():
        if getattr(dataset, "automatic_download", False):
            errors.append(
                _missing_problems_message(
                    dataset_config,
                    names,
                    suggested,
                    repo_root=repo_root,
                    prepare_data_command=prepare_data_command,
                )
            )
        else:
            warnings.append(
                (
                    f"Dataset {dataset_config.id!r} does not have {_described_each(names, suggested)} "
                    "locally: not prepared yet, or misspelled; they are not solved."
                    + _prepare_advice(dataset_config, dataset, names, repo_root),
                    {"dataset": dataset_config.id, "problems": names},
                )
            )
    if errors:
        raise RuntimeError("\n".join(errors))
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


def _missing_problems_message(
    dataset_config: DatasetConfig,
    names: list[str],
    suggestions: dict[str, set[str]],
    *,
    repo_root: Path,
    prepare_data_command: str | None,
) -> str:
    """The error for requested problems a downloadable dataset does not have locally."""
    lines = [
        f"Dataset {dataset_config.id!r} ({dataset_config.name!r}) does not have the requested "
        f"problem{'s' if len(names) > 1 else ''} {_described_each(names, suggestions)} locally: "
        "not downloaded yet, or misspelled.",
        "Prepare the requested dataset data with:",
        "  " + data_prepare_command(dataset_config, problem_names=names, repo_root=repo_root),
    ]
    if prepare_data_command:
        lines.extend(
            ["Or rerun this benchmark with data preparation enabled:", f"  {prepare_data_command}"]
        )
    else:
        lines.append("Or rerun with --prepare-data (run.auto_prepare_data: true).")
    return "\n".join(lines)


def _requested_names(dataset: Any) -> set[str] | None:
    hook = getattr(dataset, "requested_problem_names", None)
    return hook() if hook is not None else None


def _outside_subset_hint(dataset_config: DatasetConfig, dataset: Any, names: list[str]) -> str:
    """How to select ``names`` that the dataset's name subset leaves out, or nothing."""
    requested = _requested_names(dataset)
    if requested is None:
        return ""
    outside = [name for name in names if name not in requested]
    if not outside:
        return ""
    to_subset = getattr(dataset, "subset_name", None)
    entries = sorted({to_subset(name) if to_subset is not None else name for name in outside})
    them = "it" if len(outside) == 1 else "them"
    if dataset_config.dataset_options.get("subset") is None:
        return (
            f" {_quoted(outside)} {'is' if len(outside) == 1 else 'are'} not in the dataset's "
            f"default subset, which is what it lists without a `subset` option: set "
            f"dataset_options.subset to select {them} (subset: {', '.join(entries)}, with any "
            "other problems you want)."
        )
    return (
        f" {_quoted(outside)} {'is' if len(outside) == 1 else 'are'} outside "
        f"dataset_options.subset: add {', '.join(entries)} to it to select {them}."
    )


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


def _described_each(names: list[str], suggestions: dict[str, set[str]]) -> str:
    return ", ".join(_described([name], suggestions.get(name, set())) for name in names)


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


class _RejectedSolvers:
    """Solvers whose settings an adapter rejected during the run.

    The worker marks a solve that raised SolverConfigError (an unknown setting
    name, a value the adapter or its solver refuses) with
    ``metadata.error_kind = "config"``. The solver id is then stopped: that
    result is not written, its queued solves are not run and the results of
    its solves already running are dropped, while the other solvers' solves
    go on. A setting refused only for some problems stops the solver's other
    solves too, including ones its adapter would have accepted. The run then
    fails (``raise_if_any``) naming every rejected solver and the run
    directory; its solves stay missing, so a re-run with the same settings
    fails again after one solve per rejected solver in a serial run (a few,
    up to about ``parallelism``, in a parallel run, for the solves already
    started) and a corrected config runs them.
    """

    def __init__(
        self, store: ResultStore, stream_output: bool, *, environment_id: str | None = None
    ) -> None:
        self._store = store
        self._stream_output = stream_output
        self._environment = {} if environment_id is None else {"environment_id": environment_id}
        self._messages: dict[str, str] = {}

    def rejects(self, solver_id: str) -> bool:
        return solver_id in self._messages

    def discard(self, result: ProblemResult) -> bool:
        """Whether ``result`` is not to be written (and record a rejection)."""
        if result.solver_id in self._messages:
            return True
        reason = _config_error_reason(result)
        if reason is None:
            return False
        message = (
            f"solver {result.solver_id!r} rejected its settings "
            f"({result.dataset}/{result.problem}): {reason}"
        )
        self._messages[result.solver_id] = message
        self._store.append_event(
            "error",
            f"{message}; its remaining solves are not run",
            solver_id=result.solver_id,
            dataset=result.dataset,
            problem=result.problem,
            **self._environment,
        )
        _emit_progress(self._stream_output, f"error: {message}; its remaining solves are not run")
        return True

    def raise_if_any(self) -> None:
        if not self._messages:
            return
        lines = "\n".join(f"- {message}" for message in self._messages.values())
        run_dir = self._store.run_dir
        raise RuntimeError(
            "Settings rejected, so these solvers' solves were not run:\n"
            f"{lines}\n"
            f"The other solvers' results are in {run_dir}; fix the settings and resume "
            f"with --run-dir {run_dir}"
        )


def _config_error_reason(result: ProblemResult) -> str | None:
    """The first line of the adapter's message for a SolverConfigError
    result (``metadata.error_kind == "config"``), else None."""
    if result.status != status.WORKER_ERROR:
        return None
    if (result.metadata or {}).get("error_kind") != CONFIG_ERROR_KIND:
        return None
    message = str(result.error or "").strip()
    prefix = f"{SolverConfigError.__name__}: "
    if message.startswith(prefix):
        message = message[len(prefix):]
    first_line = message.splitlines()[0].strip() if message else "no message"
    return first_line.rstrip(".") or "no message"


def _retire_replaced_rows(store: ResultStore) -> None:
    """Remove the rows replaced during the run from results.jsonl.

    Superseded rows move to results.superseded.jsonl; replaced retry and
    unavailable-skip rows are dropped. A failure is logged, not raised: the
    rows stay next to their replacements, and the next resume removes them.
    """
    try:
        retired = store.retire_replaced_rows()
    except Exception as exc:  # noqa: BLE001 - logged; the next resume retries.
        logger.warning("Could not remove replaced rows from %s: %s",
                       store.results_jsonl_path, exc)
        store.append_event(
            "error",
            "Could not remove replaced rows from results.jsonl; "
            "the next resume removes them",
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

    Problems already listed locally, and problems the run excludes, are not
    requested. Listed ones are not requested again because a dataset's
    problem names are not always its download names (libsvm_qp lists
    ``svm_dual_<name>``), so asking for a bundled problem could request a
    URL that does not exist. A failure is raised as RuntimeError naming the
    dataset, which ``bench run`` reports without a traceback.
    """
    problem_names, all_problems = data_prepare_selection(config, dataset_config)
    if problem_names and not all_problems:
        listed = {problem.name for problem in dataset.list_problems()}
        # Nor are the problems the run excludes: it would not solve them.
        _, exclude = config.effective_filters(dataset_config)
        problem_names = [name for name in problem_names if name not in listed and name not in exclude]
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
    run_task: Callable[[DatasetConfig, ProblemSpec, SolverConfig], ProblemResult | None],
    store: ResultStore,
    progress: _ProgressReporter,
    parallelism: int,
    *,
    check_result: Callable[[ProblemResult], None] | None = None,
    discard_result: Callable[[ProblemResult], bool] | None = None,
) -> None:
    """Run ``tasks`` on a thread pool and write each result as it lands.

    ``run_task`` may return None for a solve it did not run. A result for
    which ``discard_result`` returns True is not written (also when it
    finishes while stopping). ``check_result`` sees each result once it is
    written and may raise to stop the run. On any exception in this thread, KeyboardInterrupt
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
            if result is None or (discard_result is not None and discard_result(result)):
                continue
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
            finished = future.result()
            if finished is None:
                continue
            if discard_result is not None:
                try:
                    if discard_result(finished):
                        continue
                except Exception:  # noqa: BLE001 - keep the original exception.
                    logger.exception("Could not check a finished result while stopping")
            try:
                store.write_result(finished)
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
    configured_timeout: float | None
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
    # -P: the working directory (the data root) is not put on sys.path; the
    # worker imports this harness through harness_pythonpath().
    cmd = [
        sys.executable, "-P", "-m", "solver_benchmarks.worker", "--payload", str(payload_path)
    ]
    _emit_progress(stream_output, f"starting {problem.name} with {solver_config.id}")
    completed = _run_subprocess(
        cmd,
        cwd=repo_root,
        timeout=subprocess_timeout,
        stdout_path=artifacts_dir / "stdout.log",
        stderr_path=artifacts_dir / "stderr.log",
        stream_output=stream_solver_output,
        env=worker_thread_environ(solver_config.settings),
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
            f"finished {problem.name} with {solver_config.id}: {result.status}"
            + _solver_error_reason(result),
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
    except (OSError, UnicodeDecodeError) as exc:
        error = f"Could not read worker_result.json: {exc}"
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            error = f"Could not parse worker_result.json: {exc}"
        else:
            if isinstance(parsed, dict):
                record = parsed
            else:
                error = (
                    "worker_result.json did not hold a result object "
                    f"(it held {type(parsed).__name__} {text.strip()[:60]})"
                )
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


# Characters of a solver_error's reason the progress line quotes.
_PROGRESS_REASON_CHARS = 200


def _solver_error_reason(result: ProblemResult) -> str:
    """`` (reason)`` for a solver_error row with an error text, else empty.

    Whitespace runs become one space and other characters a terminal would
    act on (escape sequences, bell, backspace) become ``?``."""
    if result.status != status.SOLVER_ERROR or not result.error:
        return ""
    reason = " ".join(str(result.error).split())
    reason = "".join(char if char.isprintable() else "?" for char in reason)
    if not reason:
        return ""
    if len(reason) > _PROGRESS_REASON_CHARS:
        reason = reason[: _PROGRESS_REASON_CHARS - 3] + "..."
    return f" ({reason})"


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


def worker_thread_environ(
    settings: Mapping[str, Any] | None, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """BLAS / OpenMP thread caps for the worker of a solver with ``settings``.

    Every variable of THREAD_LIMIT_ENV_VARS is set to the solver's
    ``threads`` setting, one when it is not set, as adapters size the
    solver's own threads. That bounds the harness's KKT check
    (``numpy.linalg.eigh`` per PSD block) and the BLAS of solvers without a
    thread setting, which would otherwise start one thread per core in each
    of ``run.parallelism`` workers. ``threads: 0`` (the solver's own choice)
    sets none of them. If ``environ`` (default: this process's environment)
    already sets any of them to a value that is not blank, the user is
    managing the thread pools and none is set (an empty or whitespace-only
    value counts as unset, as OpenBLAS and OpenMP read it): a mix of the
    user's values and the runner's would leave one library on a different
    count from another. An invalid ``threads``
    is the adapter's to reject; the default applies meanwhile.
    """
    if environ is None:
        environ = os.environ
    if any((environ.get(name) or "").strip() for name in THREAD_LIMIT_ENV_VARS):
        return {}
    try:
        threads = pop_threads(dict(settings or {}))
    except SolverConfigError:
        threads = None
    count = threads_or_default(threads)
    if count == 0:
        return {}
    return {name: str(count) for name in THREAD_LIMIT_ENV_VARS}


def harness_pythonpath(inherited: str | None = None) -> str:
    """PYTHONPATH for a child process that must run this harness's code.

    The directories this process imports ``solver_benchmarks`` and
    ``problem_classes`` from come first, then ``inherited``. Children are
    started with ``python -P`` from the data root (``--repo-root``), so a
    ``solver_benchmarks`` package or a module named like a standard one
    there is not what they run. A site-packages directory is left out:
    the child's interpreter finds its own installed packages, and an
    environment of ``bench env run`` must use its own solver packages.

    Empty entries of ``inherited`` are dropped: one means the working
    directory, which ``-P`` does not remove and which in a child is the data
    root. Relative entries are made absolute against this process's working
    directory at the time the child is started.
    """
    roots = list(_harness_import_roots())
    for entry in (inherited or "").split(os.pathsep):
        if entry:
            roots.append(os.path.abspath(entry))
    return os.pathsep.join(roots)


def harness_import_roots() -> tuple[str, ...]:
    """The directories harness_pythonpath hands to children (none when this
    process imports the harness from site-packages)."""
    return _harness_import_roots()


@functools.cache
def _harness_import_roots() -> tuple[str, ...]:
    site_dirs = set()
    for directory in [
        *site.getsitepackages(),
        site.getusersitepackages(),
        sysconfig.get_paths()["purelib"],
        sysconfig.get_paths()["platlib"],
    ]:
        site_dirs.add(os.path.realpath(directory))
    roots: list[str] = []
    for package in ("solver_benchmarks", "problem_classes"):
        module = sys.modules.get(package)
        origin = getattr(module, "__file__", None)
        if origin is None:
            spec = importlib.util.find_spec(package)
            origin = spec.origin if spec is not None else None
        if origin is None:
            continue
        root = os.path.dirname(os.path.dirname(os.path.abspath(origin)))
        if os.path.realpath(root) not in site_dirs and root not in roots:
            roots.append(root)
    return tuple(roots)


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: float | None,
    stdout_path: Path,
    stderr_path: Path,
    stream_output: bool,
    env: Mapping[str, str] | None = None,
):
    """Run one worker in its own process group, teeing its output to the logs.

    ``env`` adds variables to the worker's environment (the thread caps).
    The worker also gets the harness's import roots on PYTHONPATH and a
    private temporary directory as TMPDIR / TEMP / TMP, which is removed
    with everything in it once the worker's process group is gone, however
    the worker ended: a worker killed while it holds a decompressed problem
    file or a solver's scratch files leaves nothing behind.
    """
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
    worker_tmp: str | None = None
    try:
        # Under the temporary directory the runner itself uses (an
        # inherited TMPDIR included).
        worker_tmp = tempfile.mkdtemp(prefix="bench-worker-")
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
            "env": {
                **os.environ,
                **(env or {}),
                "PYTHONUNBUFFERED": "1",
                RUNNER_PID_ENV: str(os.getpid()),
                "TMPDIR": worker_tmp,
                "TEMP": worker_tmp,
                "TMP": worker_tmp,
            },
        }
        pythonpath = harness_pythonpath(os.environ.get("PYTHONPATH"))
        if pythonpath:
            popen_kwargs["env"]["PYTHONPATH"] = pythonpath
        else:
            # An inherited value of separators only would put the data root
            # (the working directory) back on the worker's sys.path.
            popen_kwargs["env"].pop("PYTHONPATH", None)
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
        if worker_tmp is not None:
            # The process group was killed on every path above.
            shutil.rmtree(worker_tmp, ignore_errors=True)
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
            # The worker's thread variables, as worker_thread_environ sets them.
            "runtime": runtime_metadata(
                solver_config.solver,
                environ={**os.environ, **worker_thread_environ(solver_config.settings)},
            ),
        }
    )
    if resume_signature is not None:
        metadata["resume_signature"] = resume_signature
    return metadata
