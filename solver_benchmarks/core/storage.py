"""Run-directory storage for benchmark results."""

from __future__ import annotations

import itertools
import json
import logging
import math
import os
import re
import shutil
import threading
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import status
from .atomic import atomic_write_text
from .config import RunConfig, manifest_solve_signatures, slugify, solve_signatures
from .result import ProblemResult, to_jsonable
from .system_info import system_metadata

try:
    # The dtype inference pandas applies to the columns of a list of
    # records; read_results_jsonl applies it once per column.
    from pandas.core.internals.construction import convert_object_array as _convert_object_array
except ImportError:  # pragma: no cover - other pandas layouts
    _convert_object_array = None

logger = logging.getLogger(__name__)


def make_run_id(config: RunConfig) -> str:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    run_name = config.name or _datasets_slug(config)
    return f"{slugify(run_name)}_{stamp}"


def _datasets_slug(config: RunConfig) -> str:
    # Programmatic callers can omit RunConfig.name. Fall back to entry ids
    # (default: name) so those runs still get a meaningful label.
    ids = [dataset.id for dataset in config.datasets] or ["run"]
    if len(ids) == 1:
        return slugify(ids[0])
    # Multi-dataset fallback: use a deterministic, slug-friendly join. Cap
    # the length so unusual configs do not produce wildly long names.
    joined = "+".join(slugify(entry_id) for entry_id in ids)
    if len(joined) > 64:
        return f"multi-{len(ids)}"
    return joined


def _legacy_slugify(value: str) -> str:
    """``slugify`` before it kept ``+`` and ``=``; older run directories use it."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "run"


# ``metadata.skip_reason`` of the rows written when a solver is not installed.
# Unavailability belongs to the host, not the problem: once the solver is
# available, resume solves those problems instead of keeping the skips.
SKIP_SOLVER_UNAVAILABLE = "solver_unavailable"
# Releases before the tag wrote this message, and nothing else, on such rows.
_LEGACY_UNAVAILABLE_SKIP = re.compile(r"Solver .+ is unavailable; skipping ")


# Manifest key mapping each resume signature in a form earlier releases
# computed to the current signature of the same solve. The first such form
# (``solve_signature(..., legacy=True)``) also hashed data_root and
# max_size_mb; the second (``legacy_selection=True``) still hashed the
# options that only select a dataset's problems, such as ``subset``. Rows
# those releases wrote carry their form, and the only record of the options
# behind it is the manifest they left, which the next run overwrites. So the
# first run of this release in such a directory derives the mapping from that
# manifest's config, and every later manifest carries it forward: the old
# rows keep counting after data_root, max_size_mb or a subset changes.
LEGACY_SIGNATURES_KEY = "legacy_resume_signatures"


def legacy_signature_aliases(manifest: dict[str, Any] | None) -> dict[str, str]:
    """Return ``{legacy signature: current signature}`` for a run's manifest.

    A manifest without the key was written by a release that computed the
    first legacy form: its rows carry that form of its config. A manifest
    with the key may have been written by a release that computed the
    second form: its rows carry that form of its config, and the values of
    its mapping are in that form too. Both are mapped to the current form of
    the manifest's config. For a manifest this release wrote, the mapping
    gains an entry from the earlier form of its config, which no row carries:
    harmless, and simpler than telling the manifests apart.
    """
    if not manifest:
        return {}
    recorded = manifest.get(LEGACY_SIGNATURES_KEY)
    config = manifest.get("config")
    if not isinstance(config, dict):
        if isinstance(recorded, dict):
            return {str(legacy): str(current) for legacy, current in recorded.items()}
        return {}
    try:
        current = manifest_solve_signatures(config)
        if isinstance(recorded, dict):
            selection = manifest_solve_signatures(config, legacy_selection=True)
        else:
            legacy = manifest_solve_signatures(config, legacy=True)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        logger.warning("Could not derive resume signatures from the manifest config: %s", exc)
        if isinstance(recorded, dict):
            return {str(legacy): str(current) for legacy, current in recorded.items()}
        return {}
    if not isinstance(recorded, dict):
        return {legacy[key]: current[key] for key in current}
    renamed = {
        selection[key]: current[key] for key in current if selection[key] != current[key]
    }
    aliases = {
        str(legacy): renamed.get(str(value), str(value)) for legacy, value in recorded.items()
    }
    aliases.update(renamed)
    return aliases


def _previous_solve_signatures(
    previous_manifest: dict[str, Any] | None,
) -> dict[tuple[str, str], str]:
    if not previous_manifest:
        return {}
    config = previous_manifest.get("config")
    if not isinstance(config, dict):
        return {}
    return manifest_solve_signatures(config)


def _is_resume_compatible(
    record: dict[str, Any],
    *,
    current_signature: str,
    legacy_signature: str,
    legacy_aliases: dict[str, str],
    previous_signature: str | None,
    selection_signature: str | None = None,
) -> bool:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        recorded_signature = metadata.get("resume_signature")
        if recorded_signature is not None:
            recorded_signature = str(recorded_signature)
            return (
                recorded_signature in (current_signature, legacy_signature, selection_signature)
                or legacy_aliases.get(recorded_signature) == current_signature
            )
    return previous_signature == current_signature


def _is_unavailable_skip(record: dict[str, Any]) -> bool:
    if record.get("status") != status.SKIPPED_UNSUPPORTED:
        return False
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("skip_reason") == SKIP_SOLVER_UNAVAILABLE:
        return True
    return bool(_LEGACY_UNAVAILABLE_SKIP.match(str(record.get("error") or "")))


def _is_replaceable(record: dict[str, Any]) -> bool:
    """A worker-error row or a skip written while the solver was not installed.

    Resume solves such a key again, and the row the new attempt writes
    replaces it (see ``ReplacedRows``).
    """
    return record.get("status") == status.WORKER_ERROR or _is_unavailable_skip(record)


class ReplacedRows:
    """Finds the rows of results.jsonl that a later row has replaced.

    A run appends the result of a retried worker error or of a recovered
    unavailable skip next to the row it replaces and drops that row in its
    end-of-run rewrite (``ResultStore.retire_replaced_rows``). Until then (and
    for good if the run is killed first) both rows are in the file: a
    replaceable row followed by a later row of its key is not a result.
    ``ResultStore.completed_keys`` uses this with ``(dataset, problem,
    solver_id)`` over the rows compatible with the current config; readers
    (``read_results_jsonl``, ``replaced_positions``) key rows by
    ``reader_key``, which adds the row's resume signature and run id, so
    rows solved under other settings are left alone, and so are rows of
    different runs in a ``bench merge`` output (a resume keeps its run
    directory's run id).
    """

    def __init__(self) -> None:
        self._last: dict[Any, Any] = {}

    def observe(self, key: Any, record: dict[str, Any], ident: Any) -> Any | None:
        """Note the row ``ident`` of ``key`` (in file order); return the
        identifier of the earlier row it replaces, if any."""
        earlier = self._last.pop(key, None)
        if _is_replaceable(record):
            self._last[key] = ident
        return earlier

    @staticmethod
    def reader_key(record: dict[str, Any]) -> tuple[str, str, str, str, str | None]:
        metadata = record.get("metadata")
        signature = metadata.get("resume_signature") if isinstance(metadata, dict) else None
        return (
            str(record.get("run_id")),
            str(record.get("dataset", "")),
            str(record.get("problem")),
            str(record.get("solver_id")),
            None if signature is None else str(signature),
        )


def replaced_positions(records: Iterable[dict[str, Any]]) -> set[int]:
    """Positions of the rows in ``records`` (results.jsonl order) that a later
    row of the same solve and resume signature replaced (see ``ReplacedRows``)."""
    replaced = ReplacedRows()
    positions: set[int] = set()
    for position, record in enumerate(records):
        earlier = replaced.observe(ReplacedRows.reader_key(record), record, position)
        if earlier is not None:
            positions.add(earlier)
    return positions


def _append_line(path: Path, line: str) -> None:
    """Append ``line`` to ``path``, starting a new line after a torn one.

    A process killed mid-write leaves a partial last line with no newline.
    Appending straight after it would merge the next record into the
    fragment, and readers would skip both as one unparseable line.
    """
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) > 0:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                line = "\n" + line
        handle.write(line.encode("utf-8"))


@dataclass
class ResultStore:
    run_dir: Path
    run_id: str
    # Per-store lock for the write paths. Using `field` with a default
    # factory keeps `cls(root, run_id)` calls compatible.
    _write_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    # Exact rows selected by resume planning for replacement, by key. Each is
    # retained until its replacement result is durable. Retry rows are
    # compatible worker errors; superseded rows were solved under another
    # resume signature and move to results.superseded.jsonl; unavailable
    # rows are skips written while the solver, now available, was missing.
    _retry_rows: dict[tuple[str, str, str], set[str]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _retry_counts: dict[tuple[str, str, str], int] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _superseded_rows: dict[tuple[str, str, str], set[str]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _unavailable_rows: dict[tuple[str, str, str], set[str]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    # Superseded rows whose replacement is already in results.jsonl. Moving
    # them out rewrites the file, so it happens once, in
    # ``retire_replaced_rows`` at the end of the run, not once per result.
    _replaced_rows: set[str] = field(
        default_factory=set, init=False, repr=False, compare=False
    )
    # Retry and unavailable rows whose replacement is already in
    # results.jsonl, and those a run stopped before that rewrite left next to
    # a later row of their key. They are dropped in the same rewrite (a retry
    # row's copy is in its retry directory's previous_results.jsonl).
    _stale_rows: set[str] = field(
        default_factory=set, init=False, repr=False, compare=False
    )

    @classmethod
    def create(cls, config: RunConfig, run_dir: str | Path | None = None) -> ResultStore:
        if run_dir is None:
            run_id = make_run_id(config)
            base_run_id = run_id
            root = config.output_dir / run_id
            suffix = 2
            # Use mkdir(exist_ok=False) inside a retry loop so two
            # concurrent `bench run` invocations cannot both pick the
            # same suffix and stomp on each other.
            while True:
                try:
                    root.parent.mkdir(parents=True, exist_ok=True)
                    root.mkdir(exist_ok=False)
                    break
                except FileExistsError:
                    run_id = f"{base_run_id}_{suffix}"
                    root = config.output_dir / run_id
                    suffix += 1
        else:
            root = Path(run_dir).resolve()
            # A run directory keeps the run id it was created with, also after
            # it is renamed or copied: the rows a resume writes must carry the
            # run id of the rows they replace (see ReplacedRows.reader_key).
            # Without a readable manifest the directory's name is the run id.
            existing = cls.read_manifest(root)
            recorded = existing.get("run_id") if existing is not None else None
            run_id = recorded if isinstance(recorded, str) and recorded else root.name
            root.mkdir(parents=True, exist_ok=True)
        store = cls(root, run_id)
        store.write_manifest(config)
        return store

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def results_jsonl_path(self) -> Path:
        return self.run_dir / "results.jsonl"

    @property
    def results_parquet_path(self) -> Path:
        return self.run_dir / "results.parquet"

    @property
    def superseded_jsonl_path(self) -> Path:
        return self.run_dir / "results.superseded.jsonl"

    @staticmethod
    def read_manifest(run_dir: str | Path) -> dict[str, Any] | None:
        """The run's manifest, or None when it is missing, torn or not an object."""
        path = Path(run_dir).resolve() / "manifest.json"
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            manifest = None
        if not isinstance(manifest, dict):
            logger.warning("Could not parse existing manifest %s", path)
            return None
        return manifest

    def write_manifest(self, config: RunConfig) -> None:
        manifest = {
            "run_id": self.run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": config.to_manifest(),
            "system": system_metadata(),
        }
        # A torn manifest (a run killed while writing it) is written anew.
        existing = self.read_manifest(self.run_dir)
        if existing is not None:
            manifest["created_at_utc"] = existing.get(
                "created_at_utc", manifest["created_at_utc"]
            )
            # Preserve the system snapshot from the original write so
            # rewrites (e.g. when the runner re-saves the manifest on
            # resume) don't capture a different system blob if the
            # second invocation runs on a different host. Re-running
            # on a different machine should be visible as a separate
            # run, not as a silent overwrite of provenance.
            existing_system = existing.get("system")
            if existing_system:
                manifest["system"] = existing_system
            manifest[LEGACY_SIGNATURES_KEY] = legacy_signature_aliases(existing)
        else:
            manifest[LEGACY_SIGNATURES_KEY] = {}
        atomic_write_text(self.manifest_path, json.dumps(to_jsonable(manifest), indent=2))

    def copy_source_config(self, config_path: str | Path, *, name: str = "run_config") -> Path:
        source = Path(config_path)
        suffix = source.suffix or ".txt"
        target = self.run_dir / f"{slugify(name)}{suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.tmp")
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
        return target

    def problem_solver_dir(
        self, dataset: str, problem: str, solver_id: str
    ) -> Path:
        path = (
            self.run_dir
            / "problems"
            / slugify(dataset)
            / slugify(problem)
            / slugify(solver_id)
        )
        path.mkdir(parents=True, exist_ok=True)
        key = (dataset, problem, solver_id)
        # Keep prior stdout/stderr and raw solver output intact. A failed or
        # interrupted retry must not overwrite the previous attempt, and a
        # superseded result keeps the artifacts its row points to.
        for prefix, pending in (("retry", self._retry_rows), ("rerun", self._superseded_rows)):
            previous_rows = pending.get(key)
            if not previous_rows:
                continue
            attempt = 1
            while True:
                attempt_dir = path / f"{prefix}-{attempt}"
                try:
                    attempt_dir.mkdir()
                    break
                except FileExistsError:
                    attempt += 1
            atomic_write_text(
                attempt_dir / "previous_results.jsonl",
                "\n".join(sorted(previous_rows)) + "\n",
            )
            return attempt_dir
        return path

    def completed_keys(
        self,
        *,
        config: RunConfig | None = None,
        previous_manifest: dict[str, Any] | None = None,
        available_solver_ids: set[str] | None = None,
    ) -> set[tuple[str, str, str]]:
        """Resume keys are ``(dataset, problem, solver_id)`` tuples.

        Including the dataset name avoids collisions when two datasets
        share a problem name (e.g. ``afiro`` in NETLIB vs another LP
        bundle), so resume cannot conflate them. Tolerant of partially
        written / unparseable JSONL lines (skips with a warning).

        When ``config`` is supplied, only rows whose solve signature is
        compatible with the current dataset/solver definition are treated
        as complete. A recorded signature is compatible when it equals the
        current one, one of those earlier releases computed for the same
        config (``solve_signature(..., legacy=True)`` or
        ``legacy_selection=True``), or a legacy signature that
        ``previous_manifest`` maps to the current one (see
        ``legacy_signature_aliases``). Legacy rows without an embedded
        signature are checked against ``previous_manifest`` when available.
        An incompatible row of a dataset/solver pair the config still plans
        is superseded: it moves to results.superseded.jsonl once the next
        result for its key is written (``retire_replaced_rows``), or through
        ``retire_superseded_rows`` when its key already has a compatible
        result.

        Worker failures are retried up to ``max_worker_error_retries`` times
        (default two). This also bounds repeated deterministic exceptions
        reported as ``worker_error``. Skips written because a solver was not
        installed are not complete for a solver in ``available_solver_ids``.
        Rows remain until their replacement is written; ``problem_solver_dir``
        preserves the previous attempt's artifacts. A worker-error or
        unavailable-skip row followed by a later compatible row of its key
        was replaced by a run that stopped before ``retire_replaced_rows``;
        it is dropped in this run's ``retire_replaced_rows`` (it does not
        change which keys are complete).
        """
        self._retry_rows.clear()
        self._retry_counts.clear()
        self._superseded_rows.clear()
        self._unavailable_rows.clear()
        self._replaced_rows.clear()
        self._stale_rows.clear()
        max_retries = config.max_worker_error_retries if config is not None else 2
        if not self.results_jsonl_path.exists():
            return set()
        current_signatures = solve_signatures(config) if config is not None else None
        legacy_signatures = solve_signatures(config, legacy=True) if config is not None else {}
        selection_signatures = (
            solve_signatures(config, legacy_selection=True) if config is not None else {}
        )
        previous_signatures = _previous_solve_signatures(previous_manifest)
        legacy_aliases = legacy_signature_aliases(previous_manifest)
        keys: set[tuple[str, str, str]] = set()
        # Replaced worker-error / unavailable-skip rows, and how often each
        # such row occurs and is followed by a later row.
        replaced = ReplacedRows()
        occurrences: Counter[str] = Counter()
        followed: Counter[str] = Counter()
        with self.results_jsonl_path.open() as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping unparseable line %d in %s",
                        lineno,
                        self.results_jsonl_path,
                    )
                    continue
                problem = record.get("problem")
                solver_id = record.get("solver_id")
                if problem is None or solver_id is None:
                    logger.warning(
                        "Skipping line %d in %s: missing required keys",
                        lineno,
                        self.results_jsonl_path,
                    )
                    continue
                key = (
                    str(record.get("dataset", "")),
                    str(problem),
                    str(solver_id),
                )
                if current_signatures is not None:
                    signature_key = (key[0], key[2])
                    current_signature = current_signatures.get(signature_key)
                    if current_signature is None:
                        continue
                    previous_signature = previous_signatures.get(signature_key)
                    if not _is_resume_compatible(
                        record,
                        current_signature=current_signature,
                        legacy_signature=legacy_signatures[signature_key],
                        legacy_aliases=legacy_aliases,
                        previous_signature=previous_signature,
                        selection_signature=selection_signatures[signature_key],
                    ):
                        self._superseded_rows.setdefault(key, set()).add(line.strip())
                        continue
                earlier = replaced.observe(key, record, line.strip())
                if earlier is not None:
                    followed[earlier] += 1
                if _is_replaceable(record):
                    occurrences[line.strip()] += 1
                if (
                    available_solver_ids is not None
                    and key[2] in available_solver_ids
                    and _is_unavailable_skip(record)
                ):
                    self._unavailable_rows.setdefault(key, set()).add(line.strip())
                    continue
                if record.get("status") == status.WORKER_ERROR:
                    metadata = record.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    retries = metadata.get("worker_error_retries")
                    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
                        # Rows from the initial retry implementation have no
                        # count, but their artifact path identifies the attempt.
                        artifacts = Path(record.get("artifact_dir") or "")
                        match = re.fullmatch(r"retry-(\d+)", artifacts.name)
                        is_retry_dir = artifacts.parent.parts[-4:] in {
                            ("problems", *(slug(part) for part in key))
                            for slug in (slugify, _legacy_slugify)
                        }
                        retries = int(match.group(1)) if match and is_retry_dir else 0
                    if retries >= max_retries:
                        keys.add(key)
                        logger.warning(
                            "Worker-error retry limit reached for %s/%s/%s; "
                            "increase run.max_worker_error_retries to retry again", *key
                        )
                        self.append_event(
                            "warning",
                            "Worker-error retry limit reached; increase "
                            "run.max_worker_error_retries to retry again",
                            dataset=key[0],
                            problem=key[1],
                            solver_id=key[2],
                            worker_error_retries=retries,
                            max_worker_error_retries=max_retries,
                        )
                        continue
                    self._retry_rows.setdefault(key, set()).add(line.strip())
                    self._retry_counts[key] = max(retries, self._retry_counts.get(key, 0))
                else:
                    keys.add(key)
        # If an older file already contains a successful compatible retry,
        # it is complete even when the superseded error is still present.
        for key in keys:
            self._retry_rows.pop(key, None)
            self._retry_counts.pop(key, None)
            self._unavailable_rows.pop(key, None)
        # A row whose every copy has a later row of its key is stale; an
        # identical copy that is still the latest row of its key keeps it.
        self._stale_rows = {
            row for row, count in followed.items() if count == occurrences[row]
        }
        return keys

    def has_pending_retry(self, dataset: str, problem: str, solver_id: str) -> bool:
        """Whether resume selected a failed attempt that must stay retryable."""
        return (dataset, problem, solver_id) in self._retry_rows

    def retire_superseded_rows(self, keys: set[tuple[str, str, str]]) -> int:
        """Move superseded rows of already-complete ``keys`` out of results.jsonl.

        Releases before superseded rows were replaced appended the new
        result and kept the stale one beside it, as does a run that stops
        before ``retire_replaced_rows``. Returns the number of rows moved to
        results.superseded.jsonl.
        """
        with self._write_lock:
            rows: set[str] = set()
            for key in keys:
                rows |= self._superseded_rows.pop(key, set())
            if rows:
                self._replace_rows(rows, superseded=rows, line="")
        return len(rows)

    def append_event(self, level: str, message: str, **fields: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "level": level,
            "message": message,
            **fields,
        }
        line = json.dumps(to_jsonable(record), sort_keys=True) + "\n"
        # Lock the append so concurrent emitters can't interleave bytes
        # mid-line beyond the OS PIPE_BUF guarantee.
        with self._write_lock:
            _append_line(self.events_path, line)

    def write_result(self, result: ProblemResult) -> None:
        key = (result.dataset, result.problem, result.solver_id)
        if key in self._retry_counts:
            result.metadata = {
                **result.metadata,
                "worker_error_retries": self._retry_counts[key] + 1,
            }
        record = result.to_record()
        artifact_dir = Path(result.artifact_dir) if result.artifact_dir else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(artifact_dir / "result.json", json.dumps(record, indent=2))
        line = json.dumps(record, sort_keys=True) + "\n"
        with self._write_lock:
            # The rows this result replaces stay until retire_replaced_rows
            # rewrites the file once for the whole run. If the run stops
            # before that, the next resume recognises them again: a
            # superseded row by its signature, a retry or unavailable row
            # by the later row of its key (see completed_keys).
            self._replaced_rows |= self._superseded_rows.get(key, set())
            replaced = self._retry_rows.get(key, set()) | self._unavailable_rows.get(key, set())
            if line.strip() in replaced:
                # The new row is the replaced row, text for text: it is
                # already in the file, and dropping the old row by its text
                # would drop the new one too.
                replaced = replaced - {line.strip()}
            else:
                _append_line(self.results_jsonl_path, line)
            self._stale_rows |= replaced
            for pending in (self._retry_rows, self._superseded_rows, self._unavailable_rows):
                pending.pop(key, None)
            self._retry_counts.pop(key, None)

    def retire_replaced_rows(self) -> int:
        """Remove the rows replaced during this run from results.jsonl.

        One rewrite of results.jsonl for the whole run: superseded rows move
        to results.superseded.jsonl, stale retry and unavailable rows are
        dropped. A run that stops before calling this leaves both rows of
        such a key in results.jsonl; the next resume removes the replaced
        one. Returns the number of rows moved to results.superseded.jsonl.
        """
        with self._write_lock:
            superseded, self._replaced_rows = self._replaced_rows, set()
            stale, self._stale_rows = self._stale_rows, set()
            if superseded or stale:
                self._replace_rows(superseded | stale, superseded=superseded, line="")
        return len(superseded)

    def _replace_rows(self, rows: set[str], *, superseded: set[str], line: str) -> None:
        """Atomically drop ``rows`` from results.jsonl and append ``line``.

        ``superseded`` rows are first appended to results.superseded.jsonl,
        so an interruption can at worst leave a row in both files, never in
        neither. Callers hold the write lock.
        """
        if superseded:
            _append_line(
                self.superseded_jsonl_path,
                "".join(row + "\n" for row in sorted(superseded)),
            )
        with self.results_jsonl_path.open() as handle:
            retained = "".join(existing for existing in handle if existing.strip() not in rows)
        if retained and not retained.endswith("\n"):
            retained += "\n"
        atomic_write_text(self.results_jsonl_path, retained + line)

    def write_parquet(self) -> None:
        """Materialize results.parquet from results.jsonl.

        Called once at end of a successful run by run_benchmark; the
        parquet file's presence is the success sentinel. Safe to call
        manually to rebuild a stale or missing parquet from jsonl.

        If the jsonl is missing or contains no parseable records the
        method returns without creating a parquet — consumers that need
        a result frame should use ``analysis.load.load_results``, which
        falls back to the jsonl when the parquet is absent.
        """
        with self._write_lock:
            if not self.results_jsonl_path.exists():
                return
            # A torn last line shouldn't kill the run: it is skipped.
            df = read_results_jsonl(self.results_jsonl_path)
            if not len(df.index):
                return
            df = normalize_table_for_parquet(df)
            tmp = self.results_parquet_path.with_suffix(".parquet.tmp")
            try:
                df.to_parquet(tmp, index=False)
                os.replace(tmp, self.results_parquet_path)
            except Exception:
                # Don't leave a half-written .parquet.tmp behind.
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise


# Lines of results.jsonl parsed and flattened at a time by read_results_jsonl.
_JSONL_CHUNK_LINES = 2000


def read_results_jsonl(path: str | Path, *, chunk_lines: int = _JSONL_CHUNK_LINES) -> pd.DataFrame:
    """The records of a results.jsonl file as one flat frame.

    The frame equals ``pd.json_normalize`` of every parseable record
    (columns, their order, dtypes and values), without holding every parsed
    record at once: lines are parsed and flattened ``chunk_lines`` at a time
    into untyped (object) columns, and each column's dtype is then inferred
    once over the whole column, with the conversion pandas applies to a list
    of records. Blank lines are ignored; unparseable lines (a torn last
    line) and lines that are not a JSON object (``null``, a number, a list)
    are skipped with a warning, whatever the pandas version. No records give
    an empty frame. Where that pandas conversion is missing or has another
    signature, the records are normalized in one go instead (the same frame,
    with the memory of all records at once).

    Rows a later row replaced are left out (see ``ReplacedRows``): a
    worker-error or unavailable-skip row followed by a row of the same solve
    and resume signature, which a resume appends next to it and drops only at
    the end of the run. The frame is the one the file gives once they are
    dropped; when there are any, the file is read a second time without them.
    """
    path = Path(path)
    # One open file for both reads: a resume appending rows or replacing the
    # file meanwhile changes neither what the first read saw nor the second.
    with path.open("rb") as handle:
        if _convert_object_array is not None:
            try:
                replaced: set[int] = set()
                seen = _ReadExtent()
                frame = _read_results_jsonl_chunked(
                    path, handle, chunk_lines, replaced=replaced, extent=seen
                )
                if replaced:
                    del frame  # not held while the file is read again
                    return _read_results_jsonl_chunked(
                        path, handle, chunk_lines,
                        skip=frozenset(replaced), extent=_ReadExtent(limit=seen.end),
                        warn=False,  # the first read warned about bad lines
                    )
                return frame
            except TypeError:
                # The internal conversion's signature changed in this pandas.
                pass
        records = list(_parsed_records(path, handle))
    stale = replaced_positions(records)
    return pd.json_normalize([record for i, record in enumerate(records) if i not in stale])


@dataclass
class _ReadExtent:
    """How far a read of results.jsonl goes: up to ``limit`` bytes (all when
    None); ``end`` is where it stopped."""

    limit: int | None = None
    end: int = 0


def _read_results_jsonl_chunked(
    path: Path,
    handle: Any,
    chunk_lines: int,
    *,
    replaced: set[int] | None = None,
    skip: frozenset[int] = frozenset(),
    extent: _ReadExtent | None = None,
    warn: bool = True,
) -> pd.DataFrame:
    assert _convert_object_array is not None
    columns: dict[str, list[np.ndarray]] = {}
    rows = 0
    records = _parsed_records(
        path, handle, replaced=replaced, skip=skip, extent=extent, warn=warn
    )
    for chunk in _batched(records, chunk_lines):
        # Missing keys are NaN, as in a frame built from all records at once.
        frame = pd.DataFrame([_flatten_record(record) for record in chunk], dtype=object)
        size = len(chunk)
        for name in frame.columns:
            if name not in columns:
                columns[name] = [np.full(rows, np.nan, dtype=object)]
            columns[name].append(frame[name].to_numpy(dtype=object, copy=True))
        for name, parts in columns.items():
            if name not in frame.columns:
                parts.append(np.full(size, np.nan, dtype=object))
        rows += size
        del frame
    if not rows:
        return pd.DataFrame()
    if not columns:
        # Only empty records: rows without columns, as json_normalize gives.
        return pd.DataFrame([{} for _ in range(rows)])
    data = {}
    for name in list(columns):
        values = np.concatenate(columns.pop(name))
        data[name] = _convert_object_array([values], dtype=None)[0]
    return pd.DataFrame(data, index=pd.RangeIndex(rows))


def _parsed_records(
    path: Path,
    handle: Any,
    *,
    replaced: set[int] | None = None,
    skip: frozenset[int] = frozenset(),
    extent: _ReadExtent | None = None,
    warn: bool = True,
) -> Iterator[dict[str, Any]]:
    """The records of ``handle`` (``path`` opened in binary mode), from its
    start and within ``extent``, without the lines in ``skip``; the line
    numbers of rows a later row replaced are added to ``replaced``.

    Nothing past a line without a newline is read: it is the last line, or
    one a writer is still appending, and the rest of it would be a second
    fragment, or a whole row a later read sees and this one did not."""
    tracker = ReplacedRows()
    limit = extent.limit if extent is not None else None
    offset = 0
    torn = False
    handle.seek(0)
    for lineno, line in enumerate(handle, start=1):
        if torn:
            break
        if limit is not None:
            if offset >= limit:
                break
            line = line[: limit - offset]
        offset += len(line)
        torn = not line.endswith(b"\n")
        if lineno in skip or not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            record = None
        if not isinstance(record, dict):
            if warn:
                logger.warning("Skipping unparseable line %d in %s", lineno, path)
            continue
        if replaced is not None:
            earlier = tracker.observe(ReplacedRows.reader_key(record), record, lineno)
            if earlier is not None:
                replaced.add(earlier)
        yield record
    if extent is not None:
        extent.end = offset


def _batched(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    iterator = iter(items)
    while batch := list(itertools.islice(iterator, size)):
        yield batch


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    """One record flattened as ``pd.json_normalize`` does: top-level
    non-dict values first, then nested dicts as ``outer.inner`` keys."""
    flat = {key: value for key, value in record.items() if not isinstance(value, dict)}
    nested: dict[str, Any] = {}
    _flatten_into(nested, "", {key: value for key, value in record.items() if isinstance(value, dict)})
    return {**flat, **nested}


def _flatten_into(out: dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten_into(out, f"{prefix}.{key}" if prefix else str(key), item)
    else:
        out[prefix] = value


_NUMERIC_COLUMNS: tuple[str, ...] = (
    "objective_value",
    "iterations",
    "run_time_seconds",
    "setup_time_seconds",
    "solve_time_seconds",
)

_NONFINITE_STRINGS = {
    "nan",
    "NaN",
    "NAN",
    "inf",
    "Inf",
    "INF",
    "+inf",
    "+Inf",
    "+INF",
    "-inf",
    "-Inf",
    "-INF",
    "Infinity",
    "+Infinity",
    "-Infinity",
}


def normalize_table_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    # Rows can disagree on value types for shared keys (Gurobi and CPLEX
    # rows recorded by older versions have an integer info.raw_status, the
    # others a string), and Arrow rejects a column that mixes them. Store such columns as text, with
    # lists and dicts as JSON.
    for column in normalized.columns:
        if column in _NUMERIC_COLUMNS or normalized[column].dtype != object:
            continue
        values = normalized[column]
        if len({_type_family(value) for value in values if not _is_missing(value)}) > 1:
            normalized[column] = values.map(_as_text)
    # Restrict the string-NaN scrub to numeric columns so a problem named
    # "nan" or a free-text error message containing "inf" is not silently
    # nulled out.
    for column in _NUMERIC_COLUMNS:
        if column not in normalized:
            continue
        col = normalized[column]
        if col.dtype == object:
            col = col.map(lambda v: None if isinstance(v, str) and v in _NONFINITE_STRINGS else v)
        col = pd.to_numeric(col, errors="coerce")
        normalized[column] = col.map(_finite_or_none)
    return normalized


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _type_family(value: Any) -> type:
    # Arrow stores ints and floats together as floats; bool stays apart.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float
    return type(value)


def _as_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _finite_or_none(value):
    if pd.isna(value):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value
