"""Run-directory storage for benchmark results."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from . import status
from .atomic import atomic_write_text
from .config import RunConfig, manifest_solve_signatures, slugify, solve_signatures
from .result import ProblemResult, to_jsonable
from .system_info import system_metadata

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
    accepted_signatures: frozenset[str],
    current_signature: str,
    previous_signature: str | None,
) -> bool:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        recorded_signature = metadata.get("resume_signature")
        if recorded_signature is not None:
            return str(recorded_signature) in accepted_signatures
    return previous_signature == current_signature


def _is_unavailable_skip(record: dict[str, Any]) -> bool:
    if record.get("status") != status.SKIPPED_UNSUPPORTED:
        return False
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("skip_reason") == SKIP_SOLVER_UNAVAILABLE:
        return True
    return bool(_LEGACY_UNAVAILABLE_SKIP.match(str(record.get("error") or "")))


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
            run_id = root.name
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
        path = Path(run_dir).resolve() / "manifest.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Could not parse existing manifest %s", path)
            return None

    def write_manifest(self, config: RunConfig) -> None:
        manifest = {
            "run_id": self.run_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": config.to_manifest(),
            "system": system_metadata(),
        }
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text())
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
        current one or the one earlier releases computed for the same
        config (``solve_signature(..., legacy=True)``). Legacy rows without
        an embedded signature are checked against ``previous_manifest``
        when available. An incompatible row of a dataset/solver pair the
        config still plans is superseded: it is replaced by the next result
        for its key, or moved out by ``retire_superseded_rows`` when its key
        already has a compatible result.

        Worker failures are retried up to ``max_worker_error_retries`` times
        (default two). This also bounds repeated deterministic exceptions
        reported as ``worker_error``. Skips written because a solver was not
        installed are not complete for a solver in ``available_solver_ids``.
        Rows remain until atomically replaced; ``problem_solver_dir``
        preserves the previous attempt's artifacts.
        """
        self._retry_rows.clear()
        self._retry_counts.clear()
        self._superseded_rows.clear()
        self._unavailable_rows.clear()
        max_retries = config.max_worker_error_retries if config is not None else 2
        if not self.results_jsonl_path.exists():
            return set()
        current_signatures = solve_signatures(config) if config is not None else None
        legacy_signatures = solve_signatures(config, legacy=True) if config is not None else {}
        previous_signatures = _previous_solve_signatures(previous_manifest)
        keys: set[tuple[str, str, str]] = set()
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
                        accepted_signatures=frozenset(
                            {current_signature, legacy_signatures[signature_key]}
                        ),
                        current_signature=current_signature,
                        previous_signature=previous_signature,
                    ):
                        self._superseded_rows.setdefault(key, set()).add(line.strip())
                        continue
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
        return keys

    def has_pending_retry(self, dataset: str, problem: str, solver_id: str) -> bool:
        """Whether resume selected a failed attempt that must stay retryable."""
        return (dataset, problem, solver_id) in self._retry_rows

    def retire_superseded_rows(self, keys: set[tuple[str, str, str]]) -> int:
        """Move superseded rows of already-complete ``keys`` out of results.jsonl.

        Releases before superseded rows were replaced appended the new
        result and kept the stale one beside it. Returns the number of rows
        moved to results.superseded.jsonl.
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
            superseded = self._superseded_rows.get(key, set())
            previous_rows = (
                self._retry_rows.get(key, set())
                | superseded
                | self._unavailable_rows.get(key, set())
            )
            if previous_rows:
                self._replace_rows(previous_rows, superseded=superseded, line=line)
                for pending in (self._retry_rows, self._superseded_rows, self._unavailable_rows):
                    pending.pop(key, None)
                self._retry_counts.pop(key, None)
            else:
                _append_line(self.results_jsonl_path, line)

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
            records: list[dict] = []
            with self.results_jsonl_path.open() as handle:
                for lineno, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A torn last line shouldn't kill the run.
                        logger.warning(
                            "Skipping unparseable line %d in %s while writing parquet",
                            lineno,
                            self.results_jsonl_path,
                        )
            if not records:
                return
            df = pd.json_normalize(records)
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
    # Adapters disagree on value types for shared keys (Gurobi and CPLEX
    # report an integer info.raw_status, the others a string), and Arrow
    # rejects a column that mixes them. Store such columns as text, with
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
