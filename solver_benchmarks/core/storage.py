"""Run-directory storage for benchmark results."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig, manifest_solve_signatures, solve_signatures
from .result import ProblemResult, to_jsonable

logger = logging.getLogger(__name__)

# Minimum interval between parquet rewrites in `write_result`. Tight loops
# of fast solves now do at most one rewrite per second, instead of one per
# solve. The final rewrite is still forced via `flush_parquet`.
_PARQUET_REWRITE_INTERVAL_SECONDS = 1.0


def make_run_id(config: RunConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
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


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-") or "run"


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
    previous_signature: str | None,
) -> bool:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        recorded_signature = metadata.get("resume_signature")
        if recorded_signature is not None:
            return str(recorded_signature) == current_signature
    return previous_signature == current_signature


@dataclass
class ResultStore:
    run_dir: Path
    run_id: str
    # Per-store lock and bookkeeping for the write paths. Using `field`
    # with a default factory keeps `cls(root, run_id)` calls compatible.
    _write_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _last_parquet_rewrite: float = field(default=0.0, repr=False, compare=False)

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
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config.to_manifest(),
        }
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text())
            manifest["created_at_utc"] = existing.get(
                "created_at_utc", manifest["created_at_utc"]
            )
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
        return path

    def completed_keys(
        self,
        *,
        config: RunConfig | None = None,
        previous_manifest: dict[str, Any] | None = None,
    ) -> set[tuple[str, str, str]]:
        """Resume keys are ``(dataset, problem, solver_id)`` tuples.

        Including the dataset name avoids collisions when two datasets
        share a problem name (e.g. ``afiro`` in NETLIB vs another LP
        bundle), so resume cannot conflate them. Tolerant of partially
        written / unparseable JSONL lines (skips with a warning).

        When ``config`` is supplied, only rows whose solve signature is
        compatible with the current dataset/solver definition are treated
        as complete. Legacy rows without an embedded signature are checked
        against ``previous_manifest`` when available.
        """
        if not self.results_jsonl_path.exists():
            return set()
        current_signatures = solve_signatures(config) if config is not None else None
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
                        current_signature=current_signature,
                        previous_signature=previous_signature,
                    ):
                        continue
                keys.add(key)
        return keys

    def append_event(self, level: str, message: str, **fields: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            **fields,
        }
        line = json.dumps(to_jsonable(record), sort_keys=True) + "\n"
        # Lock the append so concurrent emitters can't interleave bytes
        # mid-line beyond the OS PIPE_BUF guarantee.
        with self._write_lock, self.events_path.open("a") as handle:
            handle.write(line)

    def write_result(self, result: ProblemResult) -> None:
        record = result.to_record()
        artifact_dir = Path(result.artifact_dir) if result.artifact_dir else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(artifact_dir / "result.json", json.dumps(record, indent=2))
        line = json.dumps(record, sort_keys=True) + "\n"
        with self._write_lock:
            with self.results_jsonl_path.open("a") as handle:
                handle.write(line)
            now = time.monotonic()
            # Amortize parquet rewrites: avoid the O(N^2) cost of
            # re-serializing all completed records on every solve.
            if now - self._last_parquet_rewrite >= _PARQUET_REWRITE_INTERVAL_SECONDS:
                self._rewrite_parquet_locked()
                self._last_parquet_rewrite = now

    def flush_parquet(self) -> None:
        """Force a parquet rewrite. Call at end of run."""
        with self._write_lock:
            self._rewrite_parquet_locked()
            self._last_parquet_rewrite = time.monotonic()

    def rewrite_parquet(self) -> None:
        with self._write_lock:
            self._rewrite_parquet_locked()
            self._last_parquet_rewrite = time.monotonic()

    def _rewrite_parquet_locked(self) -> None:
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
                    # A torn last line shouldn't kill the run; the next
                    # rewrite (or a manual `bench` step) can recover once
                    # the line is whole.
                    logger.warning(
                        "Skipping unparseable line %d in %s while rewriting parquet",
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


def _finite_or_none(value):
    if pd.isna(value):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, encoding="utf-8"
        ) as handle:
            handle.write(text)
            handle.flush()
            # Force the data to disk before the rename so a power loss
            # cannot leave the rename visible with a zero-length payload.
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
            tmp_name = handle.name
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
