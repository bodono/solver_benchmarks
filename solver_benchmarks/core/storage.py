"""Run-directory storage for benchmark results."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RunConfig
from .result import ProblemResult, to_jsonable


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


@dataclass
class ResultStore:
    run_dir: Path
    run_id: str

    @classmethod
    def create(cls, config: RunConfig, run_dir: str | Path | None = None) -> ResultStore:
        if run_dir is None:
            run_id = make_run_id(config)
            root = config.output_dir / run_id
            base_run_id = run_id
            suffix = 2
            while root.exists():
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

    def completed_keys(self) -> set[tuple[str, str, str]]:
        """Resume keys are ``(dataset, problem, solver_id)`` tuples.

        Including the dataset name avoids collisions when two datasets
        share a problem name (e.g. ``afiro`` in NETLIB vs another LP
        bundle), so resume cannot conflate them.
        """
        if not self.results_jsonl_path.exists():
            return set()
        keys: set[tuple[str, str, str]] = set()
        with self.results_jsonl_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                keys.add(
                    (
                        str(record.get("dataset", "")),
                        str(record["problem"]),
                        str(record["solver_id"]),
                    )
                )
        return keys

    def append_event(self, level: str, message: str, **fields: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            **fields,
        }
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")

    def write_result(self, result: ProblemResult) -> None:
        record = result.to_record()
        artifact_dir = Path(result.artifact_dir) if result.artifact_dir else None
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(artifact_dir / "result.json", json.dumps(record, indent=2))
        with self.results_jsonl_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.rewrite_parquet()

    def rewrite_parquet(self) -> None:
        if not self.results_jsonl_path.exists():
            return
        records = []
        with self.results_jsonl_path.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not records:
            return
        df = pd.json_normalize(records)
        df = normalize_table_for_parquet(df)
        tmp = self.results_parquet_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, self.results_parquet_path)


def normalize_table_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized = normalized.replace(
        {
            "nan": None,
            "NaN": None,
            "NAN": None,
            "inf": None,
            "Inf": None,
            "INF": None,
            "+inf": None,
            "+Inf": None,
            "+INF": None,
            "-inf": None,
            "-Inf": None,
            "-INF": None,
            "Infinity": None,
            "+Infinity": None,
            "-Infinity": None,
        }
    )
    for column in [
        "objective_value",
        "iterations",
        "run_time_seconds",
        "setup_time_seconds",
        "solve_time_seconds",
    ]:
        if column not in normalized:
            continue
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized[column] = normalized[column].map(_finite_or_none)
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
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(text)
        tmp_name = handle.name
    os.replace(tmp_name, path)
