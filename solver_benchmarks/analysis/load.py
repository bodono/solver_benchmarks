"""Load benchmark run results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_results(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    parquet = run_dir / "results.parquet"
    jsonl = run_dir / "results.jsonl"
    if parquet.exists() and _parquet_is_current(parquet, jsonl):
        return pd.read_parquet(parquet)
    if not jsonl.exists():
        return pd.DataFrame()
    return _load_jsonl(jsonl)


def _parquet_is_current(parquet: Path, jsonl: Path) -> bool:
    if not jsonl.exists():
        return True
    return parquet.stat().st_mtime_ns >= jsonl.stat().st_mtime_ns


def _load_jsonl(jsonl: Path) -> pd.DataFrame:
    records = []
    with jsonl.open() as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping unparseable line %d in %s", lineno, jsonl)
    return pd.json_normalize(records)


def solver_summary(run_dir: str | Path) -> pd.DataFrame:
    df = load_results(run_dir)
    if df.empty:
        return df
    grouped = (
        df.groupby(["solver_id", "status"], dropna=False, observed=True)
        .size()
        .reset_index(name="count")
    )
    return grouped.sort_values(["solver_id", "status"])
