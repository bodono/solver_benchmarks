"""Load benchmark run results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from solver_benchmarks.core.storage import read_results_jsonl


def load_results(run_dir: str | Path) -> pd.DataFrame:
    """The run's results: results.parquet when it is at least as new as
    results.jsonl, else the jsonl (see ``read_results_jsonl``)."""
    run_dir = Path(run_dir)
    parquet = run_dir / "results.parquet"
    jsonl = run_dir / "results.jsonl"
    if parquet.exists() and _parquet_is_current(parquet, jsonl):
        return pd.read_parquet(parquet)
    if not jsonl.exists():
        return pd.DataFrame()
    return read_results_jsonl(jsonl)


def _parquet_is_current(parquet: Path, jsonl: Path) -> bool:
    if not jsonl.exists():
        return True
    return parquet.stat().st_mtime_ns >= jsonl.stat().st_mtime_ns


def solver_summary(run_dir: str | Path) -> pd.DataFrame:
    return status_counts(load_results(run_dir))


def status_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Recorded rows per ``(solver_id, status)``."""
    if df.empty:
        return df
    grouped = (
        df.groupby(["solver_id", "status"], dropna=False, observed=True)
        .size()
        .reset_index(name="count")
    )
    return grouped.sort_values(["solver_id", "status"])
