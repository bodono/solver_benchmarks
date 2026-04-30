"""Load benchmark run results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_results(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    parquet = run_dir / "results.parquet"
    jsonl = run_dir / "results.jsonl"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if not jsonl.exists():
        return pd.DataFrame()
    with jsonl.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]
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
