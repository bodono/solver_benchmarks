"""Which rows' timed regions were warmed up (README, "Warm-up").

Rows timed with and without the warm-up time different things: a cold row's
``run_time_seconds`` can include a process's one-time costs (imports, library
loading, thread pools, a GPU's context and kernels) that a warmed row leaves
out. Nothing in the tables tells them apart, so the analysis commands and the
report warn when the loaded rows include timed rows that were not fully warmed
(``metadata.warmup``; ``core/warmup.timing_category`` says how each row is
counted). A warning only: the rows are kept, as they were recorded.
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from solver_benchmarks.core import status, warmup

_FIELDS = ("enabled", "complete", "rerun_without_warmup")


def warmup_counts(df: pd.DataFrame) -> Counter[str]:
    """Timed rows per ``warmup.timing_category``.

    A timed row has a finite ``run_time_seconds`` and is not a skip. Its
    record is read from the frame's ``metadata.warmup.*`` columns; a row whose
    ``metadata.warmup.enabled`` is missing has no record (every record has it).
    """
    counts: Counter[str] = Counter()
    if df.empty or "run_time_seconds" not in df.columns:
        return counts
    seconds = pd.to_numeric(df["run_time_seconds"], errors="coerce").tolist()
    status_column = "status_reported" if "status_reported" in df.columns else "status"
    reported = df[status_column].tolist() if status_column in df.columns else [None] * len(df)
    columns = {
        field: df[f"metadata.warmup.{field}"].tolist()
        for field in _FIELDS
        if f"metadata.warmup.{field}" in df.columns
    }
    for position, value in enumerate(seconds):
        if pd.isna(value) or not math.isfinite(float(value)):
            continue
        if reported[position] == status.SKIPPED_UNSUPPORTED:
            continue
        record = {field: _flag(values[position]) for field, values in columns.items()}
        counts[warmup.timing_category(record if record.get("enabled") is not None else None)] += 1
    return counts


def warmup_warnings(df: pd.DataFrame) -> list[str]:
    """The warning about the frame's timed rows that were not fully warmed
    (``warmup.not_warmed_warning``), as a list: empty when all were."""
    message = warmup.not_warmed_warning(warmup_counts(df))
    return [message] if message is not None else []


def _flag(value) -> bool | None:
    """A ``metadata.warmup`` flag as read from a frame: None when missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return bool(value)
