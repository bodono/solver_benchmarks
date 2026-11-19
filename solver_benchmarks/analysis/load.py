"""Load benchmark run results."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from solver_benchmarks.analysis.verify import (
    VerifyInputError,
    VerifyPolicy,
    unverified_frame,
    verify_frame,
)
from solver_benchmarks.core.storage import read_results_jsonl


def load_results(
    run_dir: str | Path,
    *,
    verify: bool | VerifyPolicy = True,
    select: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """The run's results, with KKT-verified statuses by default.

    Rows come from results.parquet when it is at least as new as
    results.jsonl, else from the jsonl (see ``read_results_jsonl``). A parquet
    a starting resume removes while it is being looked at is not an error:
    the jsonl is read instead.

    ``status`` is the verified status (see ``analysis/verify.py``): every
    claimed solution and certificate is checked against the harness's own
    KKT residuals at ``VERIFY_SLACK`` times the tolerance its configuration
    requested, and inaccurate or limit stops that pass are promoted, except
    those returned after the run's time limit. ``status_reported`` keeps the
    status the solver reported and the ``verify.*`` columns say what was
    decided and why; ``df.attrs["verification"]`` sums it up. A run written by
    ``bench kkt-verify`` keeps the verification it recorded. ``verify`` may be
    a :class:`VerifyPolicy` for another rule, or False for the statuses (and
    objectives) the solvers reported, which also undoes a ``bench kkt-verify``
    copy's. Raises ``RequestedToleranceUnknown`` (a ``VerifyInputError``)
    when a row needs a requested tolerance the run does not record.

    ``select``, given the rows as read, returns the ones to keep; only those
    are verified, so rows left out (solvers a figure does not show) never
    need a requested tolerance.
    """
    run_dir = Path(run_dir)
    df = _read_rows(run_dir)
    if select is not None and not df.empty:
        df = select(df)
    if verify is False:
        return unverified_frame(df)
    policy = VerifyPolicy() if verify is True else verify
    if not isinstance(policy, VerifyPolicy):
        raise TypeError(f"verify must be True, False or a VerifyPolicy (got {verify!r})")
    return verify_frame(df, _read_manifest(run_dir), policy)


def _read_rows(run_dir: Path) -> pd.DataFrame:
    parquet = run_dir / "results.parquet"
    jsonl = run_dir / "results.jsonl"
    try:
        if parquet.exists() and _parquet_is_current(parquet, jsonl):
            return pd.read_parquet(parquet)
    except FileNotFoundError:
        pass
    if not jsonl.exists():
        return pd.DataFrame()
    return read_results_jsonl(jsonl)


def _read_manifest(run_dir: Path) -> dict:
    """The run's manifest ({} without one); one that is not valid JSON is an error."""
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifyInputError(f"cannot read {path} to verify the run: {exc}") from exc
    return manifest if isinstance(manifest, dict) else {}


def _parquet_is_current(parquet: Path, jsonl: Path) -> bool:
    if not jsonl.exists():
        return True
    return parquet.stat().st_mtime_ns >= jsonl.stat().st_mtime_ns


def solver_summary(run_dir: str | Path, *, verify: bool | VerifyPolicy = True) -> pd.DataFrame:
    return status_counts(load_results(run_dir, verify=verify))


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
