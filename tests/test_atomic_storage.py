"""Recovery / atomicity tests for the result store.

These tests pin behaviors that the runner relies on but that no other
test exercises directly: torn-write tolerance in resume parsing and
parquet rewrite, parquet temp-file cleanup on failure, and the
all-string-NaN scrub being numeric-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.problem import QP
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.storage import ResultStore, atomic_write_text


def _make_store(tmp_path: Path) -> ResultStore:
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )
    return ResultStore.create(config, run_dir=tmp_path / "run")


def test_atomic_write_text_recovers_from_replace_failure(tmp_path: Path, monkeypatch):
    """If os.replace raises, the temp file must not linger."""
    target = tmp_path / "file.txt"
    target.write_text("original")

    import os as _os

    real_replace = _os.replace

    def failing_replace(src, dst):  # noqa: ARG001
        raise OSError("simulated rename failure")

    monkeypatch.setattr("solver_benchmarks.core.storage.os.replace", failing_replace)

    with pytest.raises(OSError):
        atomic_write_text(target, "new content")

    # Original is intact (rename was the last step), and no .tmp* files
    # have been left orphaned alongside it.
    assert target.read_text() == "original"
    assert not list(tmp_path.glob("tmp*"))

    # Sanity-check: with the real os.replace restored, the write succeeds.
    monkeypatch.setattr("solver_benchmarks.core.storage.os.replace", real_replace)
    atomic_write_text(target, "new content")
    assert target.read_text() == "new content"


def test_completed_keys_skips_torn_jsonl_lines(tmp_path: Path):
    """A truncated last line should not crash resume planning."""
    store = _make_store(tmp_path)
    # Write one valid record, then a torn line with no closing brace.
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="p1",
            problem_kind=QP,
            solver_id="scs",
            solver="scs",
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
        )
    )
    with store.results_jsonl_path.open("a") as handle:
        handle.write('{"problem": "p2", "solver_id": "scs", "dataset": "synth')

    keys = store.completed_keys()
    assert keys == {("synthetic_qp", "p1", "scs")}


def test_rewrite_parquet_recovers_from_torn_last_line(tmp_path: Path):
    """A torn jsonl line must not block subsequent parquet rewrites."""
    store = _make_store(tmp_path)
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="p1",
            problem_kind=QP,
            solver_id="scs",
            solver="scs",
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
        )
    )
    with store.results_jsonl_path.open("a") as handle:
        handle.write('{"problem": "p2", "torn": ')
    store.write_parquet()

    import pandas as pd

    df = pd.read_parquet(store.results_parquet_path)
    assert len(df) == 1
    assert df.loc[0, "problem"] == "p1"


def test_rewrite_parquet_cleans_up_tmp_on_failure(tmp_path: Path, monkeypatch):
    """If df.to_parquet raises, no .parquet.tmp file should remain."""
    store = _make_store(tmp_path)
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="p1",
            problem_kind=QP,
            solver_id="scs",
            solver="scs",
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
        )
    )

    import pandas as pd

    def failing_to_parquet(self, *args, **kwargs):  # noqa: ARG001
        raise OSError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_to_parquet)

    with pytest.raises(OSError):
        store.write_parquet()

    leftover = list(store.run_dir.glob("results.parquet.tmp"))
    assert leftover == []


def test_normalize_table_for_parquet_does_not_scrub_string_columns(tmp_path: Path):
    """A free-text problem name like ``nan`` or an error containing
    ``inf`` must not be silently nulled by the numeric scrubber."""
    store = _make_store(tmp_path)
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="nan",  # legitimate problem name happens to be the string "nan"
            problem_kind=QP,
            solver_id="scs",
            solver="scs",
            status="solver_error",
            objective_value=None,
            iterations=None,
            run_time_seconds=1.0,
            error="solver hit inf during step",
        )
    )
    store.write_parquet()

    import pandas as pd

    df = pd.read_parquet(store.results_parquet_path)
    assert df.loc[0, "problem"] == "nan"
    assert "inf" in df.loc[0, "error"]


def test_completed_keys_skips_record_missing_required_keys(tmp_path: Path):
    """Older / partially-corrupt jsonl rows missing problem/solver_id
    should be skipped with a warning rather than KeyError."""
    store = _make_store(tmp_path)
    with store.results_jsonl_path.open("a") as handle:
        # Missing solver_id
        handle.write(json.dumps({"problem": "p1", "dataset": "d"}) + "\n")
        # Complete row
        handle.write(
            json.dumps(
                {"problem": "p2", "solver_id": "s", "dataset": "d"}
            )
            + "\n"
        )

    keys = store.completed_keys()
    assert keys == {("d", "p2", "s")}
