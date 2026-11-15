"""Recovery / atomicity tests for the result store.

These tests pin behaviors that the runner relies on but that no other
test exercises directly: torn-write tolerance in resume parsing and
parquet rewrite, parquet temp-file cleanup on failure, and the
all-string-NaN scrub being numeric-only.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from solver_benchmarks.core.config import parse_run_config, solve_signature
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


def test_retry_replaces_compatible_worker_errors_and_supersedes_incompatible_rows(
    tmp_path: Path,
):
    store = _make_store(tmp_path)
    config = parse_run_config(
        {"run": {"dataset": "synthetic_qp"}, "solvers": [{"id": "scs", "solver": "scs"}]}
    )
    signature = solve_signature(config, config.datasets[0], config.solvers[0])
    failed = ProblemResult(
        run_id=store.run_id,
        dataset="synthetic_qp",
        problem="p",
        problem_kind=QP,
        solver_id="scs",
        solver="scs",
        status="worker_error",
        objective_value=None,
        iterations=None,
        run_time_seconds=None,
        metadata={"resume_signature": signature},
    )
    incompatible = replace(failed, metadata={"resume_signature": "previous-settings"})
    other_dataset = replace(failed, dataset="other_dataset")
    time_limited = replace(failed, problem="limited", status="time_limit")
    for result in (failed, incompatible, other_dataset, time_limited):
        store.write_result(result)

    assert store.completed_keys(config=config) == {("synthetic_qp", "limited", "scs")}
    store.write_result(replace(failed, status="optimal", run_time_seconds=0.1))

    rows = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    assert len(rows) == 3
    # A dataset the config does not plan keeps its rows untouched.
    assert rows[:2] == [other_dataset.to_record(), time_limited.to_record()]
    assert rows[2]["status"] == "optimal"
    assert rows[2]["metadata"]["resume_signature"] == signature
    superseded = store.superseded_jsonl_path.read_text().splitlines()
    assert [json.loads(line) for line in superseded] == [incompatible.to_record()]


def test_retry_result_replace_failure_preserves_original_row(tmp_path: Path, monkeypatch):
    store = _make_store(tmp_path)
    failed = ProblemResult(
        run_id=store.run_id,
        dataset="synthetic_qp",
        problem="p",
        problem_kind=QP,
        solver_id="scs",
        solver="scs",
        status="worker_error",
        objective_value=None,
        iterations=None,
        run_time_seconds=None,
    )
    store.write_result(failed)
    assert store.completed_keys() == set()
    original = store.results_jsonl_path.read_text()

    def fail_replace(*_args):
        raise OSError("simulated disk failure")

    with monkeypatch.context() as patcher:
        patcher.setattr("solver_benchmarks.core.storage.os.replace", fail_replace)
        with pytest.raises(OSError, match="simulated disk failure"):
            store.write_result(replace(failed, status="optimal"))
    assert store.results_jsonl_path.read_text() == original

    # The pending replacement is still available after the interrupted write.
    store.write_result(replace(failed, status="optimal"))
    rows = store.results_jsonl_path.read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["status"] == "optimal"


def test_legacy_retry_artifact_path_counts_toward_resume_limit(tmp_path: Path):
    store = _make_store(tmp_path)
    failed = ProblemResult(
        run_id=store.run_id, dataset="synthetic_qp", problem="p", problem_kind=QP,
        solver_id="scs", solver="scs", status="worker_error", objective_value=None,
        iterations=None, run_time_seconds=None,
        artifact_dir=str(store.run_dir / "problems/synthetic_qp/p/scs/retry-2"),
    )
    store.write_result(failed)
    assert store.completed_keys() == {("synthetic_qp", "p", "scs")}
    assert not store.has_pending_retry("synthetic_qp", "p", "scs")


def test_solver_named_retry_does_not_consume_legacy_retry_budget(tmp_path: Path):
    store = _make_store(tmp_path)
    artifacts = store.problem_solver_dir("synthetic_qp", "p", "retry-2")
    failed = ProblemResult(
        run_id=store.run_id, dataset="synthetic_qp", problem="p", problem_kind=QP,
        solver_id="retry-2", solver="scs", status="worker_error", objective_value=None,
        iterations=None, run_time_seconds=None, artifact_dir=str(artifacts),
    )
    store.write_result(failed)
    assert store.completed_keys() == set()
    assert store.has_pending_retry("synthetic_qp", "p", "retry-2")


def test_write_result_does_not_materialize_parquet_until_write_parquet(tmp_path: Path):
    """Pin the central contract of the end-of-run model: ``write_result``
    appends to jsonl only and never touches results.parquet. The parquet
    is materialized exclusively by ``write_parquet`` so that the file's
    presence on disk can be used as a "run completed" sentinel."""
    store = _make_store(tmp_path)
    for problem in ("p1", "p2", "p3"):
        store.write_result(
            ProblemResult(
                run_id=store.run_id,
                dataset="synthetic_qp",
                problem=problem,
                problem_kind=QP,
                solver_id="scs",
                solver="scs",
                status="optimal",
                objective_value=0.0,
                iterations=1,
                run_time_seconds=0.01,
            )
        )
        assert not store.results_parquet_path.exists()

    store.write_parquet()
    assert store.results_parquet_path.exists()


def test_write_parquet_recovers_from_torn_last_line(tmp_path: Path):
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


def test_write_parquet_cleans_up_tmp_on_failure(tmp_path: Path, monkeypatch):
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


def test_write_parquet_handles_columns_mixing_python_types(tmp_path: Path):
    """Gurobi and CPLEX report an integer raw_status; most adapters a string."""
    import pandas as pd

    store = _make_store(tmp_path)
    infos = [
        {"raw_status": 2, "nested": {"a": 1}},
        {"raw_status": "Solved", "nested": [1, 2]},
        {"raw_status": None, "nested": "text"},
    ]
    for index, info in enumerate(infos):
        store.write_result(
            ProblemResult(
                run_id=store.run_id, dataset="synthetic_qp", problem=f"p{index}",
                problem_kind=QP, solver_id="scs", solver="scs", status="optimal",
                objective_value=1.0, iterations=index, run_time_seconds=0.1, info=info,
            )
        )
    store.write_parquet()
    df = pd.read_parquet(store.results_parquet_path)
    assert list(df["info.raw_status"].iloc[:2]) == ["2", "Solved"]
    assert pd.isna(df["info.raw_status"].iloc[2])
    assert list(df["iterations"]) == [0, 1, 2]
    assert list(df["info.nested"].iloc[1:]) == ["[1, 2]", "text"]


def test_run_completes_when_parquet_cannot_be_written(tmp_path: Path, monkeypatch, repo_root):
    from types import SimpleNamespace

    from solver_benchmarks.core.runner import run_benchmark

    def fake(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        record = ProblemResult(
            run_id=payload["run_id"], dataset=payload["dataset"], problem=payload["problem"],
            problem_kind=payload["problem_kind"], solver_id="scs", solver="scs",
            status="optimal", objective_value=0.0, iterations=1, run_time_seconds=0.01,
            artifact_dir=payload["artifacts_dir"],
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, timed_out=False)

    def broken_parquet(self):
        raise TypeError("Conversion failed for column info.raw_status")

    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", fake)
    monkeypatch.setattr(ResultStore, "write_parquet", broken_parquet)
    config = parse_run_config(
        {"run": {"dataset": "synthetic_qp"}, "solvers": [{"id": "scs", "solver": "scs"}]}
    )
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    assert len(store.results_jsonl_path.read_text().splitlines()) == 2
    events = [json.loads(line) for line in store.events_path.read_text().splitlines()]
    failure = next(event for event in events if event["level"] == "error")
    assert "results.parquet" in failure["message"] and "raw_status" in failure["error"]
    assert events[-1]["message"] == "benchmark_complete"


@pytest.mark.parametrize("writer", ["text", "bytes", "stream"])
def test_atomic_writes_honour_the_umask_and_keep_an_existing_mode(tmp_path: Path, writer):
    import os
    import stat

    from solver_benchmarks.core.atomic import atomic_write_stream
    from solver_benchmarks.datasets.base import atomic_write_bytes

    def write(path):
        if writer == "text":
            atomic_write_text(path, "content")
        elif writer == "bytes":
            atomic_write_bytes(path, b"content")
        else:
            atomic_write_stream(path, lambda handle: handle.write(b"content"))

    previous = os.umask(0o022)
    try:
        fresh = tmp_path / "fresh.json"
        write(fresh)
        assert stat.S_IMODE(fresh.stat().st_mode) == 0o644
        os.umask(0o027)
        write(tmp_path / "group.json")
        assert stat.S_IMODE((tmp_path / "group.json").stat().st_mode) == 0o640
        existing = tmp_path / "existing.json"
        existing.write_text("old")
        existing.chmod(0o664)
        write(existing)
        assert stat.S_IMODE(existing.stat().st_mode) == 0o664
        assert existing.read_text() == "content"
    finally:
        os.umask(previous)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "existing.json", "fresh.json", "group.json"
    ]
