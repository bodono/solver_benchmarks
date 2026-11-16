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
    # The replaced rows leave in the one rewrite at the end of the run.
    store.retire_replaced_rows()

    rows = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    assert len(rows) == 3
    # A dataset the config does not plan keeps its rows untouched.
    assert rows[:2] == [other_dataset.to_record(), time_limited.to_record()]
    assert rows[2]["status"] == "optimal"
    assert rows[2]["metadata"]["resume_signature"] == signature
    superseded = store.superseded_jsonl_path.read_text().splitlines()
    assert [json.loads(line) for line in superseded] == [incompatible.to_record()]


def test_a_failed_end_of_run_rewrite_loses_no_row(tmp_path: Path, monkeypatch):
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
    store.write_result(replace(failed, status="optimal"))
    written = store.results_jsonl_path.read_text()

    def fail_replace(*_args):
        raise OSError("simulated disk failure")

    with monkeypatch.context() as patcher:
        patcher.setattr("solver_benchmarks.core.storage.os.replace", fail_replace)
        with pytest.raises(OSError, match="simulated disk failure"):
            store.retire_replaced_rows()
    # Both rows stay: the error and the result that replaces it.
    assert store.results_jsonl_path.read_text() == written
    assert [json.loads(line)["status"] for line in written.splitlines()] == [
        "worker_error", "optimal"
    ]

    # The next resume counts the key complete and drops the stale error.
    assert store.completed_keys() == {("synthetic_qp", "p", "scs")}
    store.retire_replaced_rows()
    rows = store.results_jsonl_path.read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["status"] == "optimal"


def _identical_rows_store(tmp_path: Path, lines: list[str]) -> ResultStore:
    store = _make_store(tmp_path)
    store.results_jsonl_path.write_text("".join(line + "\n" for line in lines))
    return store


def _replaceable_rows(store: ResultStore) -> dict[str, ProblemResult]:
    base = ProblemResult(
        run_id=store.run_id, dataset="synthetic_qp", problem="p", problem_kind=QP,
        solver_id="scs", solver="scs", status="worker_error", objective_value=None,
        iterations=None, run_time_seconds=None,
    )
    return {
        "skip": replace(base, status="skipped_unsupported",
                        metadata={"skip_reason": "solver_unavailable"}),
        "error": base,
        "ok": replace(base, status="optimal", objective_value=0.0, iterations=1),
    }


@pytest.mark.parametrize("stale", ["skip", "error"])
def test_an_identical_copy_that_is_the_latest_row_keeps_its_text(tmp_path: Path, stale: str):
    """[skip, ok, skip]: the first copy has a later row, the second is the
    latest row of its key. Rows are dropped by their text, so neither goes."""
    probe = _make_store(tmp_path / "probe")
    rows = {name: json.dumps(row.to_record(), sort_keys=True)
            for name, row in _replaceable_rows(probe).items()}
    lines = [rows[stale], rows["ok"], rows[stale]]
    store = _identical_rows_store(tmp_path, lines)
    store.completed_keys(available_solver_ids=set())
    store.retire_replaced_rows()
    assert store.results_jsonl_path.read_text().splitlines() == lines


def test_a_skip_identical_to_the_skip_it_replaces_is_kept_once(tmp_path: Path):
    # (A retried error is never identical: it records its retry count.)
    store = _make_store(tmp_path)
    row = _replaceable_rows(store)["skip"]
    store.write_result(row)
    assert store.completed_keys(available_solver_ids={"scs"}) == set()
    store.write_result(replace(row))
    store.retire_replaced_rows()
    rows = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    assert rows == [row.to_record()]


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
    """Gurobi and CPLEX rows recorded by older versions have an integer
    raw_status; the others a string."""
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

    def fake(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, env=None):
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


# --- results.jsonl is read in chunks into the frame one json_normalize gives ---


def _heterogeneous_lines() -> list[str]:
    records = [
        {"problem": "p0", "status": "optimal", "iterations": 3, "flag": True,
         "info": {"raw_status": 2, "x": {"deep": 1.5}}, "maybe": None, "text": "a"},
        {"problem": "p1", "status": "solver_error", "iterations": 4.5, "flag": None,
         "info": 5, "maybe": None, "text": None},
        # keys that first appear in a later chunk, one of them nested
        {"problem": "p2", "status": "optimal", "late": [1, 2], "info": {"raw_status": "Solved", "y": {}},
         "maybe": 1.25},
        {"problem": "p3", "status": "optimal", "iterations": 2**60, "info.raw_status": "top",
         "info": {"raw_status": "nested"}, "maybe": 2.0, "flag": False, "text": "b"},
        {},
        None,
        {"problem": "p6", "status": "optimal", "maybe": float("nan"), "late": "s", "flag": True},
        {"problem": "p7", "status": "optimal", "maybe": None, "text": "c", "iterations": -1},
    ]
    lines = [json.dumps(record) for record in records]
    lines.insert(3, "")  # a blank line
    lines.append('{"problem": "p8", "torn": ')
    return lines


def _expected_frame(lines):
    """json_normalize of the lines that are JSON objects (the others are skipped)."""
    import pandas as pd

    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return pd.json_normalize(records)


def _assert_identical_frames(got, expected):
    import pandas as pd

    assert list(got.columns) == list(expected.columns)
    assert got.columns.dtype == expected.columns.dtype
    assert got.index.equals(expected.index) and type(got.index) is type(expected.index)
    assert list(got.dtypes) == list(expected.dtypes)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    for column in got.columns:
        if got[column].dtype == object:
            assert [type(v) for v in got[column]] == [type(v) for v in expected[column]], column
            assert [repr(v) for v in got[column]] == [repr(v) for v in expected[column]], column


@pytest.mark.parametrize("chunk_lines", [1, 2, 3, 4, 1000])
def test_read_results_jsonl_equals_one_shot_json_normalize(tmp_path: Path, chunk_lines):

    from solver_benchmarks.core.storage import read_results_jsonl

    lines = _heterogeneous_lines()
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(lines) + "\n")
    expected = _expected_frame(lines)
    got = read_results_jsonl(path, chunk_lines=chunk_lines)
    _assert_identical_frames(got, expected)
    # in chunks of two, 'maybe' is all null in the second chunk and 'late' first appears in the third
    assert expected["maybe"].dtype == float and expected["info.raw_status"].dtype == object


def test_read_results_jsonl_edge_files(tmp_path: Path):
    import pandas as pd

    from solver_benchmarks.core.storage import read_results_jsonl

    path = tmp_path / "results.jsonl"
    path.write_text("")
    assert read_results_jsonl(path).empty
    path.write_text('\n{"torn": \n')
    assert read_results_jsonl(path).empty
    path.write_text("{}\n{}\n{}\n")
    _assert_identical_frames(read_results_jsonl(path, chunk_lines=2), pd.json_normalize([{}, {}, {}]))


@pytest.mark.parametrize("conversion", ["chunked", "missing", "changed signature"])
def test_read_results_jsonl_skips_lines_that_are_not_objects(tmp_path: Path, monkeypatch, caplog, conversion):
    import pandas as pd

    from solver_benchmarks.core import storage

    if conversion == "missing":
        monkeypatch.setattr(storage, "_convert_object_array", None)
    elif conversion == "changed signature":
        def changed(arrays):
            raise AssertionError("not reached")

        monkeypatch.setattr(storage, "_convert_object_array", changed)
    path = tmp_path / "results.jsonl"
    path.write_text('{"problem": "p1", "x": 1}\nnull\n5\n[1, 2]\n"text"\n{"problem": "p2", "x": 2.5}\n')
    with caplog.at_level("WARNING", logger="solver_benchmarks.core.storage"):
        frame = storage.read_results_jsonl(path, chunk_lines=2)
    _assert_identical_frames(frame, pd.json_normalize([{"problem": "p1", "x": 1}, {"problem": "p2", "x": 2.5}]))
    skipped = [r.getMessage() for r in caplog.records if "Skipping unparseable line" in r.getMessage()]
    # the fallback reads the file again, so a line can be reported twice
    assert {message.split()[3] for message in skipped} == {"2", "3", "4", "5"}


def test_read_results_jsonl_without_the_pandas_conversion(tmp_path: Path, monkeypatch):

    from solver_benchmarks.core import storage

    lines = _heterogeneous_lines()
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(storage, "_convert_object_array", None)
    expected = _expected_frame(lines)
    _assert_identical_frames(storage.read_results_jsonl(path, chunk_lines=2), expected)


def test_load_results_and_write_parquet_share_the_chunked_reader(tmp_path: Path, monkeypatch):
    import pandas as pd

    from solver_benchmarks.analysis import load
    from solver_benchmarks.core import storage

    store = _make_store(tmp_path)
    lines = _heterogeneous_lines()
    store.results_jsonl_path.write_text("\n".join(lines) + "\n")
    chunks = []
    original = storage._batched

    def spy(items, size):
        for batch in original(items, size):
            chunks.append(len(batch))
            yield batch

    monkeypatch.setattr(storage, "_batched", spy)
    monkeypatch.setattr(storage.read_results_jsonl, "__kwdefaults__", {"chunk_lines": 3})
    frame = load.load_results(store.run_dir)
    assert chunks == [3, 3, 1]
    expected = _expected_frame(lines)
    _assert_identical_frames(frame, expected)
    chunks.clear()
    store.write_parquet()
    assert chunks == [3, 3, 1]
    assert len(pd.read_parquet(store.results_parquet_path)) == len(expected)


def test_atomic_write_stream_whose_temporary_was_taken_by_a_finished_writer(tmp_path: Path):
    """A stalled writer whose temporary another process's sweep removed, after
    that process wrote the target: the write counts as done."""
    from solver_benchmarks.core.atomic import atomic_write_stream

    target = tmp_path / "big.jld2"

    def stalled(handle):
        handle.write(b"partial")
        (temporary,) = tmp_path.glob(".big.jld2.*.tmp")
        temporary.unlink()  # another worker's remove_stale_temporaries
        target.write_bytes(b"complete")  # and its finished copy

    atomic_write_stream(target, stalled)

    assert target.read_bytes() == b"complete"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["big.jld2"]


def test_atomic_write_stream_whose_temporary_vanished_without_a_target_fails(tmp_path: Path):
    from solver_benchmarks.core.atomic import atomic_write_stream

    target = tmp_path / "big.jld2"

    def stalled(handle):
        handle.write(b"partial")
        (temporary,) = tmp_path.glob(".big.jld2.*.tmp")
        temporary.unlink()

    with pytest.raises(FileNotFoundError):
        atomic_write_stream(target, stalled)
    assert not target.exists()


# ---------------------------------------------------------------------------
# The reader while a resume writes: a torn tail, bad lines, merged runs.
# ---------------------------------------------------------------------------


def _reader_row(problem: str, status: str, run_id: str = "r") -> str:
    return json.dumps({"run_id": run_id, "dataset": "d", "problem": problem, "solver_id": "s",
                       "status": status, "metadata": {"resume_signature": "A"}})


def _problems_and_statuses(frame) -> list[tuple[str, str]]:
    return list(zip(frame["problem"], frame["status"]))


def test_a_last_line_completed_during_the_first_read_is_not_read_past(tmp_path: Path, monkeypatch):
    """The writer finishes the torn last row while the first read is at it:
    the first read stops at the fragment, and so does the second, so no
    frame holds the row without the first read having weighed it."""
    import types

    from solver_benchmarks.core import storage

    path = tmp_path / "results.jsonl"
    last = _reader_row("K", "optimal") + "\n"
    path.write_text("".join(_reader_row(*row) + "\n" for row in (
        ("J", "worker_error"), ("J", "optimal"), ("K", "worker_error"))) + last[:40])
    completed = []

    def loads(line, *args, **kwargs):
        if not completed and not line.endswith(b"\n"):
            completed.append(True)
            with path.open("a") as handle:  # the writer's write() finishes now
                handle.write(last[40:])
        return json.loads(line, *args, **kwargs)

    monkeypatch.setattr(storage, "json", types.SimpleNamespace(
        loads=loads, dumps=json.dumps, JSONDecodeError=json.JSONDecodeError))
    frame = storage.read_results_jsonl(path)
    assert completed
    assert _problems_and_statuses(frame) == [("J", "optimal"), ("K", "worker_error")]


def test_a_last_line_completed_between_the_reads_is_cut_where_the_first_read_stopped(
    tmp_path: Path, monkeypatch
):
    from solver_benchmarks.core import storage

    path = tmp_path / "results.jsonl"
    last = _reader_row("K", "optimal") + "\n"
    path.write_text("".join(_reader_row(*row) + "\n" for row in (
        ("J", "worker_error"), ("J", "optimal"), ("K", "worker_error"))) + last[:40])
    first_read = storage._read_results_jsonl_chunked
    reads = []

    def read_then_complete(*args, **kwargs):
        frame = first_read(*args, **kwargs)
        if not reads:
            with path.open("a") as handle:
                handle.write(last[40:])
        reads.append(kwargs)
        return frame

    monkeypatch.setattr(storage, "_read_results_jsonl_chunked", read_then_complete)
    frame = storage.read_results_jsonl(path)
    assert len(reads) == 2
    assert _problems_and_statuses(frame) == [("J", "optimal"), ("K", "worker_error")]


def test_a_bad_line_is_warned_about_once(tmp_path: Path, caplog):
    from solver_benchmarks.core.storage import read_results_jsonl

    path = tmp_path / "results.jsonl"
    path.write_text("\n".join([
        _reader_row("J", "worker_error"), "{not json", _reader_row("J", "optimal"),
        "[1, 2]", _reader_row("K", "optimal"),
    ]) + "\n")
    with caplog.at_level("WARNING", logger="solver_benchmarks.core.storage"):
        frame = read_results_jsonl(path)
    assert _problems_and_statuses(frame) == [("J", "optimal"), ("K", "optimal")]
    warnings = [record.getMessage() for record in caplog.records]
    assert warnings == [f"Skipping unparseable line {n} in {path}" for n in (2, 4)]


def test_a_line_that_is_not_utf8_is_skipped(tmp_path: Path, caplog):
    from solver_benchmarks.core.storage import read_results_jsonl

    path = tmp_path / "results.jsonl"
    path.write_bytes(
        (_reader_row("J", "optimal") + "\n").encode()
        + b'{"problem": "\xff\xfe"}\n'
        + (_reader_row("K", "optimal") + "\n").encode()
    )
    with caplog.at_level("WARNING", logger="solver_benchmarks.core.storage"):
        frame = read_results_jsonl(path)
    assert _problems_and_statuses(frame) == [("J", "optimal"), ("K", "optimal")]
    assert [record.getMessage() for record in caplog.records] == [
        f"Skipping unparseable line 2 in {path}"
    ]


@pytest.mark.parametrize("order", ["AB", "BA"])
def test_rows_of_different_runs_never_replace_each_other(tmp_path: Path, order: str):
    """A bench merge output holds the rows of several runs; a worker error of
    one run next to a result of another is a duplicate merge reports, not a
    replaced row."""
    from solver_benchmarks.core.storage import read_results_jsonl

    runs = {
        "A": [_reader_row("J", "worker_error", "A"), _reader_row("K", "worker_error", "A")],
        "B": [_reader_row("J", "optimal", "B"), _reader_row("K", "worker_error", "B")],
    }
    lines = runs[order[0]] + runs[order[1]]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(lines) + "\n")
    frame = read_results_jsonl(path)
    assert len(frame) == 4
    assert sorted(zip(frame["run_id"], frame["problem"], frame["status"])) == [
        ("A", "J", "worker_error"), ("A", "K", "worker_error"),
        ("B", "J", "optimal"), ("B", "K", "worker_error"),
    ]


def test_load_results_reads_the_jsonl_when_a_resume_removes_the_parquet(
    tmp_path: Path, monkeypatch
):
    from solver_benchmarks.analysis import load

    store = _make_store(tmp_path)
    store.write_result(ProblemResult(
        run_id=store.run_id, dataset="synthetic_qp", problem="p", problem_kind=QP,
        solver_id="scs", solver="scs", status="optimal", objective_value=0.0,
        iterations=1, run_time_seconds=0.01,
    ))
    store.write_parquet()
    read_parquet = load.pd.read_parquet

    def removed_meanwhile(path, *args, **kwargs):
        Path(path).unlink()  # a resume starts between the check and the read
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(load.pd, "read_parquet", removed_meanwhile)
    frame = load.load_results(store.run_dir)
    assert list(zip(frame["problem"], frame["status"])) == [("p", "optimal")]
