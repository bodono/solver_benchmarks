"""Resume keeps one row per planned solve, and that row matches the config.

A row whose resume signature no longer matches the config is superseded:
it stays until the new result is durable, then moves to
``results.superseded.jsonl`` at the end of the run. Rows written by older
releases, whose signatures still hashed the dataset's location options,
keep counting, also after those options change.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.core import runner as runner_module
from solver_benchmarks.core.config import parse_run_config, solve_signature
from solver_benchmarks.core.problem import QP, ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.core.storage import LEGACY_SIGNATURES_KEY, ResultStore
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.datasets.base import Dataset
from solver_benchmarks.solvers import registry as solver_registry
from solver_benchmarks.solvers.base import SolverAdapter


class _TwoProblemDataset(Dataset):
    dataset_id = "resume_fixture"
    description = "two QPs"

    def list_problems(self):
        return [
            ProblemSpec(dataset_id="resume_fixture", name=name, kind=QP)
            for name in ("p1", "p2")
        ]

    def load_problem(self, name):  # pragma: no cover - the worker is stubbed
        raise AssertionError


class _ToggleSolver(SolverAdapter):
    solver_name = "toggle"
    supported_problem_kinds = {QP}
    available = True

    @classmethod
    def is_available(cls):
        return cls.available

    def solve(self, problem, artifacts_dir):  # pragma: no cover - the worker is stubbed
        raise AssertionError


@pytest.fixture
def fake_worker(monkeypatch):
    """Stub the worker subprocess; each call records the payload and writes
    a result whose status and run time come from ``outcome``."""
    calls: list[dict] = []
    outcome = {"status": "optimal", "run_time_seconds": 0.01}

    def run(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output, env=None):
        payload = json.loads(Path(cmd[-1]).read_text())
        calls.append(payload)
        stdout_path.write_text(f"attempt {len(calls)}\n")
        stderr_path.write_text("")
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            objective_value=0.0,
            iterations=1,
            artifact_dir=payload["artifacts_dir"],
            **outcome,
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, timed_out=False)

    monkeypatch.setitem(dataset_registry.DATASETS, "resume_fixture", _TwoProblemDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "toggle", _ToggleSolver)
    monkeypatch.setattr(_ToggleSolver, "available", True)
    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", run)
    return SimpleNamespace(calls=calls, outcome=outcome)


def _config(tmp_path: Path, *, settings=None, dataset_options=None, timeout=None, resume=True):
    return parse_run_config(
        {
            "run": {
                "dataset": "resume_fixture",
                "output_dir": str(tmp_path / "runs"),
                "dataset_options": dataset_options or {},
                "timeout_seconds": timeout,
                "resume": resume,
            },
            "solvers": [{"id": "s", "solver": "toggle", "settings": settings or {}}],
        }
    )


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_resume_supersedes_rows_solved_under_other_settings(
    fake_worker, tmp_path: Path, repo_root: Path
):
    store = run_benchmark(_config(tmp_path, settings={"eps": 1e-4}), run_dir=tmp_path / "run",
                          repo_root=repo_root)
    old_rows = _rows(store.results_jsonl_path)
    old_artifacts = {Path(row["artifact_dir"]) / "stdout.log": None for row in old_rows}
    for path in old_artifacts:
        old_artifacts[path] = path.read_text()

    # Tighter settings under the same id: the old results no longer describe it.
    fake_worker.outcome["status"] = "max_iter_reached"
    run_benchmark(_config(tmp_path, settings={"eps": 1e-8}), run_dir=store.run_dir,
                  repo_root=repo_root)

    rows = _rows(store.results_jsonl_path)
    assert len(fake_worker.calls) == 4
    assert [row["status"] for row in rows] == ["max_iter_reached"] * 2
    assert _rows(store.run_dir / "results.superseded.jsonl") == old_rows
    # Analysis sees only results for the current settings.
    results = load_results(store.run_dir)
    assert len(results) == 2 and set(results["status"]) == {"max_iter_reached"}
    # The superseded attempts keep their logs; the new attempts write elsewhere.
    for path, text in old_artifacts.items():
        assert path.read_text() == text
    assert not {row["artifact_dir"] for row in rows} & {row["artifact_dir"] for row in old_rows}

    # Resuming again with the same settings solves nothing new.
    run_benchmark(_config(tmp_path, settings={"eps": 1e-8}), run_dir=store.run_dir,
                  repo_root=repo_root)
    assert len(fake_worker.calls) == 4
    assert len(_rows(store.results_jsonl_path)) == 2


def test_resume_retires_superseded_rows_next_to_a_current_result(
    fake_worker, tmp_path: Path, repo_root: Path
):
    """Older releases appended a new row and kept the stale one. A resume
    moves the stale row out without solving the problem again."""
    config = _config(tmp_path, settings={"eps": 1e-8})
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    current = _rows(store.results_jsonl_path)
    stale = [
        {**row, "status": "optimal", "metadata": {**row["metadata"], "resume_signature": "eps-1e-4"}}
        for row in current
    ]
    store.results_jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in stale + current)
    )

    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)

    assert len(fake_worker.calls) == 2
    assert _rows(store.results_jsonl_path) == current
    assert _rows(store.run_dir / "results.superseded.jsonl") == stale


@pytest.mark.parametrize(
    "dataset_options",
    [{"data_root": "/somewhere/else"}, {"max_size_mb": 100.0}],
)
def test_location_options_do_not_invalidate_results(
    fake_worker, tmp_path: Path, repo_root: Path, dataset_options
):
    store = run_benchmark(_config(tmp_path), run_dir=tmp_path / "run", repo_root=repo_root)
    run_benchmark(_config(tmp_path, dataset_options=dataset_options), run_dir=store.run_dir,
                  repo_root=repo_root)
    assert len(fake_worker.calls) == 2
    assert len(_rows(store.results_jsonl_path)) == 2


def test_rows_with_signatures_from_older_releases_stay_complete(
    fake_worker, tmp_path: Path, repo_root: Path
):
    """Older releases hashed data_root and max_size_mb into the signature."""
    config = _config(tmp_path, dataset_options={"data_root": str(tmp_path), "max_size_mb": 5.0})
    legacy = solve_signature(config, config.datasets[0], config.solvers[0], legacy=True)
    assert legacy != solve_signature(config, config.datasets[0], config.solvers[0])
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    for problem in ("p1", "p2"):
        store.write_result(
            ProblemResult(
                run_id=store.run_id, dataset="resume_fixture", problem=problem, problem_kind=QP,
                solver_id="s", solver="toggle", status="optimal", objective_value=0.0,
                iterations=1, run_time_seconds=0.01, metadata={"resume_signature": legacy},
            )
        )

    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)

    assert fake_worker.calls == []
    assert len(_rows(store.results_jsonl_path)) == 2
    assert not (store.run_dir / "results.superseded.jsonl").exists()


def _older_release_run(tmp_path: Path, dataset_options: dict) -> tuple[ResultStore, str]:
    """A run directory as earlier releases left it: rows carrying the legacy
    signature and a manifest without the signature mapping."""
    config = _config(tmp_path, dataset_options=dataset_options)
    legacy = solve_signature(config, config.datasets[0], config.solvers[0], legacy=True)
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    manifest = json.loads(store.manifest_path.read_text())
    del manifest[LEGACY_SIGNATURES_KEY]
    store.manifest_path.write_text(json.dumps(manifest))
    for problem in ("p1", "p2"):
        store.write_result(
            ProblemResult(
                run_id=store.run_id, dataset="resume_fixture", problem=problem, problem_kind=QP,
                solver_id="s", solver="toggle", status="optimal", objective_value=0.0,
                iterations=1, run_time_seconds=0.01, metadata={"resume_signature": legacy},
            )
        )
    return store, legacy


def test_rows_from_older_releases_survive_later_location_changes(
    fake_worker, tmp_path: Path, repo_root: Path
):
    """Raising max_size_mb to add problems, or pointing data_root elsewhere,
    keeps those rows, however many resumes later. The relative data_root
    those releases stored is now made absolute, which also changes the
    legacy signature."""
    store, legacy = _older_release_run(tmp_path, {"data_root": "mydata", "max_size_mb": 5.0})
    original = _rows(store.results_jsonl_path)

    for options in (
        {"data_root": "mydata", "max_size_mb": 100.0},
        {"data_root": "mydata", "max_size_mb": 200.0},
        {"data_root": str(tmp_path / "copy")},
    ):
        run_benchmark(_config(tmp_path, dataset_options=options), run_dir=store.run_dir,
                      repo_root=repo_root)
        assert fake_worker.calls == []
        assert _rows(store.results_jsonl_path) == original
        assert not (store.run_dir / "results.superseded.jsonl").exists()

    config = _config(tmp_path)
    current = solve_signature(config, config.datasets[0], config.solvers[0])
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest[LEGACY_SIGNATURES_KEY] == {legacy: current}

    # The mapping covers only the settings those rows were solved with.
    run_benchmark(_config(tmp_path, settings={"eps": 1e-8}), run_dir=store.run_dir,
                  repo_root=repo_root)
    assert len(fake_worker.calls) == 2
    assert _rows(store.run_dir / "results.superseded.jsonl") == original


def test_superseded_rows_move_out_in_one_rewrite_at_the_end_of_the_run(
    fake_worker, monkeypatch, tmp_path: Path, repo_root: Path
):
    store = run_benchmark(_config(tmp_path, settings={"eps": 1e-4}), run_dir=tmp_path / "run",
                          repo_root=repo_root)
    old_rows = _rows(store.results_jsonl_path)
    rewrites = []
    original_replace = ResultStore._replace_rows

    def counting_replace(self, rows, *, superseded, line):
        rewrites.append(len(rows))
        return original_replace(self, rows, superseded=superseded, line=line)

    monkeypatch.setattr(ResultStore, "_replace_rows", counting_replace)
    run_benchmark(_config(tmp_path, settings={"eps": 1e-8}), run_dir=store.run_dir,
                  repo_root=repo_root)
    assert rewrites == [2]
    assert len(_rows(store.results_jsonl_path)) == 2
    assert _rows(store.run_dir / "results.superseded.jsonl") == old_rows


def test_an_interrupted_run_still_moves_the_rows_it_replaced(
    fake_worker, monkeypatch, tmp_path: Path, repo_root: Path
):
    store = run_benchmark(_config(tmp_path, settings={"eps": 1e-4}), run_dir=tmp_path / "run",
                          repo_root=repo_root)
    old_rows = {row["problem"]: row for row in _rows(store.results_jsonl_path)}
    solve = runner_module._run_subprocess

    def interrupt_second(cmd, **kwargs):
        if len(fake_worker.calls) == 3:
            raise KeyboardInterrupt
        return solve(cmd, **kwargs)

    monkeypatch.setattr(runner_module, "_run_subprocess", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(_config(tmp_path, settings={"eps": 1e-8}), run_dir=store.run_dir,
                      repo_root=repo_root)
    rows = _rows(store.results_jsonl_path)
    assert [row["problem"] for row in rows] == ["p2", "p1"]
    assert rows[0] == old_rows["p2"]
    assert _rows(store.run_dir / "results.superseded.jsonl") == [old_rows["p1"]]

    monkeypatch.setattr(runner_module, "_run_subprocess", solve)
    run_benchmark(_config(tmp_path, settings={"eps": 1e-8}), run_dir=store.run_dir,
                  repo_root=repo_root)
    assert sorted(row["problem"] for row in _rows(store.results_jsonl_path)) == ["p1", "p2"]
    assert _rows(store.run_dir / "results.superseded.jsonl") == [old_rows["p1"], old_rows["p2"]]


def test_unavailable_solver_skips_are_solved_once_the_solver_is_installed(
    fake_worker, tmp_path: Path, repo_root: Path
):
    config = _config(tmp_path)
    _ToggleSolver.available = False
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    skips = _rows(store.results_jsonl_path)
    assert [row["status"] for row in skips] == ["skipped_unsupported"] * 2
    assert {row["metadata"]["skip_reason"] for row in skips} == {"solver_unavailable"}

    # Still unavailable: the skips stand and nothing is rewritten.
    original = store.results_jsonl_path.read_bytes()
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert store.results_jsonl_path.read_bytes() == original

    _ToggleSolver.available = True
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    rows = _rows(store.results_jsonl_path)
    assert len(fake_worker.calls) == 2
    assert [row["status"] for row in rows] == ["optimal"] * 2


def test_unavailable_skips_from_older_releases_are_recognised(
    fake_worker, tmp_path: Path, repo_root: Path
):
    config = _config(tmp_path)
    _ToggleSolver.available = False
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    rows = _rows(store.results_jsonl_path)
    for row in rows:
        del row["metadata"]["skip_reason"]
    store.results_jsonl_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    _ToggleSolver.available = True
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert [row["status"] for row in _rows(store.results_jsonl_path)] == ["optimal"] * 2


def _count_rewrites(monkeypatch) -> list[int]:
    rewrites: list[int] = []
    original_replace = ResultStore._replace_rows

    def counting_replace(self, rows, *, superseded, line):
        rewrites.append(len(rows))
        return original_replace(self, rows, superseded=superseded, line=line)

    monkeypatch.setattr(ResultStore, "_replace_rows", counting_replace)
    return rewrites


@pytest.mark.parametrize("replaced", ["worker_error", "unavailable_skip"])
def test_replacing_many_retry_or_skip_rows_rewrites_results_once(
    fake_worker, monkeypatch, tmp_path: Path, replaced: str
):
    config = _config(tmp_path)
    signature = solve_signature(config, config.datasets[0], config.solvers[0])
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    problems = [f"q{i:02d}" for i in range(40)]

    def result(problem, status, **metadata):
        return ProblemResult(
            run_id=store.run_id, dataset="resume_fixture", problem=problem, problem_kind=QP,
            solver_id="s", solver="toggle", status=status, objective_value=None,
            iterations=None, run_time_seconds=None,
            metadata={"resume_signature": signature, **metadata},
        )

    for problem in problems:
        if replaced == "worker_error":
            store.write_result(result(problem, "worker_error"))
        else:
            store.write_result(result(problem, "skipped_unsupported",
                                      skip_reason="solver_unavailable"))
    assert store.completed_keys(config=config, available_solver_ids={"s"}) == set()

    rewrites = _count_rewrites(monkeypatch)
    for problem in problems:
        store.write_result(result(problem, "optimal"))
    assert rewrites == []  # each result is appended
    store.retire_replaced_rows()
    assert rewrites == [len(problems)]  # one rewrite drops every replaced row
    rows = _rows(store.results_jsonl_path)
    assert [row["problem"] for row in rows] == problems
    assert {row["status"] for row in rows} == {"optimal"}
    assert not store.superseded_jsonl_path.exists()


@pytest.mark.parametrize("available_on_resume", [True, False])
def test_a_resume_drops_skip_rows_a_stopped_run_left_next_to_their_results(
    fake_worker, monkeypatch, tmp_path: Path, repo_root: Path, available_on_resume: bool
):
    config = _config(tmp_path)
    _ToggleSolver.available = False
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)

    # Killed after the results were written, before the end-of-run rewrite.
    _ToggleSolver.available = True
    retire = runner_module._retire_replaced_rows
    monkeypatch.setattr(runner_module, "_retire_replaced_rows", lambda store: None)
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    monkeypatch.setattr(runner_module, "_retire_replaced_rows", retire)
    assert len(fake_worker.calls) == 2

    _ToggleSolver.available = available_on_resume
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    rows = _rows(store.results_jsonl_path)
    assert len(fake_worker.calls) == 2
    assert sorted((row["problem"], row["status"]) for row in rows) == [
        ("p1", "optimal"), ("p2", "optimal")
    ]


def test_a_resume_drops_error_rows_a_stopped_run_left_next_to_their_retries(
    fake_worker, monkeypatch, tmp_path: Path, repo_root: Path
):
    config = _config(tmp_path)
    fake_worker.outcome.update(status="worker_error", run_time_seconds=None)
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    errors = _rows(store.results_jsonl_path)

    retire = runner_module._retire_replaced_rows
    monkeypatch.setattr(runner_module, "_retire_replaced_rows", lambda store: None)
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)  # second errors
    monkeypatch.setattr(runner_module, "_retire_replaced_rows", retire)

    fake_worker.outcome.update(status="optimal", run_time_seconds=0.01)
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    rows = _rows(store.results_jsonl_path)
    assert len(fake_worker.calls) == 6
    assert [(row["status"], row["metadata"]["worker_error_retries"]) for row in rows] == [
        ("optimal", 2), ("optimal", 2)
    ]
    # Every dropped error row is kept in a retry directory.
    kept = {
        line
        for path in store.run_dir.glob("problems/**/previous_results.jsonl")
        for line in path.read_text().splitlines()
    }
    assert {json.dumps(row, sort_keys=True) for row in errors} <= kept

    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert len(fake_worker.calls) == 6 and _rows(store.results_jsonl_path) == rows


def test_a_resume_drops_error_rows_a_stopped_successful_retry_left(
    fake_worker, monkeypatch, tmp_path: Path, repo_root: Path
):
    config = _config(tmp_path)
    fake_worker.outcome.update(status="worker_error", run_time_seconds=None)
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)

    # The retries succeed; the run is killed before its end-of-run rewrite.
    fake_worker.outcome.update(status="optimal", run_time_seconds=0.01)
    retire = runner_module._retire_replaced_rows
    monkeypatch.setattr(runner_module, "_retire_replaced_rows", lambda store: None)
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    monkeypatch.setattr(runner_module, "_retire_replaced_rows", retire)
    assert [row["status"] for row in _rows(store.results_jsonl_path)] == [
        "worker_error", "worker_error", "optimal", "optimal"
    ]

    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert len(fake_worker.calls) == 4  # nothing solved again
    rows = _rows(store.results_jsonl_path)
    assert sorted((row["problem"], row["status"]) for row in rows) == [
        ("p1", "optimal"), ("p2", "optimal")
    ]


def test_a_resume_keeps_the_run_id_so_readers_pair_its_rows(
    fake_worker, monkeypatch, tmp_path: Path, repo_root: Path
):
    """Readers pair a replaced row with its replacement only within one run
    id; the rows a resume appends carry the run directory's run id."""
    config = _config(tmp_path)
    fake_worker.outcome.update(status="worker_error", run_time_seconds=None)
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    fake_worker.outcome.update(status="optimal", run_time_seconds=0.01)
    monkeypatch.setattr(runner_module, "_retire_replaced_rows", lambda store: None)
    resumed = run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)

    rows = _rows(store.results_jsonl_path)
    assert len(rows) == 4 and resumed.run_id == store.run_id
    assert {row["run_id"] for row in rows} == {store.run_id}
    frame = load_results(store.run_dir)
    assert sorted(zip(frame["problem"], frame["status"])) == [("p1", "optimal"), ("p2", "optimal")]


@pytest.mark.parametrize("move", ["rename", "copy"])
def test_a_renamed_run_directory_keeps_its_run_id(
    fake_worker, monkeypatch, tmp_path: Path, repo_root: Path, move: str
):
    """The rows a resume writes into a renamed (or copied) run directory carry
    the run id of the rows they replace, so readers still pair them."""
    import shutil

    from click.testing import CliRunner

    from solver_benchmarks.cli import main

    config = _config(tmp_path)
    fake_worker.outcome.update(status="worker_error", run_time_seconds=None)
    store = run_benchmark(config, run_dir=tmp_path / "run_2026-11-16_12-00-00_UTC",
                          repo_root=repo_root)
    original = store.run_id
    moved = tmp_path / "renamed"
    if move == "rename":
        store.run_dir.rename(moved)
    else:
        shutil.copytree(store.run_dir, moved)

    # The retries succeed; the run is killed before its end-of-run rewrite.
    fake_worker.outcome.update(status="optimal", run_time_seconds=0.01)
    monkeypatch.setattr(runner_module, "_retire_replaced_rows", lambda store: None)
    resumed = run_benchmark(config, run_dir=moved, repo_root=repo_root)

    assert resumed.run_id == original
    rows = _rows(moved / "results.jsonl")
    assert len(rows) == 4 and {row["run_id"] for row in rows} == {original}
    assert json.loads((moved / "manifest.json").read_text())["run_id"] == original
    frame = load_results(moved)
    assert sorted(zip(frame["problem"], frame["status"])) == [("p1", "optimal"), ("p2", "optimal")]
    result = CliRunner().invoke(main, ["summary", str(moved), "--repo-root", str(repo_root)])
    assert result.exit_code == 0, result.output
    completion = result.output.split("Completion:")[1].splitlines()[2].split()
    assert completion[-2:] == ["0", "True"]


def test_a_run_directory_with_a_torn_manifest_resumes_under_its_name(
    fake_worker, tmp_path: Path, repo_root: Path
):
    config = _config(tmp_path)
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    store.manifest_path.write_text(store.manifest_path.read_text()[:50])  # torn

    resumed = run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert resumed.run_id == "run"
    assert json.loads(store.manifest_path.read_text())["run_id"] == "run"
    assert len(fake_worker.calls) == 2  # the rows still count


def test_unsupported_kind_skips_stay_complete(fake_worker, monkeypatch, tmp_path: Path,
                                              repo_root: Path):
    config = _config(tmp_path)
    monkeypatch.setattr(_ToggleSolver, "supported_problem_kinds", set())
    store = run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    original = store.results_jsonl_path.read_bytes()
    assert "skip_reason" not in _rows(store.results_jsonl_path)[0]["metadata"]
    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert store.results_jsonl_path.read_bytes() == original
    assert fake_worker.calls == []


def test_resume_false_refuses_a_run_directory_with_results(
    fake_worker, tmp_path: Path, repo_root: Path
):
    store = run_benchmark(_config(tmp_path), run_dir=tmp_path / "run", repo_root=repo_root)
    original = store.results_jsonl_path.read_bytes()
    manifest = store.manifest_path.read_bytes()
    with pytest.raises(RuntimeError, match="resume"):
        run_benchmark(_config(tmp_path, resume=False), run_dir=store.run_dir, repo_root=repo_root)
    assert store.results_jsonl_path.read_bytes() == original
    assert store.manifest_path.read_bytes() == manifest
    assert len(fake_worker.calls) == 2


def test_resume_false_still_adds_other_solvers_to_a_shared_run_directory(
    fake_worker, tmp_path: Path, repo_root: Path
):
    """Environment matrices run each environment's solvers into one directory."""
    store = run_benchmark(_config(tmp_path), run_dir=tmp_path / "run", repo_root=repo_root)
    other = parse_run_config(
        {
            "run": {"dataset": "resume_fixture", "resume": False},
            "solvers": [{"id": "other", "solver": "toggle"}],
        }
    )
    run_benchmark(other, run_dir=store.run_dir, repo_root=repo_root)
    assert [row["solver_id"] for row in _rows(store.results_jsonl_path)] == ["s", "s", "other", "other"]


def test_append_after_a_torn_line_starts_a_new_line(fake_worker, tmp_path: Path):
    config = _config(tmp_path)
    store = ResultStore.create(config, run_dir=tmp_path / "run")

    def result(problem):
        return ProblemResult(
            run_id=store.run_id, dataset="resume_fixture", problem=problem, problem_kind=QP,
            solver_id="s", solver="toggle", status="optimal", objective_value=0.0,
            iterations=1, run_time_seconds=0.01,
        )

    store.write_result(result("a"))
    with store.results_jsonl_path.open("a") as handle:
        handle.write('{"problem": "b", "solver_id": "s", "dat')  # killed mid-write
    store.write_result(result("b"))
    store.write_result(result("c"))
    assert [row["problem"] for row in load_results(store.run_dir).to_dict("records")] == [
        "a", "b", "c"
    ]


class _SubsetDataset(Dataset):
    """Three QPs and a ``subset`` option that only chooses which are listed."""

    dataset_id = "resume_subset_fixture"
    description = "three QPs, selected by name"
    option_names = frozenset({"subset", "scale"})
    selection_options = frozenset({"subset"})

    def list_problems(self):
        subset = self.options.get("subset")
        return [
            ProblemSpec(dataset_id="resume_subset_fixture", name=name, kind=QP)
            for name in ("p1", "p2", "p3")
            if subset is None or name in subset
        ]

    def load_problem(self, name):  # pragma: no cover - the worker is stubbed
        raise AssertionError


def _subset_config(tmp_path: Path, subset, **options):
    # max_size_mb makes the signature that hashed ``subset`` differ from the
    # first legacy form too, as for most bundled configs.
    return parse_run_config(
        {
            "run": {
                "dataset": "resume_subset_fixture",
                "output_dir": str(tmp_path / "runs"),
                "dataset_options": {"subset": subset, "max_size_mb": 5.0, **options},
            },
            "solvers": [{"id": "s", "solver": "toggle"}],
        }
    )


@pytest.fixture
def subset_dataset(monkeypatch, fake_worker):
    monkeypatch.setitem(dataset_registry.DATASETS, "resume_subset_fixture", _SubsetDataset)
    return fake_worker


def _solved(calls) -> list[str]:
    return sorted(call["problem"] for call in calls)


def test_widening_a_subset_solves_only_the_new_problems(
    subset_dataset, tmp_path: Path, repo_root: Path
):
    store = run_benchmark(_subset_config(tmp_path, ["p1"]), run_dir=tmp_path / "run",
                          repo_root=repo_root)
    first = _rows(store.results_jsonl_path)

    run_benchmark(_subset_config(tmp_path, ["p1", "p2"]), run_dir=store.run_dir,
                  repo_root=repo_root)

    assert _solved(subset_dataset.calls) == ["p1", "p2"]  # p1 once, in the first run
    assert [row["problem"] for row in _rows(store.results_jsonl_path)] == ["p1", "p2"]
    assert _rows(store.results_jsonl_path)[0] == first[0]
    assert not (store.run_dir / "results.superseded.jsonl").exists()


def test_an_option_that_changes_the_problem_still_supersedes(
    subset_dataset, tmp_path: Path, repo_root: Path
):
    store = run_benchmark(_subset_config(tmp_path, ["p1"], scale=1), run_dir=tmp_path / "run",
                          repo_root=repo_root)
    run_benchmark(_subset_config(tmp_path, ["p1"], scale=2), run_dir=store.run_dir,
                  repo_root=repo_root)
    assert _solved(subset_dataset.calls) == ["p1", "p1"]


def _run_before_selection_options_left_the_signature(tmp_path: Path, subset) -> ResultStore:
    """A run directory as the previous release left it: rows carrying the
    signature that still hashed ``subset``, and a manifest with the
    signature mapping of the release before (empty for a new directory)."""
    config = _subset_config(tmp_path, subset)
    old = solve_signature(config, config.datasets[0], config.solvers[0], legacy_selection=True)
    assert old != solve_signature(config, config.datasets[0], config.solvers[0])
    assert old != solve_signature(config, config.datasets[0], config.solvers[0], legacy=True)
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest[LEGACY_SIGNATURES_KEY] == {}
    for problem in subset:
        store.write_result(
            ProblemResult(
                run_id=store.run_id, dataset="resume_subset_fixture", problem=problem,
                problem_kind=QP, solver_id="s", solver="toggle", status="optimal",
                objective_value=0.0, iterations=1, run_time_seconds=0.01,
                metadata={"resume_signature": old},
            )
        )
    return store


def test_rows_that_hashed_the_subset_stay_complete(
    subset_dataset, tmp_path: Path, repo_root: Path
):
    store = _run_before_selection_options_left_the_signature(tmp_path, ["p1", "p2"])
    original = _rows(store.results_jsonl_path)

    run_benchmark(_subset_config(tmp_path, ["p1", "p2"]), run_dir=store.run_dir,
                  repo_root=repo_root)

    assert subset_dataset.calls == []
    assert _rows(store.results_jsonl_path) == original
    assert not (store.run_dir / "results.superseded.jsonl").exists()


def test_rows_that_hashed_the_subset_survive_widening_it(
    subset_dataset, tmp_path: Path, repo_root: Path
):
    store = _run_before_selection_options_left_the_signature(tmp_path, ["p1"])
    original = _rows(store.results_jsonl_path)

    run_benchmark(_subset_config(tmp_path, ["p1", "p2"]), run_dir=store.run_dir,
                  repo_root=repo_root)
    assert _solved(subset_dataset.calls) == ["p2"]
    # The manifest carries the mapping forward to later resumes.
    run_benchmark(_subset_config(tmp_path, ["p1", "p2", "p3"]), run_dir=store.run_dir,
                  repo_root=repo_root)

    assert _solved(subset_dataset.calls) == ["p2", "p3"]
    assert _rows(store.results_jsonl_path)[0] == original[0]
    assert not (store.run_dir / "results.superseded.jsonl").exists()


def test_selection_options_leave_the_signature_but_netlib_subset_does_not():
    def signature(dataset, options):
        config = parse_run_config(
            {
                "run": {"dataset": dataset, "dataset_options": options},
                "solvers": [{"id": "s", "solver": "scs"}],
            }
        )
        return solve_signature(config, config.datasets[0], config.solvers[0])

    # ``greenbea`` is in both NETLIB folders, as different problems.
    assert signature("netlib", {"subset": "feasible"}) != signature("netlib", {"subset": "infeasible"})
    assert signature("qplib", {"subset": ["8790"]}) == signature("qplib", {"subset": "8790,8515"})
    assert signature("sdplib", {"subset": "feasible"}) == signature("sdplib", {"subset": "all"})
    assert signature("sdplib", {"blocks": "merged"}) != signature("sdplib", {})
    assert signature("cblib", {"subset_kind": "lp", "include_unsupported": True}) == signature(
        "cblib", {}
    )
    assert signature("liu_pataki", {"constraint_count": 10, "classification": "weak"}) == signature(
        "liu_pataki", {}
    )
    assert signature("libsvm_qp", {"subset": "heart"}) == signature("libsvm_qp", {})
    assert signature("libsvm_qp", {"kind": "markowitz"}) != signature("libsvm_qp", {})
    assert signature("libsvm_qp", {"C": 10.0}) != signature("libsvm_qp", {})


def test_a_row_in_the_form_that_hashed_the_subset_counts_without_a_manifest_mapping(
    subset_dataset, tmp_path: Path, repo_root: Path
):
    """Each row is also checked against that earlier form of the current
    config, as for the first legacy form, when the manifest cannot say."""
    store = _run_before_selection_options_left_the_signature(tmp_path, ["p1"])
    manifest = json.loads(store.manifest_path.read_text())
    del manifest["config"]
    store.manifest_path.write_text(json.dumps(manifest))

    run_benchmark(_subset_config(tmp_path, ["p1"]), run_dir=store.run_dir, repo_root=repo_root)

    assert subset_dataset.calls == []
    assert not (store.run_dir / "results.superseded.jsonl").exists()


def _run_with_an_ignored_option(tmp_path: Path, ignored: dict) -> ResultStore:
    """A run directory the previous release wrote from a config with an option
    the dataset did not read (accepted then, rejected now): its manifest
    records the option and its rows carry the signature that hashed it."""
    from solver_benchmarks.core.config import manifest_solve_signatures

    store = ResultStore.create(_subset_config(tmp_path, ["p1"]), run_dir=tmp_path / "run")
    manifest = json.loads(store.manifest_path.read_text())
    manifest["config"]["datasets"][0]["dataset_options"].update(ignored)
    store.manifest_path.write_text(json.dumps(manifest))
    (old,) = manifest_solve_signatures(manifest["config"], legacy_selection=True).values()
    store.write_result(
        ProblemResult(
            run_id=store.run_id, dataset="resume_subset_fixture", problem="p1",
            problem_kind=QP, solver_id="s", solver="toggle", status="optimal",
            objective_value=0.0, iterations=1, run_time_seconds=0.01,
            metadata={"resume_signature": old},
        )
    )
    return store


def test_removing_an_option_the_dataset_ignored_keeps_its_rows(
    subset_dataset, tmp_path: Path, repo_root: Path
):
    store = _run_with_an_ignored_option(tmp_path, {"sclae": 2})

    run_benchmark(_subset_config(tmp_path, ["p1"]), run_dir=store.run_dir, repo_root=repo_root)

    assert subset_dataset.calls == []
    assert not (store.run_dir / "results.superseded.jsonl").exists()


def test_correcting_a_typo_to_an_option_that_changes_the_problem_supersedes(
    subset_dataset, tmp_path: Path, repo_root: Path
):
    store = _run_with_an_ignored_option(tmp_path, {"sclae": 2})

    run_benchmark(_subset_config(tmp_path, ["p1"], scale=2), run_dir=store.run_dir,
                  repo_root=repo_root)

    assert _solved(subset_dataset.calls) == ["p1"]
    assert len(_rows(store.run_dir / "results.superseded.jsonl")) == 1


# ---------------------------------------------------------------------------
# Readers while replaced rows are still in results.jsonl (a resume running, or
# killed before its end-of-run rewrite).
# ---------------------------------------------------------------------------


def _pair_run_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """A run directory holding replaced rows next to their replacements, and
    the same directory as the end-of-run rewrite leaves it."""
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "timeout_seconds": 10},
            "solvers": [{"id": "s", "solver": "scs"}, {"id": "t", "solver": "osqp"}],
        }
    )
    signature = {
        solver.id: solve_signature(config, config.datasets[0], solver) for solver in config.solvers
    }

    def row(problem, solver_id, status, signature_of=None, **metadata):
        solved = status == "optimal"
        return ProblemResult(
            run_id="r", dataset="synthetic_qp", problem=problem, problem_kind=QP,
            solver_id=solver_id, solver={"s": "scs", "t": "osqp"}[solver_id], status=status,
            objective_value=1.0 if solved else None, iterations=5 if solved else None,
            run_time_seconds=0.01 if solved else None,
            error=None if solved else f"{status} of {problem}",
            metadata={"resume_signature": signature_of or signature[solver_id], **metadata},
        ).to_record()

    rows = [
        (row("one_variable_eq", "s", "worker_error"), True),
        (row("one_variable_lp", "s", "skipped_unsupported", skip_reason="solver_unavailable"), True),
        # Solved under other settings: superseded handling, not replacement.
        (row("one_variable_eq", "t", "worker_error", signature_of="older-settings"), False),
        (row("one_variable_lp", "t", "optimal"), False),
        (row("one_variable_eq", "s", "optimal"), False),
        (row("one_variable_lp", "s", "worker_error"), False),
        (row("one_variable_eq", "t", "optimal"), False),
    ]
    dirs = []
    for name, keep_replaced in (("with_pairs", True), ("rewritten", False)):
        store = ResultStore.create(config, run_dir=tmp_path / name)
        store.results_jsonl_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n"
                    for record, replaced in rows if keep_replaced or not replaced)
        )
        dirs.append(store.run_dir)
    return dirs[0], dirs[1]


def test_load_results_leaves_out_replaced_rows(tmp_path: Path, monkeypatch):
    import pandas as pd

    from solver_benchmarks.core import storage

    with_pairs, rewritten = _pair_run_dirs(tmp_path)
    frame = load_results(with_pairs)
    pd.testing.assert_frame_equal(frame, load_results(rewritten))
    for chunk_lines in (1, 2, 3):
        pd.testing.assert_frame_equal(
            storage.read_results_jsonl(with_pairs / "results.jsonl", chunk_lines=chunk_lines),
            frame,
        )
    with monkeypatch.context() as patch:  # pandas without the chunked conversion
        patch.setattr(storage, "_convert_object_array", None)
        pd.testing.assert_frame_equal(
            storage.read_results_jsonl(with_pairs / "results.jsonl"),
            storage.read_results_jsonl(rewritten / "results.jsonl"),
        )
    assert len(frame) == 5
    assert sorted(zip(frame["problem"], frame["solver_id"], frame["status"])) == [
        ("one_variable_eq", "s", "optimal"),
        ("one_variable_eq", "t", "optimal"),
        ("one_variable_eq", "t", "worker_error"),
        ("one_variable_lp", "s", "worker_error"),
        ("one_variable_lp", "t", "optimal"),
    ]


def test_load_results_reads_one_state_of_a_file_a_resume_is_writing(tmp_path: Path, monkeypatch):
    """Rows left out are found in a first read and the frame built in a
    second; rows appended, or the file replaced, in between change neither."""
    import pandas as pd

    from solver_benchmarks.core import storage

    with_pairs, rewritten = _pair_run_dirs(tmp_path)
    path = with_pairs / "results.jsonl"
    expected = load_results(rewritten)
    first_read = storage._read_results_jsonl_chunked
    calls = []

    def read_then_write(*args, **kwargs):
        frame = first_read(*args, **kwargs)
        if not calls:
            last = json.loads(path.read_text().splitlines()[-1])
            with path.open("a") as handle:  # a resume appends a replacement
                handle.write(json.dumps({**last, "status": "optimal"}, sort_keys=True) + "\n")
            replacement = tmp_path / "rewrite.jsonl"
            replacement.write_text(path.read_text().splitlines()[-1] + "\n")
            os.replace(replacement, path)  # and rewrites the file at its end
        calls.append(kwargs)
        return frame

    monkeypatch.setattr(storage, "_read_results_jsonl_chunked", read_then_write)
    pd.testing.assert_frame_equal(storage.read_results_jsonl(path), expected)
    assert len(calls) == 2


def test_cli_readers_see_the_rows_the_end_of_run_rewrite_leaves(tmp_path: Path, repo_root: Path):
    from click.testing import CliRunner

    from solver_benchmarks.cli import main

    with_pairs, rewritten = _pair_run_dirs(tmp_path)
    runner = CliRunner()

    def output(*args):
        result = runner.invoke(main, [*args, "--repo-root", str(repo_root)])
        assert result.exit_code == 0, result.output
        return result.output.replace(str(with_pairs), "RUN").replace(str(rewritten), "RUN")

    for command in ("summary", "failures"):
        assert output(command, str(with_pairs)) == output(command, str(rewritten))
    summary = output("summary", str(with_pairs))
    completion = {
        line.split()[0]: line.split()
        for line in summary.split("Completion:")[1].splitlines()[2:] if line.strip()
    }
    # duplicate_rows 0, complete True for the solver with replaced rows; the
    # other-settings pair of t is superseded handling and stays as it is.
    assert completion["s"][-2:] == ["0", "True"]
    assert completion["t"][-2:] == ["1", "False"]
    assert "worker_error" in output("failures", str(with_pairs))

    for run in (with_pairs, rewritten):
        output("report", str(run), "--output-dir", str(tmp_path / f"report_{run.name}"))
    def report_of(name: str) -> str:
        text = (tmp_path / f"report_{name}" / "index.md").read_text().replace(name, "RUN")
        return re.sub(r"\d{4}-\d\d-\d\dT[0-9:.]+\+00:00", "TIME", text)

    report = report_of("with_pairs")
    assert report == report_of("rewritten")
    assert "and 1 duplicate row entries" in report  # only the other-settings pair of t

    def derived_rows(path: Path) -> list[dict]:
        return [
            {key: value for key, value in row.items() if key != "run_id"}
            for row in _rows(path / "results.jsonl")
        ]

    for run in (with_pairs, rewritten):
        result = runner.invoke(main, ["merge", str(tmp_path / f"merged_{run.name}"), str(run)])
        assert result.exit_code == 0, result.output
        result = runner.invoke(main, ["kkt-verify", str(run), "--tol", "1e-3", "--output-dir",
                                      str(tmp_path / f"kkt_{run.name}")])
        assert result.exit_code == 0, result.output
    for kind in ("merged", "kkt"):
        assert derived_rows(tmp_path / f"{kind}_with_pairs") == derived_rows(
            tmp_path / f"{kind}_rewritten")
        assert len(derived_rows(tmp_path / f"{kind}_with_pairs")) == 5
