"""Resume keeps one row per planned solve, and that row matches the config.

A row whose resume signature no longer matches the config is superseded:
it stays until the new result is durable, then moves to
``results.superseded.jsonl``. Rows written by older releases, whose
signatures still hashed the dataset's location options, keep counting.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.core.config import parse_run_config, solve_signature
from solver_benchmarks.core.problem import QP, ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.core.storage import ResultStore
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

    def run(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
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
