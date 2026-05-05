"""Pin the ThreadPoolExecutor branch of run_benchmark.

Every existing runner test uses parallelism=1, so the parallel branch
was unexercised. This test stubs out the worker subprocess with a
sleep-and-write fake adapter and runs many tasks across multiple
worker threads, then asserts that:

- All results land in results.jsonl with exactly one record each.
- No two records share the same (problem, solver_id).
- The on-disk JSONL has no torn / interleaved bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.problem import QP, ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.solvers import registry as solver_registry
from solver_benchmarks.solvers.base import SolverAdapter

PROBLEM_COUNT = 8


class _ManyProblemDataset:
    def __init__(self, repo_root=None, **options):
        pass

    def list_problems(self):
        return [
            ProblemSpec(dataset_id="parallel_fixture", name=f"p{i:03d}", kind=QP)
            for i in range(PROBLEM_COUNT)
        ]


class _StubSolverAdapter(SolverAdapter):
    solver_name = "parallel_stub"
    supported_problem_kinds = {QP}

    def solve(self, problem, artifacts_dir):  # pragma: no cover - subprocess stub
        raise AssertionError("Subprocess is stubbed in this test")


def _fake_run_subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
    """Pretend to invoke the worker by reading the payload and writing
    a deterministic ProblemResult-shaped JSON next to it."""
    payload = json.loads(Path(cmd[-1]).read_text())
    artifact_dir = Path(payload["artifacts_dir"])
    record = ProblemResult(
        run_id=payload["run_id"],
        dataset=payload["dataset"],
        problem=payload["problem"],
        problem_kind=payload["problem_kind"],
        solver_id=payload["solver"]["id"],
        solver=payload["solver"]["solver"],
        status="optimal",
        objective_value=0.0,
        iterations=1,
        run_time_seconds=0.001,
        artifact_dir=str(artifact_dir),
    ).to_record()
    (artifact_dir / "worker_result.json").write_text(json.dumps(record))
    stdout_path.write_text("")
    stderr_path.write_text("")
    return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)


@pytest.mark.parametrize("parallelism", [2, 4])
def test_parallel_run_writes_one_record_per_solve_no_torn_lines(
    monkeypatch, tmp_path: Path, repo_root: Path, parallelism: int
):
    monkeypatch.setitem(
        dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset
    )
    monkeypatch.setitem(
        solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter
    )
    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", _fake_run_subprocess
    )

    config = parse_run_config(
        {
            "run": {
                "dataset": "parallel_fixture",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": parallelism,
            },
            "solvers": [
                {"id": "parallel_stub", "solver": "parallel_stub", "settings": {}}
            ],
        }
    )
    store = run_benchmark(config, repo_root=repo_root)

    text = store.results_jsonl_path.read_text()
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == PROBLEM_COUNT

    records = []
    for line in lines:
        # Every line must be parseable JSON; a torn append from
        # concurrent writes would surface here as a JSONDecodeError.
        records.append(json.loads(line))

    keys = {(r["dataset"], r["problem"], r["solver_id"]) for r in records}
    assert len(keys) == PROBLEM_COUNT  # no duplicates and no losses
    assert all(r["status"] == "optimal" for r in records)


def test_parallel_run_writes_parquet_at_end(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """write_result only appends to jsonl during the run; the runner
    must call store.write_parquet() at the end so the parquet reflects
    the complete jsonl."""
    monkeypatch.setitem(
        dataset_registry.DATASETS, "parallel_fixture", _ManyProblemDataset
    )
    monkeypatch.setitem(
        solver_registry.SOLVERS, "parallel_stub", _StubSolverAdapter
    )
    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", _fake_run_subprocess
    )

    config = parse_run_config(
        {
            "run": {
                "dataset": "parallel_fixture",
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 4,
            },
            "solvers": [
                {"id": "parallel_stub", "solver": "parallel_stub", "settings": {}}
            ],
        }
    )
    store = run_benchmark(config, repo_root=repo_root)

    import pandas as pd

    df = pd.read_parquet(store.results_parquet_path)
    assert len(df) == PROBLEM_COUNT
