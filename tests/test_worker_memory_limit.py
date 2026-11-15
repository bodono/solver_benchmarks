"""``run.worker_memory_limit_mb`` caps each solve's worker with RLIMIT_DATA.

The option goes from the config into the worker payload, and the worker
sets the limit before it loads the problem. A solve that needs more memory
than the cap fails, and its row is a ``worker_error``.
"""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

import pytest

from solver_benchmarks import worker
from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.core import status
from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.solvers import get_solver


def _config(tmp_path: Path, limit_mb):
    return parse_run_config({
        "run": {
            "dataset": "synthetic_qp",
            "include": ["one_variable_lp"],
            "output_dir": str(tmp_path / "runs"),
            "timeout_seconds": 60,
            "worker_memory_limit_mb": limit_mb,
        },
        "solvers": [{"id": "clarabel", "solver": "clarabel", "settings": {"verbose": False}}],
    })


def test_limit_memory_sets_rlimit_data(monkeypatch):
    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limits: calls.append((kind, limits)))

    worker._limit_memory(64)

    assert calls == [(resource.RLIMIT_DATA, (64 * 1024 * 1024, 64 * 1024 * 1024))]


def test_worker_applies_the_payload_limit_before_solving(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(worker, "_limit_memory", lambda limit: order.append(("limit", limit)))

    class _Result:
        def to_record(self):
            return {}

    def fake_run_payload(payload):
        order.append(("solve", payload["problem"]))
        return _Result()

    monkeypatch.setattr(worker, "run_payload", fake_run_payload)
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"memory_limit_mb": 256, "problem": "p", "artifacts_dir": str(tmp_path)}))

    assert worker.main(["--payload", str(payload)]) == 0
    assert order == [("limit", 256), ("solve", "p")]


@pytest.mark.skipif(sys.platform != "linux", reason="only Linux enforces RLIMIT_DATA for all allocations")
def test_a_worker_over_its_memory_limit_is_a_worker_error(tmp_path, repo_root):
    if not get_solver("clarabel").is_available():
        pytest.skip("clarabel is not installed")
    store = run_benchmark(_config(tmp_path, 1), repo_root=repo_root)
    (row,) = load_results(store.run_dir).to_dict("records")
    payload = json.loads(next(Path(store.run_dir).rglob("payload.json")).read_text())

    assert payload["memory_limit_mb"] == 1
    assert row["status"] == status.WORKER_ERROR, row


def test_a_generous_memory_limit_lets_the_solve_finish(tmp_path, repo_root):
    if not get_solver("clarabel").is_available():
        pytest.skip("clarabel is not installed")
    store = run_benchmark(_config(tmp_path, 8192), repo_root=repo_root)
    (row,) = load_results(store.run_dir).to_dict("records")

    assert row["status"] == status.OPTIMAL, row
