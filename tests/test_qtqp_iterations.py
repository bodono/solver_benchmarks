"""The qtqp adapter reports completed IPM steps, not the zero-based label."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.solvers.qtqp_adapter import QTQPSolverAdapter


def _install_fake_qtqp(monkeypatch, solution_fields: dict) -> None:
    rows = [{"iter": k, "pcost": 1.0} for k in range(4)]

    class FakeQTQP:
        def __init__(self, a, b, c, z, p):
            self.n, self.m = a.shape[1], a.shape[0]

        def solve(self, **kwargs):
            return SimpleNamespace(
                status=SimpleNamespace(value="solved"),
                stats=rows,
                x=np.zeros(self.n),
                y=np.zeros(self.m),
                s=np.zeros(self.m),
                **solution_fields,
            )

    monkeypatch.setitem(sys.modules, "qtqp", SimpleNamespace(QTQP=FakeQTQP))


def _problem() -> ProblemData:
    data = {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([1.0, 1.0]),
        "A": sp.csc_matrix(np.eye(2)),
        "l": np.array([-5.0, -5.0]),
        "u": np.array([5.0, 5.0]),
        "n": 2,
        "m": 2,
        "obj_type": "min",
    }
    return ProblemData("test", "p", QP, data)


@pytest.mark.parametrize(
    ("solution_fields", "expected"),
    [
        # qtqp with Solution.iterations: four completed steps, labels 0..3.
        ({"iterations": 4}, 4),
        # Older qtqp without the field: the zero-based label of the last row.
        ({}, 3),
    ],
)
def test_qtqp_iterations_prefer_completed_steps(
    monkeypatch, tmp_path, solution_fields, expected
):
    _install_fake_qtqp(monkeypatch, solution_fields)
    result = QTQPSolverAdapter({}).solve(_problem(), tmp_path)
    assert result.iterations == expected


def _install_recording_qtqp(monkeypatch, calls: list, x) -> None:
    class FakeQTQP:
        def __init__(self, a, b, c, z, p):
            self.n, self.m = a.shape[1], a.shape[0]

        def solve(self, max_iter=100, collect_stats=False, **kwargs):
            calls.append({"collect_stats": collect_stats, **kwargs})
            rows = [{"iter": 0, "pcost": 123.0}] if collect_stats else []
            return SimpleNamespace(
                status=SimpleNamespace(value="solved"),
                stats=rows,
                iterations=1,
                x=np.asarray(x, dtype=float),
                y=np.zeros(self.m),
                s=np.ones(self.m),
            )

    monkeypatch.setitem(sys.modules, "qtqp", SimpleNamespace(QTQP=FakeQTQP))


def test_qtqp_does_not_collect_stats_inside_the_timed_solve(monkeypatch, tmp_path):
    # Per-iteration statistics are harness instrumentation (2-5 ms per solve
    # on small problems); they are collected only on request.
    calls: list = []
    _install_recording_qtqp(monkeypatch, calls, [-1.0, -1.0])
    result = QTQPSolverAdapter({}).solve(_problem(), tmp_path)
    assert calls == [{}] or calls[0]["collect_stats"] is False
    # The objective comes from the returned point, not from a trace row.
    assert result.objective_value == pytest.approx(-1.0)
    assert result.trace == []
    assert not (tmp_path / "trace.jsonl").exists()


def test_qtqp_collects_stats_when_asked(monkeypatch, tmp_path):
    calls: list = []
    _install_recording_qtqp(monkeypatch, calls, [-1.0, -1.0])
    result = QTQPSolverAdapter({"collect_stats": True}).solve(_problem(), tmp_path)
    assert calls[0]["collect_stats"] is True
    assert result.trace == [{"iter": 0, "pcost": 123.0}]
    assert (tmp_path / "trace.jsonl").exists()
    assert result.objective_value == pytest.approx(-1.0)
