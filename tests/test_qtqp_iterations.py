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
