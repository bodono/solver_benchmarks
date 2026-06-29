from __future__ import annotations

import sys
import types

import pytest

from solver_benchmarks.solvers.base import SolverUnavailable


def test_pdlp_binary_import_error_marks_solver_unavailable(monkeypatch):
    import solver_benchmarks.solvers.pdlp_adapter as pdlp_mod

    _install_fake_ortools_modules(monkeypatch)

    def broken_helper():
        raise ImportError("undefined symbol: setLocalOptionValue")

    monkeypatch.setattr(pdlp_mod, "_import_model_builder_helper", broken_helper)

    assert pdlp_mod.PDLPSolverAdapter.is_available() is False
    with pytest.raises(SolverUnavailable, match="OR-Tools could not be imported"):
        pdlp_mod._import_ortools()


def _install_fake_ortools_modules(monkeypatch) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    protobuf = types.ModuleType("google.protobuf")
    google.protobuf = protobuf

    ortools = types.ModuleType("ortools")
    ortools.__version__ = "9.15.6755"
    ortools.__path__ = []

    linear_solver = types.ModuleType("ortools.linear_solver")
    linear_solver.__path__ = []
    linear_solver_pb2 = types.ModuleType("ortools.linear_solver.linear_solver_pb2")
    linear_solver.linear_solver_pb2 = linear_solver_pb2

    pdlp = types.ModuleType("ortools.pdlp")
    pdlp.__path__ = []
    solve_log_pb2 = types.ModuleType("ortools.pdlp.solve_log_pb2")
    solvers_pb2 = types.ModuleType("ortools.pdlp.solvers_pb2")
    pdlp.solve_log_pb2 = solve_log_pb2
    pdlp.solvers_pb2 = solvers_pb2

    ortools.linear_solver = linear_solver
    ortools.pdlp = pdlp

    modules = {
        "google": google,
        "google.protobuf": protobuf,
        "ortools": ortools,
        "ortools.linear_solver": linear_solver,
        "ortools.linear_solver.linear_solver_pb2": linear_solver_pb2,
        "ortools.pdlp": pdlp,
        "ortools.pdlp.solve_log_pb2": solve_log_pb2,
        "ortools.pdlp.solvers_pb2": solvers_pb2,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
