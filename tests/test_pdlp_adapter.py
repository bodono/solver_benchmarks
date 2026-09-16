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


@pytest.mark.parametrize("serialized_api", [False, True])
@pytest.mark.parametrize("limit_delta", [-1, 0, 1])
def test_pdlp_request_size_boundary(monkeypatch, tmp_path, serialized_api, limit_delta):
    """Include the envelope and varint length, and protect both OR-Tools APIs."""
    linear_solver_pb2 = pytest.importorskip("ortools.linear_solver.linear_solver_pb2")
    solve_log_pb2 = pytest.importorskip("ortools.pdlp.solve_log_pb2")
    from google.protobuf import text_format

    from solver_benchmarks.core import status
    from solver_benchmarks.solvers import pdlp_adapter as mod

    model = linear_solver_pb2.MPModelProto(name="x" * 128)
    expected_request = linear_solver_pb2.MPModelRequest(
        model=model,
        enable_internal_solver_output=False,
        solver_type=linear_solver_pb2.MPModelRequest.PDLP_LINEAR_PROGRAMMING,
        solver_specific_parameters=text_format.MessageToString(
            mod._pdlp_parameters_from_settings({})
        ),
    )
    request_bytes = expected_request.ByteSize()
    monkeypatch.setattr(mod, "MAX_PROTOBUF_BYTES", request_bytes + limit_delta)
    response = linear_solver_pb2.MPSolutionResponse(
        solver_specific_info=solve_log_pb2.SolveLog(
            termination_reason=solve_log_pb2.TERMINATION_REASON_OPTIMAL
        ).SerializeToString(),
    )
    calls = []

    def solve(request):
        calls.append(request)
        assert request == expected_request
        return response

    def solve_serialized(data):
        solve(linear_solver_pb2.MPModelRequest.FromString(data))
        return response.SerializeToString()

    helper = types.SimpleNamespace(
        **({"solve_serialized_request": solve_serialized} if serialized_api else {"Solve": solve})
    )
    monkeypatch.setattr(
        mod, "_import_model_builder_helper",
        lambda: types.SimpleNamespace(ModelSolverHelper=lambda: helper),
    )
    result = mod._solve_model(model, {}, tmp_path)
    if limit_delta < 0:
        assert result.status == status.SKIPPED_UNSUPPORTED
        assert result.info["protobuf_request_bytes"] == request_bytes
        assert result.info["protobuf_model_bytes"] == model.ByteSize()
        assert result.info["protobuf_limit_bytes"] == request_bytes - 1
        assert not calls
    else:
        assert result.status == status.OPTIMAL
        assert len(calls) == 1


def test_pdlp_oversized_model_is_not_copied_or_serialized(monkeypatch, tmp_path):
    pytest.importorskip("ortools.linear_solver.linear_solver_pb2")
    from solver_benchmarks.core import status
    from solver_benchmarks.solvers import pdlp_adapter as mod

    monkeypatch.setattr(mod, "_import_model_builder_helper", lambda: None)
    # Not a protobuf message: any attempt to copy/serialize it would fail.
    model = types.SimpleNamespace(ByteSize=lambda: 1 << 31)
    result = mod._solve_model(model, {}, tmp_path)
    assert result.status == status.SKIPPED_UNSUPPORTED
    assert "protobuf" in result.info["reason"]
    assert result.info["protobuf_request_bytes"] > result.info["protobuf_model_bytes"]


@pytest.mark.parametrize("serialized_api", [False, True])
def test_pdlp_does_not_mask_unrelated_decode_errors(monkeypatch, tmp_path, serialized_api):
    linear_solver_pb2 = pytest.importorskip("ortools.linear_solver.linear_solver_pb2")
    from google.protobuf.message import DecodeError

    from solver_benchmarks.solvers import pdlp_adapter as mod

    def broken_solve(_request):
        raise DecodeError("malformed solver response")

    helper = types.SimpleNamespace(
        **({"solve_serialized_request": broken_solve} if serialized_api else {"Solve": broken_solve})
    )
    monkeypatch.setattr(
        mod, "_import_model_builder_helper",
        lambda: types.SimpleNamespace(ModelSolverHelper=lambda: helper),
    )
    with pytest.raises(DecodeError, match="malformed solver response"):
        mod._solve_model(linear_solver_pb2.MPModelProto(), {}, tmp_path)
