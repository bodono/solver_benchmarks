from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult
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


def test_pdlp_oversized_model_skips_before_request_copy(monkeypatch, tmp_path):
    pytest.importorskip("ortools.linear_solver.linear_solver_pb2")
    from solver_benchmarks.core import status
    from solver_benchmarks.solvers import pdlp_adapter as mod

    monkeypatch.setattr(mod, "_import_model_builder_helper", lambda: None)
    # Not a protobuf message: attempting to copy it into the request would fail.
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


def _lp_problem(kind, a=None):
    if a is None:
        a = sp.csc_matrix([[1.0, 0.0], [2.0, 3.0]])
    data = {"A": a, "q": np.zeros(a.shape[1])}
    if kind == QP:
        data.update(P=sp.csc_matrix((a.shape[1], a.shape[1])),
                    l=np.zeros(a.shape[0]), u=np.ones(a.shape[0]))
    else:
        data.update(b=np.ones(a.shape[0]), cone={"z": 1, "l": a.shape[0] - 1})
    return ProblemData(dataset_id="test", name="lp", kind=kind, data=data)


@pytest.mark.parametrize("kind", [QP, CONE])
@pytest.mark.parametrize("limit_delta", [-1, 0, 1])
def test_pdlp_preflight_skips_before_model_build(monkeypatch, tmp_path, kind, limit_delta):
    from solver_benchmarks.solvers import pdlp_adapter as mod

    # Three stored coefficients, two variables and two constraints. Setting a
    # small boundary exercises the real early return without a GiB allocation.
    lower_bound = 117
    monkeypatch.setattr(mod, "MAX_PROTOBUF_BYTES", lower_bound + limit_delta)
    monkeypatch.setattr(mod, "_import_ortools", lambda: None)
    builds = []

    def build(data):
        builds.append(data)
        assert sp.issparse(data["A"])
        return object()

    builder = "_build_lp_model_from_qp" if kind == QP else "_build_lp_model_from_linear_cone"
    monkeypatch.setattr(mod, builder, build)
    solved = SolverResult(status=status.OPTIMAL)
    monkeypatch.setattr(mod, "_solve_model", lambda *_args: solved)
    result = mod.PDLPSolverAdapter({}).solve(_lp_problem(kind), tmp_path)
    if limit_delta < 0:
        assert not builds
        assert result.status == status.SKIPPED_UNSUPPORTED
        assert result.info["protobuf_model_bytes_lower_bound"] == lower_bound
        assert result.info["protobuf_limit_bytes"] == lower_bound - 1
        assert "protobuf_model_bytes" not in result.info  # No exact size was computed.
    else:
        assert len(builds) == 1
        assert result is solved


@pytest.mark.parametrize("kind", [QP, CONE])
def test_pdlp_oversized_canonical_input_skips_before_sparse_copy(monkeypatch, tmp_path, kind):
    from solver_benchmarks.solvers import pdlp_adapter as mod

    problem = _lp_problem(kind)
    monkeypatch.setattr(mod, "MAX_PROTOBUF_BYTES", 116)
    monkeypatch.setattr(mod, "_import_ortools", lambda: None)
    monkeypatch.setattr(mod.sp, "csr_matrix", lambda *_args: pytest.fail("unnecessary sparse copy"))
    assert mod.PDLPSolverAdapter({}).solve(problem, tmp_path).status == status.SKIPPED_UNSUPPORTED


@pytest.mark.parametrize("kind", [QP, CONE])
@pytest.mark.parametrize("matrix_format", ["csr", "csc", "coo", "dense"])
def test_pdlp_preflight_is_a_lower_bound_on_actual_proto(monkeypatch, tmp_path, kind, matrix_format):
    pytest.importorskip("ortools.linear_solver.linear_solver_pb2")
    from solver_benchmarks.solvers import pdlp_adapter as mod

    # Include explicit zeros and repeated coordinates, which CSR conversion may
    # coalesce. Zero objective/bound doubles must still be present in proto2.
    a = sp.coo_matrix(([0.0, 2.0, -2.0, 4.0], ([0, 0, 0, 1], [0, 1, 1, 1])), shape=(3, 2))
    a = a.toarray() if matrix_format == "dense" else a.asformat(matrix_format)
    problem = _lp_problem(kind, a)
    if kind == QP:
        problem.data["l"][0], problem.data["u"][0] = -np.inf, np.inf
    preflight = mod._preflight_model_size
    bounds = []

    def capture_bound(n, m, nnz):
        with monkeypatch.context() as patch:
            patch.setattr(mod, "MAX_PROTOBUF_BYTES", 0)
            bounds.append(preflight(n, m, nnz).info["protobuf_model_bytes_lower_bound"])
        return None

    def check_model(model, *_args):
        assert 0 < bounds[0] <= len(model.SerializeToString())
        for variable in model.variable:
            assert variable.HasField("objective_coefficient")
        return SolverResult(status=status.OPTIMAL)

    monkeypatch.setattr(mod, "_preflight_model_size", capture_bound)
    monkeypatch.setattr(mod, "_import_ortools", lambda: None)
    monkeypatch.setattr(mod, "_solve_model", check_model)
    assert mod.PDLPSolverAdapter({}).solve(problem, tmp_path).status == status.OPTIMAL
    assert len(bounds) == 1


@pytest.mark.parametrize("case", ["unbounded_qp_rows", "coo_duplicates"])
def test_pdlp_preflight_does_not_count_entries_omitted_by_builder(monkeypatch, tmp_path, case):
    pytest.importorskip("ortools.linear_solver.linear_solver_pb2")
    from solver_benchmarks.solvers import pdlp_adapter as mod

    if case == "unbounded_qp_rows":
        problem = _lp_problem(QP, sp.csc_matrix(np.ones((100, 2))))
        problem.data["l"][:] = -mod.INF_BOUND
        problem.data["u"][:] = mod.INF_BOUND
        model = mod._build_lp_model_from_qp(problem.data)
        assert not model.constraint
    else:
        a = sp.coo_matrix((np.ones(200), (np.zeros(200, dtype=int), np.zeros(200, dtype=int))))
        problem = _lp_problem(CONE, a)
        model = mod._build_lp_model_from_linear_cone(problem.data)
        assert len(model.constraint[0].coefficient) == 1
    limit = model.ByteSize() + 64
    assert 9 * problem.data["A"].nnz > limit  # A raw nnz bound would reject this model.
    monkeypatch.setattr(mod, "MAX_PROTOBUF_BYTES", limit)
    monkeypatch.setattr(mod, "_import_ortools", lambda: None)
    monkeypatch.setattr(mod, "_solve_model", lambda *_args: SolverResult(status=status.OPTIMAL))
    assert mod.PDLPSolverAdapter({}).solve(problem, tmp_path).status == status.OPTIMAL
