"""The GPU part of the warm-up before a solve's timed region (core/gpu_warmup.py,
run by core/warmup.py for a solve whose settings select a GPU).

The CUDA driver and cuBLAS are replaced by fakes that record their calls, so
these tests run without a GPU; the real libraries are exercised by hand on a
GPU machine (see the module docstring).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from solver_benchmarks.core import gpu_warmup, status
from solver_benchmarks.core.problem import QP
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.solvers import get_solver, list_solvers
from solver_benchmarks.solvers.base import SolverAdapter, WarmupModules
from solver_benchmarks.solvers.registry import SOLVERS
from solver_benchmarks.worker import run_payload

CUBLAS = "/opt/cuda/lib64/libcublas.so.13.1.1.3"
# How long the fake driver's cuInit takes: the stand-in for the context.
CONTEXT_SECONDS = 0.2


def _maps_line(path: str) -> str:
    return f"7f0000000000-7f0000001000 r-xp 00000000 103:02 1234567                    {path}\n"


class FakeDriver:
    """The CUDA driver API calls the warm-up makes; each one is recorded."""

    def __init__(self, events: list, *, fail: str | None = None, delay: float = 0.0):
        self.events = events
        self.fail = fail
        self.delay = delay

    def _call(self, name: str) -> int:
        self.events.append(name)
        return 100 if name == self.fail else 0  # 100: CUDA_ERROR_NO_DEVICE

    def cuInit(self, flags):
        time.sleep(self.delay)
        return self._call("cuInit")

    def cuDeviceGet(self, device, ordinal):
        device._obj.value = 0
        return self._call("cuDeviceGet")

    def cuDevicePrimaryCtxRetain(self, context, device):
        context._obj.value = 1
        return self._call("cuDevicePrimaryCtxRetain")

    def cuCtxSetCurrent(self, context):
        return self._call("cuCtxSetCurrent")

    def cuMemAlloc_v2(self, pointer, size):
        pointer._obj.value = 1 << 20
        return self._call("cuMemAlloc")

    def cuMemsetD8_v2(self, pointer, value, size):
        return self._call("cuMemsetD8")

    def cuCtxSynchronize(self):
        return self._call("cuCtxSynchronize")

    def cuMemFree_v2(self, pointer):
        return self._call("cuMemFree")

    def cuGetErrorName(self, result, name):
        name._obj.value = b"CUDA_ERROR_NO_DEVICE"
        return 0


class FakeCublas:
    """cuBLAS's handle, gemm and gemv calls, recorded with their operations."""

    def __init__(self, events: list, *, fail: str | None = None):
        self.events = events
        self.fail = fail

    def cublasCreate_v2(self, handle):
        handle._obj.value = 7
        self.events.append("cublasCreate")
        return 0

    def cublasDgemm_v2(self, handle, op_a, op_b, *args):
        self.events.append(("gemm", op_a.value, op_b.value))
        return 13 if self.fail == "gemm" else 0  # CUBLAS_STATUS_EXECUTION_FAILED

    def cublasDgemv_v2(self, handle, op, *args):
        self.events.append(("gemv", op.value))
        return 0

    def cublasDestroy_v2(self, handle):
        self.events.append("cublasDestroy")
        return 0


@pytest.fixture
def fake_gpu(monkeypatch, tmp_path):
    """A process with a CUDA driver and the cuBLAS at CUBLAS mapped (fakes)."""

    class Fake:
        events: list = []
        maps = tmp_path / "maps"
        driver_fail: str | None = None
        cublas_fail: str | None = None
        delay = 0.0

    Fake.maps.write_text(_maps_line("/usr/lib/libc.so.6") + _maps_line(CUBLAS))

    def open_library(name):
        if name == gpu_warmup.CUDA_DRIVER_LIBRARY:
            return FakeDriver(Fake.events, fail=Fake.driver_fail, delay=Fake.delay)
        if name == CUBLAS:
            return FakeCublas(Fake.events, fail=Fake.cublas_fail)
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(gpu_warmup, "open_library", open_library)
    monkeypatch.setattr(gpu_warmup, "PROC_MAPS", Fake.maps)
    return Fake


CONTEXT_CALLS = ["cuInit", "cuDeviceGet", "cuDevicePrimaryCtxRetain", "cuCtxSetCurrent"]
# One gemm (N,N and T,N) and one gemv (N and T) of each kind, on a handle and
# a buffer of their own, freed after a synchronization.
CUBLAS_CALLS = [
    "cublasCreate", "cuMemAlloc", "cuMemsetD8",
    ("gemm", 0, 0), ("gemm", 1, 0), ("gemv", 0), ("gemv", 1),
    "cuCtxSynchronize", "cuMemFree", "cublasDestroy",
]


def test_warm_up_creates_the_context_then_warms_the_loaded_cublas(fake_gpu):
    record = gpu_warmup.warm_up(["json"])

    assert fake_gpu.events == CONTEXT_CALLS + CUBLAS_CALLS
    assert record["modules"] == ["json"]
    assert record["cublas_libraries"] == [CUBLAS]
    assert record["notes"] == []
    for key in ("import_seconds", "cuda_context_seconds", "cublas_seconds"):
        assert record[key] >= 0
    assert record["seconds"] >= record["cuda_context_seconds"] + record["cublas_seconds"]


def test_cublas_is_looked_up_after_the_modules_are_imported(fake_gpu, tmp_path, monkeypatch):
    # The solver's module is what loads cuBLAS: a process that has not
    # imported it yet has none mapped.
    fake_gpu.maps.write_text(_maps_line("/usr/lib/libc.so.6"))
    (tmp_path / "fake_gpu_extension.py").write_text(
        "import pathlib, os\n"
        "path = pathlib.Path(os.environ['FAKE_MAPS'])\n"
        f"path.write_text(path.read_text() + {_maps_line(CUBLAS)!r})\n"
    )
    monkeypatch.setenv("FAKE_MAPS", str(fake_gpu.maps))
    monkeypatch.syspath_prepend(str(tmp_path))

    record = gpu_warmup.warm_up(["fake_gpu_extension"])

    assert record["cublas_libraries"] == [CUBLAS]
    assert fake_gpu.events == CONTEXT_CALLS + CUBLAS_CALLS


def test_without_a_cuda_driver_nothing_is_warmed_and_the_record_says_why(fake_gpu, monkeypatch):
    def no_driver(name):
        raise OSError(f"{name}: cannot open shared object file: No such file or directory")

    monkeypatch.setattr(gpu_warmup, "open_library", no_driver)

    record = gpu_warmup.warm_up(["json"])

    assert record["cuda_context_seconds"] is None
    assert record["cublas_libraries"] == [] and record["cublas_seconds"] is None
    assert len(record["notes"]) == 1 and "no CUDA context created" in record["notes"][0]
    assert "libcuda.so.1" in record["notes"][0]
    assert record["seconds"] >= 0


def test_the_real_warm_up_never_raises_on_this_machine():
    # On CI (no GPU) this is the no-GPU path end to end; on a GPU machine it
    # warms the real libraries.
    record = gpu_warmup.warm_up([])
    assert record["seconds"] >= 0
    assert (record["cuda_context_seconds"] is None) == any("no CUDA context" in n for n in record["notes"])


def test_a_cuda_error_is_a_note_not_a_failure(fake_gpu):
    fake_gpu.driver_fail = "cuInit"

    record = gpu_warmup.warm_up(["json"])

    assert fake_gpu.events == ["cuInit"]
    assert record["cuda_context_seconds"] is None and record["cublas_libraries"] == []
    assert record["notes"] == [
        "no CUDA context created: WarmupError: cuInit returned CUDA_ERROR_NO_DEVICE (100)"
    ]


def test_a_cublas_error_frees_what_it_allocated_and_is_a_note(fake_gpu):
    fake_gpu.cublas_fail = "gemm"

    record = gpu_warmup.warm_up(["json"])

    assert fake_gpu.events == CONTEXT_CALLS + ["cublasCreate", "cuMemAlloc", "cuMemsetD8", ("gemm", 0, 0),
                                               "cuMemFree", "cublasDestroy"]
    assert record["cuda_context_seconds"] is not None
    assert record["cublas_libraries"] == [] and record["cublas_seconds"] is None
    assert record["notes"] == [
        f"cuBLAS {CUBLAS} not warmed: WarmupError: cublasDgemm returned cublasStatus_t 13"
    ]


def test_a_module_that_does_not_import_and_no_loaded_cublas_are_notes(fake_gpu):
    fake_gpu.maps.write_text(_maps_line("/usr/lib/libc.so.6"))

    record = gpu_warmup.warm_up(["solver_benchmarks_no_such_module"])

    assert record["cuda_context_seconds"] is not None  # the context is still created
    assert record["cublas_libraries"] == []
    assert record["notes"][0].startswith("import solver_benchmarks_no_such_module: ModuleNotFoundError")
    assert record["notes"][1].startswith("no cuBLAS library is loaded")
    assert fake_gpu.events == CONTEXT_CALLS


def test_mapped_libraries_reads_each_cublas_once_and_skips_cublaslt_and_deleted_files(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        _maps_line("/usr/lib/libc.so.6")
        + _maps_line(CUBLAS)
        + _maps_line(CUBLAS)
        + _maps_line("/opt/cuda/lib64/libcublasLt.so.13.1.1.3")
        + _maps_line("/site-packages/nvidia/cublas/lib/libcublas.so.12")
        + _maps_line("/tmp/libcublas.so.11 (deleted)")
        + "7f0000000000-7f0000001000 rw-p 00000000 00:00 0\n"
    )
    assert gpu_warmup.mapped_libraries("libcublas.so", maps) == [
        CUBLAS, "/site-packages/nvidia/cublas/lib/libcublas.so.12",
    ]
    assert gpu_warmup.mapped_libraries("libcublas.so", tmp_path / "missing") == []


# The worker: the GPU part of the warm-up (core/warmup.py) runs before the
# adapter's solve, outside its timer, after the modules are imported and
# before the tiny solve.

class _Recorder(SolverAdapter):
    solver_name = "warmup_probe"
    supported_problem_kinds = {QP}
    events: list = []
    modules: tuple[str, ...] = ()
    gpu = False
    raise_in_solve = False

    @classmethod
    def requested_tolerance(cls, settings, **kwargs):
        from solver_benchmarks.solvers.base import loosest_tolerance

        return loosest_tolerance({}, {"eps": 1e-4})

    def warmup_modules(self):
        return WarmupModules(modules=self.modules, gpu=self.gpu)

    def solve(self, problem, artifacts_dir: Path) -> SolverResult:
        start = time.perf_counter()
        self.events.append("tiny solve" if problem.dataset_id == "warmup" else "solve")
        if self.raise_in_solve:
            raise RuntimeError("solver crashed")
        return SolverResult(status=status.OPTIMAL, objective_value=0.0, iterations=1,
                            run_time_seconds=time.perf_counter() - start, info={"raw_status": "ok"})


def _payload(repo_root, tmp_path) -> dict:
    return {
        "run_id": "run", "dataset": "synthetic_qp", "problem": "one_variable_lp", "repo_root": str(repo_root),
        "solver": {"id": "probe", "solver": "warmup_probe", "settings": {}},
        "artifacts_dir": str(tmp_path / "artifacts"),
    }


@pytest.fixture
def probe(monkeypatch, fake_gpu):
    class Probe(_Recorder):
        events = fake_gpu.events

    monkeypatch.setitem(SOLVERS, "warmup_probe", Probe)
    return Probe


def test_the_worker_warms_up_a_gpu_solve_before_its_timed_region(probe, fake_gpu, repo_root, tmp_path):
    probe.modules, probe.gpu = ("json",), True
    fake_gpu.delay = CONTEXT_SECONDS

    record = run_payload(_payload(repo_root, tmp_path)).to_record()

    # The context and cuBLAS first, then the tiny solve, then the real one.
    assert fake_gpu.events == CONTEXT_CALLS + CUBLAS_CALLS + ["tiny solve", "solve"]
    warmup = record["metadata"]["warmup"]
    assert warmup["modules"] == ["json"] and warmup["gpu"] is True
    assert warmup["cublas_libraries"] == [CUBLAS] and warmup["complete"] is True
    assert warmup["seconds"] >= warmup["gpu_seconds"] >= warmup["cuda_context_seconds"] >= CONTEXT_SECONDS
    # The adapter's timer starts after the warm-up.
    assert record["run_time_seconds"] < CONTEXT_SECONDS
    assert record["status"] == status.OPTIMAL


def test_the_worker_does_not_warm_a_gpu_for_a_cpu_solve(probe, fake_gpu, repo_root, tmp_path, monkeypatch):
    def not_called(modules):
        raise AssertionError("gpu_warmup.warm_up called for a CPU solve")

    monkeypatch.setattr(gpu_warmup, "warm_up", not_called)

    record = run_payload(_payload(repo_root, tmp_path)).to_record()

    assert fake_gpu.events == ["tiny solve", "solve"]
    warmup = record["metadata"]["warmup"]
    assert warmup["gpu"] is False and warmup["cuda_context_seconds"] is None and warmup["complete"] is True
    assert record["status"] == status.OPTIMAL


def test_a_worker_error_row_keeps_the_warm_up_record(probe, fake_gpu, repo_root, tmp_path):
    probe.modules, probe.gpu = ("json",), True
    probe.raise_in_solve = True

    record = run_payload(_payload(repo_root, tmp_path)).to_record()

    assert record["status"] == status.WORKER_ERROR
    warmup = record["metadata"]["warmup"]
    assert warmup["cublas_libraries"] == [CUBLAS]
    # The tiny solve raised too: a note, not a failure of the run.
    assert warmup["complete"] is False
    assert warmup["notes"] == ["tiny solve raised RuntimeError: solver crashed"]


def test_the_worker_runs_a_gpu_solve_without_a_gpu(probe, repo_root, tmp_path, monkeypatch):
    def no_driver(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(gpu_warmup, "open_library", no_driver)
    probe.modules, probe.gpu = ("json",), True

    record = run_payload(_payload(repo_root, tmp_path)).to_record()

    assert record["status"] == status.OPTIMAL
    warmup = record["metadata"]["warmup"]
    assert warmup["cuda_context_seconds"] is None and warmup["complete"] is False
    assert "no CUDA context created" in warmup["notes"][0]
    assert warmup["tiny_status"] == status.OPTIMAL  # the tiny solve still ran


def test_a_cublas_that_is_mapped_but_not_warmed_leaves_the_row_not_fully_warmed(
    probe, fake_gpu, repo_root, tmp_path
):
    probe.modules, probe.gpu = ("json",), True
    fake_gpu.cublas_fail = "gemm"

    warmup = run_payload(_payload(repo_root, tmp_path)).to_record()["metadata"]["warmup"]

    assert warmup["cuda_context_seconds"] is not None and warmup["cublas_libraries"] == []
    assert warmup["complete"] is False
    assert warmup["notes"][0].startswith(f"cuBLAS {CUBLAS} not warmed")


def test_no_cublas_loaded_is_a_note_but_the_row_is_fully_warmed(probe, fake_gpu, repo_root, tmp_path):
    # SCS 3.3.2's cuDSS module links no cuBLAS: there is none to warm.
    probe.modules, probe.gpu = ("json",), True
    fake_gpu.maps.write_text(_maps_line("/usr/lib/libc.so.6"))

    warmup = run_payload(_payload(repo_root, tmp_path)).to_record()["metadata"]["warmup"]

    assert warmup["complete"] is True and warmup["cublas_libraries"] == []
    assert warmup["notes"][0].startswith("no cuBLAS library is loaded")


# Which adapters warm a GPU: those whose settings select one.

class _Member:
    def __init__(self, name, value):
        self.name, self.value = name, value


@pytest.mark.parametrize(
    "settings, modules, gpu",
    [
        ({"linear_solver": "cudss"}, ("scs", "scs._scs_cudss"), True),
        ({"linear_solver": "gpu_indirect"}, ("scs", "scs._scs_gpu"), True),
        ({"linear_solver": _Member("CUDSS", "cudss")}, ("scs", "scs._scs_cudss"), True),
        ({}, ("scs",), False),
        ({"linear_solver": None}, ("scs",), False),
        ({"linear_solver": "auto"}, ("scs",), False),
        ({"linear_solver": "mkl"}, ("scs", "scs._scs_mkl"), False),
        ({"linear_solver": "qdldl"}, ("scs", "scs._scs_direct"), False),
        ({"linear_solver": "cpu_indirect"}, ("scs", "scs._scs_indirect"), False),
        ({"linear_solver": "accelerate"}, ("scs", "scs._scs_accelerate"), False),
        ({"linear_solver": "cpu_dense"}, ("scs", "scs._scs_dense"), False),
        ({"linear_solver": 3}, ("scs",), False),
        ({"linear_solver": ["cudss"]}, ("scs",), False),
    ],
)
def test_scs_names_the_module_of_its_linear_solver(settings, modules, gpu):
    assert get_solver("scs")(settings).warmup_modules() == WarmupModules(modules=modules, gpu=gpu)


@pytest.mark.parametrize(
    "settings, modules",
    [
        ({"linear_solver": "cudss"}, ("cupy", "cupyx.scipy.sparse", "nvmath")),
        ({"linear_solver": " CUDSS "}, ("cupy", "cupyx.scipy.sparse", "nvmath")),
        ({"linear_solver": _Member("CUDSS", object)}, ("cupy", "cupyx.scipy.sparse", "nvmath")),
        ({"linear_solver": "cupy_dense"}, ("cupy", "cupyx.scipy.linalg")),
        ({"linear_solver": "qdldl"}, ()),
        ({}, ()),
    ],
)
def test_qtqp_warms_up_its_gpu_linear_solvers(settings, modules):
    plan = get_solver("qtqp")(settings).warmup_modules()
    assert plan == WarmupModules(modules=("qtqp", *modules), gpu=bool(modules))


def test_cuopt_always_warms_up_the_gpu():
    plan = get_solver("cuopt")({}).warmup_modules()
    assert plan.gpu is True and plan.modules[0] == "cuopt.linear_programming.solver"


@pytest.mark.parametrize("name", [name for name in list_solvers() if name != "cuopt"])
def test_other_solvers_at_default_settings_run_on_the_cpu(name):
    plan = get_solver(name)({}).warmup_modules()
    assert plan.gpu is False
    assert plan.modules  # every adapter names its solver's package
