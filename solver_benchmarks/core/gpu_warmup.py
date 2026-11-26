"""GPU warm-up before a GPU solver's timed region.

Every solve runs in a worker process of its own (``runner._run_one``), so
without a warm-up a GPU solver pays, inside ``run_time_seconds``, costs that
a process pays once whatever it then solves:

- loading the solver's GPU extension module and the CUDA libraries it links
  (scs-python, for one, imports its cuDSS or GPU module inside ``scs.SCS``);
- creating the CUDA context on the device: 0.245 s on an NVIDIA GB10 with
  CUDA 13 (measured 2026-11-26);
- cuBLAS loading its matrix-product kernels at the first ``gemm`` or
  ``gemv`` of the process: cuBLAS 13 runs both through cuBLASLt, which then
  loads 15 kernel libraries, 0.128 s on the GB10 whatever the problem size
  (same measurement; no CUDA loader setting avoided it).

A Python session that solves many problems pays these once, so the
benchmark times every GPU solve the way such a session's later solves see
it. ``warm_up`` is the GPU part of the worker's warm-up (``core/warmup.py``),
which runs it before the adapter's ``solve`` for every solver whose settings
select a GPU (``SolverAdapter.warmup_modules``), once it has imported the
solver's modules itself (so it passes no modules here):

1. it imports the modules the adapter names, so their libraries are loaded;
2. it creates the primary CUDA context of device 0 through the driver API
   (``libcuda.so.1``, the system's one driver library), the context every
   CUDA runtime in the process then uses;
3. for each cuBLAS library mapped into the process at that point (the ones
   the solver's modules loaded, read from ``/proc/self/maps``), it runs on a
   handle of its own one ``gemm`` of each kind (N,N and T,N) and one
   ``gemv`` of each kind (N and T) on 10 x 10 operands, then frees what it
   allocated. These are the calls of the 2026-11-26 measurement, where
   they took the GPU indirect solver's extra cost on the first solve of a
   process from 160 ms to 28 ms (32 ms for a build of the solver that
   makes no such call). They are double precision, as the benchmarked
   solvers are: the first single-precision ``gemm`` after them still took
   137 ms on the GB10, as long as in a process without them.

The warm-up never loads a cuBLAS of its own: a copy loaded first by name
could take the place of a solver's own (cuOpt bundles its CUDA libraries)
and change what the solver runs. It uses only ``ctypes`` and the libraries
the solver already depends on.

Measured through the worker on the GB10 (SCS eeef3109, tiny Maros-Meszaros
QPs, 5 fresh processes each): the warm-up took 0.42 s, of which the context
0.24 s, cuBLAS 0.13 s and the import 0.04 s; ``run_time_seconds`` fell by
0.41 s for the GPU indirect solver and by 0.29 s for cuDSS, whose solve
makes no ``gemm`` or ``gemv`` call and so never paid cuBLAS's load.

What stays in the timed region: a library a solver loads only during its
solve (a cuBLAS loaded then is not mapped at step 3, which the record's
notes say), a cuBLAS linked statically into a solver, kernels that other
libraries load at their first use (cuDSS's own; cuSPARSE's first
``csr2csc`` in an SCS build that transposes A on the GPU, 21.7 ms on the
GB10), cuBLAS kernels of other shapes (10 to 15 ms at sizes of a few
thousand on the GB10) and types, and the solver's own handles and memory.
In the measurement above the first solve of a process, after the warm-up,
still took 6 to 9 ms longer than a second solve for the GPU indirect solver
and 34 to 36 ms longer for cuDSS. The tradeoff: the context and the loaded
kernels hold GPU memory during the solve that a solver making no ``gemm``
or ``gemv`` call would not have held, and runs recorded before the warm-up
are slower by these costs (see the README's "Warm-up").

The warm-up never fails a run. A part that cannot run (no CUDA driver, no
device, a module that does not import, a CUDA error) is skipped with the
reason in the record's ``notes``, and the solve runs as it would have
without it. It needs Linux, where the GPU solvers run: elsewhere it records
that the CUDA driver library cannot be loaded.
"""

from __future__ import annotations

import ctypes
import importlib
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CUDA_DRIVER_LIBRARY = "libcuda.so.1"
# File names of cuBLAS's shared library (libcublas.so.13.1.1.3 and the like;
# libcublasLt is not matched, cuBLAS calls it itself).
CUBLAS_LIBRARY_PREFIX = "libcublas.so"
PROC_MAPS = Path("/proc/self/maps")

# The operands of the 2026-11-26 measurement's warm-up: one buffer of 4096
# doubles, matrices of 10 x 10 at offsets 0, 100 and 200, vectors at 300 and
# 400. Any size triggers the load (0.127 to 0.138 s from n = 10 to 1e5).
_N = 10
_BUFFER_DOUBLES = 4096
_DOUBLE_BYTES = 8
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1


class WarmupError(RuntimeError):
    """A CUDA or cuBLAS call of the warm-up returned an error."""


def warm_up(modules: Sequence[str]) -> dict[str, Any]:
    """Warm the GPU for a solve through ``modules``; returns the row's record.

    The record (``metadata.gpu_warmup``): ``seconds``, the warm-up's whole
    wall-clock time, outside ``run_time_seconds``; ``modules`` and
    ``import_seconds``; ``cuda_context_seconds``, None when no context was
    created; ``cublas_libraries``, the paths of the cuBLAS libraries warmed,
    and ``cublas_seconds``, None when none was; ``notes``, one line for each
    part that did not run and why (empty when every part ran).
    """
    start = time.perf_counter()
    notes: list[str] = []
    record: dict[str, Any] = {
        "modules": list(modules),
        "import_seconds": None,
        "cuda_context_seconds": None,
        "cublas_libraries": [],
        "cublas_seconds": None,
        "notes": notes,
    }
    began = time.perf_counter()
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # the solve then reports its own failure
            notes.append(f"import {name}: {type(exc).__name__}: {exc}")
    record["import_seconds"] = time.perf_counter() - began

    driver = None
    try:
        driver = CudaDriver.load()
        began = time.perf_counter()
        driver.create_context()
        record["cuda_context_seconds"] = time.perf_counter() - began
    except Exception as exc:  # OSError: no driver; WarmupError: a CUDA error
        notes.append(f"no CUDA context created: {type(exc).__name__}: {exc}")

    if record["cuda_context_seconds"] is not None and driver is not None:
        libraries = mapped_libraries(CUBLAS_LIBRARY_PREFIX)
        if not libraries:
            notes.append(
                "no cuBLAS library is loaded after importing the modules, so none was warmed "
                "(a solver that loads cuBLAS during its solve pays its kernel loading there)"
            )
        began = time.perf_counter()
        for path in libraries:
            try:
                warm_cublas(driver, path)
            except Exception as exc:  # recorded; the solve runs as it would have without it
                notes.append(f"cuBLAS {path} not warmed: {type(exc).__name__}: {exc}")
            else:
                record["cublas_libraries"].append(path)
        if record["cublas_libraries"]:
            record["cublas_seconds"] = time.perf_counter() - began
    record["seconds"] = time.perf_counter() - start
    return record


def mapped_libraries(prefix: str, maps: Path | None = None) -> list[str]:
    """Paths of the files mapped into this process whose name starts with ``prefix``.

    In the order the process mapped them, each once; a file deleted since it
    was mapped is left out. ``maps`` defaults to ``/proc/self/maps``; empty
    where it cannot be read (no ``/proc``, as on macOS).
    """
    try:
        text = (PROC_MAPS if maps is None else maps).read_text()
    except OSError:
        return []
    paths: list[str] = []
    for line in text.splitlines():
        # address perms offset dev inode pathname
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5]
        if path.endswith(" (deleted)") or not os.path.basename(path).startswith(prefix):
            continue
        if path not in paths:
            paths.append(path)
    return paths


def open_library(name: str) -> Any:
    """``ctypes.CDLL(name)``: a library already loaded (the same file, or the
    same soname) comes back as that library, not as a second copy."""
    return ctypes.CDLL(name)


class CudaDriver:
    """The few CUDA driver API calls the warm-up makes, through ``ctypes``."""

    def __init__(self, library: Any):
        self._lib = library

    @classmethod
    def load(cls) -> CudaDriver:
        """The driver library; OSError where it is not installed."""
        return cls(open_library(CUDA_DRIVER_LIBRARY))

    def create_context(self) -> None:
        """Create the primary context of device 0 and make it current.

        The primary context is the one the CUDA runtime of every library in
        the process uses on that device, so this is the context the solver
        runs in. It is retained and never released: releasing the last
        reference would destroy it, and the kernels loaded into it, before
        the solve.
        """
        lib = self._lib
        self._check(lib.cuInit(ctypes.c_uint(0)), "cuInit")
        device = ctypes.c_int()
        self._check(lib.cuDeviceGet(ctypes.byref(device), ctypes.c_int(0)), "cuDeviceGet")
        context = ctypes.c_void_p()
        self._check(lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
        self._check(lib.cuCtxSetCurrent(context), "cuCtxSetCurrent")

    def alloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self._check(self._lib.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(size)), "cuMemAlloc")
        return int(pointer.value)

    def zero(self, pointer: int, size: int) -> None:
        self._check(
            self._lib.cuMemsetD8_v2(ctypes.c_uint64(pointer), ctypes.c_ubyte(0), ctypes.c_size_t(size)),
            "cuMemsetD8",
        )

    def synchronize(self) -> None:
        self._check(self._lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def free(self, pointer: int) -> None:
        self._check(self._lib.cuMemFree_v2(ctypes.c_uint64(pointer)), "cuMemFree")

    def _check(self, result: int, call: str) -> None:
        if result != 0:
            raise WarmupError(f"{call} returned {self._error_name(result)}")

    def _error_name(self, result: int) -> str:
        name = ctypes.c_char_p()
        try:
            if self._lib.cuGetErrorName(ctypes.c_int(result), ctypes.byref(name)) == 0 and name.value:
                return f"{name.value.decode(errors='replace')} ({result})"
        except (AttributeError, OSError):
            pass
        return f"CUresult {result}"


def warm_cublas(driver: CudaDriver, path: str) -> None:
    """Make the first matrix-product calls of the cuBLAS at ``path``.

    ``path`` is a library already mapped into the process, so opening it
    gives that same library, not a second copy. The calls run on a handle
    and a buffer of their own, both freed before returning; their results
    are not used.
    """
    lib = open_library(path)
    handle = ctypes.c_void_p()
    _check_cublas(lib.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate")
    try:
        size = _BUFFER_DOUBLES * _DOUBLE_BYTES
        buffer = driver.alloc(size)
        try:
            driver.zero(buffer, size)
            a, b, c, x, y = (ctypes.c_void_p(buffer + k * _DOUBLE_BYTES) for k in (0, 100, 200, 300, 400))
            one, zero = ctypes.c_double(1.0), ctypes.c_double(0.0)
            n = ctypes.c_int(_N)
            for op_a in (_CUBLAS_OP_N, _CUBLAS_OP_T):
                _check_cublas(
                    lib.cublasDgemm_v2(
                        handle, ctypes.c_int(op_a), ctypes.c_int(_CUBLAS_OP_N), n, n, n,
                        ctypes.byref(one), a, n, b, n, ctypes.byref(zero), c, n,
                    ),
                    "cublasDgemm",
                )
            for op, beta in ((_CUBLAS_OP_N, one), (_CUBLAS_OP_T, zero)):
                _check_cublas(
                    lib.cublasDgemv_v2(
                        handle, ctypes.c_int(op), n, n, ctypes.byref(one), a, n, x, ctypes.c_int(1),
                        ctypes.byref(beta), y, ctypes.c_int(1),
                    ),
                    "cublasDgemv",
                )
            driver.synchronize()
        finally:
            driver.free(buffer)
    finally:
        lib.cublasDestroy_v2(handle)


def _check_cublas(result: int, call: str) -> None:
    if result != 0:
        raise WarmupError(f"{call} returned cublasStatus_t {result}")
