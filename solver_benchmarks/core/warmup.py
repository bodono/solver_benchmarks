"""Warm-up before a solve's timed region.

Every solve runs in a worker process of its own (``runner._run_one``), so
without a warm-up the adapter's timer would include costs that a process
pays once, whatever it then solves: Python imports of the solver's modules
and backends (scs-python imports its MKL, cuDSS or GPU module inside
``scs.SCS``; QTQP imports its linear-solver backend inside its first solve;
Clarabel imports ``scipy.linalg`` at its first PSD cone), loading and
binding shared libraries, a BLAS's dispatch and thread pools, the CUDA
context, cuBLAS's kernels, and the first run of the solver's own code. A
Python session that solves many problems pays these once, so the benchmark
times every solve the way such a session's later solves see it.

``run`` does, in the worker, before the adapter's ``solve`` (``worker.py``):

1. it imports the modules the adapter names for these settings
   (``SolverAdapter.warmup_modules``: the solver's package and the backend
   module a setting names);
2. for a solve on a GPU, it creates the CUDA context and warms the cuBLAS
   those modules loaded (``gpu_warmup.warm_up``);
3. it solves a tiny problem (``tiny_problem``) through the same adapter
   class and ``solve`` method, with the real solve's settings
   (``SolverAdapter.warmup_settings``: the same threads, linear solver,
   precision, device, tolerances and limits, with output off), in an
   artifacts directory of its own that is then deleted, with the process's
   standard output sent to the null device. Its result is discarded.

The tiny problem has the real problem's form (``qp`` or ``cone``), the same
kind of objective (a ``P`` with entries, an all-zero ``P``, or none) and one
small block of every cone type the real problem has, so the tiny solve runs
the real solve's code paths: whatever the solver imports, loads or
initialises at its first solve of that form happens here. It is built from
the structure alone (the form, whether ``P`` has stored entries, the cone
keys, the signs of the power cone exponents), never from the data's values,
and it is feasible and bounded with a unique optimum.

Measured with these steps run by a measurement script around the worker
(2026-11-26, harness a3091dcb, 5 fresh processes per arm, the second solve
of a problem in one process as the warm floor): on a Mac
(Apple Silicon, 14 solver configurations, 80 problem pairs) every one-time
cost above noise was an import inside the timed call (QTQP's backend 52 to
55 ms, Clarabel's ``scipy.linalg`` on SDPs 35 ms, CVXOPT's ``conelp``
modules 2.4 to 7.4 ms, SCS's backend module 0.8 to 1.7 ms), and the imports
plus the tiny solve left -2.3 to +1.3 ms against the warm floor on the pairs
whose run-to-run noise was small. On an NVIDIA GB10 (SCS only): CPU backends
0.4 to 1.1 ms cold, 0.0 to 0.8 ms left after the tiny solve; cuDSS 317 to
358 ms cold, 32 to 48 ms left after the GPU warm-up alone, -2.8 to +7.9 ms
after the tiny solve too; GPU indirect 286 to 424 ms cold, -1.7 to +13.1 ms
left on the problems whose GPU CG runs did not vary from run to run. The
tiny solve itself took 0.7 to 20 ms on the CPU, 15 to 24 ms (GPU indirect)
and 42 to 57 ms (cuDSS). No status, iteration count or objective of a CPU
solve changed (bit for bit), and no KKT-verified status changed anywhere.

What stays in the timed region: work that depends on the problem's size or
takes a path the tiny problem does not (a backend chosen by size, threaded
code entered only for large blocks, MKL Pardiso's threading, cuBLAS kernels
of other shapes), first touch of the problem's own memory, per-object
set-up that every solve repeats (the solver's own handles; HiGHS's thread
pool, which the adapter resets before each solve; MOSEK's licence checkout,
which a new environment repeats), and, for GPU iterative solvers, run-to-run
variation that is larger than any of this.

The warm-up never fails a run. An exception in any part (a module that does
not import, no CUDA device, a cone the tiny problem cannot build, a tiny
solve that raises or does not solve) is recorded in the row's
``metadata.warmup.notes``, ``complete`` is then false, and the real solve
runs as it would have without that part. A warm-up that kills or hangs the
worker is handled by the runner (``runner._run_one``): it reads this
module's state file (``STATE_FILE``), and when the worker ended before the
warm-up finished it runs the solve again in a new worker without the
warm-up; that row records why. The warm-up's time is not in
``run_time_seconds``, and the runner adds it to the worker's timeout.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Collection, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core import gpu_warmup, status
from solver_benchmarks.core.problem import CONE, QP, ProblemData, as_list, cone_items_in_row_order

# The worker's state file in the solve's artifacts directory, which the
# runner reads when a worker ends without a result: {"state": "running",
# "part": ...} while the warm-up runs, {"state": "done", "record": {...}}
# once it has finished.
STATE_FILE = "warmup.json"
RUNNING = "running"
DONE = "done"

# Statuses of the tiny solve that count as solved: it ran the solver's
# iterations to the end. Any other status (an error, a limit, a skip) is
# recorded as a note and the row is not fully warmed.
_SOLVED = frozenset({status.OPTIMAL, status.OPTIMAL_INACCURATE})

# Block sizes of the tiny problem's cones, each the smallest that takes the
# general code path rather than a closed form or special case: a
# second-order cone of 4 (SCS projects sizes 1 to 3 in closed form), a PSD
# block of order 3 (orders 1 and 2 are closed forms in SCS; 3 needs LAPACK's
# eigensolver), a complex PSD block of order 2 (4 rows; order 1 is real).
_SOC_SIZE = 4
_PSD_ORDER = 3
_COMPLEX_PSD_ORDER = 2
# Power cone exponents of the tiny problem: one primal (alpha > 0) and one
# dual (SCS's alpha < 0) block as the real problem has them. Any exponent in
# (0, 1) takes the same code path; 0.5 keeps the optimum at a round number.
_POWER_ALPHA = 0.5


class TinyProblemUnsupported(ValueError):
    """The real problem has a structure the tiny problem cannot mirror."""


def tiny_problem(problem: ProblemData) -> ProblemData:
    """A tiny problem of ``problem``'s form, objective kind and cone types.

    Built from the structure only: the form, whether ``P`` is absent, has no
    stored entries or has some, the cone keys with a block in them and the
    signs of the power cone exponents. Raises :class:`TinyProblemUnsupported`
    for a form or a cone key it does not know.
    """
    if problem.kind == QP:
        data = tiny_qp(_p_kind(problem.qp.get("P")) == "nonzero")
    elif problem.kind == CONE:
        data = tiny_cone(problem.cone["cone"], _p_kind(problem.cone.get("P")))
    else:
        raise TinyProblemUnsupported(f"problem kind {problem.kind!r}")
    return ProblemData("warmup", f"tiny_{problem.kind}", problem.kind, data)


def _p_kind(p: Any) -> str:
    """'none', 'zero' (no stored entries) or 'nonzero', as adapters test it (``P.nnz``)."""
    if p is None:
        return "none"
    nnz = p.nnz if sp.issparse(p) else int(np.count_nonzero(np.asarray(p)))
    return "nonzero" if nnz else "zero"


def tiny_qp(has_p: bool) -> dict[str, Any]:
    """The harness's QP form ``l <= A x <= u`` with two variables.

    Rows: an equality (x1 + x2 = 1), a two-sided row, a row bounded above
    only, a row bounded below only, and a two-sided bound on each variable,
    so the adapters' conversions see every kind of bound. Minimises
    x1 + 3 x2 (plus (x1^2 + x2^2) / 2 when ``has_p``); the optimum is
    x = (1, 0) in both cases, with only the equality and x2's lower bound
    active.
    """
    a = sp.csc_matrix(np.array([
        [1.0, 1.0],
        [1.0, -1.0],
        [1.0, 2.0],
        [1.0, -2.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]))
    p = sp.identity(2, format="csc") if has_p else sp.csc_matrix((2, 2))
    return {
        "P": p,
        "q": np.array([1.0, 3.0]),
        "r": 0.0,
        "A": a,
        "l": np.array([1.0, -2.0, -math.inf, -3.0, 0.0, 0.0]),
        "u": np.array([1.0, 2.0, 5.0, math.inf, 10.0, 10.0]),
        "n": 2,
        "m": 6,
        "obj_type": "min",
    }


def tiny_cone(cone: Mapping[str, Any], p_kind: str = "none") -> dict[str, Any]:
    """One small block of each cone type in ``cone``, in row order.

    Each block has variables of its own and a unique optimum away from the
    cone's apex (s = b - A x in the cone, minimising q'x, plus x'x / 2 when
    ``p_kind`` is 'nonzero'):

    - z / f: x = 1.
    - l: -1 <= x <= 1, minimise x: x = -1.
    - bl / bu: one box of width 1 (t = 1, -1 <= x <= 1), minimise x: x = -1.
    - q: ||(x1, x2, x3)|| <= 1, minimise their sum.
    - s: [[1, a, b], [a, 1, c], [b, c, 1]] PSD, minimise a + b + c (svec,
      off-diagonals times sqrt 2).
    - cs: [[1, a + ib], [a - ib, 1]] PSD (a^2 + b^2 <= 1), minimise a + b.
    - ep: (x, 1, 1) in the exponential cone, maximise x: x = 0.
    - ed: (-1, x, 1) in its dual, minimise x: x = -1.
    - p: (1, 1, x) in the power cone of exponent 0.5 (a dual one, -0.5, when
      the problem has exponents < 0), minimise x.

    ``p_kind`` 'none' leaves ``P`` out, 'zero' gives an all-zero ``P``.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b: list[float] = []
    c: list[float] = []
    out_cone: dict[str, Any] = {}
    n = 0

    def variables(count: int) -> list[int]:
        nonlocal n
        n += count
        return list(range(n - count, n))

    def row(entries: list[tuple[int, float]], rhs: float) -> None:
        for column, value in entries:
            rows.append(len(b))
            cols.append(column)
            vals.append(value)
        b.append(rhs)

    root2 = math.sqrt(2.0)
    box_done = False
    for key, value in cone_items_in_row_order(dict(cone)):
        if key in ("z", "f", "l", "ep", "ed"):
            if not _positive_count(key, value):
                continue
        elif key in ("q", "s", "cs", "p", "bl", "bu"):
            if not as_list(value):
                continue
        else:
            raise TinyProblemUnsupported(f"cone key {key!r}")
        if key in ("z", "f"):
            (x,) = variables(1)
            row([(x, 1.0)], 1.0)
            c.append(1.0)
            out_cone[key] = 1
        elif key == "l":
            (x,) = variables(1)
            row([(x, 1.0)], 1.0)
            row([(x, -1.0)], 1.0)
            c.append(1.0)
            out_cone["l"] = 2
        elif key in ("bl", "bu"):
            if box_done:
                continue
            box_done = True
            (x,) = variables(1)
            row([], 1.0)  # t = 1
            row([(x, -1.0)], 0.0)  # s = x, with -t <= s <= t
            c.append(1.0)
            out_cone["bl"] = [-1.0]
            out_cone["bu"] = [1.0]
        elif key == "q":
            xs = variables(_SOC_SIZE - 1)
            row([], 1.0)
            for x in xs:
                row([(x, -1.0)], 0.0)
            c.extend([1.0] * len(xs))
            out_cone["q"] = [_SOC_SIZE]
        elif key == "s":
            a12, a13, a23 = variables(3)
            # svec of an order-3 block: the lower triangle column by column.
            row([], 1.0)
            row([(a12, -root2)], 0.0)
            row([(a13, -root2)], 0.0)
            row([], 1.0)
            row([(a23, -root2)], 0.0)
            row([], 1.0)
            c.extend([1.0, 1.0, 1.0])
            out_cone["s"] = [_PSD_ORDER]
        elif key == "cs":
            re, im = variables(2)
            # An order-2 Hermitian block: X11, sqrt 2 (Re, Im) of X21, X22.
            row([], 1.0)
            row([(re, -root2)], 0.0)
            row([(im, -root2)], 0.0)
            row([], 1.0)
            c.extend([1.0, 1.0])
            out_cone["cs"] = [_COMPLEX_PSD_ORDER]
        elif key == "ep":
            (x,) = variables(1)
            row([(x, -1.0)], 0.0)
            row([], 1.0)
            row([], 1.0)
            c.append(-1.0)
            out_cone["ep"] = 1
        elif key == "ed":
            (x,) = variables(1)
            row([], -1.0)
            row([(x, -1.0)], 0.0)
            row([], 1.0)
            c.append(1.0)
            out_cone["ed"] = 1
        elif key == "p":
            exponents = []
            for sign in _power_signs(value):
                (x,) = variables(1)
                row([], 1.0)
                row([], 1.0)
                row([(x, -1.0)], 0.0)
                c.append(1.0)
                exponents.append(sign * _POWER_ALPHA)
            out_cone["p"] = exponents
    if not b:
        raise TinyProblemUnsupported(f"cone {dict(cone)!r} has no block")
    m = len(b)
    data: dict[str, Any] = {
        "A": sp.csc_matrix((vals, (rows, cols)), shape=(m, n)),
        "b": np.array(b),
        "q": np.array(c),
        "r": 0.0,
        "cone": out_cone,
        "n": n,
        "m": m,
        "obj_type": "min",
    }
    if p_kind == "nonzero":
        data["P"] = sp.identity(n, format="csc")
    elif p_kind == "zero":
        data["P"] = sp.csc_matrix((n, n))
    return data


def _positive_count(key: str, value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError) as exc:
        raise TinyProblemUnsupported(f"cone {key!r} size {value!r}") from exc


def _power_signs(value: Any) -> list[int]:
    """+1 and/or -1, in the order the exponents' signs first appear."""
    signs: list[int] = []
    for alpha in as_list(value):
        try:
            sign = -1 if float(alpha) < 0 else 1
        except (TypeError, ValueError) as exc:
            raise TinyProblemUnsupported(f"power cone exponent {alpha!r}") from exc
        if sign not in signs:
            signs.append(sign)
    return signs


def describe(problem: ProblemData) -> dict[str, Any]:
    """The tiny problem's shape for the row's record."""
    data = problem.data
    record: dict[str, Any] = {
        "kind": problem.kind,
        "n": int(data["n"]),
        "m": int(data["m"]),
        "p": _p_kind(data.get("P")),
    }
    if problem.kind == CONE:
        record["cone"] = {
            key: list(value) if isinstance(value, list) else value for key, value in data["cone"].items()
        }
    return record


def run(solver: Any, problem: ProblemData, state_path: Path | None = None) -> dict[str, Any]:
    """Warm up for ``solver``'s solve of ``problem``; returns the row's record.

    ``solver`` is the adapter instance the worker will time. The record
    (``metadata.warmup``) has ``seconds``, the whole warm-up; ``modules``
    and ``import_seconds``; for a GPU solve (``gpu``) ``gpu_seconds``,
    ``cuda_context_seconds``, ``cublas_libraries`` and ``cublas_seconds``;
    ``tiny_problem`` (its shape), ``tiny_status``, ``tiny_iterations``,
    ``tiny_run_time_seconds`` (the adapter's timer) and
    ``tiny_solve_seconds`` (the whole call); ``tiny_solve_imported``, the
    modules the tiny solve imported first; ``notes``, one line per part that
    did not run or did not succeed; and ``complete``, true when every part
    ran and the tiny problem was solved. ``state_path`` gets the progress
    for the runner (see ``STATE_FILE``).
    """
    start = time.perf_counter()
    notes: list[str] = []
    record: dict[str, Any] = {
        "enabled": True,
        "complete": False,
        "seconds": None,
        "modules": [],
        "import_seconds": None,
        "gpu": False,
        "gpu_seconds": None,
        "cuda_context_seconds": None,
        "cublas_libraries": [],
        "cublas_seconds": None,
        "tiny_problem": None,
        "tiny_status": None,
        "tiny_iterations": None,
        "tiny_run_time_seconds": None,
        "tiny_solve_seconds": None,
        "tiny_solve_imported": [],
        "notes": notes,
    }
    ok = True

    write_state(state_path, {"state": RUNNING, "part": "imports"})
    try:
        plan = solver.warmup_modules()
        modules = tuple(plan.modules)
        gpu = bool(plan.gpu)
    except Exception as exc:  # recorded; the tiny solve still imports what the solve needs
        notes.append(f"warmup_modules raised {type(exc).__name__}: {exc}")
        ok, modules, gpu = False, (), False
    record["modules"] = list(modules)
    record["gpu"] = gpu
    began = time.perf_counter()
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # the solve then reports its own failure
            notes.append(f"import {name}: {type(exc).__name__}: {exc}")
            ok = False
    record["import_seconds"] = time.perf_counter() - began

    if gpu:
        write_state(state_path, {"state": RUNNING, "part": "gpu"})
        ok = _warm_gpu(record, notes) and ok

    write_state(state_path, {"state": RUNNING, "part": "tiny_solve"})
    ok = _tiny_solve(solver, problem, record, notes) and ok

    record["complete"] = ok
    record["seconds"] = time.perf_counter() - start
    write_state(state_path, {"state": DONE, "record": record})
    return record


def _warm_gpu(record: dict[str, Any], notes: list[str]) -> bool:
    """The CUDA context and cuBLAS (``gpu_warmup``), after the modules were
    imported; True when the context was created and every cuBLAS mapped was
    warmed (a solver whose modules load no cuBLAS has none to warm)."""
    try:
        gpu = gpu_warmup.warm_up(())
    except Exception as exc:  # warm_up records its own failures; this is a bug guard
        notes.append(f"GPU warm-up raised {type(exc).__name__}: {exc}")
        return False
    record["gpu_seconds"] = gpu["seconds"]
    record["cuda_context_seconds"] = gpu["cuda_context_seconds"]
    record["cublas_libraries"] = list(gpu["cublas_libraries"])
    record["cublas_seconds"] = gpu["cublas_seconds"]
    notes.extend(gpu["notes"])
    if gpu["cuda_context_seconds"] is None:
        return False
    mapped = gpu_warmup.mapped_libraries(gpu_warmup.CUBLAS_LIBRARY_PREFIX)
    return set(mapped) <= set(gpu["cublas_libraries"])


def _tiny_solve(solver: Any, problem: ProblemData, record: dict[str, Any], notes: list[str]) -> bool:
    try:
        tiny = tiny_problem(problem)
    except TinyProblemUnsupported as exc:
        notes.append(f"no tiny problem: {exc}")
        return False
    record["tiny_problem"] = describe(tiny)
    began = time.perf_counter()
    before = set(sys.modules.copy())
    try:
        settings = solver.warmup_settings() if hasattr(solver, "warmup_settings") else dict(solver.settings)
        tiny_solver = type(solver)(settings)
        with tempfile.TemporaryDirectory(prefix="warmup-") as artifacts, _stdout_to_null():
            result = tiny_solver.solve(tiny, Path(artifacts))
    except Exception as exc:  # recorded; the real solve runs as it would have
        notes.append(f"tiny solve raised {type(exc).__name__}: {exc}")
        return False
    finally:
        record["tiny_solve_seconds"] = time.perf_counter() - began
        record["tiny_solve_imported"] = imported_since(before)
    record["tiny_status"] = result.status
    record["tiny_iterations"] = result.iterations
    record["tiny_run_time_seconds"] = result.run_time_seconds
    if result.status not in _SOLVED:
        reason = (result.info or {}).get("error") or (result.info or {}).get("reason")
        notes.append(f"tiny solve ended {result.status}" + (f": {reason}" if reason else ""))
        return False
    return True


def imported_since(before: Collection[str]) -> list[str]:
    """Modules imported since ``before`` (a copy of ``sys.modules``'s names),
    each package once: a module whose parent package is also new is left out."""
    new = sorted(name for name in sys.modules.copy() if name not in before)
    fresh = set(new)
    return [name for name in new if name.rpartition(".")[0] not in fresh]


def off_record() -> dict[str, Any]:
    """``metadata.warmup`` of a row run with the warm-up switched off (``run.warmup: false``)."""
    return {"enabled": False, "complete": False, "notes": []}


def rerun_record(failure: str) -> dict[str, Any]:
    """``metadata.warmup`` of a row the runner ran again without the warm-up,
    because the first worker ended (``failure`` says how) before it finished."""
    return {"enabled": True, "complete": False, "rerun_without_warmup": True, "notes": [failure]}


def write_state(path: Path | None, state: dict[str, Any]) -> None:
    """Write the warm-up's progress for the runner; a failure is ignored (the
    runner then reads no state, as for a worker that died before the warm-up).

    Atomic (a temporary file renamed over ``path``) but not synced to disk:
    only the runner, which outlives the worker, reads it, so a power loss
    need not be survived, and an fsync per part would cost real time on a
    network file system (the campaign's artifacts are on a Modal volume).
    """
    if path is None:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(state, default=str))
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError):
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def read_state(path: Path) -> dict[str, Any] | None:
    """The state a worker last wrote to ``path``; None when there is none or it does not parse."""
    try:
        state = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return state if isinstance(state, dict) else None


def finished_seconds(path: Path) -> float | None:
    """How long a finished warm-up took, from its state file; None while it
    runs, or without one. The runner adds it to the worker's timeout."""
    state = read_state(path)
    if not state or state.get("state") != DONE:
        return None
    seconds = (state.get("record") or {}).get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return None
    return max(0.0, float(seconds))


@contextlib.contextmanager
def _stdout_to_null() -> Iterator[None]:
    """Send standard output to the null device, so the tiny solve adds
    nothing to the solve's ``stdout.log``: file descriptor 1, which C
    libraries write to, and ``sys.stdout``, which Python code and extension
    modules writing through Python (scs-python's ``PySys_WriteStdout``) use.
    Standard error is kept: a crash in the tiny solve shows there. Where the
    descriptor cannot be redirected, C output passes."""
    try:
        sys.stdout.flush()
    except (AttributeError, OSError, ValueError):
        pass
    _flush_c_stdio()
    with open(os.devnull, "w") as null_text, contextlib.redirect_stdout(null_text):
        try:
            saved = os.dup(1)
        except OSError:
            yield
            return
        try:
            os.dup2(null_text.fileno(), 1)
            yield
        finally:
            _flush_c_stdio()
            os.dup2(saved, 1)
            os.close(saved)


def _flush_c_stdio() -> None:
    """``fflush(NULL)``: write out what C code left in its stdio buffers."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (AttributeError, OSError, TypeError):
        pass
