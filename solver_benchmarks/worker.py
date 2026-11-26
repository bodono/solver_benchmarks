"""Subprocess worker for one problem/solver solve."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import traceback
from pathlib import Path

from solver_benchmarks.core import status, warmup
from solver_benchmarks.core.atomic import atomic_write_text
from solver_benchmarks.core.environment import runtime_metadata
from solver_benchmarks.core.parent_watch import watch_parent
from solver_benchmarks.core.problem import CONE, QP, cone_dimensions, qp_dimensions
from solver_benchmarks.core.result import ProblemResult, to_jsonable
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.solvers import get_solver
from solver_benchmarks.solvers.base import CONFIG_ERROR_KIND, SolverConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)
    _die_with_runner(_runner_pid())
    payload = json.loads(Path(args.payload).read_text())
    if payload.get("memory_limit_mb"):
        _limit_memory(payload["memory_limit_mb"])
    result = run_payload(payload)
    output = Path(payload["artifacts_dir"]) / "worker_result.json"
    # Atomic write so the parent never observes a half-serialized record.
    atomic_write_text(output, json.dumps(result.to_record(), indent=2))
    return 0


# Set by the runner in the worker's environment (runner.RUNNER_PID_ENV); the
# module run by hand has none and does not watch its parent.
_RUNNER_PID_ENV = "SOLVER_BENCHMARKS_RUNNER_PID"


def _die_with_runner(runner_pid: int | None) -> str | None:
    """Make this worker die when the runner that started it dies.

    The worker runs in its own session, so neither a terminal hang-up nor
    the runner's death reaches it, and the time limit is enforced by the
    runner: a runner killed hard (SIGKILL, the OOM killer) would leave the
    worker solving unsupervised. See ``parent_watch.watch_parent``: on
    Linux the kernel sends SIGKILL when the runner dies, also while the
    solver holds the GIL in a long C call (the runner starts each worker
    from the thread that then waits for it, so that thread outlives the
    worker); elsewhere, or behind a launcher, a daemon thread checks once a
    second. Without ``runner_pid`` (no runner set it in the environment),
    and on Windows, nothing is installed.
    """
    return watch_parent(runner_pid, signum=signal.SIGKILL, on_gone=lambda: _die(runner_pid))


def _runner_pid() -> int | None:
    try:
        return int(os.environ[_RUNNER_PID_ENV])
    except (KeyError, ValueError):
        return None


def _die(runner_pid: int | None) -> None:
    """Say why, then kill this worker and the helpers in its process group."""
    try:
        print(f"worker: the runner (pid {runner_pid}) is gone; exiting", file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass
    if hasattr(os, "killpg") and os.getpgrp() == os.getpid():
        os.killpg(0, signal.SIGKILL)
    os._exit(1)


def _limit_memory(limit_mb: int) -> None:
    """Cap this process's data segment."""
    import resource

    limit_bytes = int(limit_mb) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))


def run_payload(payload: dict) -> ProblemResult:
    artifacts_dir = Path(payload["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    warmup_record: dict | None = None
    try:
        dataset_cls = get_dataset(payload.get("dataset_name", payload["dataset"]))
        dataset = dataset_cls(
            repo_root=payload.get("repo_root"),
            **payload.get("dataset_options", {}),
        )
        problem = dataset.load_problem(payload["problem"])
        solver_config = payload["solver"]
        solver_cls = get_solver(solver_config["solver"])
        solver = solver_cls(solver_config.get("settings", {}))
        # The per-process loading (imports, libraries, a GPU's context and
        # kernels, the solver's first run) is done here, before the adapter's
        # timed region: see core/warmup.py. The record says what ran.
        warmup_record = _warm_up(payload, solver, problem, artifacts_dir)
        before_solve = set(sys.modules.copy())
        try:
            solver_result = solver.solve(problem, artifacts_dir)
        finally:
            # Modules first imported during the adapter's whole ``solve`` call
            # (its timed region and its own post-processing): what the
            # warm-up did not load, empty when it loaded everything. With the
            # warm-up off it also lists what adapters import before their
            # timers (the solver's package), so there it is not a list of
            # what was timed.
            warmup_record["imported_during_solve"] = warmup.imported_since(before_solve)
        objective = _reported_objective(problem, solver_result.objective_value)
        _write_trace_if_needed(artifacts_dir, solver_result.trace)
        dimensions = _problem_dimensions(problem)
        runtime = runtime_metadata(solver_config["solver"])
        metadata = {
            **problem.metadata,
            **dimensions,
            "environment_id": payload.get("environment_id"),
            "environment_metadata": payload.get("environment_metadata", {}),
            "runtime": runtime,
            "solver_extra": to_jsonable(solver_result.extra),
            "requested_tolerance": _requested_tolerance(
                solver_cls, solver_config.get("settings", {}), problem.kind, dimensions, runtime
            ),
        }
        if payload.get("resume_signature") is not None:
            metadata["resume_signature"] = payload["resume_signature"]
        metadata["warmup"] = warmup_record
        error = None
        if solver_result.status == status.SKIPPED_UNSUPPORTED:
            error = str(solver_result.info.get("reason", "Unsupported solve"))
        elif solver_result.status == status.SOLVER_ERROR and solver_result.info.get("error"):
            # Adapters put the reason of a solver_error on info["error"]; the
            # row's error field is what reports show.
            error = str(solver_result.info["error"])
        kkt = to_jsonable(solver_result.kkt) if solver_result.kkt else None
        point_objective = _point_objective(problem, kkt)
        if point_objective is not None and kkt is not None:
            kkt["primal_obj_reported"] = point_objective
        return ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=problem.name,
            problem_kind=problem.kind,
            solver_id=solver_config["id"],
            solver=solver_config["solver"],
            status=solver_result.status,
            objective_value=objective,
            iterations=solver_result.iterations,
            run_time_seconds=solver_result.run_time_seconds,
            setup_time_seconds=solver_result.setup_time_seconds,
            solve_time_seconds=solver_result.solve_time_seconds,
            error=error,
            artifact_dir=str(artifacts_dir),
            metadata=metadata,
            info=to_jsonable(solver_result.info),
            kkt=kkt,
        )
    except Exception as exc:
        traceback.print_exc()
        solver_config = payload["solver"]
        metadata = {
            "environment_id": payload.get("environment_id"),
            "environment_metadata": payload.get("environment_metadata", {}),
            "runtime": runtime_metadata(solver_config["solver"]),
        }
        if payload.get("resume_signature") is not None:
            metadata["resume_signature"] = payload["resume_signature"]
        if warmup_record is not None:
            metadata["warmup"] = warmup_record
        if isinstance(exc, SolverConfigError):
            metadata["error_kind"] = CONFIG_ERROR_KIND
        return ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload.get("problem_kind", "unknown"),
            solver_id=solver_config["id"],
            solver=solver_config["solver"],
            status=status.WORKER_ERROR,
            objective_value=None,
            iterations=None,
            run_time_seconds=None,
            error=f"{type(exc).__name__}: {exc}",
            artifact_dir=str(artifacts_dir),
            metadata=metadata,
        )


def _warm_up(payload: dict, solver, problem, artifacts_dir: Path) -> dict:
    """``metadata.warmup`` for this solve, after running the warm-up unless
    the run switched it off (``run.warmup``, the payload's ``warmup``) or the
    runner is running the solve again because a first worker ended in it
    (``warmup_failure``: how)."""
    failure = payload.get("warmup_failure")
    if failure:
        return warmup.rerun_record(str(failure))
    if not payload.get("warmup", True):
        return warmup.off_record()
    return warmup.run(solver, problem, artifacts_dir / warmup.STATE_FILE)


def _requested_tolerance(solver_cls, settings, problem_kind, dimensions, runtime) -> dict:
    """``metadata.requested_tolerance``: the tolerance the settings asked the
    solver for (``SolverAdapter.requested_tolerance``), which the KKT
    verification of the row multiplies by its slack, or ``{"unknown": why}``.
    Never fails the solve: the loader then looks in the run's manifest."""
    nnz_p = dimensions.get("nnz_p")
    try:
        return solver_cls.requested_tolerance(
            settings or {},
            problem_kind=problem_kind,
            is_lp=None if nnz_p is None else nnz_p == 0,
            versions=(runtime or {}).get("solver_package_versions") or {},
        ).to_record()
    except Exception as exc:  # recorded, not raised: the solve itself succeeded
        return {"unknown": f"{type(exc).__name__}: {exc}"}


def _reported_objective(problem, objective):
    if objective is None:
        return None
    data = problem.data
    value = float(objective) + float(data.get("r", 0.0) or 0.0)
    if data.get("obj_type") == "max":
        value = -value
    return value


def _point_objective(problem, kkt):
    """The returned point's objective (``kkt.primal_obj``) as ``objective_value`` reports it.

    ``objective_value`` is only reported for a solution, so this is what
    the KKT verification (analysis/verify.py) gives a limit stop it promotes
    to optimal. None when the record has no finite primal objective.
    """
    if not kkt:
        return None
    value = kkt.get("primal_obj")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return _reported_objective(problem, value)


def _problem_dimensions(problem):
    if problem.kind == QP:
        return qp_dimensions(problem.qp)
    if problem.kind == CONE:
        return cone_dimensions(problem.cone)
    return {}


def _write_trace_if_needed(artifacts_dir: Path, trace: list[dict]) -> None:
    # Always overwrite so a re-run on the same artifacts_dir does not
    # leave a stale trace from a previous attempt.
    path = artifacts_dir / "trace.jsonl"
    if not trace:
        path.unlink(missing_ok=True)
        return
    body = "".join(
        json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in trace
    )
    atomic_write_text(path, body)


if __name__ == "__main__":
    raise SystemExit(main())
