"""Subprocess worker for one problem/solver solve."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from solver_benchmarks.core import status
from solver_benchmarks.core.environment import runtime_metadata
from solver_benchmarks.core.problem import CONE, QP, cone_dimensions, qp_dimensions
from solver_benchmarks.core.result import ProblemResult, to_jsonable
from solver_benchmarks.core.storage import atomic_write_text
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.solvers import get_solver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.payload).read_text())
    result = run_payload(payload)
    output = Path(payload["artifacts_dir"]) / "worker_result.json"
    # Atomic write so the parent never observes a half-serialized record.
    atomic_write_text(output, json.dumps(result.to_record(), indent=2))
    return 0


def run_payload(payload: dict) -> ProblemResult:
    artifacts_dir = Path(payload["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
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
        solver_result = solver.solve(problem, artifacts_dir)
        objective = _reported_objective(problem, solver_result.objective_value)
        _write_trace_if_needed(artifacts_dir, solver_result.trace)
        metadata = {
            **problem.metadata,
            **_problem_dimensions(problem),
            "environment_id": payload.get("environment_id"),
            "environment_metadata": payload.get("environment_metadata", {}),
            "runtime": runtime_metadata(solver_config["solver"]),
            "solver_extra": to_jsonable(solver_result.extra),
        }
        if payload.get("resume_signature") is not None:
            metadata["resume_signature"] = payload["resume_signature"]
        error = None
        if solver_result.status == status.SKIPPED_UNSUPPORTED:
            error = str(solver_result.info.get("reason", "Unsupported solve"))
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
            kkt=to_jsonable(solver_result.kkt) if solver_result.kkt else None,
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


def _reported_objective(problem, objective):
    if objective is None:
        return None
    data = problem.data
    value = float(objective) + float(data.get("r", 0.0) or 0.0)
    if data.get("obj_type") == "max":
        value = -value
    return value


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
