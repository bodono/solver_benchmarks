"""Solver result structures and JSON-safe conversion helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class SolverResult:
    status: str
    objective_value: float | None = None
    iterations: int | None = None
    run_time_seconds: float | None = None
    setup_time_seconds: float | None = None
    solve_time_seconds: float | None = None
    info: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    kkt: dict[str, Any] | None = None


@dataclass
class ProblemResult:
    run_id: str
    dataset: str
    problem: str
    problem_kind: str
    solver_id: str
    solver: str
    status: str
    objective_value: float | None
    iterations: int | None
    run_time_seconds: float | None
    setup_time_seconds: float | None = None
    solve_time_seconds: float | None = None
    error: str | None = None
    artifact_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    kkt: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_jsonable(value.item())
        if value.size > 100:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "preview": to_jsonable(value.reshape(-1)[:100]),
            }
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
