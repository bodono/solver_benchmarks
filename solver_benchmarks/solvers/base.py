"""Solver adapter interface."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from solver_benchmarks.core import status as canonical
from solver_benchmarks.core.problem import ProblemData
from solver_benchmarks.core.result import SolverResult

logger = logging.getLogger(__name__)


class SolverUnavailable(RuntimeError):
    pass


class SolverAdapter(ABC):
    solver_name: str
    supported_problem_kinds: set[str]

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = dict(settings or {})

    @classmethod
    def is_available(cls) -> bool:
        return True

    def supports(self, problem_kind: str) -> bool:
        return problem_kind in self.supported_problem_kinds

    @abstractmethod
    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        raise NotImplementedError


# Default to non-verbose. Batch benchmark runs would otherwise have every
# solver print its own header/footer and intermediate progress, which
# makes the per-solve stdout.log unreadable. The runner's
# stream_solver_output knob already controls whether the worker's stdout
# is mirrored to the operator console; individual adapters should not
# also opt in by default.
DEFAULT_VERBOSE = False


def settings_with_defaults(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(settings)
    normalized.setdefault("verbose", DEFAULT_VERBOSE)
    return normalized


# Common aliases for the solver-agnostic time-limit setting. Adapters
# look at both forms because some users carry over conventions from
# other libraries (e.g. SCS uses time_limit_secs, Gurobi/MOSEK use
# different native names which their adapters translate).
TIME_LIMIT_KEYS: tuple[str, ...] = ("time_limit", "time_limit_sec", "time_limit_secs")


def pop_time_limit(settings: dict[str, Any]) -> float | None:
    """Pop the common time-limit aliases from ``settings`` and return
    the resolved value (or ``None`` if not configured).

    If multiple aliases are set, the first non-None encountered (in the
    order of ``TIME_LIMIT_KEYS``) wins. All aliases are popped to keep
    later forwarding to the solver clean.
    """
    resolved: float | None = None
    for key in TIME_LIMIT_KEYS:
        value = settings.pop(key, None)
        if resolved is None and value is not None:
            try:
                resolved = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Setting {key!r} must be a number or null (got {value!r})"
                ) from exc
    return resolved


def mark_time_limit_ignored(info: dict[str, Any], time_limit: float | None) -> None:
    """Record on the info dict that a configured time limit could not be
    forwarded to the solver, so callers can detect the case from
    inspecting result.info instead of relying on undocumented behavior.
    """
    if time_limit is not None:
        info["time_limit_ignored"] = True
        info["time_limit_seconds"] = float(time_limit)


THREADS_KEYS: tuple[str, ...] = ("threads", "num_threads")


def pop_threads(settings: dict[str, Any]) -> int | None:
    """Pop the solver-agnostic thread-count alias from ``settings``.

    Returns the integer value (or ``None`` if not set). Adapters whose
    backend exposes a thread count knob should pass the result through
    to the solver-specific option (e.g. ``Threads`` for Gurobi,
    ``MSK_IPAR_NUM_THREADS`` for MOSEK, ``threads`` for HiGHS).
    """
    resolved: int | None = None
    for key in THREADS_KEYS:
        value = settings.pop(key, None)
        if resolved is None and value is not None:
            try:
                resolved = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Setting {key!r} must be an integer (got {value!r})"
                ) from exc
    if resolved is not None and resolved < 0:
        raise ValueError(f"threads must be >= 0 (got {resolved})")
    return resolved


def mark_threads_ignored(info: dict[str, Any], threads: int | None) -> None:
    """Record an unsupported `threads` request, mirroring
    ``mark_time_limit_ignored`` for solvers that have no thread knob."""
    if threads is not None:
        info["threads_ignored"] = True
        info["threads_requested"] = int(threads)


OLD_STATUS_MAP = {
    "optimal": canonical.OPTIMAL,
    "optimal inaccurate": canonical.OPTIMAL_INACCURATE,
    "optimal_inaccurate": canonical.OPTIMAL_INACCURATE,
    "primal infeasible": canonical.PRIMAL_INFEASIBLE,
    "primal_infeasible": canonical.PRIMAL_INFEASIBLE,
    "primal infeasible inaccurate": canonical.PRIMAL_INFEASIBLE_INACCURATE,
    "dual infeasible": canonical.DUAL_INFEASIBLE,
    "dual_infeasible": canonical.DUAL_INFEASIBLE,
    "dual infeasible inaccurate": canonical.DUAL_INFEASIBLE_INACCURATE,
    "primal or dual infeasible": canonical.PRIMAL_OR_DUAL_INFEASIBLE,
    "solver_error": canonical.SOLVER_ERROR,
    "max_iter_reached": canonical.MAX_ITER_REACHED,
    "time_limit": canonical.TIME_LIMIT,
}


def normalize_status(status: Any) -> str:
    key = str(status)
    if key in OLD_STATUS_MAP:
        return OLD_STATUS_MAP[key]
    lower = key.lower()
    if lower in OLD_STATUS_MAP:
        return OLD_STATUS_MAP[lower]
    # Log so an adapter that forgets to extend its mapping shows up in
    # the debug stream rather than silently producing SOLVER_ERROR.
    logger.warning("normalize_status: unmapped solver status %r", status)
    return canonical.SOLVER_ERROR


def qp_namespace(problem: ProblemData):
    return SimpleNamespace(qp_problem=problem.qp, prob_name=problem.name)
