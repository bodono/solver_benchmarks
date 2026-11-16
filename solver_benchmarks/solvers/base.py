"""Solver adapter interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import ProblemData
from solver_benchmarks.core.result import SolverResult


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


# Cross-adapter reporting rules. A solver may stop with a point for any
# status in ``status.ANY_FEASIBLE`` (optimal, inaccurate, iteration or time
# limit, error); adapters compute KKT residuals for that point whenever it is
# finite, so ``bench kkt-verify --promote`` judges every solver's limit-hit
# iterate by the same check. Only a solution, though, has an objective value:
# ``objective_value`` is reported for these statuses and is None for limits,
# certificates and errors (the point's own objective is on kkt.primal_obj,
# and the worker records it in the convention of objective_value as
# kkt.primal_obj_reported, which kkt-verify --promote uses for a row it
# promotes to optimal).
OBJECTIVE_STATUSES = frozenset({status.OPTIMAL, status.OPTIMAL_INACCURATE})


def reported_objective(mapped_status: str, value: Any) -> float | None:
    """``objective_value`` for a result: ``value`` if the status has a solution."""
    if mapped_status not in OBJECTIVE_STATUSES or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def finite_point(*arrays: Any) -> bool:
    """True when every array is present and holds only finite numbers."""
    for array in arrays:
        if array is None:
            return False
        values = np.asarray(array, dtype=float)
        if not np.all(np.isfinite(values)):
            return False
    return True


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

    The limit must be a finite number of seconds greater than zero. Zero is
    refused rather than forwarded because solvers disagree on what it means:
    SCS reads it as "no limit", Clarabel, cpo3, HiGHS and PDLP stop before
    the first iteration, and OSQP rejects it as an invalid setting. To run
    without a limit, leave the setting out (or set it to null).
    """
    resolved: float | None = None
    for key in TIME_LIMIT_KEYS:
        value = settings.pop(key, None)
        if resolved is None and value is not None:
            resolved = _coerce_time_limit(key, value)
    return resolved


def _coerce_time_limit(key: str, value: Any) -> float:
    message = (
        f"Setting {key!r} must be a finite positive number of seconds or null "
        f"(got {value!r})"
    )
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(coerced) or coerced < 0:
        raise ValueError(message)
    if coerced == 0:
        raise ValueError(
            f"{message}; a zero limit means 'no limit' to some solvers and 'stop "
            "immediately' to others, so to run without a limit omit the setting"
        )
    return coerced


def mark_time_limit_ignored(info: dict[str, Any], time_limit: float | None) -> None:
    """Record on the info dict that a configured time limit could not be
    forwarded to the solver, so callers can detect the case from
    inspecting result.info instead of relying on undocumented behavior.
    """
    if time_limit is not None:
        info["time_limit_ignored"] = True
        info["time_limit_seconds"] = float(time_limit)


# The shared tolerance aliases. ``eps`` sets both ``eps_abs`` and ``eps_rel``.
EPS_ALIASES: tuple[str, ...] = ("eps", "eps_abs", "eps_rel")


def translate_eps_alias(settings: dict[str, Any]) -> dict[str, float]:
    """Expand ``eps`` into ``eps_abs`` and ``eps_rel`` for solvers that take both.

    An explicit ``eps_abs`` / ``eps_rel`` wins. Returns the settings actually
    inserted, for adapters to record as ``info["settings_translated"]``.
    """
    eps = settings.pop("eps", None)
    translated: dict[str, float] = {}
    if eps is not None:
        for key in ("eps_abs", "eps_rel"):
            if settings.get(key) is None:
                settings[key] = float(eps)
                translated[key] = float(eps)
    return translated


def refuse_settings(settings: dict[str, Any], names, solver: str, hint: str) -> None:
    """Raise ValueError if ``settings`` uses any of ``names``.

    For shared aliases a solver has no faithful equivalent for: refusing them
    keeps a config from silently running the solver at its defaults.
    """
    present = sorted(name for name in names if name in settings)
    if present:
        raise ValueError(f"{solver} does not accept {present}: {hint}")


THREADS_KEYS: tuple[str, ...] = ("threads", "num_threads")


def pop_threads(settings: dict[str, Any]) -> int | None:
    """Pop the solver-agnostic thread-count alias from ``settings``.

    Returns the integer value (or ``None`` if not set). Adapters whose
    backend exposes a thread count knob should pass the result through
    to the solver-specific option (e.g. ``Threads`` for Gurobi,
    ``MSK_IPAR_NUM_THREADS`` for MOSEK, ``threads`` for HiGHS).

    Booleans and non-integral floats are rejected explicitly. Without
    that, ``int(True)`` would silently become ``1`` and ``int(1.9)``
    would silently truncate to ``1`` even though the error message
    promises an integer.
    """
    resolved: int | None = None
    for key in THREADS_KEYS:
        value = settings.pop(key, None)
        if resolved is None and value is not None:
            resolved = _coerce_threads(key, value)
    if resolved is not None and resolved < 0:
        raise ValueError(f"threads must be >= 0 (got {resolved})")
    return resolved


def _coerce_threads(key: str, value: Any) -> int:
    # ``bool`` is a subclass of ``int``; reject it explicitly so YAML
    # ``threads: true`` doesn't quietly become ``threads = 1``.
    if isinstance(value, bool):
        raise ValueError(
            f"Setting {key!r} must be an integer (got {value!r})"
        )
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Setting {key!r} must be an integer (got {value!r})"
            )
        return int(value)
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Setting {key!r} must be an integer (got {value!r})"
        ) from exc
    return coerced


def mark_threads_ignored(info: dict[str, Any], threads: int | None) -> None:
    """Record an unsupported `threads` request, mirroring
    ``mark_time_limit_ignored`` for solvers that have no thread knob."""
    if threads is not None:
        info["threads_ignored"] = True
        info["threads_requested"] = int(threads)

