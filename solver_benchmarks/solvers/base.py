"""Solver adapter interface."""

from __future__ import annotations

import math
import numbers
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import ProblemData
from solver_benchmarks.core.result import SolverResult


class SolverUnavailable(RuntimeError):
    pass


class SolverConfigError(ValueError):
    """A setting name, value or alias that an adapter or its solver rejects.

    A configuration error, not the solver failing a problem: adapters raise it
    (never return ``solver_error``) so the row records the mistake and the
    setting can be fixed. It is a ``ValueError``, so callers catching that
    still see it.
    """


# ``metadata.error_kind`` the worker gives the worker_error row of a solve that
# raised SolverConfigError; the runner stops the run on it, since every other
# solve with the same settings would fail the same way.
CONFIG_ERROR_KIND = "config"


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
            resolved = coerce_time_limit(key, value)
    return resolved


def coerce_time_limit(key: str, value: Any) -> float:
    message = (
        f"Setting {key!r} must be a finite positive number of seconds or null "
        f"(got {value!r})"
    )
    if isinstance(value, bool):
        raise SolverConfigError(message)
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise SolverConfigError(message) from exc
    if not math.isfinite(coerced) or coerced < 0:
        raise SolverConfigError(message)
    if coerced == 0:
        raise SolverConfigError(
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


def translate_tolerance_aliases(
    settings: dict[str, Any],
    abs_keys: tuple[str, ...] = ("eps_abs",),
    rel_keys: tuple[str, ...] = ("eps_rel",),
) -> dict[str, float]:
    """Map the shared tolerance aliases onto a solver's own names, in place.

    ``eps_abs`` (or ``eps``) supplies every name in ``abs_keys`` and
    ``eps_rel`` (or ``eps``) every name in ``rel_keys``. The aliases are
    popped. An alias set to null counts as unset (so ``eps`` still applies
    when ``eps_abs: null``), and a solver name the config gives a non-null
    value always wins. For a solver whose own names are ``eps_abs`` /
    ``eps_rel`` an explicitly given value is passed through unchanged, for
    the solver to validate. Returns the settings inserted from an alias,
    for ``info["settings_translated"]``; a value that is not a number raises
    ``SolverConfigError`` naming the alias.
    """
    eps = settings.pop("eps", None)
    given = {"eps_abs": settings.pop("eps_abs", None), "eps_rel": settings.pop("eps_rel", None)}
    translated: dict[str, float] = {}
    for alias, keys in (("eps_abs", abs_keys), ("eps_rel", rel_keys)):
        source = alias if given[alias] is not None else "eps"
        value = given[alias] if given[alias] is not None else eps
        if value is None:
            continue
        for key in keys:
            if settings.get(key) is not None:
                continue
            if key == source:
                settings[key] = value
                continue
            settings[key] = _coerce_tolerance(source, value)
            translated[key] = settings[key]
    return translated


def translate_eps_alias(settings: dict[str, Any]) -> dict[str, float]:
    """Expand ``eps`` into ``eps_abs`` and ``eps_rel`` for solvers that take both.

    :func:`translate_tolerance_aliases` for the solver names ``eps_abs`` /
    ``eps_rel``.
    """
    return translate_tolerance_aliases(settings, ("eps_abs",), ("eps_rel",))


def _coerce_tolerance(alias: str, value: Any) -> float:
    if isinstance(value, bool):
        raise SolverConfigError(f"Setting {alias!r} must be a number (got {value!r})")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SolverConfigError(f"Setting {alias!r} must be a number (got {value!r})") from exc


def refuse_settings(settings: dict[str, Any], names, solver: str, hint: str) -> None:
    """Raise SolverConfigError if ``settings`` uses any of ``names``.

    For shared aliases a solver has no faithful equivalent for: refusing them
    keeps a config from silently running the solver at its defaults. A name
    set to null counts as unset and is dropped.
    """
    for name in names:
        if name in settings and settings[name] is None:
            del settings[name]
    present = sorted(name for name in names if name in settings)
    if present:
        raise SolverConfigError(f"{solver} does not accept {present}: {hint}")


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
        raise SolverConfigError(f"threads must be >= 0 (got {resolved})")
    return resolved


# Thread count for solvers that have a thread setting, when neither
# ``threads`` nor the solver's own name for it is configured. One thread
# keeps timings comparable with the single-threaded solvers, and the runner
# may run several workers in parallel, which a solver using every core would
# oversubscribe. ``threads: 0`` leaves the choice to the solver.
DEFAULT_THREADS = 1


def threads_or_default(threads: int | None) -> int:
    """The thread count to give a solver: ``threads`` if set, else one."""
    return DEFAULT_THREADS if threads is None else int(threads)


def _coerce_threads(key: str, value: Any) -> int:
    return coerce_int_setting(key, value)


def coerce_int_setting(key: str, value: Any, *, minimum: int | None = None) -> int:
    """The integer value of setting ``key``, else ``SolverConfigError`` naming it.

    An ``int``, a numpy integer or an integer-valued float (YAML reads ``1e3``
    as ``1000.0``) is accepted. A bool (``threads: true`` is not 1), a
    non-integral or non-finite float and a string are refused rather than
    truncated or parsed, as is a value below ``minimum``.
    """
    message = f"Setting {key!r} must be an integer"
    if minimum is not None:
        message += f" >= {minimum}"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SolverConfigError(f"{message} (got {value!r})")
    if isinstance(value, numbers.Integral):
        coerced = int(value)
    else:
        number = float(value)
        if not (math.isfinite(number) and number.is_integer()):
            raise SolverConfigError(f"{message} (got {value!r})")
        coerced = int(number)
    if minimum is not None and coerced < minimum:
        raise SolverConfigError(f"{message} (got {value!r})")
    return coerced


def pop_max_iter(settings: dict[str, Any], *, minimum: int = 1) -> int | None:
    """Pop the shared ``max_iter`` alias and return it as an integer.

    ``max_iter: null`` counts as unset (None); anything else must pass
    :func:`coerce_int_setting` with ``minimum``.
    """
    value = settings.pop("max_iter", None)
    if value is None:
        return None
    return coerce_int_setting("max_iter", value, minimum=minimum)


def normalize_max_iter(settings: dict[str, Any], *, minimum: int = 1) -> None:
    """For solvers whose own name is ``max_iter``: validate it in place
    (:func:`pop_max_iter`), dropping an explicit null."""
    value = pop_max_iter(settings, minimum=minimum)
    if value is not None:
        settings["max_iter"] = value


# Text of the TypeError a solver binding raises for a keyword argument it does
# not know or a value of the wrong type; a TypeError about problem data (for
# example SCS's "Ax must be a 1-D numpy array of floats") has none of these.
_KEYWORD_TYPE_ERROR_MARKERS = (
    "keyword argument",
    "incompatible function arguments",
    "cannot be interpreted as an integer",
    "must be real number",
    "argument ",
)


def is_setting_type_error(exc: BaseException) -> bool:
    """Whether a solver's TypeError is about a keyword (a setting), not data."""
    text = str(exc)
    return any(marker in text for marker in _KEYWORD_TYPE_ERROR_MARKERS)


def mark_threads_ignored(info: dict[str, Any], threads: int | None) -> None:
    """Record an unsupported `threads` request, mirroring
    ``mark_time_limit_ignored`` for solvers that have no thread knob."""
    if threads is not None:
        info["threads_ignored"] = True
        info["threads_requested"] = int(threads)

