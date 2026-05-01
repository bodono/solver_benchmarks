"""Cross-adapter knob translation helpers in solvers/base.py.

These helpers (pop_time_limit, pop_threads, mark_*_ignored) are the
shared point that every adapter routes through. Pinning the contract
here lets the per-adapter migrations from the audit be small and
mechanical.
"""

from __future__ import annotations

import pytest

from solver_benchmarks.solvers.base import (
    mark_threads_ignored,
    mark_time_limit_ignored,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)


def test_pop_time_limit_accepts_each_alias():
    for alias in ("time_limit", "time_limit_sec", "time_limit_secs"):
        settings = {alias: 1.5}
        assert pop_time_limit(settings) == 1.5
        assert alias not in settings  # consumed


def test_pop_time_limit_first_alias_wins_and_pops_others():
    settings = {"time_limit": 1.0, "time_limit_sec": 2.0, "time_limit_secs": 3.0}
    assert pop_time_limit(settings) == 1.0
    # All three are consumed even when only the first contributed the value.
    assert settings == {}


def test_pop_time_limit_returns_none_when_unset():
    settings = {"eps_abs": 1e-6}
    assert pop_time_limit(settings) is None
    assert settings == {"eps_abs": 1e-6}


def test_pop_time_limit_rejects_non_numeric():
    with pytest.raises(ValueError, match="must be a number"):
        pop_time_limit({"time_limit": "fast"})


def test_pop_threads_basic():
    settings = {"threads": 4}
    assert pop_threads(settings) == 4
    assert settings == {}


def test_pop_threads_accepts_num_threads_alias():
    settings = {"num_threads": 8}
    assert pop_threads(settings) == 8
    assert settings == {}


def test_pop_threads_rejects_negative():
    with pytest.raises(ValueError, match=">= 0"):
        pop_threads({"threads": -1})


def test_pop_threads_rejects_bool():
    # ``bool`` is a subclass of ``int``; a YAML ``threads: true`` would
    # otherwise silently become ``threads = 1``.
    with pytest.raises(ValueError, match="must be an integer"):
        pop_threads({"threads": True})
    with pytest.raises(ValueError, match="must be an integer"):
        pop_threads({"threads": False})


def test_pop_threads_rejects_non_integer_float():
    # ``int(1.9)`` would silently truncate to 1; the helper should
    # refuse rather than guessing the user's intent.
    with pytest.raises(ValueError, match="must be an integer"):
        pop_threads({"threads": 1.9})


def test_pop_threads_accepts_integer_valued_float():
    # ``threads: 4.0`` (e.g. parsed from a JSON number) is unambiguous.
    assert pop_threads({"threads": 4.0}) == 4


def test_mark_time_limit_ignored_records_value():
    info: dict = {}
    mark_time_limit_ignored(info, 30.0)
    assert info == {"time_limit_ignored": True, "time_limit_seconds": 30.0}


def test_mark_time_limit_ignored_noop_when_none():
    info: dict = {}
    mark_time_limit_ignored(info, None)
    assert info == {}


def test_mark_threads_ignored_records_request():
    info: dict = {}
    mark_threads_ignored(info, 4)
    assert info == {"threads_ignored": True, "threads_requested": 4}


def test_settings_with_defaults_default_quiet():
    assert settings_with_defaults({})["verbose"] is False
