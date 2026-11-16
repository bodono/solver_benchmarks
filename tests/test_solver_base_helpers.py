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
    with pytest.raises(ValueError, match="finite positive"):
        pop_time_limit({"time_limit": "fast"})


def test_pop_time_limit_rejects_bool():
    with pytest.raises(ValueError, match="finite positive"):
        pop_time_limit({"time_limit": True})
    with pytest.raises(ValueError, match="finite positive"):
        pop_time_limit({"time_limit": False})


def test_pop_time_limit_rejects_nonfinite_and_negative():
    for value in (float("nan"), float("inf"), -float("inf"), "nan", "inf", -1):
        with pytest.raises(ValueError, match="finite positive"):
            pop_time_limit({"time_limit": value})


@pytest.mark.parametrize("alias", ["time_limit", "time_limit_sec", "time_limit_secs"])
@pytest.mark.parametrize("value", [0, 0.0, "0", -0.0])
def test_pop_time_limit_rejects_zero(alias, value):
    # Solvers disagree on what a zero limit means (SCS: no limit; Clarabel,
    # cpo3, HiGHS, PDLP: stop before the first iteration; OSQP: a settings
    # error), so the shared helper refuses it instead of passing it on.
    with pytest.raises(ValueError, match=r"positive.*omit the setting"):
        pop_time_limit({alias: value})


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


# --- SolverConfigError and the shared tolerance aliases -----------------------


def test_config_errors_from_the_shared_helpers_are_solver_config_errors():
    from solver_benchmarks.solvers.base import SolverConfigError, refuse_settings

    assert issubclass(SolverConfigError, ValueError)
    with pytest.raises(SolverConfigError):
        pop_time_limit({"time_limit": 0})
    with pytest.raises(SolverConfigError):
        pop_threads({"threads": -1})
    with pytest.raises(SolverConfigError):
        pop_threads({"threads": 1.5})
    with pytest.raises(SolverConfigError):
        refuse_settings({"eps": 1e-6}, ("eps",), "X", "hint")


def test_tolerance_aliases_map_onto_solver_names():
    from solver_benchmarks.solvers.base import translate_tolerance_aliases

    settings = {"eps": 1e-6, "eps_rel": 1e-4, "tol_feas": 1e-9}
    translated = translate_tolerance_aliases(
        settings, ("tol_feas", "tol_gap_abs"), ("tol_gap_rel",)
    )
    # eps supplies the absolute names, eps_rel the relative one; a native
    # value wins and is not reported.
    assert settings == {"tol_feas": 1e-9, "tol_gap_abs": 1e-6, "tol_gap_rel": 1e-4}
    assert translated == {"tol_gap_abs": 1e-6, "tol_gap_rel": 1e-4}


def test_tolerance_alias_set_to_null_counts_as_unset():
    # eps_abs: null disabled the eps fallback in Clarabel, cpo3, ECOS, CVXOPT
    # and SDPA, and reached SCS as None.
    from solver_benchmarks.solvers.base import translate_tolerance_aliases

    settings = {"eps": 1e-6, "eps_abs": None, "tol_feas": None}
    translated = translate_tolerance_aliases(settings, ("tol_feas",), ("tol_gap_rel",))
    assert settings == {"tol_feas": 1e-6, "tol_gap_rel": 1e-6}
    assert translated == {"tol_feas": 1e-6, "tol_gap_rel": 1e-6}
    native = {"eps": 1e-6, "eps_abs": None}
    assert translate_tolerance_aliases(native) == {"eps_abs": 1e-6, "eps_rel": 1e-6}
    assert native == {"eps_abs": 1e-6, "eps_rel": 1e-6}


def test_native_eps_names_pass_through_unchanged():
    from solver_benchmarks.solvers.base import translate_tolerance_aliases

    settings = {"eps_abs": "1e-3"}
    assert translate_tolerance_aliases(settings) == {}
    assert settings == {"eps_abs": "1e-3"}  # left for the solver to judge


def test_tolerance_alias_that_is_not_a_number_is_a_config_error():
    from solver_benchmarks.solvers.base import SolverConfigError, translate_tolerance_aliases

    with pytest.raises(SolverConfigError, match="'eps'"):
        translate_tolerance_aliases({"eps": "tight"})
    with pytest.raises(SolverConfigError, match="'eps_abs'"):
        translate_tolerance_aliases({"eps_abs": True}, ("tol_feas",), ())


def test_refused_alias_set_to_null_counts_as_unset():
    from solver_benchmarks.solvers.base import EPS_ALIASES, refuse_settings

    settings = {"eps": None, "max_iter": 5}
    refuse_settings(settings, EPS_ALIASES, "X", "hint")
    assert settings == {"max_iter": 5}


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "cpo3", "ecos", "qtqp"])
def test_eps_with_null_eps_abs_sets_every_tolerance(solver_name):
    # One rule in every adapter: SCS passed None on (TypeError), Clarabel set
    # only tol_gap_rel.
    import importlib

    pytest.importorskip(solver_name)
    module = importlib.import_module(f"solver_benchmarks.solvers.{solver_name}_adapter")
    settings = {"eps": 1e-6, "eps_abs": None}
    if solver_name == "qtqp":
        module._translate_tolerances(settings, {"tol_feas", "tol_gap_abs", "tol_gap_rel"})
    else:
        module.translate_settings(settings)
    assert None not in settings.values()
    assert sorted(settings.values()) == [1e-6] * len(settings)
    assert len(settings) >= 2


def test_integer_setting_rule():
    import numpy as np

    from solver_benchmarks.solvers.base import (
        SolverConfigError,
        coerce_int_setting,
        pop_max_iter,
    )

    assert coerce_int_setting("k", 3) == 3
    assert coerce_int_setting("k", np.int32(3)) == 3
    assert coerce_int_setting("k", 1000.0) == 1000
    for value in (True, 2.5, float("nan"), "3", None, [3]):
        with pytest.raises(SolverConfigError, match="'k'"):
            coerce_int_setting("k", value)
    with pytest.raises(SolverConfigError, match=">= 1"):
        coerce_int_setting("k", 0, minimum=1)
    settings = {"max_iter": None}
    assert pop_max_iter(settings) is None and settings == {}
    assert pop_max_iter({"max_iter": 0}, minimum=0) == 0
    with pytest.raises(SolverConfigError):
        pop_threads({"threads": "4"})


def test_threads_or_default():
    from solver_benchmarks.solvers.base import DEFAULT_THREADS, threads_or_default

    assert DEFAULT_THREADS == 1
    assert threads_or_default(None) == 1
    assert threads_or_default(0) == 0
    assert threads_or_default(8) == 8
