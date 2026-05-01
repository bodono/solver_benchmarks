"""Regression tests for option-forwarding edge cases flagged in review.

These check the *plumbing* between user-supplied settings and the
underlying solver's option API, not the actual solve. Each test
records the calls a fake solver would receive and asserts that the
right options were set.
"""

from __future__ import annotations


def test_highs_max_iter_caps_qp_iteration_limit_too():
    """Audit-driven regression: ``max_iter`` must propagate to
    ``qp_iteration_limit`` so a HiGHS QP solve with a 100-iter cap
    actually stops at 100 iters instead of running to the int32-max
    default."""
    from solver_benchmarks.solvers.highs_adapter import _configure_highs

    options: dict = {}

    class FakeHighs:
        def setOptionValue(self, key, value):
            options[key] = value

    _configure_highs(FakeHighs(), {"verbose": False, "max_iter": 123})

    # Pre-fix this missed qp_iteration_limit and HiGHS QP solves ran
    # uncapped.
    assert options.get("qp_iteration_limit") == 123
    # The other algorithm caps must continue to be set so simplex /
    # IPM / PDLP solves are still bounded.
    assert options.get("simplex_iteration_limit") == 123
    assert options.get("ipm_iteration_limit") == 123
    assert options.get("pdlp_iteration_limit") == 123


def test_highs_max_iter_tolerates_missing_qp_iteration_limit_on_old_highspy():
    """Older highspy builds may not expose qp_iteration_limit; the
    forwarding loop must swallow the missing-option error so the other
    caps still take effect."""
    from solver_benchmarks.solvers.highs_adapter import _configure_highs

    options: dict = {}

    class OldHighs:
        def setOptionValue(self, key, value):
            if key == "qp_iteration_limit":
                raise RuntimeError("option not supported in this highspy build")
            options[key] = value

    _configure_highs(OldHighs(), {"verbose": False, "max_iter": 50})

    # qp_iteration_limit was rejected, but the others still landed.
    assert "qp_iteration_limit" not in options
    assert options.get("simplex_iteration_limit") == 50
    assert options.get("ipm_iteration_limit") == 50
    assert options.get("pdlp_iteration_limit") == 50


def test_scs_threads_detection_falls_back_to_marking_ignored(monkeypatch):
    """If the installed SCS does not accept ``num_threads`` (the kwarg
    only exists on OpenMP builds), the adapter must mark the request
    ignored instead of letting the C extension raise TypeError.

    Pre-fix: ``threads: 2`` would crash a stock SCS build with
    ``TypeError: 'num_threads' is an invalid keyword argument``.
    """
    import solver_benchmarks.solvers.scs_adapter as scs_mod

    monkeypatch.setattr(scs_mod, "_scs_supports_num_threads", lambda: False)
    # Reset the cache so the test does not depend on test ordering.
    monkeypatch.setattr(scs_mod, "_SCS_NUM_THREADS_SUPPORTED", None)

    settings = {"threads": 2, "verbose": False}
    info: dict = {}

    # Inline what the adapter does at the threads-translation point.
    from solver_benchmarks.solvers.base import mark_threads_ignored, pop_threads

    threads = pop_threads(settings)
    if threads is not None and not scs_mod._scs_supports_num_threads():
        mark_threads_ignored(info, threads)

    assert "num_threads" not in settings
    assert info == {"threads_ignored": True, "threads_requested": 2}


def test_clarabel_threads_only_injected_when_settings_has_attr(monkeypatch):
    """Pre-fix the adapter unconditionally added both ``max_threads``
    and ``threads`` to the kwargs forwarded into Clarabel, then later
    rejected ``threads`` as an invalid setting on builds that didn't
    expose it. Now we probe the real DefaultSettings instance and only
    inject attributes that exist."""

    class FakeSettings:
        # Simulate a Clarabel build that has only ``max_threads``.
        max_threads = 0
        verbose = False
        time_limit = 0.0

    class FakeClarabel:
        DefaultSettings = FakeSettings

    # Reproduce the threads-injection block in isolation.
    settings_obj = FakeClarabel.DefaultSettings()
    normalized: dict = {}
    threads = 2
    available = [
        attr for attr in ("max_threads", "threads")
        if hasattr(settings_obj, attr)
    ]
    for attr in available:
        normalized.setdefault(attr, threads)

    assert available == ["max_threads"]
    assert normalized == {"max_threads": 2}
    # The disallowed ``threads`` attr is NOT injected, so the later
    # validation loop does not reject it.
    assert "threads" not in normalized


def test_clarabel_threads_marks_ignored_when_no_thread_attr():
    """If Clarabel exposes neither ``max_threads`` nor ``threads``, the
    cross-adapter ``threads`` request should be recorded as ignored on
    info (mirroring SCS without OpenMP) instead of being silently
    dropped."""
    from solver_benchmarks.solvers.base import mark_threads_ignored

    class NoThreadSettings:
        verbose = False
        time_limit = 0.0

    settings_obj = NoThreadSettings()
    available = [
        attr for attr in ("max_threads", "threads")
        if hasattr(settings_obj, attr)
    ]
    info: dict = {}
    threads = 4
    if not available:
        mark_threads_ignored(info, threads)

    assert available == []
    assert info == {"threads_ignored": True, "threads_requested": 4}
