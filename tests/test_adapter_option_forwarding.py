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
