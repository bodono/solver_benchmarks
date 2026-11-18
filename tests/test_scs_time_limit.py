"""The SCS adapter's time limit covers setup as well as the solve.

SCS's own ``time_limit_secs`` covers only its iterations. With an SCS whose
``SCS.update`` takes settings (bodono/scs-python#267) the adapter constructs
the solver, times that, and gives the solve the limit minus the setup time;
with an older SCS it passes the whole limit at construction. A fake SCS and
a fake clock keep these tests independent of real timings; the tests at the
end run the real SCS when the installed one has the new ``update``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import scipy.sparse as sp

import solver_benchmarks.solvers.scs_adapter as scs_mod
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, ProblemData

_SOLVE_SETTINGS = frozenset({
    "max_iters", "eps_abs", "eps_rel", "eps_infeas", "alpha",
    "time_limit_secs", "verbose", "log_csv_filename",
})


def _cone_problem() -> ProblemData:
    # min x s.t. x >= 1, as -x + s = -1, s >= 0.
    data = {
        "P": None,
        "A": sp.csc_matrix([[-1.0]]),
        "b": np.array([-1.0]),
        "q": np.array([1.0]),
        "r": 0.0,
        "cone": {"l": 1},
    }
    return ProblemData("test", "p", CONE, data)


class _Clock:
    """Stands in for the adapter's ``time`` module; the fake SCS moves it."""

    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now


def _raw(status_val=1, text="solved", solve_ms=1000.0):
    return {
        # x=None: no KKT computation; these tests are about the time limit.
        "x": None, "y": None, "s": None,
        "info": {"status_val": status_val, "status": text, "iter": 40,
                 "pobj": 1.0, "setup_time": 1.0, "solve_time": solve_ms},
    }


def _fake_scs(clock, *, setup=3.0, solve=1.0, with_update=True, update_error=None, raw=None):
    """A fake scs module recording its calls.

    Constructing advances the clock by ``setup``, solving by ``solve``. With
    ``with_update`` it mirrors scs-python#267 (``SOLVE_SETTINGS`` exported,
    ``update(b=None, c=None, **settings)``); without, the older
    ``update(b=None, c=None)``.
    """
    calls = SimpleNamespace(constructed=[], updates=[], solves=0, module_solves=[])
    result = raw or _raw()

    class FakeSolver:
        def __init__(self, data, cone, **settings):
            calls.constructed.append(dict(settings))
            clock.now += setup
            self.settings = dict(settings)

        def solve(self, warm_start=True, x=None, y=None, s=None):
            calls.solves += 1
            clock.now += solve
            return result

        if with_update:
            def update(self, b=None, c=None, **settings):
                calls.updates.append(dict(settings))
                if update_error is not None:
                    raise update_error
                self.settings.update(settings)
        else:
            def update(self, b=None, c=None):
                calls.updates.append({"b": b, "c": c})

    def module_solve(data, cone, **settings):
        calls.module_solves.append(dict(settings))
        return FakeSolver(data, cone, **settings).solve()

    module = SimpleNamespace(SCS=FakeSolver, solve=module_solve)
    if with_update:
        module.SOLVE_SETTINGS = _SOLVE_SETTINGS
    return module, calls


def _run(monkeypatch, tmp_path, module, clock, settings):
    monkeypatch.setitem(sys.modules, "scs", module)
    monkeypatch.setattr(scs_mod, "time", clock)
    adapter = scs_mod.SCSSolverAdapter({"verbose": False, **settings})
    return adapter.solve(_cone_problem(), tmp_path)


def test_solve_gets_the_time_limit_minus_setup(monkeypatch, tmp_path):
    clock = _Clock()
    module, calls = _fake_scs(clock, setup=3.0, solve=1.0)
    result = _run(monkeypatch, tmp_path, module, clock, {"time_limit": 10.0})

    assert "time_limit_secs" not in calls.constructed[0]
    assert calls.updates == [{"time_limit_secs": pytest.approx(7.0)}]
    assert calls.solves == 1
    assert calls.module_solves == []
    assert result.status == status.OPTIMAL
    # run_time_seconds covers construction and solve.
    assert result.run_time_seconds == pytest.approx(4.0)
    assert result.info["scs_time_limit_mode"] == "solve_minus_setup"
    assert result.info["scs_setup_seconds"] == pytest.approx(3.0)
    assert result.info["scs_time_limit_secs_given"] == pytest.approx(7.0)


def test_limit_status_is_read_against_the_limit_the_solve_was_given(monkeypatch, tmp_path):
    # An inaccurate stop whose text does not name the limit, after 7 s of
    # solve: that is the remaining limit, though less than the 10 s limit.
    clock = _Clock()
    raw = _raw(status_val=2, text="solved (inaccurate)", solve_ms=7000.0)
    module, _ = _fake_scs(clock, setup=3.0, solve=7.0, raw=raw)
    result = _run(monkeypatch, tmp_path, module, clock, {"time_limit": 10.0})
    assert result.status == status.TIME_LIMIT


@pytest.mark.parametrize("setup", [10.0, 12.0])
def test_solve_is_skipped_when_setup_uses_the_whole_limit(monkeypatch, tmp_path, setup):
    # time_limit_secs=0 would mean no limit to SCS; the solve is not started.
    clock = _Clock()
    module, calls = _fake_scs(clock, setup=setup)
    result = _run(monkeypatch, tmp_path, module, clock, {"time_limit": 10.0})

    assert calls.updates == []
    assert calls.solves == 0
    assert result.status == status.TIME_LIMIT
    assert result.objective_value is None
    assert result.kkt is None
    assert result.iterations == 0
    assert result.run_time_seconds == pytest.approx(setup)
    assert result.setup_time_seconds == pytest.approx(setup)
    assert result.info["scs_time_limit_mode"] == "solve_minus_setup"
    assert result.info["scs_time_limit_secs_given"] is None
    assert result.info["scs_setup_seconds"] == pytest.approx(setup)
    assert "solve_skipped" in result.info


def test_scs_without_settings_in_update_gets_the_limit_at_construction(monkeypatch, tmp_path):
    clock = _Clock()
    module, calls = _fake_scs(clock, setup=3.0, solve=1.0, with_update=False)
    result = _run(monkeypatch, tmp_path, module, clock, {"time_limit": 10.0})

    assert calls.updates == []
    assert calls.module_solves == [{"verbose": False, "time_limit_secs": 10.0}]
    assert result.status == status.OPTIMAL
    assert result.run_time_seconds == pytest.approx(4.0)
    assert result.info["scs_time_limit_mode"] == "construction"
    assert result.info["scs_time_limit_secs_given"] == 10.0
    assert "scs_setup_seconds" not in result.info


def test_a_failing_update_falls_back_to_the_construction_limit(monkeypatch, tmp_path):
    clock = _Clock()
    module, calls = _fake_scs(clock, setup=3.0, solve=1.0, update_error=ValueError("refused"))
    result = _run(monkeypatch, tmp_path, module, clock, {"time_limit": 10.0})

    # The rebuild's solve gets the limit minus two setups (10 - 2 * 3).
    assert calls.module_solves == [{"verbose": False, "time_limit_secs": 4.0}]
    assert result.status == status.OPTIMAL
    # Both constructions and the solve.
    assert result.run_time_seconds == pytest.approx(7.0)
    assert result.info["scs_time_limit_mode"] == "construction_after_update_error"
    assert result.info["scs_time_limit_secs_given"] == 4.0
    assert result.info["scs_time_limit_update_error"] == "ValueError: refused"


def test_a_failing_update_skips_the_solve_when_two_setups_use_the_limit(monkeypatch, tmp_path):
    clock = _Clock()
    module, calls = _fake_scs(clock, setup=6.0, solve=1.0, update_error=ValueError("refused"))
    result = _run(monkeypatch, tmp_path, module, clock, {"time_limit": 10.0})

    assert calls.module_solves == []
    assert result.status == status.TIME_LIMIT
    assert result.info["scs_time_limit_mode"] == "construction_after_update_error"
    assert result.info["scs_time_limit_secs_given"] is None
    assert "two setups" in result.info["solve_skipped"]
    # Only one setup ran.
    assert result.setup_time_seconds == pytest.approx(6.0)


def test_no_time_limit_constructs_once_and_records_no_mode(monkeypatch, tmp_path):
    clock = _Clock()
    module, calls = _fake_scs(clock)
    result = _run(monkeypatch, tmp_path, module, clock, {})

    assert calls.updates == []
    assert calls.module_solves == [{"verbose": False}]
    assert result.status == status.OPTIMAL
    assert "scs_time_limit_mode" not in result.info


def test_update_support_detection():
    clock = _Clock()
    new, _ = _fake_scs(clock)
    old, _ = _fake_scs(clock, with_update=False)
    assert scs_mod._scs_update_takes_settings(new) is True
    assert scs_mod._scs_update_takes_settings(old) is False
    # An update taking keyword settings without the SOLVE_SETTINGS export.
    unexported = SimpleNamespace(SCS=new.SCS)
    assert scs_mod._scs_update_takes_settings(unexported) is True
    # SOLVE_SETTINGS without time_limit_secs.
    assert scs_mod._scs_update_takes_settings(
        SimpleNamespace(SOLVE_SETTINGS=frozenset({"max_iters"}), SCS=new.SCS)
    ) is False
    assert scs_mod._scs_update_takes_settings(SimpleNamespace()) is False
    # The MagicMock modules other tests install: not a settings update.
    assert scs_mod._scs_update_takes_settings(MagicMock()) is False


# ---------------------------------------------------------------------------
# The real SCS, when the installed one has the new update().
# ---------------------------------------------------------------------------


def _real_scs_with_update():
    scs = pytest.importorskip("scs")
    if not scs_mod._scs_update_takes_settings(scs):
        pytest.skip("the installed SCS's update() takes no settings (before scs-python#267)")
    return scs


def test_real_scs_solve_gets_the_rest_of_the_limit(tmp_path):
    _real_scs_with_update()
    adapter = scs_mod.SCSSolverAdapter({"verbose": False, "time_limit": 100.0})
    result = adapter.solve(_cone_problem(), tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(1.0, abs=1e-3)
    info = result.info
    assert info["scs_time_limit_mode"] == "solve_minus_setup"
    assert 0 < info["scs_time_limit_secs_given"] < 100.0
    assert info["scs_time_limit_secs_given"] + info["scs_setup_seconds"] == pytest.approx(100.0)


def test_real_scs_solve_is_skipped_when_setup_uses_the_limit(tmp_path):
    _real_scs_with_update()
    adapter = scs_mod.SCSSolverAdapter({"verbose": False, "time_limit": 1e-9})
    result = adapter.solve(_cone_problem(), tmp_path)
    assert result.status == status.TIME_LIMIT, result.info
    assert result.info["scs_time_limit_secs_given"] is None
    assert result.kkt is None
