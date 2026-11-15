"""Signals during `bench env run` reach the child `bench run` so that it can
kill its workers, and only the first stop signal unwinds a run."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from solver_benchmarks.cli import terminate_like_interrupt
from solver_benchmarks.core import env_runner

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signals")

# A stand-in for a child `bench run`: on SIGTERM it records that it cleaned up
# (as the real child kills its workers) and exits; otherwise it sleeps.
_CHILD = """
import pathlib, signal, sys, time
marker = pathlib.Path(sys.argv[1])
ignore = sys.argv[2] == "ignore"
def stop(signum, frame):
    marker.write_text("cleaned up")
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN if ignore else stop)
marker.with_suffix(".ready").write_text("ready")
time.sleep(60)
"""


def _signal_self_when_ready(marker: Path, signum: int) -> threading.Thread:
    def send():
        deadline = time.monotonic() + 20
        while not marker.with_suffix(".ready").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        os.kill(os.getpid(), signum)

    thread = threading.Thread(target=send, daemon=True)
    thread.start()
    return thread


@pytest.mark.parametrize("signum, expected", [
    (signal.SIGTERM, SystemExit),
    (signal.SIGINT, KeyboardInterrupt),
])
def test_a_stop_signal_is_forwarded_so_the_child_can_clean_up(tmp_path, signum, expected):
    marker = tmp_path / "child"
    _signal_self_when_ready(marker, signum)
    started = time.monotonic()
    with pytest.raises(expected):
        with terminate_like_interrupt():
            env_runner._run_child([sys.executable, "-c", _CHILD, str(marker), "obey"], cwd=tmp_path)
    assert marker.read_text() == "cleaned up"
    assert time.monotonic() - started < 15


def test_a_child_that_ignores_sigterm_is_killed_after_the_grace(tmp_path, monkeypatch):
    monkeypatch.setattr(env_runner, "_CHILD_SHUTDOWN_SECONDS", 0.5)
    marker = tmp_path / "child"
    _signal_self_when_ready(marker, signal.SIGTERM)
    started = time.monotonic()
    with pytest.raises(SystemExit):
        with terminate_like_interrupt():
            env_runner._run_child([sys.executable, "-c", _CHILD, str(marker), "ignore"], cwd=tmp_path)
    assert not marker.exists()
    assert time.monotonic() - started < 15


def test_run_child_reports_a_failing_child(tmp_path):
    import subprocess

    with pytest.raises(subprocess.CalledProcessError) as raised:
        env_runner._run_child([sys.executable, "-c", "raise SystemExit(3)"], cwd=tmp_path)
    assert raised.value.returncode == 3


def test_only_the_first_stop_signal_unwinds_and_handlers_are_restored():
    previous = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    with pytest.raises(SystemExit) as raised:
        with terminate_like_interrupt():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(1)
            finally:
                # A second signal during cleanup is ignored, not raised.
                os.kill(os.getpid(), signal.SIGTERM)
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(0.2)
    assert raised.value.code == 128 + signal.SIGTERM
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == previous
