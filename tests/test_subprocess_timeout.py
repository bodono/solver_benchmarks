"""Pin the subprocess timeout / cleanup contract used by the runner.

The previous implementation never had a test for the timeout path,
which is the entire reason the audit-driven hardening (process
group, bounded waits, drain on teardown) was necessary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from solver_benchmarks.core.runner import _run_subprocess


def test_run_subprocess_timeout_returns_timed_out_and_writes_logs(tmp_path: Path):
    """A child that hangs past the timeout is killed cleanly and the
    log files reflect the partial output, with timed_out=True."""
    script = (
        "import sys, time;"
        "sys.stdout.write('starting\\n'); sys.stdout.flush();"
        "time.sleep(60)"
    )
    result = _run_subprocess(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        timeout=1.0,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stream_output=False,
    )
    assert result.timed_out is True
    # SIGTERM/SIGKILL produce a non-zero / negative returncode; the
    # important contract is that we don't hang.
    assert result.returncode != 0
    # Log file must contain at least the line emitted before the sleep.
    stdout_log = (tmp_path / "stdout.log").read_text()
    assert "starting" in stdout_log


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only: relies on process groups")
def test_run_subprocess_timeout_kills_grandchildren(tmp_path: Path):
    """If the worker forks helpers, killing the direct child must not
    leave them holding the stdout pipe and blocking wait()."""
    script_path = tmp_path / "fork_and_sleep.py"
    script_path.write_text(
        "import os, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "else:\n"
        "    sys.stdout.write('parent_started\\n')\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(60)\n"
    )
    result = _run_subprocess(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        timeout=1.5,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stream_output=False,
    )
    assert result.timed_out is True
