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
    leave them running. The grandchild writes its PID to a file
    before sleeping; after _run_subprocess returns we probe that PID
    with kill(pid, 0) and assert it's gone. Without process-group
    escalation the grandchild would survive and the probe would
    succeed.
    """
    import errno
    import time as _time

    pid_file = tmp_path / "grandchild.pid"
    script_path = tmp_path / "fork_and_sleep.py"
    script_path.write_text(
        "import os, pathlib, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    pathlib.Path(" + repr(str(pid_file)) + ").write_text(str(os.getpid()))\n"
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

    # The grandchild had time to write its PID before sleeping. The
    # file must exist; otherwise the test fixture itself misfired and
    # we cannot prove anything about cleanup.
    assert pid_file.exists(), "grandchild did not write its PID file"
    grandchild_pid = int(pid_file.read_text().strip())

    # Probe up to 2 seconds for the grandchild to disappear. SIGKILL
    # delivery to the process group is asynchronous; the probe gives
    # the OS a moment to reap it, then asserts it's gone. Without
    # process-group escalation the grandchild would still be alive
    # 60s later.
    deadline = _time.monotonic() + 2.0
    while _time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            return  # grandchild is gone — pass.
        except OSError as exc:
            if exc.errno == errno.ESRCH:
                return
            raise
        _time.sleep(0.05)
    # Probe never raised → grandchild still alive. Fail loudly and
    # clean up so we don't orphan a 60-second sleep.
    try:
        os.kill(grandchild_pid, 9)
    except OSError:
        pass
    pytest.fail(
        f"grandchild PID {grandchild_pid} survived _run_subprocess timeout — "
        "process-group escalation did not propagate the kill."
    )
