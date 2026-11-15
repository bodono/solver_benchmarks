"""End a process when the process that started it dies.

Workers run in their own sessions, and a matrix child `bench run` has no
tie to `bench env run` either, so a parent killed outright (SIGKILL, the
out-of-memory killer) would leave them running unsupervised.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable

# How often the polling fallback checks whether the parent is still there.
POLL_SECONDS = 1.0
_PR_SET_PDEATHSIG = 1


def watch_parent(
    parent_pid: int | None, *, signum: int, on_gone: Callable[[], None]
) -> str | None:
    """Call ``on_gone`` (or have the kernel send ``signum``) when ``parent_pid`` dies.

    When ``parent_pid`` is this process's parent, Linux sends ``signum``
    when it dies (``PR_SET_PDEATHSIG``), which works even while a C call
    holds the GIL; the kernel ties it to the parent's thread that started
    this process, so the parent must start it from a thread that outlives
    it. The parent is then checked again, in case it died before the
    signal was set. Elsewhere a daemon thread checks the parent pid once a
    second. When the parent is not ``parent_pid`` (``sys.executable`` is a
    launcher that runs the interpreter as its child instead of exec'ing
    it), that process is checked for existence instead: gone, ``on_gone``
    is called at once; alive, a thread watches it, without the kernel
    signal, which would follow the launcher. On Windows nothing is
    installed: there ``os.kill(pid, 0)`` would terminate the process, and a
    venv's ``python.exe`` is such a launcher.

    Returns how the parent is watched (``"kernel"``, ``"parent"``,
    ``"pid"``), or None when nothing was installed.
    """
    if parent_pid is None or _windows():
        return None
    if os.getppid() == parent_pid:
        if sys.platform.startswith("linux") and _set_parent_death_signal(signum):
            mode = "kernel"
        else:
            _start(lambda: os.getppid() == parent_pid, on_gone)
            mode = "parent"
        if os.getppid() != parent_pid:
            on_gone()
        return mode
    if not pid_exists(parent_pid):
        on_gone()
        return None
    _start(lambda: pid_exists(parent_pid), on_gone)
    return "pid"


def pid_exists(pid: int) -> bool:
    """Whether a process ``pid`` exists (POSIX only)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows() -> bool:
    return os.name == "nt"


def _set_parent_death_signal(signum: int) -> bool:
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(_PR_SET_PDEATHSIG, int(signum), 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def _start(alive: Callable[[], bool], on_gone: Callable[[], None]) -> None:
    def watch() -> None:
        while alive():
            time.sleep(POLL_SECONDS)
        on_gone()

    threading.Thread(target=watch, name="bench-watch-parent", daemon=True).start()
