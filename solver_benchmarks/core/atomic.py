"""Atomic file writes shared by the result store and the dataset downloaders."""

from __future__ import annotations

import os
import re
import secrets
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` so readers see the old or the new file, never a mix.

    The file gets the mode of the file it replaces, or, for a new file, the
    mode ``open()`` would give it (0o666 less the umask). A temporary file
    from ``tempfile`` is always 0o600, which ``os.replace`` would carry over:
    on a shared machine colleagues could then not read manifest.json.
    """
    atomic_write_stream(path, lambda handle: handle.write(data))


def atomic_write_stream(path: Path, write: Callable[[BinaryIO], object]) -> None:
    """``atomic_write_bytes`` for content too large to hold in memory.

    ``write`` is called with the temporary file, open for binary writing,
    and writes the content to it, for instance with ``shutil.copyfileobj``
    from another stream. If ``write`` raises, or the process is interrupted
    (``KeyboardInterrupt`` included), the temporary file is removed and
    ``path`` is left as it was. The temporary file is named
    ``.<name>.<random>.tmp``, so a glob for ``path``'s suffix never matches it.
    A process killed without unwinding leaves it behind;
    ``remove_stale_temporaries`` removes such files once they are old. If
    the temporary file is gone at the final rename (another process removed
    it) and ``path`` exists, the write counts as done.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode: int | None = stat.S_IMODE(os.stat(path).st_mode)
    except FileNotFoundError:
        mode = None
    fd, tmp_name = _create_temporary(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            # Force the data to disk before the rename so a power loss
            # cannot leave the rename visible with a zero-length payload.
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        if mode is not None:
            os.chmod(tmp_name, mode)
        try:
            os.replace(tmp_name, path)
        except FileNotFoundError:
            # ``remove_stale_temporaries`` of another process took the
            # temporary file of a writer it judged dead; that process has
            # written ``path`` meanwhile, which then is complete.
            if not path.exists():
                raise
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def remove_stale_temporaries(path: Path, max_age_seconds: float) -> list[Path]:
    """Remove temporary files of ``atomic_write_stream`` for ``path`` left by a killed writer.

    A writer killed with SIGKILL (or SIGTERM without a handler) never runs
    the cleanup that removes its ``.<name>.<random>.tmp``. This removes such
    siblings of ``path`` that were last modified more than
    ``max_age_seconds`` ago; a writer still copying keeps updating its
    file's modification time, so a generous age never removes the file of a
    live writer. Files that cannot be inspected or removed are left alone.
    Returns the paths removed.
    """
    path = Path(path)
    pattern = re.compile(rf"\.{re.escape(path.name)}\.[0-9a-f]{{12}}\.tmp")
    cutoff = time.time() - max_age_seconds
    removed: list[Path] = []
    try:
        entries = list(os.scandir(path.parent))
    except OSError:
        return removed
    for entry in entries:
        if not pattern.fullmatch(entry.name):
            continue
        try:
            if not entry.is_file(follow_symlinks=False) or entry.stat(follow_symlinks=False).st_mtime > cutoff:
                continue
            os.unlink(entry.path)
        except OSError:
            continue
        removed.append(Path(entry.path))
    return removed


def _create_temporary(path: Path) -> tuple[int, str]:
    """Create a fresh file next to ``path``; the umask applies to its mode."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    for _ in range(100):
        name = str(path.parent / f".{path.name}.{secrets.token_hex(6)}.tmp")
        try:
            return os.open(name, flags, 0o666), name
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not create a temporary file next to {path}")
