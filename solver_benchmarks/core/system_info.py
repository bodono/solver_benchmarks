"""System / hardware metadata for benchmark runs.

The run manifest captures a snapshot of the system that produced the
results. This is the missing piece between "the solve took 1.2 s" and
knowing how to interpret that — was it on a 2.4 GHz laptop, a 32-core
server, or a frequency-throttled VM? The captured fields are also
surfaced in the markdown report so anyone reading the analysis can
size up the timing data before quoting it.

Only stdlib + numpy / scipy versions are captured by default. If the
optional ``psutil`` package is installed, richer fields (physical
core count, CPU frequency, total RAM, swap) are added; otherwise we
gracefully fall back to ``os.cpu_count()`` and platform-specific
``/proc/meminfo`` parsing on Linux. Capture must never raise — a
metadata blob with partial fields is always preferable to a runner
crash from a hardware-info call gone wrong.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import socket
import sys
from importlib import metadata
from typing import Any

logger = logging.getLogger(__name__)


def system_metadata(*, include_hostname: bool = False) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the host running this code.

    Captures CPU model / count / frequency, total RAM, OS / kernel
    version, Python implementation + version, and NumPy / SciPy /
    pandas versions (since solver runtimes are often dominated by the
    BLAS shipped with numpy). All fields are best-effort: if a
    library is missing or a probe raises, the field is set to None
    rather than failing the capture.

    ``include_hostname`` defaults to False so manifests don't leak
    machine names into shared reports. Callers running on isolated
    infrastructure can opt in.
    """
    payload: dict[str, Any] = {}
    payload.update(_safe(_python_metadata, "python") or {})
    payload.update(_safe(_os_metadata, "os") or {})
    payload["cpu"] = _safe(_cpu_metadata, "cpu") or {}
    payload["memory"] = _safe(_memory_metadata, "memory") or {}
    payload["library_versions"] = _safe(_library_versions, "library_versions") or {}
    if include_hostname:
        payload["hostname"] = _safe(_hostname, "hostname")
    return payload


def _python_metadata() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
    }


def _os_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _cpu_metadata() -> dict[str, Any]:
    """CPU info using psutil when available, falling back to stdlib."""
    cpu: dict[str, Any] = {
        "logical_count": os.cpu_count(),
        "physical_count": None,
        "max_frequency_mhz": None,
        "current_frequency_mhz": None,
        "model": None,
    }
    psutil_module = _try_import("psutil")
    if psutil_module is not None:
        try:
            cpu["physical_count"] = psutil_module.cpu_count(logical=False)
        except Exception:  # noqa: BLE001 — best-effort hardware probe.
            pass
        # psutil reports CPU frequency in MHz on Linux and macOS-Intel
        # but returns small integers (apparently GHz) on macOS-arm64;
        # skip the platform where the value is known unreliable rather
        # than report a misleading number.
        if not _is_apple_silicon():
            try:
                freq = psutil_module.cpu_freq()
                if freq is not None:
                    cpu["max_frequency_mhz"] = float(freq.max) if freq.max else None
                    cpu["current_frequency_mhz"] = float(freq.current) if freq.current else None
            except Exception:  # noqa: BLE001
                pass
    cpu["model"] = _detect_cpu_model()
    return cpu


def _is_apple_silicon() -> bool:
    """True on Apple Silicon (arm64 Darwin) where psutil's cpu_freq
    returns small integers instead of MHz. The check is a hot path
    during system_metadata; we keep it cheap.
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _detect_cpu_model() -> str | None:
    """Best-effort CPU model lookup across platforms.

    On Linux: parse ``/proc/cpuinfo`` for the first ``model name`` line.
    On macOS: ``sysctl -n machdep.cpu.brand_string`` (via subprocess,
    cached so we don't fork on every solve). On Windows / unknown
    systems we fall back to ``platform.processor()`` which is
    architecture-only on most hosts.
    """
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            return None
    if platform.system() == "Darwin":
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    # Windows / other: platform.processor() is the closest stdlib
    # equivalent; on Windows it usually returns the Intel CPU model.
    proc = platform.processor()
    return proc or None


def _memory_metadata() -> dict[str, Any]:
    """Memory info via psutil first, then a /proc/meminfo fallback."""
    mem: dict[str, Any] = {
        "total_bytes": None,
        "available_bytes": None,
        "swap_total_bytes": None,
    }
    psutil_module = _try_import("psutil")
    if psutil_module is not None:
        try:
            virtual = psutil_module.virtual_memory()
            mem["total_bytes"] = int(virtual.total)
            mem["available_bytes"] = int(virtual.available)
        except Exception:  # noqa: BLE001
            pass
        try:
            swap = psutil_module.swap_memory()
            mem["swap_total_bytes"] = int(swap.total)
        except Exception:  # noqa: BLE001
            pass
    if mem["total_bytes"] is None and platform.system() == "Linux":
        mem.update(_meminfo_fallback())
    if mem["total_bytes"] is None and platform.system() == "Darwin":
        total = _macos_total_memory()
        if total is not None:
            mem["total_bytes"] = total
    return mem


def _macos_total_memory() -> int | None:
    """Read total physical RAM via ``sysctl hw.memsize`` on macOS.

    Used when psutil isn't installed. Available memory isn't easily
    queried without psutil on macOS (``vm_stat`` reports pages, the
    page size needs a separate sysctl call), so we report only the
    total here — the available number is less useful for benchmark
    interpretation than the total anyway.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def _meminfo_fallback() -> dict[str, Any]:
    """Parse ``/proc/meminfo`` for total + available memory.

    Returns an empty dict if the file is unavailable. Keys mirror the
    psutil names so the merged ``memory`` dict has consistent
    semantics whichever path produced the values.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            content = handle.read()
    except OSError:
        return {}
    out: dict[str, Any] = {}
    for line, key in (("MemTotal", "total_bytes"), ("MemAvailable", "available_bytes"), ("SwapTotal", "swap_total_bytes")):
        match = re.search(rf"^{line}:\s*(\d+)\s*kB$", content, flags=re.MULTILINE)
        if match is not None:
            out[key] = int(match.group(1)) * 1024
    return out


def _library_versions() -> dict[str, Any]:
    """Versions of libraries that materially affect benchmark
    timings. NumPy / SciPy ship the BLAS / LAPACK that linear solvers
    delegate to; pandas / pyarrow are the reporting backbone.
    """
    versions: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "pandas", "pyarrow"):
        versions[package] = _package_version(package)
    return versions


def _hostname() -> str | None:
    try:
        return socket.gethostname()
    except OSError:
        return None


def _try_import(name: str):
    try:
        import importlib

        return importlib.import_module(name)
    except ImportError:
        return None


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _safe(fn, label: str) -> Any:
    """Call ``fn`` and swallow any exception, logging at debug level.

    System-info capture must never crash the runner — a partial
    metadata blob is always preferable to a benchmark abort caused by
    a hardware-introspection call gone wrong (e.g. a Linux container
    with /proc mounted differently, an exotic Python build).
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.debug("system_info.%s failed: %s", label, exc)
        return None
