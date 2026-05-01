"""Tests for the system_info capture and its integration into the
manifest + markdown report.

Covers the capture function (best-effort fields, gracefully degraded
output when probes fail), the manifest write path (system block
captured once and preserved across rewrites), and the markdown
report rendering (system summary block with the right rows).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# system_metadata smoke + structure.
# ---------------------------------------------------------------------------


def test_system_metadata_returns_required_top_level_keys():
    from solver_benchmarks.core.system_info import system_metadata

    md = system_metadata()
    # Top-level fields the report relies on.
    for key in (
        "python_executable",
        "python_version",
        "platform",
        "system",
        "machine",
        "cpu",
        "memory",
        "library_versions",
    ):
        assert key in md, f"missing field: {key}"


def test_system_metadata_cpu_block_has_logical_count():
    """``logical_count`` is the one CPU field that's always available
    via stdlib (``os.cpu_count``) — so it must never be None on a
    machine that can run pytest."""
    from solver_benchmarks.core.system_info import system_metadata

    cpu = system_metadata()["cpu"]
    assert cpu["logical_count"] is not None
    assert cpu["logical_count"] >= 1


def test_system_metadata_library_versions_includes_numpy_scipy():
    """numpy and scipy versions should always populate (they're
    runtime deps); pandas and pyarrow are also tracked."""
    from solver_benchmarks.core.system_info import system_metadata

    libs = system_metadata()["library_versions"]
    assert "numpy" in libs and libs["numpy"] is not None
    assert "scipy" in libs and libs["scipy"] is not None


def test_system_metadata_omits_hostname_by_default():
    from solver_benchmarks.core.system_info import system_metadata

    assert "hostname" not in system_metadata()


def test_system_metadata_includes_hostname_when_opt_in():
    from solver_benchmarks.core.system_info import system_metadata

    md = system_metadata(include_hostname=True)
    assert "hostname" in md
    # On a real system this is a non-empty string; the test machine
    # could in principle have an empty hostname but that's vanishingly
    # rare. Allow it just in case.
    assert md["hostname"] is None or isinstance(md["hostname"], str)


def test_system_metadata_python_executable_is_basename_by_default():
    """``python_executable`` should be the basename only (e.g.
    ``python3.12``) by default, since the full ``sys.executable``
    path commonly contains a username or private installer path.
    The full path is captured under ``python_executable_full`` only
    when the caller opts in."""
    from solver_benchmarks.core.system_info import system_metadata

    md = system_metadata()
    executable = md["python_executable"]
    # Basename has no path separators.
    assert executable is None or "/" not in executable
    assert executable is None or "\\" not in executable
    # ``python_executable_full`` is suppressed by default.
    assert "python_executable_full" not in md


def test_system_metadata_includes_full_python_path_when_opt_in():
    """Privacy opt-in: ``include_full_python_path=True`` records the
    full ``sys.executable`` under ``python_executable_full`` so
    isolated infrastructure can preserve the install location for
    reproducibility."""
    import sys as _sys

    from solver_benchmarks.core.system_info import system_metadata

    md = system_metadata(include_full_python_path=True)
    assert md["python_executable"] == _sys.executable
    assert md["python_executable_full"] == _sys.executable


def test_system_metadata_swallows_probe_exceptions(monkeypatch):
    """Every individual probe is wrapped in ``_safe`` so an
    unexpected error in one piece (e.g. a Linux container with /proc
    mounted differently) does not crash the runner."""
    from solver_benchmarks.core import system_info as si

    # Force every internal probe to raise.
    def boom():
        raise RuntimeError("simulated")

    monkeypatch.setattr(si, "_python_metadata", boom)
    monkeypatch.setattr(si, "_os_metadata", boom)
    monkeypatch.setattr(si, "_cpu_metadata", boom)
    monkeypatch.setattr(si, "_memory_metadata", boom)
    monkeypatch.setattr(si, "_library_versions", boom)

    md = si.system_metadata()
    # Capture must return a dict (possibly mostly None) rather than raise.
    assert isinstance(md, dict)
    assert md["cpu"] == {}
    assert md["memory"] == {}
    assert md["library_versions"] == {}


def test_system_metadata_is_json_serializable():
    """The manifest writer round-trips through ``json.dumps``; every
    field in ``system_metadata`` must be representable."""
    from solver_benchmarks.core.system_info import system_metadata

    md = system_metadata(include_hostname=True)
    # Round-trip; any unserializable type would raise.
    json.loads(json.dumps(md))


# ---------------------------------------------------------------------------
# CPU model detection.
# ---------------------------------------------------------------------------


def test_detect_cpu_model_returns_string_or_none():
    """``_detect_cpu_model`` is best-effort. The exact value depends on
    the platform; we just assert the type contract (str or None)."""
    from solver_benchmarks.core.system_info import _detect_cpu_model

    value = _detect_cpu_model()
    assert value is None or isinstance(value, str)


def test_detect_cpu_model_uses_proc_cpuinfo_on_linux(tmp_path: Path):
    """On Linux, the model is read from /proc/cpuinfo. Patch
    ``open`` and ``platform.system`` so the test doesn't depend on
    the host actually being Linux. Clear the LRU cache so any prior
    real-host probe doesn't leak in."""
    from solver_benchmarks.core import system_info as si

    si._detect_cpu_model.cache_clear()

    fake_cpuinfo = (
        "processor : 0\n"
        "vendor_id : GenuineIntel\n"
        "model name : Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz\n"
        "cache size : 28160 KB\n"
    )

    try:
        with (
            patch("solver_benchmarks.core.system_info.platform.system", return_value="Linux"),
            patch("builtins.open", _mock_open(fake_cpuinfo)),
        ):
            assert si._detect_cpu_model() == "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz"
    finally:
        # Don't leak the patched value into other tests.
        si._detect_cpu_model.cache_clear()


def _mock_open(read_text: str):
    """Build a context-manager-aware open() stub that yields
    ``read_text`` line-by-line on iteration, matching how
    ``_detect_cpu_model`` reads /proc/cpuinfo."""
    from io import StringIO
    from unittest.mock import MagicMock

    def fake_open(path, *args, **kwargs):
        handle = MagicMock()
        handle.__enter__.return_value = StringIO(read_text)
        handle.__exit__.return_value = False
        return handle

    return fake_open


# ---------------------------------------------------------------------------
# Memory fallbacks.
# ---------------------------------------------------------------------------


def test_meminfo_fallback_parses_proc_meminfo():
    from solver_benchmarks.core import system_info as si

    fake = (
        "MemTotal:       16384000 kB\n"
        "MemAvailable:    8192000 kB\n"
        "SwapTotal:       2048000 kB\n"
    )
    with patch("builtins.open", _mock_open(fake)):
        result = si._meminfo_fallback()
    assert result == {
        "total_bytes": 16384000 * 1024,
        "available_bytes": 8192000 * 1024,
        "swap_total_bytes": 2048000 * 1024,
    }


def test_meminfo_fallback_returns_empty_when_file_missing():
    from solver_benchmarks.core import system_info as si

    def raise_oserror(*_args, **_kwargs):
        raise OSError("no /proc/meminfo")

    with patch("builtins.open", side_effect=raise_oserror):
        assert si._meminfo_fallback() == {}


# ---------------------------------------------------------------------------
# Manifest integration.
# ---------------------------------------------------------------------------


def _minimal_run_config(tmp_path: Path):
    from solver_benchmarks.core.config import parse_run_config

    return parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
            },
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )


def test_write_manifest_includes_system_block(tmp_path: Path):
    """``ResultStore.create()`` calls ``write_manifest`` which must
    capture the system snapshot under ``manifest["system"]``."""
    from solver_benchmarks.core.storage import ResultStore

    config = _minimal_run_config(tmp_path)
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    manifest = json.loads(store.manifest_path.read_text())
    assert "system" in manifest
    system = manifest["system"]
    # Must contain the same shape as system_metadata's output.
    assert "cpu" in system
    assert "memory" in system
    assert "library_versions" in system


def test_write_manifest_preserves_existing_system_block(tmp_path: Path):
    """Re-writing the manifest (e.g. on resume) must NOT replace the
    original system snapshot. Otherwise a re-run on a different
    machine would silently overwrite the provenance of the original
    solves."""
    from solver_benchmarks.core.storage import ResultStore

    config = _minimal_run_config(tmp_path)
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    manifest_original = json.loads(store.manifest_path.read_text())
    original_system = manifest_original["system"]

    # Mutate the on-disk manifest to mark it with a sentinel so we
    # can detect overwrites, then re-call write_manifest.
    fake_marker = {**original_system, "_test_marker": "preserved"}
    manifest_original["system"] = fake_marker
    store.manifest_path.write_text(json.dumps(manifest_original, indent=2))
    store.write_manifest(config)

    rewritten = json.loads(store.manifest_path.read_text())
    assert rewritten["system"].get("_test_marker") == "preserved"


def test_runtime_metadata_includes_cpu_model():
    from solver_benchmarks.core.environment import runtime_metadata

    md = runtime_metadata("scs")
    assert "cpu_model" in md
    # str or None — either is acceptable, the field just must exist
    # so the markdown report's column wiring doesn't KeyError.
    assert md["cpu_model"] is None or isinstance(md["cpu_model"], str)


def test_runtime_metadata_python_executable_is_basename():
    """The per-row metadata must NOT publish the full
    ``sys.executable`` path, which commonly contains a username."""
    from solver_benchmarks.core.environment import runtime_metadata

    md = runtime_metadata("scs")
    executable = md["python_executable"]
    assert executable is None or "/" not in executable
    assert executable is None or "\\" not in executable


def test_detect_cpu_model_is_cached_across_calls():
    """``runtime_metadata`` is invoked once per result row, so the
    CPU-model probe needs to be cheap. ``functools.lru_cache``
    means repeated calls don't re-read ``/proc/cpuinfo`` or fork
    ``sysctl``. Verify by patching the underlying ``platform.system``
    after the first call: a non-cached implementation would hit the
    new branch on the second call."""
    from solver_benchmarks.core import system_info as si

    si._detect_cpu_model.cache_clear()
    try:
        first = si._detect_cpu_model()
        # If we patched ``platform.system`` to ``"Windows"`` here a
        # fresh call would route through ``platform.processor()``
        # instead of the macOS / Linux paths. The cache should make
        # the second call short-circuit, returning the original
        # result.
        with patch(
            "solver_benchmarks.core.system_info.platform.system",
            return_value="Windows",
        ):
            cached = si._detect_cpu_model()
        assert cached == first
    finally:
        si._detect_cpu_model.cache_clear()


# ---------------------------------------------------------------------------
# Markdown report integration.
# ---------------------------------------------------------------------------


def test_system_summary_lines_renders_cpu_and_memory():
    from solver_benchmarks.analysis.markdown_report import _system_summary_lines

    manifest = {
        "system": {
            "cpu": {
                "logical_count": 16,
                "physical_count": 8,
                "max_frequency_mhz": 3500.0,
                "model": "Intel Xeon",
            },
            "memory": {
                "total_bytes": 16 * 1024 ** 3,
                "available_bytes": 8 * 1024 ** 3,
            },
            "platform": "Linux-5.15-x86_64",
            "python_version": "3.12.0",
            "library_versions": {"numpy": "1.26.0", "scipy": "1.13.0"},
        }
    }
    lines = _system_summary_lines(manifest)
    rendered = "\n".join(lines)
    assert "### System" in rendered
    assert "Intel Xeon" in rendered
    assert "8 physical / 16 logical" in rendered
    assert "3500 MHz" in rendered
    assert "16.0 GiB" in rendered
    assert "Linux-5.15-x86_64" in rendered
    assert "Python" in rendered
    assert "numpy 1.26.0" in rendered


def test_system_summary_lines_returns_empty_when_no_system_block():
    """Legacy run directories without a ``system`` manifest block
    must render no system section rather than a half-populated one
    or a crash."""
    from solver_benchmarks.analysis.markdown_report import _system_summary_lines

    assert _system_summary_lines({}) == []
    assert _system_summary_lines({"system": None}) == []


def test_system_summary_lines_handles_partial_data():
    """Some hosts (no psutil, exotic Linux) only fill a subset of the
    fields. The summary should render whatever is available."""
    from solver_benchmarks.analysis.markdown_report import _system_summary_lines

    manifest = {
        "system": {
            "cpu": {"logical_count": 4, "model": "ARM Neoverse-N1"},
            "memory": {"total_bytes": None},  # missing memory.
            "platform": None,  # missing platform.
        }
    }
    rendered = "\n".join(_system_summary_lines(manifest))
    assert "ARM Neoverse-N1" in rendered
    assert "4 logical" in rendered
    # No memory row, no OS row.
    assert "Total RAM" not in rendered


def test_system_summary_lines_escapes_pipes_and_newlines_in_values():
    """A CPU model / platform / library string carrying a literal
    ``|`` or newline must not break the Markdown table layout. Other
    report tables route values through ``_escape_cell``; pre-fix
    ``_system_summary_lines`` did not."""
    from solver_benchmarks.analysis.markdown_report import _system_summary_lines

    manifest = {
        "system": {
            "cpu": {
                "logical_count": 4,
                "model": "Weird | CPU\nwith pipe and newline",
            },
            "platform": "Linux | special",
        }
    }
    rendered = "\n".join(_system_summary_lines(manifest))
    # Escaped pipe and newline should appear in the output.
    assert r"\|" in rendered
    assert "<br>" in rendered
    # No raw unescaped pipe in any data cell. A row has 3 unescaped
    # pipes (leading, between Field and Value, trailing); any extra
    # would mean an unescaped pipe leaked from the value.
    body_lines = [
        line for line in rendered.splitlines()
        if line.startswith("| ") and "Field" not in line and "---" not in line
    ]
    for line in body_lines:
        stripped = line.replace(r"\|", "")
        assert stripped.count("|") == 3, line


def test_format_bytes_uses_binary_prefixes():
    from solver_benchmarks.analysis.markdown_report import _format_bytes

    assert _format_bytes(0) == "0 B"
    assert _format_bytes(512) == "512 B"
    assert _format_bytes(2 * 1024) == "2.0 KiB"
    assert _format_bytes(int(1.5 * 1024 ** 2)) == "1.5 MiB"
    assert _format_bytes(16 * 1024 ** 3) == "16.0 GiB"
    assert _format_bytes(2 * 1024 ** 4) == "2.0 TiB"


# ---------------------------------------------------------------------------
# Report end-to-end smoke.
# ---------------------------------------------------------------------------


def test_provenance_block_renders_system_section_when_manifest_has_it(tmp_path: Path):
    """The full provenance block must include the System table when
    the manifest carries one. We render directly rather than through
    the CLI to keep the test fast."""
    pytest.importorskip("pandas")
    import pandas as pd

    from solver_benchmarks.analysis.markdown_report import _render_provenance_block

    manifest = {
        "system": {
            "cpu": {"logical_count": 8, "model": "M1"},
            "memory": {"total_bytes": 16 * 1024 ** 3},
            "platform": "macOS-14.0-arm64",
            "python_version": "3.12.0",
        },
        "config": {"name": "smoke"},
    }
    block = _render_provenance_block(tmp_path, manifest, pd.DataFrame())
    rendered = "\n".join(block)
    assert "## Provenance" in rendered
    assert "### System" in rendered
    assert "M1" in rendered
    assert "16.0 GiB" in rendered
