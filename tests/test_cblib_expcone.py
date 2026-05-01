"""CBLib EXP / EXP* cone parsing and ``subset_kind`` filter.

Pre-fix the CBF parser rejected the EXP / EXP* cone kinds with
``UnsupportedCBFError``, so any CBLib instance using the exponential
cone was silently filtered out at ``list_problems`` time. Now those
instances are loaded with their cones translated into the canonical
schema's ``ep`` / ``ed`` keys (count of 3-tuples), and a new
``subset_kind`` dataset option lets callers select instances by cone
shape — ``expcone``, ``socp``, or ``lp``.

These tests build synthetic CBF files in a tmp directory rather than
relying on a downloaded CBLib snapshot, so they run without network
access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_cbf(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.cbf"
    path.write_text(body)
    return path


_LP_ONLY_CBF = """VER
1

OBJSENSE
MIN

VAR
2 1
L+ 2

CON
1 1
L= 1

OBJACOORD
2
0 1.0
1 1.0

ACOORD
2
0 0 1.0
0 1 1.0

BCOORD
1
0 1.0
"""


_SOCP_CBF = """VER
1

OBJSENSE
MIN

VAR
3 1
F 3

CON
1 1
Q 3

OBJACOORD
1
0 1.0

ACOORD
3
0 0 -1.0
1 1 -1.0
2 2 -1.0

BCOORD
0
"""


_EXPCONE_CBF = """VER
1

OBJSENSE
MIN

VAR
3 1
EXP 3

CON
2 1
L= 2

OBJACOORD
1
2 1.0

ACOORD
2
0 0 1.0
1 1 1.0

BCOORD
2
0 0.0
1 1.0
"""


_DUAL_EXPCONE_CBF = """VER
1

OBJSENSE
MIN

VAR
3 1
EXP* 3

CON
2 1
L= 2

OBJACOORD
1
2 1.0

ACOORD
2
0 0 1.0
1 1 1.0

BCOORD
2
0 0.0
1 1.0
"""


_PSD_CBF = """VER
1

OBJSENSE
MIN

VAR
1 1
F 1

PSDCON
1
2

OBJACOORD
1
0 1.0
"""


# ---------------------------------------------------------------------------
# Parser: EXP / EXP* are recognized and translated into ep / ed.
# ---------------------------------------------------------------------------


def test_inspect_cbf_recognizes_exp_cone(tmp_path: Path):
    from solver_benchmarks.datasets.cblib import inspect_cbf

    path = _write_cbf(tmp_path, "expcone_synthetic", _EXPCONE_CBF)
    metadata = inspect_cbf(path)
    assert metadata["supported"] is True
    assert metadata["variable_cones"] == {"ep": 1}
    # No constraint EXP cones in this fixture; only L= rows.
    assert metadata["constraint_cones"] == {"l=": 2}


def test_inspect_cbf_recognizes_dual_exp_cone(tmp_path: Path):
    from solver_benchmarks.datasets.cblib import inspect_cbf

    path = _write_cbf(tmp_path, "dual_expcone_synthetic", _DUAL_EXPCONE_CBF)
    metadata = inspect_cbf(path)
    assert metadata["variable_cones"] == {"ed": 1}


def test_read_cbf_translates_exp_to_ep_in_cone_schema(tmp_path: Path):
    """The cone schema in the ProblemData uses ``ep`` for the count
    of EXP cones (each contributing 3 rows). Verify the translation
    explicitly rather than only via the higher-level inspect helper."""
    from solver_benchmarks.datasets.cblib import read_cbf_cone_problem

    path = _write_cbf(tmp_path, "expcone_synthetic", _EXPCONE_CBF)
    problem, _meta = read_cbf_cone_problem(path)
    cone = problem["cone"]
    # Two equality rows (from L= 2) plus one 3-tuple EXP cone.
    assert cone.get("z") == 2
    assert cone.get("ep") == 1
    # A has 5 rows total (2 equality + 3 expcone), 3 columns (3 vars).
    assert problem["A"].shape == (5, 3)


def test_read_cbf_translates_dual_exp_to_ed(tmp_path: Path):
    from solver_benchmarks.datasets.cblib import read_cbf_cone_problem

    path = _write_cbf(tmp_path, "dual_expcone_synthetic", _DUAL_EXPCONE_CBF)
    problem, _meta = read_cbf_cone_problem(path)
    assert problem["cone"].get("ed") == 1


def test_read_cbf_rejects_exp_cone_with_wrong_dimension(tmp_path: Path):
    """CBF EXP cones are always 3-dim per cone. A misformed ``EXP 5``
    entry must raise ``UnsupportedCBFError`` rather than silently
    miscount rows downstream."""
    from solver_benchmarks.datasets.cblib import (
        UnsupportedCBFError,
        inspect_cbf,
    )

    bad = """VER
1

OBJSENSE
MIN

VAR
5 1
EXP 5

CON
0 0

OBJACOORD
0

ACOORD
0

BCOORD
0
"""
    path = _write_cbf(tmp_path, "expcone_bad_dim", bad)
    with pytest.raises(UnsupportedCBFError, match="EXP cone must have dim 3"):
        inspect_cbf(path)


def test_inspect_cbf_still_rejects_psd_section(tmp_path: Path):
    """Adding EXP support must not silently accept PSDCON; PSD support
    is a separate, larger change."""
    from solver_benchmarks.datasets.cblib import (
        UnsupportedCBFError,
        inspect_cbf,
    )

    path = _write_cbf(tmp_path, "psd_synthetic", _PSD_CBF)
    with pytest.raises(UnsupportedCBFError):
        inspect_cbf(path)


# ---------------------------------------------------------------------------
# subset_kind filter end-to-end via CBLIBDataset.list_problems.
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_cblib_folder(tmp_path: Path) -> Path:
    """Fake CBLib data folder with one instance of each cone shape."""
    folder = tmp_path / "cblib_data"
    _write_cbf(folder, "lp_only", _LP_ONLY_CBF)
    _write_cbf(folder, "socp_only", _SOCP_CBF)
    _write_cbf(folder, "expcone_only", _EXPCONE_CBF)
    return folder


def _make_dataset(folder: Path, options: dict):
    """Build a CBLIBDataset that reads from ``folder`` regardless of
    ``problem_classes_dir``. We subclass per-call rather than mutating
    the class attribute so test fixtures don't leak the patched
    ``folder`` across tests.
    """
    from solver_benchmarks.datasets.cblib import CBLIBDataset

    class _LocalCBLIB(CBLIBDataset):
        @property
        def folder(self) -> Path:  # type: ignore[override]
            return folder

        @property
        def data_dir(self) -> Path:  # type: ignore[override]
            return folder

    return _LocalCBLIB(**options)


def test_subset_kind_expcone_filters_to_expcone_instances(
    mixed_cblib_folder: Path,
):
    dataset = _make_dataset(mixed_cblib_folder, {"subset_kind": "expcone"})
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["expcone_only"]


def test_subset_kind_socp_filters_to_socp_instances(mixed_cblib_folder: Path):
    dataset = _make_dataset(mixed_cblib_folder, {"subset_kind": "socp"})
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["socp_only"]


def test_subset_kind_lp_filters_to_lp_instances(mixed_cblib_folder: Path):
    dataset = _make_dataset(mixed_cblib_folder, {"subset_kind": "lp"})
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["lp_only"]


def test_subset_kind_none_returns_all_instances(mixed_cblib_folder: Path):
    dataset = _make_dataset(mixed_cblib_folder, {})
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["expcone_only", "lp_only", "socp_only"]


def test_subset_kind_lp_does_not_include_unsupported_instances(
    tmp_path: Path,
):
    """Pre-fix: when ``include_unsupported=True`` was set, an
    unsupported instance had empty ``variable_cones`` /
    ``constraint_cones`` metadata, so ``_matches_kind`` for ``"lp"``
    falsely returned True. Now unsupported instances never match a
    kind filter, regardless of ``include_unsupported``."""
    folder = tmp_path / "cblib_data"
    _write_cbf(folder, "lp_only", _LP_ONLY_CBF)
    _write_cbf(folder, "psd_synthetic", _PSD_CBF)
    dataset = _make_dataset(
        folder,
        {"subset_kind": "lp", "include_unsupported": True},
    )
    names = sorted(spec.name for spec in dataset.list_problems())
    # The PSD instance is unsupported by the parser; it must not be
    # reclassified as LP just because its cone summary is empty.
    assert names == ["lp_only"]


def test_unsupported_instance_passthrough_when_no_kind_filter(
    tmp_path: Path,
):
    """``include_unsupported=True`` without a ``subset_kind`` filter
    still surfaces the instance — kind-filter exclusion is the only
    new behavior, not blanket suppression."""
    folder = tmp_path / "cblib_data"
    _write_cbf(folder, "lp_only", _LP_ONLY_CBF)
    _write_cbf(folder, "psd_synthetic", _PSD_CBF)
    dataset = _make_dataset(folder, {"include_unsupported": True})
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["lp_only", "psd_synthetic"]


def test_subset_kind_combines_with_subset_name_filter(
    mixed_cblib_folder: Path,
):
    """``subset`` (name list) and ``subset_kind`` (cone shape) must
    intersect — only instances passing both filters are kept. Useful
    for "give me the expcone instances from this curated subset"."""
    dataset = _make_dataset(
        mixed_cblib_folder,
        {"subset": "lp_only,expcone_only", "subset_kind": "expcone"},
    )
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["expcone_only"]


def test_subset_kind_unknown_value_raises_value_error(
    mixed_cblib_folder: Path,
):
    """A typo in ``subset_kind`` (e.g. "expcones") must raise rather
    than silently returning an empty list — empty lists make it look
    like the data isn't prepared yet, which is misleading."""
    dataset = _make_dataset(mixed_cblib_folder, {"subset_kind": "nope"})
    with pytest.raises(ValueError, match="Unknown CBLib subset_kind"):
        dataset.list_problems()


# ---------------------------------------------------------------------------
# Sanity: an EXP cone instance loads and is shaped correctly so a conic
# solver could in principle take it (we don't require any specific solver
# here — just that the schema is well-formed).
# ---------------------------------------------------------------------------


def test_expcone_problem_loads_with_well_formed_cone_schema(
    mixed_cblib_folder: Path,
):
    dataset = _make_dataset(mixed_cblib_folder, {"subset_kind": "expcone"})
    [spec] = dataset.list_problems()
    problem_data = dataset.load_problem(spec.name)
    cone = problem_data.cone["cone"]
    # Total rows in A must equal sum of cone block sizes (z + 3*ep + ...).
    a_rows = problem_data.cone["A"].shape[0]
    expected_rows = (
        int(cone.get("z", 0))
        + int(cone.get("l", 0))
        + sum(int(d) for d in cone.get("q", []))
        + 3 * int(cone.get("ep", 0))
        + 3 * int(cone.get("ed", 0))
    )
    assert a_rows == expected_rows, (a_rows, cone)
    assert isinstance(problem_data.cone["b"], np.ndarray)
    assert problem_data.cone["b"].shape[0] == a_rows
