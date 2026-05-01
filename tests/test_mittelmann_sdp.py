"""Tests for the Mittelmann SDP dataset and SDPA-S parser.

Synthesizes tiny SDPA-S files in tmp directories so the tests run
without network access. Covers the parser (comments, multiple
blocks, gzipped input), the SDPA → CONE-form translation (PSD
vec layout, NN diagonal blocks, sign of dual objective), the
dataset's list_problems / load_problem flow, the subset filter, and
end-to-end solving with a real SDP solver.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Synthetic SDPA-S fixtures.
# ---------------------------------------------------------------------------


# A minimal SDP: m=1 constraint, 1 PSD block of order 2.
#   min  X[0,0] + 2 X[1,1]
#   s.t. X[0,0] + X[1,1] = 1   (trace = 1)
#        X ⪰ 0
# Optimum: X = diag(1, 0), value = 1. (Same as the canonical small
# SDP fixture used elsewhere in the codebase.)
TRACE_ONE_SDP = """\
"Comment line 1
*Comment line 2
1
1
2
1.0
0 1 1 1 1.0
0 1 2 2 2.0
1 1 1 1 1.0
1 1 2 2 1.0
"""


# A multi-block SDP: 1 NN (diagonal) block of size 2 + 1 PSD block of
# order 2. Tests both the diagonal and PSD branches of the converter.
MIXED_BLOCK_SDP = """\
*Two-block SDP
2
2
-2 2
1.0 0.5
0 1 1 1 1.0
0 1 2 2 1.0
0 2 1 1 1.0
0 2 2 2 2.0
1 1 1 1 1.0
1 1 2 2 0.0
1 2 1 1 1.0
1 2 2 2 1.0
2 1 1 1 0.0
2 1 2 2 1.0
2 2 1 2 0.5
"""


def _write_dat_s(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.dat-s"
    path.write_text(body)
    return path


def _local_dataset(folder: Path, **options):
    """Return a MittelmannSDPDataset reading from ``folder`` only."""
    from solver_benchmarks.datasets.mittelmann_sdp import MittelmannSDPDataset

    class _Local(MittelmannSDPDataset):
        @property
        def folder(self) -> Path:  # type: ignore[override]
            return folder

        @property
        def data_dir(self) -> Path:  # type: ignore[override]
            return folder

    return _Local(**options)


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


def test_mittelmann_sdp_is_registered():
    from solver_benchmarks.datasets import registry

    cls = registry.get_dataset("mittelmann_sdp")
    assert cls.dataset_id == "mittelmann_sdp"


# ---------------------------------------------------------------------------
# SDPA-S parser.
# ---------------------------------------------------------------------------


def test_parse_sdpa_s_strips_comments_and_extracts_dimensions():
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    parsed = parse_sdpa_s(TRACE_ONE_SDP)
    assert parsed.m == 1
    assert len(parsed.blocks) == 1
    assert parsed.blocks[0].order == 2
    assert parsed.blocks[0].is_psd is True
    assert parsed.b.tolist() == [1.0]


def test_parse_sdpa_s_collects_c_and_a_entries():
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    parsed = parse_sdpa_s(TRACE_ONE_SDP)
    # C is matno=0 in SDPA-S; in our parsed output, c_blocks[0] holds
    # the entries for C's block 1. Stored as (i_zero, j_zero, value).
    assert parsed.c_blocks == [[(0, 0, 1.0), (1, 1, 2.0)]]
    # A_1 has trace-one constraint X[0,0] + X[1,1] = 1.
    assert parsed.a_blocks == [[[(0, 0, 1.0), (1, 1, 1.0)]]]


def test_parse_sdpa_s_handles_negative_block_size_as_diagonal():
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    parsed = parse_sdpa_s(MIXED_BLOCK_SDP)
    assert len(parsed.blocks) == 2
    assert parsed.blocks[0].order == 2 and parsed.blocks[0].is_psd is False
    assert parsed.blocks[1].order == 2 and parsed.blocks[1].is_psd is True


def test_parse_sdpa_s_accepts_comma_and_brace_separators():
    """SDPA-S in the wild often uses commas / braces for block sizes;
    the parser must tokenize the file's permissive separator set."""
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    body = """\
*comment
2
1
{3}
1.0, 2.0
0 1 1 1 1.0
1 1 1 1 1.0
2 1 2 2 1.0
"""
    parsed = parse_sdpa_s(body)
    assert parsed.m == 2
    assert parsed.blocks[0].order == 3 and parsed.blocks[0].is_psd is True


def test_parse_sdpa_s_rejects_truncated_file():
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    # m=1, nblocks=1, blocksize=2, then EOF before b is read.
    body = "1\n1\n2\n"
    with pytest.raises(ValueError, match="Unexpected end"):
        parse_sdpa_s(body)


def test_parse_sdpa_s_rejects_empty_file():
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    with pytest.raises(ValueError, match="empty"):
        parse_sdpa_s("* only comments\n")


def test_parse_sdpa_s_rejects_trailing_partial_entry():
    """Pre-fix ``parse_sdpa_s`` consumed entries while
    ``cursor + 5 <= len(tokens)`` and silently dropped any 1-4
    trailing tokens. A truncated SDPA file with a complete header
    plus an incomplete final entry would parse "successfully" with
    the partial row dropped — masking download corruption.

    Verify the parser now refuses any leftover non-multiple-of-5
    tail."""
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    body = """\
*truncated entry
1
1
2
1.0
0 1 1 1 1.0
1 1 1
"""
    with pytest.raises(ValueError, match="ends mid-entry"):
        parse_sdpa_s(body)


def test_parse_sdpa_s_rejects_zero_indexed_entry():
    """SDPA-S indices are 1-based; a literal ``0`` would wrap into
    Python's negative-index space and silently corrupt the
    canonical vec output downstream. The parser must reject it."""
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    body = """\
*0-indexed bug
1
1
2
1.0
0 1 0 1 1.0
"""
    with pytest.raises(ValueError, match="indices"):
        parse_sdpa_s(body)


def test_parse_sdpa_s_rejects_index_above_block_order():
    """An entry with ``i`` or ``j`` exceeding the declared block
    order indicates a malformed file. The parser must raise rather
    than crash later inside the canonical vec construction."""
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s

    body = """\
*out-of-range entry
1
1
2
1.0
0 1 5 5 1.0
"""
    with pytest.raises(ValueError, match="indices"):
        parse_sdpa_s(body)


def test_parse_sdpa_s_file_handles_gzip(tmp_path: Path):
    from solver_benchmarks.transforms.sdpa import parse_sdpa_s_file

    raw = TRACE_ONE_SDP.encode()
    path = tmp_path / "trace_one.dat-s.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(raw)
    parsed = parse_sdpa_s_file(path)
    assert parsed.m == 1


# ---------------------------------------------------------------------------
# SDPA → CONE-form conversion.
# ---------------------------------------------------------------------------


def test_sdpa_to_cone_problem_translates_psd_block_to_canonical_layout():
    from solver_benchmarks.transforms.sdpa import (
        parse_sdpa_s,
        sdpa_to_cone_problem,
    )

    parsed = parse_sdpa_s(TRACE_ONE_SDP)
    cone = sdpa_to_cone_problem(parsed)
    assert cone["cone"] == {"s": [2]}
    # b = vec(C) in canonical layout: [C[0,0], √2 * C[1,0], C[1,1]]
    # but our C has only diagonal entries, so [1.0, 0.0, 2.0].
    assert cone["b"].tolist() == [1.0, 0.0, 2.0]
    # q = -b_sdpa (we minimize the SDP dual). b_sdpa = [1.0].
    assert cone["q"].tolist() == [-1.0]
    # A is one column (one constraint y_1) and 3 rows (the canonical
    # PSD-2 vec). Constraint: trace(X) = X[0,0] + X[1,1] = 1, so the
    # vec of A_1 is [1.0, 0.0, 1.0].
    assert cone["A"].toarray().flatten().tolist() == [1.0, 0.0, 1.0]


def test_sdpa_to_cone_problem_handles_off_diagonal_with_sqrt2_scaling():
    """A constraint A_k with X[1,0] = 0.5 picks up the canonical
    layout's √2 scaling on the off-diagonal. We verify by parsing a
    one-block, one-constraint SDP with an explicit off-diagonal."""
    from solver_benchmarks.transforms.sdpa import (
        parse_sdpa_s,
        sdpa_to_cone_problem,
    )

    body = """\
* off-diagonal A constraint
1
1
2
0.5
0 1 1 1 0.0
1 1 2 1 0.5
"""
    parsed = parse_sdpa_s(body)
    cone = sdpa_to_cone_problem(parsed)
    # b = vec(C); C is identically zero except (1,1) was specified
    # which after sorting i ≤ j is also (1,1) entry (zero) - but we
    # specified ``0 1 1 1 0.0`` so C is the zero matrix.
    assert cone["b"].tolist() == [0.0, 0.0, 0.0]
    # A_1 has X[1,0] = 0.5 → canonical entry at index 1 with √2 scaling.
    a_col = cone["A"].toarray().flatten()
    assert a_col[0] == 0.0
    assert a_col[1] == pytest.approx(0.5 * np.sqrt(2.0))
    assert a_col[2] == 0.0


def test_sdpa_to_cone_problem_handles_mixed_blocks():
    """NN (diagonal) block + PSD block: rows are stacked as
    [NN | PSD] so the cone dict has both ``l`` and ``s``."""
    from solver_benchmarks.transforms.sdpa import (
        parse_sdpa_s,
        sdpa_to_cone_problem,
    )

    parsed = parse_sdpa_s(MIXED_BLOCK_SDP)
    cone = sdpa_to_cone_problem(parsed)
    # NN block of size 2 (negative size in file) + PSD block of order 2.
    assert cone["cone"] == {"l": 2, "s": [2]}
    # 2 NN diag entries + 3 PSD-canonical entries = 5 rows total.
    assert cone["A"].shape == (5, 2)


def test_sdpa_to_cone_problem_rejects_offdiag_in_diagonal_block():
    """A diagonal/NN block must not have off-diagonal entries; the
    converter raises if the SDPA file says otherwise."""
    from solver_benchmarks.transforms.sdpa import (
        parse_sdpa_s,
        sdpa_to_cone_problem,
    )

    body = """\
*bad NN block
1
1
-2
1.0
0 1 1 1 1.0
1 1 2 1 0.5
"""
    parsed = parse_sdpa_s(body)
    with pytest.raises(ValueError, match="off-diagonal"):
        sdpa_to_cone_problem(parsed)


# ---------------------------------------------------------------------------
# Dataset list_problems / load_problem.
# ---------------------------------------------------------------------------


def test_list_problems_returns_well_formed_specs(tmp_path: Path):
    folder = tmp_path / "mittelmann_sdp_data"
    _write_dat_s(folder, "trace_one", TRACE_ONE_SDP)
    _write_dat_s(folder, "mixed", MIXED_BLOCK_SDP)
    dataset = _local_dataset(folder)
    specs = dataset.list_problems()
    names = sorted(spec.name for spec in specs)
    assert names == ["mixed", "trace_one"]
    for spec in specs:
        assert spec.kind == "cone"
        assert spec.metadata["format"] == "sdpa-s"


def test_list_problems_filters_by_subset(tmp_path: Path):
    folder = tmp_path / "mittelmann_sdp_data"
    _write_dat_s(folder, "trace_one", TRACE_ONE_SDP)
    _write_dat_s(folder, "mixed", MIXED_BLOCK_SDP)
    dataset = _local_dataset(folder, subset="mixed")
    names = [spec.name for spec in dataset.list_problems()]
    assert names == ["mixed"]


def test_list_problems_returns_empty_when_folder_missing(tmp_path: Path):
    dataset = _local_dataset(tmp_path / "nope")
    assert dataset.list_problems() == []


def test_load_problem_returns_canonical_cone_problem(tmp_path: Path):
    folder = tmp_path / "mittelmann_sdp_data"
    _write_dat_s(folder, "trace_one", TRACE_ONE_SDP)
    dataset = _local_dataset(folder)
    pd = dataset.load_problem("trace_one")
    cone = pd.cone
    assert cone["cone"] == {"s": [2]}
    assert cone["A"].shape == (3, 1)
    assert pd.metadata["num_constraints_primal"] == 1
    assert pd.metadata["num_blocks"] == 1
    assert pd.metadata["block_orders"] == [2]


# ---------------------------------------------------------------------------
# End-to-end: solve a small Mittelmann-style SDP through Clarabel.
# ---------------------------------------------------------------------------


def test_trace_one_sdp_solves_to_known_optimum(tmp_path: Path):
    """The trace-one SDP fixture has known optimum value = 1
    (X = diag(1, 0)). Verify Clarabel reaches it."""
    pytest.importorskip("clarabel")
    from solver_benchmarks.core import status
    from solver_benchmarks.solvers.clarabel_adapter import ClarabelSolverAdapter

    folder = tmp_path / "mittelmann_sdp_data"
    _write_dat_s(folder, "trace_one", TRACE_ONE_SDP)
    dataset = _local_dataset(folder)
    pd = dataset.load_problem("trace_one")

    adapter = ClarabelSolverAdapter({"verbose": False})
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = adapter.solve(pd, artifacts)
    assert result.status == status.OPTIMAL
    # The CONE form's ``q'x = -b'y``; at optimum y_1 = 1 (so trace
    # constraint is tight) gives q'x = -1. The "primal" SDP optimum
    # 1.0 corresponds to dual y_1 = 1, hence -1.
    assert result.objective_value == pytest.approx(-1.0, abs=1e-4)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-3
    assert result.kkt["dual_res_rel"] < 1e-3


# ---------------------------------------------------------------------------
# Subset normalization helper.
# ---------------------------------------------------------------------------


def test_normalize_subset_none_means_no_filter():
    """``subset=None`` (or "all") means *show every problem on disk* —
    matching the CBLib / Mittelmann LP/QP pattern. Use the prepare
    script to control which problems are downloaded; the dataset's
    list_problems then reflects whatever is locally available."""
    from solver_benchmarks.datasets.mittelmann_sdp import _normalize_subset

    assert _normalize_subset(None) is None
    assert _normalize_subset("all") is None


def test_normalize_subset_accepts_comma_string_and_list():
    from solver_benchmarks.datasets.mittelmann_sdp import _normalize_subset

    assert _normalize_subset("trto3, G40mc") == {"trto3", "G40mc"}
    assert _normalize_subset(["trto3", "rose13"]) == {"trto3", "rose13"}


def test_sdpa_name_recognizes_dat_s_extensions(tmp_path: Path):
    from solver_benchmarks.datasets.mittelmann_sdp import _sdpa_name

    assert _sdpa_name(tmp_path / "trto3.dat-s") == "trto3"
    assert _sdpa_name(tmp_path / "G40mc.dat-s.gz") == "G40mc"
    assert _sdpa_name(tmp_path / "not_an_sdp.txt") is None
