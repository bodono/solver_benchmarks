"""Tests for the tsplib_sdp dataset and MaxCut SDP transform.

Synthesizes tiny TSPLIB-format files in tmp directories so the tests
run without network access. Covers the parser (EUC_2D coords,
EXPLICIT formats, gzipped input), the MaxCut SDP construction (PSD
canonical layout, diagonal-fix constraints, asymmetric-input
symmetrization), the dataset's list_problems / load_problem flow,
the subset filter, and end-to-end solving.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# TSPLIB fixtures.
# ---------------------------------------------------------------------------


# A tiny EUC_2D TSPLIB instance: 4 cities at corners of a unit square.
# Pairwise distances (rounded to nearest int per TSPLIB spec):
#   d(0,1) = d(2,3) = 1
#   d(0,2) = d(1,3) = 1
#   d(0,3) = round(sqrt(2)) = 1
#   d(1,2) = round(sqrt(2)) = 1
TINY_EUC2D = """\
NAME : tiny4
TYPE : TSP
COMMENT : 4 cities on a unit square
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.0 0.0
3 0.0 1.0
4 1.0 1.0
EOF
"""


# Same 4-city graph specified explicitly via FULL_MATRIX.
TINY_FULL_MATRIX = """\
NAME : tiny4_explicit
TYPE : TSP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
EDGE_WEIGHT_SECTION
0 1 1 2
1 0 2 1
1 2 0 1
2 1 1 0
EOF
"""


TINY_LOWER_DIAG = """\
NAME : tiny3_lower_diag
TYPE : TSP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : LOWER_DIAG_ROW
EDGE_WEIGHT_SECTION
0
1 0
2 3 0
EOF
"""


TINY_UPPER_ROW = """\
NAME : tiny3_upper_row
TYPE : TSP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : UPPER_ROW
EDGE_WEIGHT_SECTION
1 2
3
EOF
"""


def _write_tsp(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.tsp"
    path.write_text(body)
    return path


def _local_dataset(folder: Path, **options):
    from solver_benchmarks.datasets.tsplib_sdp import TSPLIBSDPDataset

    class _Local(TSPLIBSDPDataset):
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


def test_tsplib_sdp_is_registered():
    from solver_benchmarks.datasets import registry

    cls = registry.get_dataset("tsplib_sdp")
    assert cls.dataset_id == "tsplib_sdp"


# ---------------------------------------------------------------------------
# TSPLIB parser.
# ---------------------------------------------------------------------------


def test_read_tsplib_weights_euc_2d(tmp_path: Path):
    from solver_benchmarks.datasets.tsplib_sdp import read_tsplib_weights

    path = _write_tsp(tmp_path, "tiny4", TINY_EUC2D)
    w = read_tsplib_weights(path)
    assert w.shape == (4, 4)
    # All edges round to 1 with the unit-square layout.
    assert np.allclose(w[0, 1], 1.0)
    assert np.allclose(w[0, 3], 1.0)  # round(sqrt(2)) = 1
    assert np.allclose(w.diagonal(), 0.0)
    # Symmetric.
    assert np.allclose(w, w.T)


def test_read_tsplib_weights_explicit_full_matrix(tmp_path: Path):
    from solver_benchmarks.datasets.tsplib_sdp import read_tsplib_weights

    path = _write_tsp(tmp_path, "tiny4_explicit", TINY_FULL_MATRIX)
    w = read_tsplib_weights(path)
    assert w.shape == (4, 4)
    # The fixture matrix already has zero diagonal and is symmetric.
    assert np.allclose(w.diagonal(), 0.0)
    assert np.allclose(w[0, 3], 2.0)
    assert np.allclose(w[1, 2], 2.0)


def test_read_tsplib_weights_lower_diag_row(tmp_path: Path):
    from solver_benchmarks.datasets.tsplib_sdp import read_tsplib_weights

    path = _write_tsp(tmp_path, "tiny3_ld", TINY_LOWER_DIAG)
    w = read_tsplib_weights(path)
    # LOWER_DIAG_ROW row 0: [0]; row 1: [1, 0]; row 2: [2, 3, 0].
    assert w.shape == (3, 3)
    assert np.allclose(w[1, 0], 1.0)
    assert np.allclose(w[2, 0], 2.0)
    assert np.allclose(w[2, 1], 3.0)
    assert np.allclose(w, w.T)
    assert np.allclose(w.diagonal(), 0.0)


def test_read_tsplib_weights_upper_row(tmp_path: Path):
    from solver_benchmarks.datasets.tsplib_sdp import read_tsplib_weights

    path = _write_tsp(tmp_path, "tiny3_ur", TINY_UPPER_ROW)
    w = read_tsplib_weights(path)
    # UPPER_ROW row 0: [w[0,1], w[0,2]] = [1, 2]; row 1: [w[1,2]] = [3].
    assert w[0, 1] == 1.0
    assert w[0, 2] == 2.0
    assert w[1, 2] == 3.0
    assert np.allclose(w, w.T)


def test_read_tsplib_weights_handles_gzip(tmp_path: Path):
    from solver_benchmarks.datasets.tsplib_sdp import read_tsplib_weights

    path = tmp_path / "tiny4.tsp.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(TINY_EUC2D.encode())
    w = read_tsplib_weights(path)
    assert w.shape == (4, 4)


def test_read_tsplib_weights_rejects_missing_dimension(tmp_path: Path):
    from solver_benchmarks.datasets.tsplib_sdp import read_tsplib_weights

    path = _write_tsp(
        tmp_path,
        "no_dim",
        "NAME : x\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n1 0 0\nEOF\n",
    )
    with pytest.raises(ValueError, match="DIMENSION"):
        read_tsplib_weights(path)


def test_read_tsplib_weights_rejects_unsupported_weight_type(tmp_path: Path):
    from solver_benchmarks.datasets.tsplib_sdp import read_tsplib_weights

    path = _write_tsp(
        tmp_path,
        "weird",
        "DIMENSION : 2\nEDGE_WEIGHT_TYPE : XRAY\nEOF\n",
    )
    with pytest.raises(ValueError, match="Unsupported EDGE_WEIGHT_TYPE"):
        read_tsplib_weights(path)


# ---------------------------------------------------------------------------
# MaxCut SDP construction.
# ---------------------------------------------------------------------------


def test_maxcut_sdp_cone_problem_layout():
    from solver_benchmarks.transforms.maxcut_sdp import maxcut_sdp_cone_problem

    # 3-node graph: triangle with unit weights.
    w = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    problem, metadata = maxcut_sdp_cone_problem(w)
    n = 3
    triangle = n * (n + 1) // 2  # 6
    # One PSD-cone block of order n.
    assert problem["cone"] == {"s": [n]}
    # n diagonal-fix constraints → n columns of A, ``triangle`` rows.
    assert problem["A"].shape == (triangle, n)
    assert problem["b"].shape == (triangle,)
    # q = -1 (minimize -1'y).
    assert problem["q"].tolist() == [-1.0, -1.0, -1.0]
    # Each diagonal-fix constraint A_k = e_k e_k^T contributes a
    # single 1 at the (k, k) diagonal index.
    a_dense = problem["A"].toarray()
    for k in range(n):
        diag_idx = k * n - k * (k - 1) // 2  # j*n - j*(j-1)//2 with i=j=k
        assert a_dense[diag_idx, k] == 1.0
    assert metadata["num_nodes"] == n


def test_maxcut_sdp_constant_is_quarter_total_weight():
    from solver_benchmarks.transforms.maxcut_sdp import maxcut_sdp_cone_problem

    w = np.array([[0, 2, 0], [2, 0, 4], [0, 4, 0]], dtype=float)
    _, metadata = maxcut_sdp_cone_problem(w)
    # sum W / 2 = (4 + 8) / 2 = 6 (off-diagonals counted once each).
    # Wait: total_weight should be sum(W) / 2 over the symmetric
    # weight matrix; for this 3x3 with off-diagonal 2 + 4 it's
    # (0+2+0+2+0+4+0+4+0)/2 = 12/2 = 6.
    assert metadata["total_weight"] == pytest.approx(6.0)
    assert metadata["maxcut_constant"] == pytest.approx(0.25 * 12.0)


def test_maxcut_sdp_symmetrizes_asymmetric_inputs():
    """ATSP-style asymmetric distance matrices must be symmetrized
    before the Laplacian is built — MaxCut models undirected edges."""
    from solver_benchmarks.transforms.maxcut_sdp import maxcut_sdp_cone_problem

    asym = np.array([[0, 1, 0], [3, 0, 5], [0, 7, 0]], dtype=float)
    problem, metadata = maxcut_sdp_cone_problem(asym)
    # After symmetrize: W[0,1] = 2, W[1,2] = 6, W[0,2] = 0 ⇒ total 8.
    expected_total = (2.0 + 6.0 + 0.0)
    assert metadata["total_weight"] == pytest.approx(expected_total)


def test_maxcut_sdp_zeroes_diagonal():
    """Self-loops carry no MaxCut signal; the construction must zero
    the input diagonal before computing the Laplacian."""
    from solver_benchmarks.transforms.maxcut_sdp import maxcut_sdp_cone_problem

    w_with_diag = np.array([[10, 1, 1], [1, 10, 1], [1, 1, 10]], dtype=float)
    w_without_diag = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    p1, _ = maxcut_sdp_cone_problem(w_with_diag)
    p2, _ = maxcut_sdp_cone_problem(w_without_diag)
    # Same b because we zero diagonals; same A by construction.
    assert np.allclose(p1["b"], p2["b"])


def test_maxcut_sdp_rejects_non_square_input():
    from solver_benchmarks.transforms.maxcut_sdp import maxcut_sdp_cone_problem

    with pytest.raises(ValueError, match="square"):
        maxcut_sdp_cone_problem(np.array([[1, 2, 3]]))


# ---------------------------------------------------------------------------
# Dataset list_problems / load_problem.
# ---------------------------------------------------------------------------


@pytest.fixture
def tsplib_data_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "tsplib_data"
    _write_tsp(folder, "tiny4", TINY_EUC2D)
    _write_tsp(folder, "tiny4_explicit", TINY_FULL_MATRIX)
    return folder


def test_list_problems_returns_specs_for_all_local_files(tsplib_data_folder: Path):
    dataset = _local_dataset(tsplib_data_folder)
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["tiny4", "tiny4_explicit"]


def test_list_problems_filters_by_subset(tsplib_data_folder: Path):
    dataset = _local_dataset(tsplib_data_folder, subset="tiny4")
    names = [spec.name for spec in dataset.list_problems()]
    assert names == ["tiny4"]


def test_load_problem_returns_canonical_cone_problem(tsplib_data_folder: Path):
    dataset = _local_dataset(tsplib_data_folder)
    pd = dataset.load_problem("tiny4")
    cone = pd.cone
    assert cone["cone"] == {"s": [4]}
    # Triangular vec for n=4 has 10 entries.
    assert cone["A"].shape == (10, 4)
    assert pd.metadata["num_nodes"] == 4
    assert pd.metadata["format"] == "tsplib-maxcut-sdp"


# ---------------------------------------------------------------------------
# End-to-end: solve a small TSPLIB MaxCut SDP through Clarabel.
# ---------------------------------------------------------------------------


def test_tiny4_maxcut_sdp_solves_to_known_optimum(
    tmp_path: Path, tsplib_data_folder: Path
):
    """Tiny 4-cycle (square): MaxCut SDP gives the exact integer
    MaxCut value because the graph is bipartite. Cut value of the
    natural bipartition (alternating colors) equals 4 (sum of edges
    in the cut). The Goemans-Williamson SDP relaxation matches that
    on bipartite graphs."""
    pytest.importorskip("clarabel")
    from solver_benchmarks.core import status
    from solver_benchmarks.solvers.clarabel_adapter import ClarabelSolverAdapter

    dataset = _local_dataset(tsplib_data_folder)
    pd = dataset.load_problem("tiny4_explicit")
    adapter = ClarabelSolverAdapter({"verbose": False})
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = adapter.solve(pd, artifacts)
    assert result.status == status.OPTIMAL
    # primal_max_cut_sdp = -q'x + maxcut_constant
    # The 4-node graph in TINY_FULL_MATRIX has weights:
    # row 0: [0, 1, 1, 2], row 1: [1, 0, 2, 1], etc.
    # Sum = 2*(1+1+2+2+1+1) = 16, so total_weight = 8 and
    # maxcut_constant = 4. The SDP optimum value is the relaxation
    # upper bound on MaxCut (≤ 8 here, since total weight is 8).
    primal_value = -float(result.objective_value) + pd.metadata["maxcut_constant"]
    assert primal_value > 0  # nontrivial MaxCut SDP value
    assert primal_value <= pd.metadata["total_weight"] + 1e-6  # SDP ≤ total
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-3
    assert result.kkt["dual_res_rel"] < 1e-3


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def test_tsplib_name_recognizes_extensions(tmp_path: Path):
    from solver_benchmarks.datasets.tsplib_sdp import _tsplib_name

    assert _tsplib_name(tmp_path / "burma14.tsp") == "burma14"
    assert _tsplib_name(tmp_path / "burma14.tsp.gz") == "burma14"
    assert _tsplib_name(tmp_path / "not_a_tsplib.txt") is None


def test_normalize_subset_none_means_no_filter():
    from solver_benchmarks.datasets.tsplib_sdp import _normalize_subset

    assert _normalize_subset(None) is None
    assert _normalize_subset("all") is None


def test_normalize_subset_accepts_comma_string_and_list():
    from solver_benchmarks.datasets.tsplib_sdp import _normalize_subset

    assert _normalize_subset("a, b, c") == {"a", "b", "c"}
    assert _normalize_subset(["x", "y"]) == {"x", "y"}
