"""SDPLIB problems read from the converted archive get their declared blocks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core.config import _stable_digest, parse_run_config, solve_signature
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.transforms.sdpa import parse_sdpa_s, sdpa_to_cone_problem
from solver_benchmarks.transforms.sdplib import _psd_vec, split_into_declared_blocks
from solver_benchmarks.transforms.sdplib_blocks import SDPLIB_BLOCK_SIZES

SQRT2 = np.sqrt(2.0)


def _merged_problem(matrices: list[np.ndarray], c: np.ndarray) -> dict:
    """The problem ``read_sdplib_jld2`` builds from F_0, F_1, ..., F_m."""
    n = matrices[0].shape[0]
    columns = [-_psd_vec(sp.csc_matrix(matrix)) for matrix in matrices[1:]]
    a = sp.hstack(columns, format="csc")
    b = -_psd_vec(sp.csc_matrix(matrices[0])).toarray().reshape(-1)
    return {
        "P": None, "A": a, "b": b, "q": np.asarray(c, dtype=float), "r": 0.0,
        "n": int(a.shape[1]), "m": int(a.shape[0]), "cone": {"s": [n]}, "obj_type": "min",
    }


def _block_diagonal(rng: np.random.Generator, sizes: tuple[int, ...]) -> np.ndarray:
    n = sum(abs(size) for size in sizes)
    matrix = np.zeros((n, n))
    offset = 0
    for size in sizes:
        k = abs(size)
        if size < 0:
            matrix[offset:offset + k, offset:offset + k] = np.diag(rng.uniform(1.0, 2.0, k))
        else:
            block = rng.uniform(1.0, 2.0, (k, k))
            matrix[offset:offset + k, offset:offset + k] = block + block.T
        offset += k
    return matrix


def _unvec(vec: np.ndarray, order: int) -> np.ndarray:
    """Symmetric matrix from its scaled column-major lower triangle."""
    matrix = np.zeros((order, order))
    cols, rows = np.triu_indices(order)
    matrix[rows, cols] = vec / np.where(rows == cols, 1.0, SQRT2)
    return matrix + np.tril(matrix, -1).T


def test_split_recovers_every_block_in_declared_order():
    sizes = (3, -2, 1, 2)
    rng = np.random.default_rng(0)
    matrices = [_block_diagonal(rng, sizes) for _ in range(4)]
    merged = _merged_problem(matrices, np.arange(3.0))
    rows_before = merged["A"].shape[0]
    split = split_into_declared_blocks(merged, sizes)

    assert merged["A"].shape[0] == rows_before and merged["cone"] == {"s": [8]}
    assert split["cone"] == {"l": 2, "s": [3, 1, 2]}
    # Consumers read the row layout from the order of the keys.
    assert list(split["cone"]) == ["l", "s"]
    assert split["m"] == split["A"].shape[0] == 2 + 6 + 1 + 3
    assert split["n"] == 3 and np.array_equal(split["q"], merged["q"])
    # Row layout: the diagonal block's two rows, then the blocks of order 3, 1, 2.
    a = split["A"].toarray()
    for column, matrix in enumerate([matrices[0]] + matrices[1:]):
        vec = -split["b"] if column == 0 else -a[:, column - 1]
        assert np.allclose(vec[0:2], np.diag(matrix)[3:5])
        assert np.allclose(_unvec(vec[2:8], 3), matrix[0:3, 0:3])
        assert np.allclose(_unvec(vec[8:9], 1), matrix[5:6, 5:6])
        assert np.allclose(_unvec(vec[9:12], 2), matrix[6:8, 6:8])


def test_split_matches_the_original_file_reader():
    # The same problem as SDPA sparse text and as merged matrices: the two
    # readers must give the same rows, in the same order, with the same signs.
    text = "\n".join([
        "2", "3", "2 -2 1", "1.0 2.0",
        "0 1 1 1 1.0", "0 1 1 2 0.5", "0 1 2 2 2.0", "0 2 1 1 3.0", "0 2 2 2 4.0", "0 3 1 1 5.0",
        "1 1 1 1 1.0", "1 1 1 2 -1.5", "1 2 2 2 1.0", "1 3 1 1 2.0",
        "2 1 2 2 1.0", "2 2 1 1 -2.0", "2 3 1 1 1.0",
    ])
    primal = parse_sdpa_s(text)
    reference = sdpa_to_cone_problem(primal)

    n = 5
    matrices = [np.zeros((n, n)) for _ in range(3)]
    entries = {
        0: [(0, 0, 1.0), (0, 1, 0.5), (1, 1, 2.0), (2, 2, 3.0), (3, 3, 4.0), (4, 4, 5.0)],
        1: [(0, 0, 1.0), (0, 1, -1.5), (3, 3, 1.0), (4, 4, 2.0)],
        2: [(1, 1, 1.0), (2, 2, -2.0), (4, 4, 1.0)],
    }
    for index, triples in entries.items():
        for i, j, value in triples:
            matrices[index][i, j] = matrices[index][j, i] = value
    split = split_into_declared_blocks(_merged_problem(matrices, np.array([1.0, 2.0])), (2, -2, 1))

    assert split["cone"] == reference["cone"] == {"l": 2, "s": [2, 1]}
    assert list(split["cone"]) == list(reference["cone"]) == ["l", "s"]
    assert split["A"].shape == reference["A"].shape
    assert np.allclose(split["A"].toarray(), reference["A"].toarray())
    assert np.allclose(split["b"], np.asarray(reference["b"], dtype=float))
    assert np.allclose(split["q"], np.asarray(reference["q"], dtype=float))


def test_split_leaves_a_single_declared_block_alone():
    rng = np.random.default_rng(1)
    merged = _merged_problem([_block_diagonal(rng, (4,)) for _ in range(3)], np.ones(2))
    assert split_into_declared_blocks(merged, (4,)) is merged


def test_split_rejects_an_entry_outside_the_declared_blocks():
    rng = np.random.default_rng(2)
    matrices = [_block_diagonal(rng, (2, 2)) for _ in range(3)]
    matrices[2][3, 0] = matrices[2][0, 3] = 0.25
    with pytest.raises(ValueError, match=r"Entry \(3, 0\)"):
        split_into_declared_blocks(_merged_problem(matrices, np.ones(2)), (2, 2))


def test_split_rejects_a_right_hand_side_entry_outside_the_declared_blocks():
    rng = np.random.default_rng(5)
    matrices = [_block_diagonal(rng, (2, 2)) for _ in range(3)]
    matrices[0][2, 1] = matrices[0][1, 2] = 0.5  # F_0 only: a nonzero in b, none in A
    with pytest.raises(ValueError, match=r"Entry \(2, 1\)"):
        split_into_declared_blocks(_merged_problem(matrices, np.ones(2)), (2, 2))


def test_split_accepts_a_stored_zero_outside_the_declared_blocks():
    rng = np.random.default_rng(6)
    merged = _merged_problem([_block_diagonal(rng, (2, 2)) for _ in range(3)], np.ones(2))
    a = merged["A"].tolil()
    a[2, 0] = 1.0  # entry (2, 0) of the merged matrix, between the two blocks
    a = a.tocsr()
    a.data[a.indptr[2]] = 0.0  # keep it stored, as an explicit zero
    merged["A"] = a
    assert a.nnz == merged["A"].nnz and (a.data == 0.0).any()
    split = split_into_declared_blocks(merged, (2, 2))
    assert split["cone"] == {"s": [2, 2]} and split["A"].shape[0] == 6
    assert (merged["A"].data == 0.0).any()  # the caller's matrix is not modified


@pytest.mark.parametrize("sizes", [(2, 3), (2, 0, 2), ()])
def test_split_rejects_sizes_that_do_not_fit(sizes):
    rng = np.random.default_rng(3)
    merged = _merged_problem([_block_diagonal(rng, (4,)) for _ in range(2)], np.ones(1))
    with pytest.raises(ValueError, match="do not add up"):
        split_into_declared_blocks(merged, sizes)


def test_split_rejects_a_problem_that_is_not_one_block():
    rng = np.random.default_rng(4)
    merged = _merged_problem([_block_diagonal(rng, (4,)) for _ in range(2)], np.ones(1))
    merged["cone"] = {"s": [2, 2]}
    with pytest.raises(ValueError, match="single semidefinite block"):
        split_into_declared_blocks(merged, (2, 2))


def test_block_table_is_well_formed():
    assert len(SDPLIB_BLOCK_SIZES) == 92
    for name, sizes in SDPLIB_BLOCK_SIZES.items():
        assert sizes and all(isinstance(size, int) and size != 0 for size in sizes), name
    assert SDPLIB_BLOCK_SIZES["arch0"] == (161, -174)
    assert SDPLIB_BLOCK_SIZES["hinf8"] == (5, 5, 6)
    assert SDPLIB_BLOCK_SIZES["truss1"] == (2, 2, 2, 2, 2, 2, 1)
    assert SDPLIB_BLOCK_SIZES["theta1"] == (50,)
    assert SDPLIB_BLOCK_SIZES["hinf13"] == (7, 9, 14)
    assert SDPLIB_BLOCK_SIZES["control5"] == (50, 25)
    assert SDPLIB_BLOCK_SIZES["ss30"] == (294, -132)
    assert SDPLIB_BLOCK_SIZES["truss5"] == (*(10,) * 33, 1)
    # Diagonal blocks: only the arch problems and ss30 have one.
    with_diagonal = {name for name, sizes in SDPLIB_BLOCK_SIZES.items() if any(s < 0 for s in sizes)}
    assert with_diagonal == {"arch0", "arch2", "arch4", "arch8", "ss30"}
    # 39 problems are declared with more than one block.
    several = {name for name, sizes in SDPLIB_BLOCK_SIZES.items() if len(sizes) > 1}
    assert len(several) == 39
    assert {name.rstrip("0123456789") for name in several} == {"arch", "control", "hinf", "ss", "truss"}


def _archive_dataset(repo_root: Path, tmp_path: Path, **options):
    """The SDPLIB dataset over the bundled archive, extracting into tmp_path, not the source tree."""
    archive = repo_root / "problem_classes/sdplib_data/sdplib.tar"
    if not archive.exists():
        pytest.skip("bundled SDPLIB archive is not present")
    folder = tmp_path / "problem_classes" / "sdplib_data"
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "sdplib.tar").exists():
        (folder / "sdplib.tar").symlink_to(archive)
    return get_dataset("sdplib")(repo_root=repo_root, data_root=tmp_path / "problem_classes", **options)


@pytest.mark.parametrize(
    ("name", "cone", "merged_order"),
    [
        ("truss1", {"s": [2, 2, 2, 2, 2, 2, 1]}, 13),
        ("hinf1", {"s": [4, 4, 6]}, 14),
        ("control1", {"s": [10, 5]}, 15),
        ("arch0", {"l": 174, "s": [161]}, 335),
        ("theta1", {"s": [50]}, 50),
    ],
)
def test_bundled_problems_load_with_declared_blocks(
    repo_root: Path, tmp_path: Path, name: str, cone: dict, merged_order: int
):
    declared = _archive_dataset(repo_root, tmp_path, subset="all").load_problem(name)
    assert declared.cone["cone"] == cone and list(declared.cone["cone"]) == list(cone)
    assert declared.metadata["block_structure"] == "declared"
    assert declared.metadata["num_blocks"] == len(SDPLIB_BLOCK_SIZES[name])
    assert declared.metadata["block_orders"] == [abs(size) for size in SDPLIB_BLOCK_SIZES[name]]
    rows = cone.get("l", 0) + sum(order * (order + 1) // 2 for order in cone["s"])
    assert declared.cone["A"].shape[0] == declared.cone["m"] == declared.cone["b"].size == rows

    merged = _archive_dataset(repo_root, tmp_path, subset="all", blocks="merged").load_problem(name)
    assert merged.cone["cone"] == {"s": [merged_order]}
    assert merged.metadata["block_structure"] == "merged"
    assert merged.metadata["num_blocks"] == 1 and merged.metadata["block_orders"] == [merged_order]
    # Same variables and objective; only rows that carry no data are gone.
    assert merged.cone["A"].shape[1] == declared.cone["A"].shape[1]
    assert np.array_equal(merged.cone["q"], declared.cone["q"])
    assert merged.cone["A"].nnz == declared.cone["A"].nnz
    assert np.count_nonzero(merged.cone["b"]) == np.count_nonzero(declared.cone["b"])


@pytest.mark.slow
def test_every_problem_declared_with_several_blocks_splits(repo_root: Path, tmp_path: Path):
    # A wrong table entry either leaves a nonzero outside the blocks (and
    # raises) or changes the row count; this loads all 39 to see neither.
    dataset = _archive_dataset(repo_root, tmp_path, subset="all")
    for name, sizes in sorted(SDPLIB_BLOCK_SIZES.items()):
        if len(sizes) == 1:
            continue
        cone = dataset.load_problem(name).cone
        assert cone["cone"].get("l", 0) == sum(-size for size in sizes if size < 0), name
        assert cone["cone"]["s"] == [size for size in sizes if size > 0], name
        rows = sum(-s if s < 0 else s * (s + 1) // 2 for s in sizes)
        assert cone["A"].shape[0] == rows, name


def test_unknown_blocks_option_is_rejected_at_planning(repo_root: Path, tmp_path: Path):
    dataset = _archive_dataset(repo_root, tmp_path, blocks="merge")
    with pytest.raises(ValueError, match="Unknown SDPLIB blocks option"):
        dataset.list_problems()


_TWO_BLOCK_SDPA = "\n".join([
    "1", "2", "2 1", "1.0",
    "0 1 1 1 1.0", "0 1 2 2 1.0", "0 2 1 1 1.0",
    "1 1 1 2 1.0", "1 2 1 1 1.0",
])


def test_original_file_records_its_form_and_has_no_merged_form(repo_root: Path, tmp_path: Path):
    folder = tmp_path / "problem_classes" / "sdplib_data"
    folder.mkdir(parents=True)
    (folder / "tiny.dat-s").write_text(_TWO_BLOCK_SDPA)
    kwargs = {"repo_root": repo_root, "data_root": tmp_path / "problem_classes", "subset": "all"}
    declared = get_dataset("sdplib")(**kwargs).load_problem("tiny")
    assert declared.cone["cone"] == {"s": [2, 1]}
    assert declared.metadata["block_structure"] == "declared"
    with pytest.raises(ValueError, match="no merged form"):
        get_dataset("sdplib")(**kwargs, blocks="merged").load_problem("tiny")


def _signature(dataset_options: dict) -> str:
    config = parse_run_config({
        "run": {"dataset": "sdplib", "dataset_options": dataset_options},
        "solvers": [{"id": "scs", "solver": "scs"}],
    })
    return solve_signature(config, config.datasets[0], config.solvers[0])


def test_resume_signature_separates_the_two_forms():
    # A run directory from before the block table holds merged-form rows whose
    # signature was computed from options without a ``blocks`` key.
    config = parse_run_config({
        "run": {"dataset": "sdplib", "dataset_options": {"subset": "all"}},
        "solvers": [{"id": "scs", "solver": "scs"}],
    })
    dataset, solver = config.datasets[0], config.solvers[0]
    before_the_change = _stable_digest(
        {
            "dataset": {"id": dataset.id, "name": dataset.name, "dataset_options": {"subset": "all"}},
            "solver": {
                "id": solver.id, "solver": solver.solver, "settings": solver.settings,
                "effective_timeout_seconds": None,
            },
        },
        length=24,
    )
    assert _signature({"subset": "all", "blocks": "merged"}) == before_the_change
    assert _signature({"subset": "all"}) != before_the_change
    assert _signature({"subset": "all"}) == _signature({"subset": "all", "blocks": "declared"})


def test_other_datasets_keep_their_resume_signature():
    config = parse_run_config({
        "run": {"dataset": "netlib", "dataset_options": {"subset": "feasible"}},
        "solvers": [{"id": "scs", "solver": "scs"}],
    })
    dataset, solver = config.datasets[0], config.solvers[0]
    unchanged = _stable_digest(
        {
            "dataset": {"id": dataset.id, "name": dataset.name, "dataset_options": {"subset": "feasible"}},
            "solver": {
                "id": solver.id, "solver": solver.solver, "settings": solver.settings,
                "effective_timeout_seconds": None,
            },
        },
        length=24,
    )
    assert solve_signature(config, dataset, solver) == unchanged
