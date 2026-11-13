"""Every bundled SDPLIB problem can be read.

Three members of the converted archive are damaged (stretches of each file
are zeros) and cannot be read: maxG55, maxG60 and thetaG51. Their original
SDPLIB files are bundled beside the archive and are what the loader reads
for these three.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import tarfile
from pathlib import Path

import numpy as np
import pytest

from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.transforms.sdplib import list_sdplib_tar, read_sdplib_jld2

# name: (constraints m, order n of the single semidefinite block, SHA-256 of
# the original file as SDPLIB distributes it)
BUNDLED_ORIGINALS = {
    "maxG55": (5000, 5000, "5298a62e46267efbc8e32912caa84eb70232817807025ec5efb285437cd3bb7f"),
    "maxG60": (7000, 7000, "b32acebafe409eb5dff1a44e61fdb46716b33f692599bbba87dee17da6a64e43"),
    "thetaG51": (6910, 1001, "e341e9f99b9f1f867c60502e9a8c5688a56dd2d5c6ec6e16793f2b2addee9021"),
}

_TINY_SDPA = "\n".join(["1", "1", "2", "1.0", "0 1 1 1 1.0", "0 1 2 2 1.0", "1 1 1 2 1.0"])


def _bundled_folder(repo_root: Path) -> Path:
    folder = repo_root / "problem_classes/sdplib_data"
    if not (folder / "sdplib.tar").exists():
        pytest.skip("bundled SDPLIB archive is not present")
    return folder


def _dataset(repo_root: Path, tmp_path: Path, *, archive: bool = True, originals: bool = True, **options):
    """The SDPLIB dataset over links to the bundled files, in a scratch data
    folder: nothing is extracted into the source tree, and nothing a developer
    keeps in their own data folder is seen."""
    bundled = _bundled_folder(repo_root)
    folder = tmp_path / "problem_classes" / "sdplib_data"
    folder.mkdir(parents=True, exist_ok=True)
    names = (["sdplib.tar"] if archive else []) + (
        [f"{name}.dat-s.gz" for name in BUNDLED_ORIGINALS] if originals else []
    )
    for name in names:
        if not (folder / name).exists():
            (folder / name).symlink_to(bundled / name)
    return get_dataset("sdplib")(repo_root=repo_root, data_root=tmp_path / "problem_classes", **options)


@pytest.mark.parametrize("name", sorted(BUNDLED_ORIGINALS))
def test_bundled_original_is_the_sdplib_file(repo_root: Path, name: str):
    path = _bundled_folder(repo_root) / f"{name}.dat-s.gz"
    with gzip.open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert digest == BUNDLED_ORIGINALS[name][2]


def test_damaged_archive_members_are_listed_from_their_originals(repo_root: Path, tmp_path: Path):
    dataset = _dataset(repo_root, tmp_path, subset="all")
    specs = {spec.name: spec for spec in dataset.list_problems()}
    member_sizes = list_sdplib_tar(dataset.tar_path)

    assert len(specs) == len(member_sizes) == 92  # the originals add no problem
    from_originals = {name for name, spec in specs.items() if spec.metadata["format"] == "sdpa-s"}
    assert from_originals == set(BUNDLED_ORIGINALS)
    for name in BUNDLED_ORIGINALS:
        assert specs[name].path.name == f"{name}.dat-s.gz"
        # Filtered by the size of the member the original replaces, not by
        # its own, which is under 0.2 MB.
        assert specs[name].metadata["size_bytes"] == member_sizes[name] > 50_000_000
        assert specs[name].path.stat().st_size < 200_000


def test_the_three_largest_problems_stay_outside_a_size_limit(repo_root: Path, tmp_path: Path):
    limited = _dataset(repo_root, tmp_path, subset="all", max_size_mb=5)
    visible = {spec.name for spec in limited.visible_problems()}
    assert visible and not visible & set(BUNDLED_ORIGINALS)
    unlimited = _dataset(repo_root, tmp_path, subset="all")
    assert set(BUNDLED_ORIGINALS) <= {spec.name for spec in unlimited.visible_problems()}


def test_an_original_is_filtered_by_the_size_of_its_converted_file(repo_root: Path, tmp_path: Path):
    dataset = _dataset(repo_root, tmp_path, originals=False, subset="all")
    folder = dataset.folder
    member_sizes = list_sdplib_tar(dataset.tar_path)
    (folder / "truss1.dat-s").write_text(_TINY_SDPA)      # replaces an archive member
    (folder / "mine.dat-s").write_text(_TINY_SDPA)        # no converted counterpart
    (folder / "mine2.dat-s").write_text(_TINY_SDPA)       # both encodings: plain wins
    with gzip.open(folder / "mine2.dat-s.gz", "wt") as handle:
        handle.write(_TINY_SDPA)

    specs = {spec.name: spec for spec in dataset.list_problems()}

    assert specs["truss1"].metadata["format"] == "sdpa-s"
    assert specs["truss1"].metadata["size_bytes"] == member_sizes["truss1"]
    assert specs["mine"].metadata["format"] == "sdpa-s" and "size_bytes" not in specs["mine"].metadata
    assert specs["mine2"].path.name == "mine2.dat-s"
    assert len(specs) == 92 + 2


def test_without_the_archive_an_original_takes_the_size_of_an_extracted_file(
    repo_root: Path, tmp_path: Path
):
    dataset = _dataset(repo_root, tmp_path, archive=False, originals=False, subset="all")
    folder = dataset.folder
    (folder / "a.dat-s").write_text(_TINY_SDPA)
    (folder / "a.jld2").write_bytes(b"\0" * 4096)  # an extracted converted file of the same name
    (folder / "b.dat-s").write_text(_TINY_SDPA)

    specs = {spec.name: spec for spec in dataset.list_problems()}

    assert set(specs) == {"a", "b"}
    assert specs["a"].metadata["format"] == "sdpa-s" and specs["a"].metadata["size_bytes"] == 4096
    assert "size_bytes" not in specs["b"].metadata  # filtered by its own size


def test_prepare_leaves_a_member_with_an_original_alone(repo_root: Path, tmp_path: Path):
    dataset = _dataset(repo_root, tmp_path, subset="all")
    dataset.prepare_data(["thetaG51", "truss1"])
    # thetaG51's member is 58 MB that would never be read; truss1 is extracted.
    assert not (dataset.folder / "thetaG51.jld2").exists()
    assert (dataset.folder / "truss1.jld2").exists()
    assert dataset.problem_by_name("thetaG51").metadata["format"] == "sdpa-s"


def _check_loaded(name: str, problem) -> None:
    m, n, _ = BUNDLED_ORIGINALS[name]
    data = problem.cone
    assert data["cone"] == {"s": [n]}
    assert data["A"].shape == (n * (n + 1) // 2, m)
    assert data["b"].shape == (n * (n + 1) // 2,)
    # All three minimise the sum of the variables.
    assert np.array_equal(data["q"], np.ones(m))
    assert problem.metadata["format"] == "sdpa-s"
    assert problem.metadata["block_orders"] == [n]


@pytest.mark.parametrize("name", sorted(BUNDLED_ORIGINALS))
def test_damaged_member_is_read_from_its_original(repo_root: Path, tmp_path: Path, name: str):
    for blocks in ("declared", "merged"):
        problem = _dataset(repo_root, tmp_path, blocks=blocks).load_problem(name)
        _check_loaded(name, problem)
    if name.startswith("maxG"):
        # MaxCut relaxation: constraint i fixes diagonal entry i.
        assert problem.cone["A"].nnz == BUNDLED_ORIGINALS[name][0]
    else:
        # The damaged member has zeros in place of most of this vector.
        assert np.count_nonzero(problem.cone["q"]) == 6910


@pytest.mark.slow
def test_every_archive_member_without_an_original_can_be_read(repo_root: Path, tmp_path: Path):
    # One pass over the archive: each member is copied to a scratch file in
    # pieces, read and removed. The three members that have a bundled
    # original are skipped; they are the ones that cannot be read.
    archive = _bundled_folder(repo_root) / "sdplib.tar"
    scratch = tmp_path / "member.jld2"
    read, unreadable = [], []
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).stem
            if not member.name.endswith(".jld2") or name in BUNDLED_ORIGINALS:
                continue
            source = tar.extractfile(member)
            assert source is not None, member.name
            with source, open(scratch, "wb") as sink:
                shutil.copyfileobj(source, sink)
            try:
                problem = read_sdplib_jld2(scratch)
            except Exception as exc:  # noqa: BLE001 - the point is to list every failure
                unreadable.append(f"{name}: {type(exc).__name__}: {exc}")
            else:
                assert problem["A"].shape == (problem["b"].size, problem["q"].size), name
                read.append(name)
            scratch.unlink()
    assert not unreadable, unreadable
    assert len(read) == 92 - len(BUNDLED_ORIGINALS)
