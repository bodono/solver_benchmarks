"""SDPLIB archive access: atomic extraction and the cached member index.

Members were extracted straight to their final path, so a second worker
loading the same problem read a half-written file, and a worker killed in
the middle of a copy left a truncated file that every later run read.
Listing the members decompressed the whole 911 MB gzip stream, in every
worker, on every load.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path

import pytest

from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.transforms import sdplib as sdplib_module
from solver_benchmarks.transforms.sdplib import extract_from_tar, list_sdplib_tar

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_TAR = REPO_ROOT / "problem_classes" / "sdplib_data" / "sdplib.tar"


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_an_interrupted_extraction_leaves_no_file_at_the_target(tmp_path: Path, monkeypatch):
    tar_path = _tar(tmp_path / "sdplib.tar", {"p.jld2": b"x" * 1000})
    cache = tmp_path / ".cache"
    target = cache / "p.jld2"

    def killed_midway(source, sink, *args, **kwargs):
        # Nothing is visible at the final path while the copy runs.
        assert not target.exists()
        sink.write(source.read(10))
        raise KeyboardInterrupt  # stands in for the worker being killed

    monkeypatch.setattr(sdplib_module.shutil, "copyfileobj", killed_midway)
    with pytest.raises(KeyboardInterrupt):
        extract_from_tar(tar_path, "p", cache)

    assert not target.exists()
    assert list(cache.iterdir()) == []


def test_a_cache_file_of_the_wrong_size_is_extracted_again(tmp_path: Path):
    tar_path = _tar(tmp_path / "sdplib.tar", {"p.jld2": b"abcdef"})
    cache = tmp_path / ".cache"
    cache.mkdir()
    (cache / "p.jld2").write_bytes(b"abc")  # left by an interrupted copy

    path = extract_from_tar(tar_path, "p", cache, expected_size=6)

    assert path.read_bytes() == b"abcdef"
    # A complete cache file is used as it is.
    assert extract_from_tar(tar_path, "p", cache, expected_size=6) == path


def test_member_index_is_cached_beside_the_archive(tmp_path: Path, monkeypatch):
    tar_path = _tar(tmp_path / "sdplib.tar", {"a.jld2": b"1", "b.jld2": b"22", "notes.txt": b""})
    index = tmp_path / ".cache" / "sdplib.tar.index.json"

    assert list_sdplib_tar(tar_path, index_path=index) == {"a": 1, "b": 2}
    assert index.exists()

    def no_scan(*args, **kwargs):
        raise AssertionError("the archive was scanned again")

    monkeypatch.setattr(sdplib_module.tarfile, "open", no_scan)
    assert list_sdplib_tar(tar_path, index_path=index) == {"a": 1, "b": 2}

    # A changed archive is scanned again.
    monkeypatch.undo()
    _tar(tar_path, {"a.jld2": b"1", "c.jld2": b"333"})
    os.utime(tar_path, ns=(1, 1))
    assert list_sdplib_tar(tar_path, index_path=index) == {"a": 1, "c": 3}


def test_an_unwritable_index_does_not_fail_the_listing(tmp_path: Path):
    tar_path = _tar(tmp_path / "sdplib.tar", {"a.jld2": b"1"})
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("")

    assert list_sdplib_tar(tar_path, index_path=blocker / "index.json") == {"a": 1}


def test_dataset_lists_from_the_index_and_loads_over_a_truncated_cache(tmp_path: Path, monkeypatch):
    pytest.importorskip("h5py")
    if not BUNDLED_TAR.exists():
        pytest.skip("bundled SDPLIB archive is missing")
    data_root = tmp_path / "problem_classes"
    folder = data_root / "sdplib_data"
    folder.mkdir(parents=True)
    (folder / "sdplib.tar").symlink_to(BUNDLED_TAR)
    dataset = get_dataset("sdplib")(repo_root=tmp_path, data_root=data_root)

    names = [spec.name for spec in dataset.list_problems()]
    assert "theta1" in names
    member_size = next(
        spec.metadata["size_bytes"] for spec in dataset.list_problems() if spec.name == "theta1"
    )

    # A half-written file from an interrupted extraction.
    cache = folder / ".cache"
    cache.mkdir(exist_ok=True)
    full = extract_from_tar(folder / "sdplib.tar", "theta1", tmp_path / "scratch")
    (cache / "theta1.jld2").write_bytes(full.read_bytes()[: member_size // 2])

    real_open = tarfile.open
    scans = []

    def counting_open(*args, **kwargs):
        scans.append(args)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(sdplib_module.tarfile, "open", counting_open)
    problem = dataset.load_problem("theta1")

    assert problem.cone["cone"]["s"] == [50]
    assert (cache / "theta1.jld2").stat().st_size == member_size
    # One opening of the archive, to extract the member again; the listing
    # came from the index.
    assert len(scans) == 1
