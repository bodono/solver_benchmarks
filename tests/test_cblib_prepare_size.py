"""CBLIB ``prepare_data`` honours ``max_size_mb`` when following the full index."""

from __future__ import annotations

from pathlib import Path

import pytest

from solver_benchmarks.datasets import cblib

SIZES = {"small_a": 1_000_000, "big": 400_000_000, "small_b": 4_900_000}


@pytest.fixture
def fake_remote(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    fetched: list[str] = []
    monkeypatch.setattr(cblib, "cblib_remote_problem_names", lambda: sorted(SIZES))
    monkeypatch.setattr(cblib, "cblib_remote_size_bytes", lambda name: SIZES[name])

    def fake_download(name: str, folder: Path) -> Path:
        fetched.append(name)
        return folder / f"{name}.cbf.gz"

    monkeypatch.setattr(cblib, "download_cblib_problem", fake_download)
    return fetched


def test_all_problems_with_max_size_skips_large_files(tmp_path: Path, fake_remote: list[str]):
    dataset = cblib.CBLIBDataset(repo_root=tmp_path, max_size_mb=5)
    dataset.prepare_data(all_problems=True)
    assert sorted(fake_remote) == ["small_a", "small_b"]


def test_all_problems_without_max_size_fetches_everything(tmp_path: Path, fake_remote: list[str]):
    dataset = cblib.CBLIBDataset(repo_root=tmp_path)
    dataset.prepare_data(all_problems=True)
    assert sorted(fake_remote) == sorted(SIZES)


def test_explicit_problem_names_ignore_max_size(tmp_path: Path, fake_remote: list[str]):
    dataset = cblib.CBLIBDataset(repo_root=tmp_path, max_size_mb=5)
    dataset.prepare_data(["big"])
    assert fake_remote == ["big"]
