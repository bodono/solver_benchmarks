"""Offline coverage for Mittelmann LP discovery and preparation."""

import io
import urllib.error

import pytest

from solver_benchmarks.datasets import mps


@pytest.mark.parametrize("index", [b"", b"<html>Service unavailable</html>"])
def test_prepare_all_rejects_an_empty_index(monkeypatch, tmp_path, index):
    monkeypatch.setattr(mps.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(index))
    downloads = []
    monkeypatch.setattr(mps, "_download_mittelmann_problem", lambda *a: downloads.append(a))

    with pytest.raises(RuntimeError, match="index contained no .bz2 instances"):
        mps.MittelmannDataset(data_root=tmp_path).prepare_data(all_problems=True)

    assert downloads == []
    assert not (tmp_path / "mittelmann").exists()


def test_prepare_all_propagates_index_failure(monkeypatch, tmp_path):
    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(mps.urllib.request, "urlopen", offline)
    dataset = mps.MittelmannDataset(data_root=tmp_path)
    with pytest.raises(urllib.error.URLError, match="offline"):
        dataset.prepare_data(all_problems=True)
    # Local discovery remains available without the remote index.
    dataset.folder.mkdir()
    (dataset.folder / "staged.mps").touch()
    assert [spec.name for spec in dataset.list_problems()] == ["staged"]


def test_default_prepare_keeps_the_smoke_subset(monkeypatch, tmp_path):
    downloads = []
    monkeypatch.setattr(mps, "_download_mittelmann_problem", lambda *a: downloads.append(a))

    mps.MittelmannDataset(data_root=tmp_path).prepare_data()

    assert downloads == [("qap15", tmp_path / "mittelmann")]
