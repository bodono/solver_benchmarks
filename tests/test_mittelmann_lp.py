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


def test_prepare_downloads_decodes_and_reuses_emps(monkeypatch, tmp_path):
    import bz2
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "emps" / "tiny.emps"
    calls = []

    def download(url, **kwargs):
        calls.append(url)
        if url.endswith(".mps.bz2"):
            raise urllib.error.HTTPError(url, 404, "Not found", None, None)
        return io.BytesIO(bz2.compress(fixture.read_bytes()))

    monkeypatch.setattr(mps.urllib.request, "urlopen", download)
    dataset = mps.MittelmannDataset(data_root=tmp_path)
    dataset.prepare_data(["tiny"])
    assert len(calls) == 2
    target = dataset.folder / "tiny.mps"
    assert "ROWS\n" in target.read_text()
    assert dataset.load_problem("tiny").data["n"] == 6
    dataset.prepare_data(["tiny"])
    assert len(calls) == 2


def test_prepare_repairs_cached_emps_without_network(monkeypatch, tmp_path):
    from pathlib import Path

    dataset = mps.MittelmannDataset(data_root=tmp_path)
    dataset.folder.mkdir()
    target = dataset.folder / "tiny.mps"
    target.write_bytes((Path(__file__).parent / "fixtures" / "emps" / "tiny.emps").read_bytes())

    def offline(*args, **kwargs):
        raise AssertionError("cached EMPS repair must not download")

    monkeypatch.setattr(mps.urllib.request, "urlopen", offline)
    dataset.prepare_data(["tiny"])
    assert dataset.load_problem("tiny").data["n"] == 6
    assert target.read_text().endswith("ENDATA\n")
    assert list(dataset.folder.iterdir()) == [target]


@pytest.mark.parametrize("cached", [False, True])
def test_failed_emps_conversion_does_not_publish_partial_output(monkeypatch, tmp_path, cached):
    import bz2
    from pathlib import Path

    broken = (Path(__file__).parent / "fixtures" / "emps" / "tiny.emps").read_bytes()
    broken = broken.replace(b"NCOST", b"NLOST")
    dataset = mps.MittelmannDataset(data_root=tmp_path)
    dataset.folder.mkdir()
    target = dataset.folder / "broken.mps"
    if cached:
        target.write_bytes(broken)
    monkeypatch.setattr(
        mps.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(bz2.compress(broken))
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dataset.prepare_data(["broken"])
    assert list(dataset.folder.iterdir()) == ([target] if cached else [])
    if cached:
        assert target.read_bytes() == broken


def test_standard_mps_download_is_preserved_and_validated(monkeypatch, tmp_path):
    import bz2

    body = b"NAME TINY\nROWS\n N COST\nCOLUMNS\n    X COST 1\nRHS\nENDATA\n"
    monkeypatch.setattr(
        mps.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(bz2.compress(body))
    )
    dataset = mps.MittelmannDataset(data_root=tmp_path)
    dataset.prepare_data(["plain"])
    target = dataset.folder / "plain.mps"
    assert target.read_bytes() == body
    assert dataset.load_problem("plain").data["n"] == 1


@pytest.mark.parametrize("cached", [False, True])
def test_invalid_mps_is_not_reported_as_prepared(monkeypatch, tmp_path, cached):
    import bz2

    body = b"This is not an MPS model\n"
    dataset = mps.MittelmannDataset(data_root=tmp_path)
    dataset.folder.mkdir()
    target = dataset.folder / "invalid.mps"
    if cached:
        target.write_bytes(body)
    monkeypatch.setattr(
        mps.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(bz2.compress(body))
    )
    with pytest.raises(RuntimeError, match="not a readable MPS model"):
        dataset.prepare_data(["invalid"])
    assert list(dataset.folder.iterdir()) == ([target] if cached else [])


@pytest.mark.parametrize("fail_validation", [False, True])
def test_preparation_never_lists_the_staging_model(monkeypatch, tmp_path, fail_validation):
    import bz2

    body = b"NAME TINY\nROWS\n N COST\nCOLUMNS\n    X COST 1\nRHS\nENDATA\n"
    dataset = mps.MittelmannDataset(data_root=tmp_path)
    validate = mps._validate_mittelmann_mps
    observed = []

    def validate_while_listing(path):
        observed.append([spec.name for spec in dataset.list_problems()])
        if fail_validation:
            raise RuntimeError("Rejected model")
        validate(path)

    monkeypatch.setattr(mps, "_validate_mittelmann_mps", validate_while_listing)
    monkeypatch.setattr(
        mps.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(bz2.compress(body))
    )
    if fail_validation:
        with pytest.raises(RuntimeError, match="Rejected model"):
            dataset.prepare_data(["plain"])
    else:
        dataset.prepare_data(["plain"])
    assert observed == [[]]
    assert list(dataset.folder.iterdir()) == ([] if fail_validation else [dataset.folder / "plain.mps"])
