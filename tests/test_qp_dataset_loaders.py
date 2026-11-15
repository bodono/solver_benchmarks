"""QP dataset loaders: the .npz schema, subsets, preparation and downloads.

- An .npz P stored as one triangle reached the solvers as it was: adapters
  that read the upper triangle, the lower triangle or the whole matrix then
  solved three different problems, all reporting "optimal".
- .npz files were loaded with allow_pickle=True, which unpickles (and so
  can run) whatever a downloaded object array holds.
- Preparation ignored a single-valued subset (a QPLIB category, one MPC
  name), and a list subset was ignored by cutest_qp and crashed qplib.
- CUTEst preparation raised even when the exported files were present.
- QPLIB and DC OPF cached whatever the server returned, and QPLIB retried
  over plain HTTP after any HTTP error, a 404 included.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from solver_benchmarks.datasets import cblib as cblib_module
from solver_benchmarks.datasets import dc_opf as dc_opf_module
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.datasets import libsvm_qp as libsvm_module
from solver_benchmarks.datasets import mittelmann_sdp as mittelmann_sdp_module
from solver_benchmarks.datasets import mpc_qpbenchmark as mpc_module
from solver_benchmarks.datasets import qplib as qplib_module
from solver_benchmarks.datasets import tsplib_sdp as tsplib_module
from solver_benchmarks.datasets.mpc_qpbenchmark import read_mpc_qpbenchmark_npz


def _dataset(name: str, tmp_path: Path, **options):
    return get_dataset(name)(repo_root=tmp_path, data_root=tmp_path / "problem_classes", **options)


def _folder(tmp_path: Path, name: str) -> Path:
    folder = tmp_path / "problem_classes" / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------------------
# The Hessian of an .npz file.
# ---------------------------------------------------------------------------

_FULL = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, -1.0], [0.0, -1.0, 3.0]])


@pytest.mark.parametrize("stored", [np.triu(_FULL), np.tril(_FULL)], ids=["upper", "lower"])
def test_npz_hessian_stored_as_one_triangle_is_mirrored(tmp_path: Path, stored):
    path = tmp_path / "tri.npz"
    np.savez(path, P=stored, q=np.ones(3))

    qp, metadata = read_mpc_qpbenchmark_npz(path)

    np.testing.assert_array_equal(qp["P"].toarray(), _FULL)
    assert metadata["nnz_p"] == 7


def test_npz_symmetric_and_diagonal_hessians_are_kept_as_stored(tmp_path: Path):
    for name, matrix in (("full", _FULL), ("diag", np.diag([1.0, 2.0, 3.0]))):
        np.savez(tmp_path / f"{name}.npz", P=matrix, q=np.ones(3))
        qp, _ = read_mpc_qpbenchmark_npz(tmp_path / f"{name}.npz")
        np.testing.assert_array_equal(qp["P"].toarray(), matrix)


def test_npz_hessian_neither_symmetric_nor_triangular_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez(path, P=np.array([[2.0, 1.0], [3.0, 2.0]]), q=np.ones(2))

    with pytest.raises(ValueError, match="neither symmetric nor triangular"):
        read_mpc_qpbenchmark_npz(path)


def test_cutest_dataset_mirrors_a_triangular_hessian(tmp_path: Path):
    folder = _folder(tmp_path, "cutest_qp_data")
    np.savez(folder / "TRI.npz", P=np.triu(_FULL), q=-np.ones(3))

    problem = _dataset("cutest_qp", tmp_path).load_problem("TRI")

    np.testing.assert_array_equal(problem.qp["P"].toarray(), _FULL)


# ---------------------------------------------------------------------------
# Object arrays in an .npz file.
# ---------------------------------------------------------------------------

UNPICKLED: list[str] = []


def _record(tag: str) -> str:
    UNPICKLED.append(tag)
    return tag


class _Payload:
    def __reduce__(self):
        return (_record, ("ran",))


def test_npz_none_placeholders_load_as_missing(tmp_path: Path):
    path = tmp_path / "none.npz"
    np.savez(path, P=np.eye(2), q=np.ones(2), G=None, h=None, A=None, b=None, lb=np.zeros(2), ub=None)

    qp, _ = read_mpc_qpbenchmark_npz(path)

    assert qp["A"].shape == (2, 2)  # the bounds only
    assert qp["l"].tolist() == [0.0, 0.0]


def test_npz_object_arrays_other_than_none_are_not_unpickled(tmp_path: Path):
    UNPICKLED.clear()
    path = tmp_path / "payload.npz"
    np.savez(path, P=np.eye(2), q=np.ones(2), G=np.array(_Payload(), dtype=object))

    with pytest.raises(ValueError, match="object array"):
        read_mpc_qpbenchmark_npz(path)
    assert UNPICKLED == []


def test_npz_sparse_hessian_is_refused_with_a_message(tmp_path: Path):
    import scipy.sparse as sp

    path = tmp_path / "sparse.npz"
    np.savez(path, P=sp.csc_matrix(np.eye(2)), q=np.ones(2))

    with pytest.raises(ValueError, match="object array"):
        read_mpc_qpbenchmark_npz(path)


def test_mpc_downloads_from_the_pinned_revision(tmp_path: Path, monkeypatch):
    urls = []
    buffer = io.BytesIO()
    np.savez(buffer, P=np.eye(1), q=np.ones(1))

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(url, timeout=None):
        urls.append(url)
        return _Response(buffer.getvalue())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    mpc_module.download_mpc_qpbenchmark_problem("LIPMWALK0", tmp_path)

    assert urls == [
        "https://raw.githubusercontent.com/qpsolvers/mpc_qpbenchmark/"
        f"{mpc_module.MPC_QPBENCHMARK_REVISION}/data/LIPMWALK0.npz"
    ]
    assert f"ref={mpc_module.MPC_QPBENCHMARK_REVISION}" in mpc_module.MPC_QPBENCHMARK_API_URL


# ---------------------------------------------------------------------------
# CUTEst preparation and subsets.
# ---------------------------------------------------------------------------


def test_cutest_prepare_returns_when_the_requested_files_are_present(tmp_path: Path):
    folder = _folder(tmp_path, "cutest_qp_data")
    for name in ("HS35", "QAFIRO"):
        np.savez(folder / f"{name}.npz", P=np.eye(1), q=np.ones(1))

    _dataset("cutest_qp", tmp_path).prepare_data()
    _dataset("cutest_qp", tmp_path).prepare_data(problem_names=["HS35"])
    _dataset("cutest_qp", tmp_path).prepare_data(all_problems=True)
    _dataset("cutest_qp", tmp_path, subset="HS35,QAFIRO").prepare_data()


def test_cutest_prepare_names_only_the_missing_exports(tmp_path: Path):
    folder = _folder(tmp_path, "cutest_qp_data")
    np.savez(folder / "HS35.npz", P=np.eye(1), q=np.ones(1))

    with pytest.raises(RuntimeError) as excinfo:
        _dataset("cutest_qp", tmp_path).prepare_data(problem_names=["HS35", "QAFIRO"])
    assert "QAFIRO" in str(excinfo.value)
    assert "HS35" not in str(excinfo.value)
    assert (folder / "EXPORT_README.md").exists()

    with pytest.raises(RuntimeError, match="HS35MOD, QAFIRO, QPCBLEND"):
        _dataset("cutest_qp", tmp_path, subset="default").prepare_data()


def test_cutest_prepare_without_any_data_still_explains_the_export(tmp_path: Path):
    with pytest.raises(RuntimeError, match="cannot be downloaded automatically"):
        _dataset("cutest_qp", tmp_path).prepare_data()


def test_cutest_list_subset_filters(tmp_path: Path):
    folder = _folder(tmp_path, "cutest_qp_data")
    for name in ("HS35", "QAFIRO", "TRI"):
        np.savez(folder / f"{name}.npz", P=np.eye(1), q=np.ones(1))

    names = [spec.name for spec in _dataset("cutest_qp", tmp_path, subset=["HS35", "TRI"]).list_problems()]

    assert names == ["HS35", "TRI"]


# ---------------------------------------------------------------------------
# QPLIB subsets.
# ---------------------------------------------------------------------------


def _qplib_folder(tmp_path: Path) -> Path:
    folder = _folder(tmp_path, "qplib_data")
    (folder / "list_convex_qps.txt").write_text(
        "CCB (3)\n-------\n8790\n8792\n8991\n\nDCL (2)\n--------\n8495\n8500\n"
    )
    for problem_id in ("8790", "8792", "8495", "9001"):
        (folder / f"QPLIB_{problem_id}.qplib").write_text("")
    return folder


def test_qplib_list_subset_accepts_ids_and_categories(tmp_path: Path):
    _qplib_folder(tmp_path)

    def names(subset):
        return sorted(spec.name for spec in _dataset("qplib", tmp_path, subset=subset).list_problems())

    assert names(["9001"]) == ["9001"]
    assert names(["9001", "8495"]) == ["8495", "9001"]
    assert names(["ccb"]) == ["8790", "8792"]
    assert names("ccb,9001") == ["8790", "8792", "9001"]
    assert names("ccb") == ["8790", "8792"]
    assert names("9001") == ["9001"]


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("ccb", ["8790", "8792", "8991"]),
        ("dcl", ["8495", "8500"]),
        ("8792", ["8792"]),
        (None, list(qplib_module.QPLIB_DEFAULT_SUBSET)),
        ("default", list(qplib_module.QPLIB_DEFAULT_SUBSET)),
    ],
)
def test_qplib_prepare_downloads_the_configured_subset(tmp_path: Path, monkeypatch, subset, expected):
    _qplib_folder(tmp_path)
    fetched = []
    monkeypatch.setattr(qplib_module, "download_qplib_problem", lambda name, folder: fetched.append(name))

    _dataset("qplib", tmp_path, subset=subset).prepare_data()

    assert sorted(fetched) == sorted(expected)


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        (["ccb", "dcl"], ["8495", "8500", "8790", "8792", "8991"]),
        ("ccb,8495", ["8495", "8790", "8792", "8991"]),
        (["DCL", "QPLIB_9001"], ["8495", "8500", "9001"]),
    ],
)
def test_qplib_auto_prepare_expands_categories_in_a_list_subset(
    tmp_path: Path, monkeypatch, subset, expected
):
    # A list or comma-separated subset is forwarded by auto-prepare as
    # problem names; a category among them raised "QPLIB problem names must
    # be numeric IDs" although listing accepts the same subset.
    from solver_benchmarks.core.config import parse_run_config
    from solver_benchmarks.core.data_prepare import data_prepare_selection

    _qplib_folder(tmp_path)
    config = parse_run_config(
        {"run": {"dataset": "qplib", "dataset_options": {"subset": subset}},
         "solvers": [{"id": "scs", "solver": "scs"}]}
    )
    problem_names, all_problems = data_prepare_selection(config, config.datasets[0])
    assert problem_names is not None and not all_problems
    fetched = []
    monkeypatch.setattr(qplib_module, "download_qplib_problem", lambda name, folder: fetched.append(name))

    _dataset("qplib", tmp_path, subset=subset).prepare_data(problem_names)

    assert fetched == expected


@pytest.mark.parametrize(
    ("dataset_id", "module", "function", "subset", "expected"),
    [
        ("mpc_qpbenchmark", mpc_module, "download_mpc_qpbenchmark_problem", "LIPMWALK1", ["LIPMWALK1"]),
        ("libsvm_qp", libsvm_module, "download_libsvm_dataset", "a1a", ["a1a"]),
        ("cblib", cblib_module, "download_cblib_problem", "nql60", ["nql60"]),
        ("mittelmann_sdp", mittelmann_sdp_module, "download_mittelmann_sdp_problem", "theta1", ["theta1"]),
        ("tsplib_sdp", tsplib_module, "download_tsplib_instance", "berlin52", ["berlin52"]),
        ("dc_opf", dc_opf_module, "download_matpower_case", "case118", ["case118"]),
        ("dc_opf", dc_opf_module, "download_matpower_case", ["case118", "case300"], ["case118", "case300"]),
    ],
)
def test_prepare_downloads_a_single_named_subset(
    tmp_path: Path, monkeypatch, dataset_id, module, function, subset, expected
):
    fetched = []
    monkeypatch.setattr(module, function, lambda name, folder: fetched.append(name))
    if dataset_id == "dc_opf":
        labels = tmp_path / "labels.json"
        labels.write_text("{}")
        monkeypatch.setattr(dc_opf_module, "download_feasibility_labels", lambda folder: labels)

    _dataset(dataset_id, tmp_path, subset=subset).prepare_data()

    assert sorted(fetched) == sorted(expected)


# ---------------------------------------------------------------------------
# Download validation.
# ---------------------------------------------------------------------------

_QPLIB_BODY = "\n".join(
    [
        "TINY", "QCB", "minimize", "1 variables", "0 constraints",
        "1 quadratic objective", "1 1 2.0", "0.0", "0", "0.0", "1.0e30",
        "0.0", "0", "1.0", "0", "0.0", "0", "0.0", "0", "0.0", "0", "",
    ]
).encode()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_urlopen(monkeypatch, replies: dict[str, object]):
    calls: list[str] = []

    def fake(url, timeout=None):
        calls.append(url)
        scheme = url.split(":", 1)[0]
        reply = replies[scheme]
        if isinstance(reply, BaseException):
            raise reply
        return _Response(reply)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return calls


def test_qplib_download_does_not_fall_back_to_http_after_an_http_error(tmp_path: Path, monkeypatch):
    not_found = urllib.error.HTTPError("https://x", 404, "Not Found", {}, None)
    calls = _fake_urlopen(monkeypatch, {"https": not_found, "http": _QPLIB_BODY})

    with pytest.raises(urllib.error.HTTPError):
        qplib_module.download_qplib_problem("8790", tmp_path)

    assert [url.split(":", 1)[0] for url in calls] == ["https"]
    assert not (tmp_path / "QPLIB_8790.qplib").exists()


def test_qplib_download_falls_back_to_http_on_a_transport_error(tmp_path: Path, monkeypatch):
    calls = _fake_urlopen(
        monkeypatch, {"https": urllib.error.URLError("TLS handshake failed"), "http": _QPLIB_BODY}
    )

    path = qplib_module.download_qplib_problem("8790", tmp_path)

    assert [url.split(":", 1)[0] for url in calls] == ["https", "http"]
    assert path.read_bytes() == _QPLIB_BODY


def test_qplib_download_rejects_a_body_that_is_not_qplib(tmp_path: Path, monkeypatch):
    _fake_urlopen(monkeypatch, {"https": b"<html><body>Service unavailable</body></html>"})

    with pytest.raises(RuntimeError, match="did not download as a readable QPLIB file"):
        qplib_module.download_qplib_problem("8790", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_dc_opf_downloads_are_checked_before_caching(tmp_path: Path, monkeypatch):
    _fake_urlopen(monkeypatch, {"https": b"<html><body>Service unavailable</body></html>"})

    with pytest.raises(RuntimeError, match="MATPOWER"):
        dc_opf_module.download_matpower_case("case9", tmp_path)
    with pytest.raises(RuntimeError, match="feasibility labels"):
        dc_opf_module.download_feasibility_labels(tmp_path)
    assert list(tmp_path.iterdir()) == []

    buffer = io.BytesIO()
    scipy.io.savemat(buffer, {"mpc": {"baseMVA": 100.0}})
    _fake_urlopen(monkeypatch, {"https": buffer.getvalue()})
    assert dc_opf_module.download_matpower_case("case9", tmp_path).exists()
    _fake_urlopen(monkeypatch, {"https": json.dumps({"case9": "feasible"}).encode()})
    assert dc_opf_module.download_feasibility_labels(tmp_path).exists()
