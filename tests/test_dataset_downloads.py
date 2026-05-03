from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from solver_benchmarks.cli import main
from solver_benchmarks.core import status
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.solvers import get_solver

pytestmark = pytest.mark.network

TERMINAL_SOLVE_STATUSES = {
    status.OPTIMAL,
    status.OPTIMAL_INACCURATE,
    *status.ANY_INFEASIBLE,
}


@pytest.mark.parametrize(
    ("dataset_id", "prepare_name", "expected_problem_name", "expected_file"),
    [
        ("cblib", "nb", "nb", "cblib_data/nb.cbf.gz"),
        ("dc_opf", "case5", "case5", "dc_opf_data/case5.m"),
        ("libsvm_qp", "heart", "svm_dual_heart", "libsvm_data/heart.libsvm"),
        (
            "miplib",
            "markshare_4_0",
            "markshare_4_0",
            "miplib_data/markshare_4_0.mps.gz",
        ),
        ("mittelmann", "qap15", "qap15", "mittelmann/qap15.mps"),
        (
            "mittelmann_sdp",
            "trto3",
            "trto3",
            "mittelmann_sdp_data/trto3.dat-s.gz",
        ),
        (
            "mpc_qpbenchmark",
            "LIPMWALK0",
            "LIPMWALK0",
            "mpc_qpbenchmark_data/LIPMWALK0.npz",
        ),
        ("qplib", "8790", "8790", "qplib_data/QPLIB_8790.qplib"),
        ("tsplib_sdp", "burma14", "burma14", "tsplib_data/burma14.tsp"),
    ],
)
def test_external_dataset_prepare_downloads_real_problem_and_uses_cache(
    monkeypatch,
    tmp_path: Path,
    dataset_id: str,
    prepare_name: str,
    expected_problem_name: str,
    expected_file: str,
):
    data_root = tmp_path / "problem_classes"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "data",
            "prepare",
            dataset_id,
            "--repo-root",
            str(tmp_path),
            "--option",
            f"data_root={data_root}",
            "--problem",
            prepare_name,
        ],
    )

    assert result.exit_code == 0, result.output

    downloaded = data_root / expected_file
    assert downloaded.exists()
    assert downloaded.stat().st_size > 0
    dataset = get_dataset(dataset_id)(repo_root=tmp_path, data_root=data_root)
    assert expected_problem_name in {spec.name for spec in dataset.list_problems()}

    def fail_if_network_is_used(*args, **kwargs):
        raise AssertionError("cached prepare_data unexpectedly used the network")

    _disable_dataset_network(monkeypatch, dataset_id, fail_if_network_is_used)
    cached_result = runner.invoke(
        main,
        [
            "data",
            "prepare",
            dataset_id,
            "--repo-root",
            str(tmp_path),
            "--option",
            f"data_root={data_root}",
            "--problem",
            prepare_name,
        ],
    )

    assert cached_result.exit_code == 0, cached_result.output


def test_miplib_prepare_max_size_uses_real_manifest_and_downloads_small_files(
    tmp_path: Path,
):
    data_root = tmp_path / "problem_classes"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "data",
            "prepare",
            "miplib",
            "--repo-root",
            str(tmp_path),
            "--option",
            f"data_root={data_root}",
            "--option",
            "max_size_mb=0.001",
        ],
    )

    assert result.exit_code == 0, result.output
    downloaded = list((data_root / "miplib_data").glob("*.mps.gz"))
    assert downloaded
    assert all(path.stat().st_size <= 1000 for path in downloaded)
    assert (data_root / "miplib_data" / "markshare_4_0.mps.gz").exists()


@pytest.mark.parametrize(
    ("dataset_id", "prepare_name", "problem_name"),
    [
        ("cblib", "nb", "nb"),
        ("dc_opf", "case5", "case5"),
        ("libsvm_qp", "heart", "svm_dual_heart"),
        ("miplib", "markshare_4_0", "markshare_4_0"),
        ("mittelmann", "qap15", "qap15"),
        ("mittelmann_sdp", "trto3", "trto3"),
        ("mpc_qpbenchmark", "LIPMWALK0", "LIPMWALK0"),
        ("qplib", "8790", "8790"),
        ("tsplib_sdp", "burma14", "burma14"),
    ],
)
def test_external_dataset_downloaded_problem_loads_and_solves(
    tmp_path: Path,
    dataset_id: str,
    prepare_name: str,
    problem_name: str,
):
    data_root = tmp_path / "problem_classes"
    dataset = get_dataset(dataset_id)(repo_root=tmp_path, data_root=data_root)
    dataset.prepare_data([prepare_name])
    problem = dataset.load_problem(problem_name)

    solver_cls = get_solver("clarabel")
    assert solver_cls.is_available()
    assert problem.kind in solver_cls.supported_problem_kinds

    artifacts = tmp_path / "artifacts" / dataset_id
    artifacts.mkdir(parents=True)
    result = solver_cls({"verbose": False, "max_iter": 200}).solve(problem, artifacts)

    assert result.status in TERMINAL_SOLVE_STATUSES


def _disable_dataset_network(monkeypatch, dataset_id: str, replacement) -> None:
    if dataset_id == "dc_opf":
        from solver_benchmarks.datasets import dc_opf

        monkeypatch.setattr(dc_opf.urllib.request, "urlopen", replacement)
    elif dataset_id in {"miplib", "mittelmann"}:
        from solver_benchmarks.datasets import mps

        monkeypatch.setattr(mps.urllib.request, "urlopen", replacement)
    elif dataset_id == "cblib":
        from solver_benchmarks.datasets import cblib

        monkeypatch.setattr(cblib.urllib.request, "urlopen", replacement)
    elif dataset_id == "mpc_qpbenchmark":
        from solver_benchmarks.datasets import mpc_qpbenchmark

        monkeypatch.setattr(mpc_qpbenchmark.urllib.request, "urlopen", replacement)
    elif dataset_id == "libsvm_qp":
        from solver_benchmarks.datasets import libsvm_qp

        monkeypatch.setattr(libsvm_qp.urllib.request, "urlopen", replacement)
    elif dataset_id == "mittelmann_sdp":
        from solver_benchmarks.datasets import mittelmann_sdp

        monkeypatch.setattr(mittelmann_sdp.urllib.request, "urlopen", replacement)
    elif dataset_id == "qplib":
        from solver_benchmarks.datasets import qplib

        monkeypatch.setattr(qplib.urllib.request, "urlopen", replacement)
    elif dataset_id == "tsplib_sdp":
        from solver_benchmarks.datasets import tsplib_sdp

        monkeypatch.setattr(tsplib_sdp.urllib.request, "urlopen", replacement)
    else:
        raise AssertionError(f"Unhandled dataset {dataset_id!r}")
