from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from solver_benchmarks.cli import main
from solver_benchmarks.datasets import get_dataset


pytestmark = pytest.mark.network


@pytest.mark.parametrize(
    ("dataset_id", "problem_name", "expected_file"),
    [
        ("cblib", "nb", "cblib_data/nb.cbf.gz"),
        ("kennington", "ken-07", "kennington/ken-07.mps.gz"),
        ("mittelmann", "qap15", "mittelmann/qap15.mps"),
        ("mpc_qpbenchmark", "LIPMWALK0", "mpc_qpbenchmark_data/LIPMWALK0.npz"),
        ("qplib", "8790", "qplib_data/QPLIB_8790.qplib"),
    ],
)
def test_external_dataset_prepare_downloads_real_problem_and_uses_cache(
    monkeypatch,
    tmp_path: Path,
    dataset_id: str,
    problem_name: str,
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
            problem_name,
        ],
    )

    assert result.exit_code == 0, result.output

    downloaded = data_root / expected_file
    assert downloaded.exists()
    assert downloaded.stat().st_size > 0
    dataset = get_dataset(dataset_id)(repo_root=tmp_path, data_root=data_root)
    assert problem_name in {spec.name for spec in dataset.list_problems()}

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
            problem_name,
        ],
    )

    assert cached_result.exit_code == 0, cached_result.output


def _disable_dataset_network(monkeypatch, dataset_id: str, replacement) -> None:
    if dataset_id in {"kennington", "mittelmann"}:
        from solver_benchmarks.datasets import mps

        monkeypatch.setattr(mps.urllib.request, "urlopen", replacement)
    elif dataset_id == "cblib":
        from solver_benchmarks.datasets import cblib

        monkeypatch.setattr(cblib.urllib.request, "urlopen", replacement)
    elif dataset_id == "mpc_qpbenchmark":
        from solver_benchmarks.datasets import mpc_qpbenchmark

        monkeypatch.setattr(mpc_qpbenchmark.urllib.request, "urlopen", replacement)
    elif dataset_id == "qplib":
        from solver_benchmarks.datasets import qplib

        monkeypatch.setattr(qplib.urllib.request, "urlopen", replacement)
    else:
        raise AssertionError(f"Unhandled dataset {dataset_id!r}")
