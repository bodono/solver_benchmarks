"""Expected failures surface as one-line Click errors, not tracebacks.

``bench run`` and ``bench data prepare`` turn the exceptions the harness
raises for a described, expected failure into ``Error: ...``. These tests
hold ``bench env run`` (an environment whose install or run failed) and
``bench list problems --prepare`` (a failed download) to the same rule, the
latter with the same message as ``bench data prepare``.
"""

from __future__ import annotations

import urllib.error

import pytest
from click.testing import CliRunner

from solver_benchmarks import cli
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.datasets.synthetic import SyntheticQPDataset


def test_env_run_reports_matrix_failures_as_a_click_error(tmp_path, monkeypatch):
    config_path = tmp_path / "envs.yaml"
    config_path.write_text("environments: []\n")
    monkeypatch.setattr(cli, "load_environment_run_config", lambda path: object())

    def failing_matrix(config, **kwargs):
        raise RuntimeError("environment matrix had failures: osqp_1_0 (install)")

    monkeypatch.setattr(cli, "run_environment_matrix", failing_matrix)

    result = CliRunner().invoke(cli.main, ["env", "run", str(config_path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error: environment matrix had failures: osqp_1_0 (install)" in result.output


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("download of example.org/data.tar failed: HTTP 503"),
        # urllib's HTTPError and URLError are OSErrors.
        urllib.error.URLError("download of example.org/data.tar failed: HTTP 503"),
    ],
    ids=["runtime-error", "url-error"],
)
def test_list_problems_prepare_reports_prepare_failures_as_a_click_error(monkeypatch, error):
    class FailingPrepare(SyntheticQPDataset):
        dataset_id = "failing_prepare"

        def prepare_data(self, *args, **kwargs):
            raise error

    monkeypatch.setitem(dataset_registry.DATASETS, "failing_prepare", FailingPrepare)

    result = CliRunner().invoke(cli.main, ["list", "problems", "failing_prepare", "--prepare"])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error: Preparing failing_prepare failed:" in result.output
    assert "download of example.org/data.tar failed: HTTP 503" in result.output
