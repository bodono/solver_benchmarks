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


@pytest.mark.parametrize(
    "args",
    [
        ["list", "problems", "netlb"],
        ["data", "status", "netlb"],
        ["data", "prepare", "netlb"],
    ],
    ids=["list-problems", "data-status", "data-prepare"],
)
def test_a_mistyped_dataset_name_is_a_click_error_with_a_suggestion(args):
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Unknown dataset 'netlb'. Did you mean 'netlib'?" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["list", "problems", "synthetic_qp", "--option", "max_size_mb=abc"],
        ["data", "status", "synthetic_qp", "--option", "max_size_mb=abc"],
    ],
    ids=["list-problems", "data-status"],
)
def test_a_bad_dataset_option_value_is_a_click_error(args):
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error:" in result.output and "abc" in result.output


@pytest.mark.parametrize(
    "text, expected",
    [
        ("run: {dataset: synthetic_qp, output_dir: }\nsolvers: [{id: s, solver: scs}]\n", "output_dir"),
        (
            "run: {dataset: synthetic_qp}\n"
            "solvers: [{id: s, solver: scs, sweep: {eps_abs: [1, 2]}, id_template: '{id}_{0}'}]\n",
            "id_template",
        ),
        (
            "run: {dataset: synthetic_qp}\n"
            "solvers: [{id: s, solver: scs, sweep: {eps_abs: [1, 2]}, id_template: '{eps_abs.real.x}'}]\n",
            "id_template",
        ),
        (
            "run: {dataset: synthetic_qp}\n"
            "solvers: [{id: s, solver: scs, sweep: {eps_abs: [1, 2]}, id_template: '{eps_abs:zz}'}]\n",
            "id_template",
        ),
    ],
    ids=["null-output-dir", "positional-field", "attribute-field", "bad-format-spec"],
)
def test_run_config_mistakes_are_click_errors(tmp_path, text, expected):
    path = tmp_path / "cfg.yaml"
    path.write_text(text)
    result = CliRunner().invoke(cli.main, ["run", str(path), "--run-dir", str(tmp_path / "run")])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "Error:" in result.output and expected in result.output
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "run: {dataset: synthetic_qp}\n"
            "environments: [{id: osqp/1.0, solvers: [{id: a, solver: osqp}]}]\n",
            "osqp/1.0",
        ),
        (
            "run: {dataset: synthetic_qp}\n"
            "environments:\n"
            "  - {id: e, solvers: [{id: a, solver: osqp}]}\n"
            "  - {id: e, solvers: [{id: b, solver: scs}]}\n",
            "Duplicate environment id",
        ),
        (
            "run: {dataset: synthetic_qp}\n"
            "solvers: [{id: never_runs, solver: scs}]\n"
            "environments: [{id: e, solvers: [{id: a, solver: osqp}]}]\n",
            "environments[].solvers",
        ),
    ],
    ids=["slash-in-id", "duplicate-id", "top-level-solvers"],
)
def test_environment_config_mistakes_are_click_errors(tmp_path, text, expected):
    path = tmp_path / "envs.yaml"
    path.write_text(text)
    result = CliRunner().invoke(cli.main, ["env", "run", str(path), "--run-dir", str(tmp_path / "run")])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit), result.exception
    assert "Error:" in result.output and expected in result.output
    assert not (tmp_path / "run").exists()
