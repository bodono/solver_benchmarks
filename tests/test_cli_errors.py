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
@pytest.mark.parametrize(
    "command",
    [["list", "problems", "failing_prepare", "--prepare"], ["data", "prepare", "failing_prepare"]],
    ids=["list-problems", "data-prepare"],
)
def test_prepare_failures_are_click_errors(monkeypatch, error, command):
    class FailingPrepare(SyntheticQPDataset):
        dataset_id = "failing_prepare"

        def prepare_data(self, *args, **kwargs):
            raise error

    monkeypatch.setitem(dataset_registry.DATASETS, "failing_prepare", FailingPrepare)

    result = CliRunner().invoke(cli.main, command)

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


@pytest.fixture
def linked_repo_root(repo_root, tmp_path):
    """A repository root whose data folders hold links to the bundled files,
    so a dataset that writes beside its data (SDPLIB caches its archive
    index there) writes under tmp_path, not into the source tree."""
    root = tmp_path / "repo"
    for folder in (repo_root / "problem_classes").iterdir():
        if not folder.is_dir() or folder.name == "__pycache__":
            continue
        target = root / "problem_classes" / folder.name
        target.mkdir(parents=True)
        for entry in folder.iterdir():
            if entry.name != ".cache":
                (target / entry.name).symlink_to(entry)
    return root


def test_data_status_of_every_dataset_gives_each_only_the_options_it_takes(linked_repo_root):
    result = CliRunner().invoke(
        cli.main, ["data", "status", "--repo-root", str(linked_repo_root), "--option", "subset=all"]
    )
    assert result.exit_code == 0, result.output
    rows = {line.split("\t")[0]: line.split("\t") for line in result.output.splitlines()}
    assert set(rows) == set(dataset_registry.DATASETS) - {"miplib_lp_relaxation"} | {"miplib"}
    # SDPLIB reads subset=all: the feasible problems and the four infeasible ones.
    assert rows["sdplib"][2] == "92"
    # Maros-Meszaros has no subset; NETLIB rejects the value and keeps its default.
    assert rows["maros_meszaros"][2] == "138"
    assert rows["netlib"][2] == "93"


def test_data_status_of_one_dataset_rejects_an_option_it_does_not_take(repo_root):
    result = CliRunner().invoke(
        cli.main, ["data", "status", "maros_meszaros", "--repo-root", str(repo_root), "--option", "subset=x"]
    )
    assert result.exit_code == 1
    assert "Unknown dataset option for 'maros_meszaros': 'subset'" in result.output
    assert "Traceback" not in result.output


def test_data_status_of_every_dataset_skips_one_that_rejects_the_value(linked_repo_root):
    result = CliRunner().invoke(
        cli.main, ["data", "status", "--repo-root", str(linked_repo_root), "--option", "subset=feasible"]
    )
    assert result.exit_code == 0, result.output
    assert "note: qplib skipped: QPLIB problem names must be numeric IDs" in result.output
    rows = [line.split("\t")[0] for line in result.output.splitlines() if "\t" in line]
    assert "qplib" not in rows
    assert {"netlib", "sdplib", "tsplib_sdp", "maros_meszaros"} <= set(rows)


def test_data_status_of_every_dataset_rejects_an_option_no_dataset_reads(repo_root):
    result = CliRunner().invoke(
        cli.main, ["data", "status", "--repo-root", str(repo_root), "--option", "subst=all"]
    )
    assert result.exit_code == 1
    assert "No dataset reads the option 'subst' (did you mean 'subset'?)" in result.output


def test_a_key_error_importing_a_dataset_loader_is_not_an_unknown_dataset(
    monkeypatch, tmp_path
):
    """A bug in a loader module that raises KeyError at import is reported as
    that error, not as a misspelled dataset name."""
    (tmp_path / "broken_loader_module.py").write_text("{}['missing_setting']\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(
        dataset_registry.DATASETS._entries, "broken_loader", ("broken_loader_module", "Broken")
    )
    result = CliRunner().invoke(cli.main, ["list", "problems", "broken_loader"])
    assert "Unknown dataset" not in result.output
    assert isinstance(result.exception, KeyError)
    assert result.exception.args == ("missing_setting",)

    result = CliRunner().invoke(cli.main, ["list", "problems", "no_such_dataset"])
    assert result.exit_code != 0 and "Unknown dataset 'no_such_dataset'" in result.output

    # The other callers that test the name report the loader's error too.
    from solver_benchmarks.core.config import _signature_dataset_options
    from solver_benchmarks.core.data_prepare import _subset_lists_problems

    for call in (lambda: _signature_dataset_options("broken_loader", {}),
                 lambda: _subset_lists_problems("broken_loader")):
        with pytest.raises(KeyError) as excinfo:
            call()
        assert excinfo.value.args == ("missing_setting",)
    assert _subset_lists_problems("no_such_dataset") is False
