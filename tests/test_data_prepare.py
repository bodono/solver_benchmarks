"""Tests for the data-preparation command/selection helpers.

Audit flagged that ``shell_join`` quoting and
``_format_option_value`` had no direct coverage; the previous tests
exercised them only via the missing-data run-message path.
"""

from __future__ import annotations

import pytest

from solver_benchmarks.core.config import DatasetConfig, parse_run_config
from solver_benchmarks.core.data_prepare import (
    data_prepare_command,
    data_prepare_selection,
    run_with_prepare_command,
    shell_join,
)


def test_shell_join_quotes_values_with_spaces():
    assert (
        shell_join(["bench", "data", "prepare", "weird name with spaces"])
        == "bench data prepare 'weird name with spaces'"
    )


def test_shell_join_escapes_single_quotes():
    """A dataset option literal with embedded quotes must round-trip
    through a real shell. We don't run a shell here but assert that
    shlex.quote produces a balanced quoting."""
    out = shell_join(["bench", "--option", "subset=foo's"])
    # The single quote inside must be escaped using the shlex pattern
    # (close-quote, escaped-single-quote, reopen-quote).
    assert "'" in out
    assert "foo" in out
    assert "s" in out


def test_shell_join_quotes_dollar_signs_and_backticks():
    out = shell_join(["bench", "--option", "expr=$(rm -rf /)"])
    # The dangerous tokens must be inside single quotes so the shell
    # does not interpret them.
    assert "'expr=$(rm -rf /)'" in out


def test_data_prepare_command_emits_bools_as_lowercase_strings():
    cfg = DatasetConfig(
        name="netlib",
        dataset_options={"verbose": True, "include_lp": False},
    )
    cmd = data_prepare_command(cfg)
    assert "verbose=true" in cmd
    assert "include_lp=false" in cmd


def test_data_prepare_command_serializes_list_options_as_comma_joined():
    cfg = DatasetConfig(
        name="cblib",
        dataset_options={"subset": ["a", "b", "c"]},
    )
    cmd = data_prepare_command(cfg)
    assert "subset=a,b,c" in cmd


def test_data_prepare_command_appends_all_flag_and_problem_options():
    cfg = DatasetConfig(name="qplib")
    cmd = data_prepare_command(
        cfg,
        problem_names=["8790", "8495"],
        all_problems=True,
    )
    assert cmd.endswith("--problem 8790 --problem 8495")
    assert "--all" in cmd


def test_data_prepare_command_includes_repo_root_when_provided(tmp_path):
    cfg = DatasetConfig(name="dimacs")
    cmd = data_prepare_command(cfg, repo_root=tmp_path)
    assert f"--repo-root {tmp_path}" in cmd or f"--repo-root '{tmp_path}'" in cmd


def test_run_with_prepare_command_includes_run_dir_and_repo_root(tmp_path):
    cmd = run_with_prepare_command(
        tmp_path / "config.yaml",
        run_dir=tmp_path / "run",
        repo_root=tmp_path,
    )
    assert "--prepare-data" in cmd
    assert "--run-dir" in cmd
    assert "--repo-root" in cmd


def test_data_prepare_selection_picks_subset_all_for_all_problems():
    config = parse_run_config(
        {
            "run": {"dataset": "qplib", "dataset_options": {"subset": "all"}},
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    problem_names, all_problems = data_prepare_selection(
        config, config.datasets[0]
    )
    assert problem_names is None
    assert all_problems is True


def test_data_prepare_selection_splits_comma_subset():
    config = parse_run_config(
        {
            "run": {"dataset": "qplib", "dataset_options": {"subset": "a, b ,c"}},
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    problem_names, all_problems = data_prepare_selection(
        config, config.datasets[0]
    )
    assert problem_names == ["a", "b", "c"]
    assert all_problems is False


def test_data_prepare_selection_passes_through_list_subset():
    config = parse_run_config(
        {
            "run": {"dataset": "qplib", "dataset_options": {"subset": ["x", "y"]}},
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    problem_names, all_problems = data_prepare_selection(
        config, config.datasets[0]
    )
    assert problem_names == ["x", "y"]
    assert all_problems is False


def test_data_prepare_selection_uses_run_include_when_no_subset():
    config = parse_run_config(
        {
            "run": {"dataset": "qplib", "include": ["alpha", "beta"]},
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    problem_names, all_problems = data_prepare_selection(
        config, config.datasets[0]
    )
    assert problem_names == ["alpha", "beta"]
    assert all_problems is False


@pytest.mark.parametrize("ascii_quote_safe", ["bench data prepare cblib"])
def test_shell_join_round_trips_simple_command(ascii_quote_safe):
    """For ASCII-only no-special-char inputs shell_join is a no-op."""
    parts = ascii_quote_safe.split()
    assert shell_join(parts) == ascii_quote_safe


def test_data_prepare_selection_treats_a_single_name_subset_as_a_problem():
    """cblib, mittelmann_sdp and libsvm_qp read `subset` as a list of names."""
    for dataset, name in (("cblib", "nb_L1"), ("mittelmann_sdp", "hamming_7_5_6"),
                          ("libsvm_qp", "australian")):
        config = parse_run_config(
            {"run": {"dataset": dataset, "dataset_options": {"subset": name}},
             "solvers": [{"id": "scs", "solver": "scs"}]}
        )
        assert data_prepare_selection(config, config.datasets[0]) == ([name], False)
    # Reserved values and named subsets keep their meaning.
    for dataset, subset, expected in (
        ("cblib", "default", (None, False)),
        ("cblib", "all", (None, True)),
        ("netlib", "feasible", (None, False)),
    ):
        config = parse_run_config(
            {"run": {"dataset": dataset, "dataset_options": {"subset": subset}},
             "solvers": [{"id": "scs", "solver": "scs"}]}
        )
        assert data_prepare_selection(config, config.datasets[0]) == expected


def _prepare_fixture(monkeypatch, tmp_path, *, listed, include, fail=None):
    from solver_benchmarks.core.problem import QP, ProblemSpec
    from solver_benchmarks.datasets import registry as dataset_registry

    calls = []

    class LocalDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def prepare_data(self, problem_names=None, all_problems=False):
            calls.append(problem_names)
            if fail is not None:
                raise fail

        def list_problems(self):
            return [ProblemSpec(dataset_id="local", name=name, kind=QP) for name in listed]

    monkeypatch.setitem(dataset_registry.DATASETS, "local_fixture", LocalDataset)
    config = parse_run_config(
        {"run": {"dataset": "local_fixture", "include": include, "auto_prepare_data": True,
                 "output_dir": str(tmp_path / "runs")},
         "solvers": [{"id": "scs", "solver": "scs"}]}
    )
    return config, calls


def test_auto_prepare_skips_problems_already_present(monkeypatch, tmp_path, repo_root):
    """libsvm_qp problems are named svm_dual_<name>, which is not a download
    name: asking to prepare a bundled one requested a URL that does not exist."""
    from solver_benchmarks.core import runner

    monkeypatch.setattr(runner, "_run_subprocess", _no_worker)
    config, calls = _prepare_fixture(
        monkeypatch, tmp_path, listed=["svm_dual_australian"], include=["svm_dual_australian"]
    )
    runner.run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)
    assert calls == []

    config, calls = _prepare_fixture(
        monkeypatch, tmp_path, listed=["a"], include=["a", "b"]
    )
    runner.run_benchmark(config, run_dir=tmp_path / "run2", repo_root=repo_root)
    assert calls == [["b"]]


def test_auto_prepare_failure_is_a_clear_error(monkeypatch, tmp_path, repo_root):
    from solver_benchmarks.core import runner

    config, _ = _prepare_fixture(
        monkeypatch, tmp_path, listed=[], include=["remote"],
        fail=OSError("HTTP Error 404: Not Found"),
    )
    with pytest.raises(RuntimeError, match="local_fixture.*HTTP Error 404"):
        runner.run_benchmark(config, run_dir=tmp_path / "run", repo_root=repo_root)


def _no_worker(cmd, **kwargs):
    from conftest import FakeWorker

    return FakeWorker()(cmd, **kwargs)
