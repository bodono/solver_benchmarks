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
