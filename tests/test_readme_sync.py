"""Keep the README's reference sections in step with the code.

The README is the user documentation for the CLI, the config fields and the
solver adapters. These checks compare its tables with what the package
registers, so a new command, option, config field or solver cannot ship
undocumented.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from solver_benchmarks.solvers import list_solvers

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()


def _section(heading: str) -> str:
    start = README.index(f"\n{heading}\n")
    level = heading.split(" ", 1)[0]
    match = re.compile(rf"\n{re.escape(level)} ").search(README, start + len(heading) + 2)
    return README[start:match.start() if match else len(README)]


@pytest.mark.parametrize("solver_name", list_solvers())
def test_every_registered_solver_has_a_supported_solvers_row(solver_name):
    rows = re.findall(r"^\| [^|]+ \| `([a-z0-9_]+)` \|", _section("## Supported Solvers"), flags=re.MULTILINE)
    assert solver_name in rows


def test_list_solvers_sample_output_names_every_solver():
    installation = _section("## Installation")
    sample = installation[installation.index("```text"):]
    sample = sample[:sample.index("```", 3)]
    listed = [line.split()[0] for line in sample.splitlines()[1:] if line.strip()]
    assert listed == list(list_solvers())  # the order `bench list solvers` prints


def _cli_commands():
    import click

    from solver_benchmarks.cli import main

    def walk(group, prefix):
        for name, command in sorted(group.commands.items()):
            path = f"{prefix} {name}".strip()
            if isinstance(command, click.Group):
                yield from walk(command, path)
            else:
                yield path, command

    return list(walk(main, ""))


def _cli_row(path: str) -> str | None:
    for line in _section("## CLI Reference").splitlines():
        if re.match(rf"^\| `bench {re.escape(path)}[ `]", line):
            return line
    return None


@pytest.mark.parametrize("path, command", _cli_commands(), ids=lambda value: value if isinstance(value, str) else "")
def test_every_cli_command_and_option_is_in_the_cli_reference(path, command):
    import click

    row = _cli_row(path)
    assert row is not None, f"README CLI Reference has no row for `bench {path}`"
    for param in command.params:
        if isinstance(param, click.Option) and not param.hidden:
            for flag in [*param.opts, *param.secondary_opts]:
                assert flag in row, f"`bench {path}` row does not mention {flag}"


def test_every_run_config_field_is_in_the_field_table():
    import dataclasses

    from solver_benchmarks.core.config import RunConfig

    table = _section("## Configuration Files")
    for field in dataclasses.fields(RunConfig):
        if field.name in {"datasets", "solvers"}:
            continue  # documented as run.dataset / run.datasets and solvers[].*
        assert f"| `run.{field.name}` |" in table, f"run.{field.name} is missing from the config field table"
