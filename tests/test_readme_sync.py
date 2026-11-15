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
