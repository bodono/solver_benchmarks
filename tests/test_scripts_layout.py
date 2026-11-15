"""``scripts/`` holds the maintained data-preparation wrappers and the CI
solver smoke script; one-off research scripts live in ``scripts/research/``.

These checks keep machine-specific paths out of the scripts and catch a
dataset constructed with a ``dataset_options=`` keyword: datasets take their
options as keyword arguments, so that call stores an option literally named
``dataset_options`` and silently ignores what it was meant to select.
"""

from __future__ import annotations

import re
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
PERSONAL_PATH = re.compile(r"""["'](/Users/|/home/)[^"']+["']""")


def test_maintained_scripts_are_prepare_wrappers_or_ci_tools():
    names = sorted(path.name for path in SCRIPTS.iterdir() if path.is_file())
    unexpected = [name for name in names if not name.startswith("prepare_") and name != "ci_solver_smoke.py"]
    assert not unexpected, f"move research scripts and patches to scripts/research/: {unexpected}"


def test_research_scripts_are_documented():
    readme = (SCRIPTS / "research" / "README.md").read_text()
    for path in sorted((SCRIPTS / "research").iterdir()):
        if path.name != "README.md":
            assert path.name in readme, f"scripts/research/README.md does not mention {path.name}"


def test_scripts_hard_code_no_personal_paths():
    for path in sorted(SCRIPTS.rglob("*.py")):
        match = PERSONAL_PATH.search(path.read_text())
        assert match is None, f"{path.relative_to(SCRIPTS)} hard-codes {match.group(0)}; take it as an argument"


def test_scripts_pass_dataset_options_as_keywords():
    for path in sorted(SCRIPTS.rglob("*.py")):
        text = path.read_text()
        assert "Dataset(dataset_options=" not in text, (
            f"{path.relative_to(SCRIPTS)}: pass dataset options as keywords, e.g. NetlibDataset(subset=...)"
        )
