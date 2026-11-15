"""``scripts/`` holds the maintained data-preparation wrappers and the CI
solver smoke script; one-off research scripts live in ``scripts/research/``.

These checks keep machine-specific paths out of the scripts and catch a
dataset constructed with a ``dataset_options=`` keyword: datasets take their
options as keyword arguments, so that call stores an option literally named
``dataset_options`` and silently ignores what it was meant to select. Every
prepare wrapper takes ``--repo-root`` (default: the repository ``bench``
would use, not the working directory) and ``--data-root``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
PERSONAL_PATH = re.compile(r"""["'](/Users/|/home/)[^"']+["']""")


def test_maintained_scripts_are_prepare_wrappers_or_ci_tools():
    names = sorted(path.name for path in SCRIPTS.iterdir() if path.is_file() and not path.name.startswith("."))
    unexpected = [name for name in names if not name.startswith("prepare_") and name != "ci_solver_smoke.py"]
    assert not unexpected, f"move research scripts and patches to scripts/research/: {unexpected}"


def _undocumented_research_files(research: Path) -> list[str]:
    # Regular, non-hidden files only: `python -m compileall scripts` (a CI
    # lint step) creates scripts/research/__pycache__/, and editors and
    # Finder leave .DS_Store and similar files behind.
    readme = (research / "README.md").read_text()
    return [
        path.name
        for path in sorted(research.iterdir())
        if path.is_file() and not path.name.startswith(".") and path.name != "README.md" and path.name not in readme
    ]


def test_research_scripts_are_documented():
    undocumented = _undocumented_research_files(SCRIPTS / "research")
    assert not undocumented, f"scripts/research/README.md does not mention {undocumented}"


def test_research_listing_ignores_bytecode_caches_and_hidden_files(tmp_path):
    research = tmp_path / "research"
    research.mkdir()
    (research / "README.md").write_text("- `aa_tool.py`\n")
    (research / "aa_tool.py").write_text("")
    (research / "__pycache__").mkdir()
    (research / "__pycache__" / "aa_tool.cpython-312.pyc").write_bytes(b"")
    (research / ".DS_Store").write_bytes(b"")
    assert _undocumented_research_files(research) == []

    (research / "aa_new_tool.py").write_text("")
    assert _undocumented_research_files(research) == ["aa_new_tool.py"]


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


@pytest.mark.parametrize("script", sorted(path.name for path in SCRIPTS.glob("prepare_*.py")))
def test_prepare_wrappers_default_to_bench_s_repository_and_take_a_data_root(script, tmp_path, monkeypatch):
    # A wrapper run from another directory must not write a problem_classes/
    # there (which default_repo_root would then take for the repository).
    spec = importlib.util.spec_from_file_location(f"_scripts_{script[:-3]}", SCRIPTS / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    created = []

    class FakeDataset:
        def __init__(self, repo_root=None, **options):
            created.append((repo_root, options))
            self.data_dir = tmp_path / "data"

        def prepare_data(self, *args, **kwargs):
            pass

        def data_status(self):
            return SimpleNamespace(dataset="fake", problem_count=0, data_dir=self.data_dir)

    (dataset_name,) = [name for name in vars(module) if name.endswith("Dataset")]
    monkeypatch.setattr(module, dataset_name, FakeDataset)
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path / "elsewhere"

    monkeypatch.setattr(sys, "argv", [script, "--data-root", str(data_root)])
    assert module.main() == 0
    monkeypatch.setattr(sys, "argv", [script, "--repo-root", str(tmp_path / "repo")])
    assert module.main() == 0

    assert created[0] == (None, {"data_root": str(data_root)})
    assert created[1][0] == tmp_path / "repo"
