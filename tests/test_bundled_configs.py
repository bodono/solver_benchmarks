"""The configs under ``configs/`` run with the solvers the package installs.

A config either runs with the released solvers from the package extras on
every platform those extras support, or it starts with an ``# EXPERIMENTAL:``
header saying what it needs (an unreleased solver build, an extra Python
package). Every config the README tells users to run must be of the first
kind. The check runs each distinct solver variant of the unmarked configs on
a one-variable LP in-process, which is what a config's settings first meet in
a real run: an unknown keyword, an out-of-range value or a linear-solver
backend whose module is not installed raises there.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from solver_benchmarks.core.config import load_run_config
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.solvers import get_solver

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
EXPERIMENTAL_HEADER = "# EXPERIMENTAL:"

# Linear-solver backends every platform's install of the extras provides.
# QTQP's other backends need packages the qtqp extra does not install (qdldl,
# scikit-sparse, ...) or exist on one platform only (pardiso on x86-64
# Linux/Windows through py-mkl-pardiso, accelerate on macOS arm64 through
# macldlt); SCS ships mkl only in its x86-64 Linux/Windows wheels and
# accelerate only in its macOS wheels.
PORTABLE_BACKENDS = {
    "qtqp": {None, "auto", "scipy"},
    "scs": {None, "auto", "qdldl", "cpu_indirect"},
}


def _config_paths() -> list[Path]:
    return sorted(
        path for path in CONFIGS.rglob("*")
        if path.suffix in {".yaml", ".yml", ".json"} and "runs" not in path.relative_to(CONFIGS).parts
    )


def _is_experimental(path: Path) -> bool:
    return path.read_text().lstrip().startswith(EXPERIMENTAL_HEADER)


def _maintained_configs() -> list[Path]:
    return [path for path in _config_paths() if not _is_experimental(path)]


def _readme_configs() -> list[str]:
    """Configs the README presents as runnable: the rows of its bundled-config
    table, and any other mention of a config that exists (other mentions such
    as ``configs/my_run.yaml`` are placeholders in prose)."""
    text = (REPO / "README.md").read_text()
    table = set(re.findall(r"^\| `configs/([^`]+)` \|", text, flags=re.MULTILINE))
    mentioned = set(re.findall(r"configs/([\w./-]+\.(?:ya?ml|json))", text))
    return sorted(table | {name for name in mentioned if (CONFIGS / name).exists()})


def _variants(paths: list[Path]) -> dict[tuple[str, str], tuple[dict, list[str]]]:
    variants: dict[tuple[str, str], tuple[dict, list[str]]] = {}
    for path in paths:
        for solver in load_run_config(path).solvers:
            key = (solver.solver, json.dumps(solver.settings, sort_keys=True, default=str))
            variants.setdefault(key, (dict(solver.settings), []))[1].append(path.name)
    return variants


@pytest.mark.parametrize("name", _readme_configs())
def test_readme_configs_exist_and_are_maintained(name):
    path = CONFIGS / name
    assert path.exists(), f"README names configs/{name}, which does not exist"
    assert not _is_experimental(path), f"README presents configs/{name} as runnable, but it is experimental"
    load_run_config(path)


def test_experimental_configs_say_what_they_need():
    for path in _config_paths():
        if _is_experimental(path):
            header = path.read_text().lstrip().splitlines()[0]
            assert len(header) > len(EXPERIMENTAL_HEADER) + 10, f"{path.name}: say what it needs"


def test_maintained_configs_use_backends_every_platform_installs():
    for (solver_name, _), (settings, files) in _variants(_maintained_configs()).items():
        allowed = PORTABLE_BACKENDS.get(solver_name)
        if allowed is None:
            continue
        backend = settings.get("linear_solver")
        backend = backend.lower() if isinstance(backend, str) else backend
        assert backend in allowed, (
            f"{sorted(set(files))}: {solver_name} linear_solver {backend!r} is not installed by the "
            f"extras on every platform; use one of {sorted(map(str, allowed))} or mark the config "
            f"{EXPERIMENTAL_HEADER!r}"
        )


def test_maintained_configs_settings_are_accepted_by_the_installed_solvers(tmp_path):
    problem = get_dataset("synthetic_qp")(repo_root=REPO).load_problem("one_variable_lp")
    checked = 0
    for (solver_name, _), (settings, files) in _variants(_maintained_configs()).items():
        solver_cls = get_solver(solver_name)
        if not solver_cls.is_available():
            continue
        settings = {**settings, "verbose": False}
        for cap in ("max_iters", "max_iter"):
            if cap in settings:
                settings[cap] = min(int(settings[cap]), 2000)
        artifacts = tmp_path / f"v{checked}"
        artifacts.mkdir()
        try:
            solver_cls(settings).solve(problem, artifacts)
        except Exception as exc:
            pytest.fail(
                f"{sorted(set(files))}: {solver_name} rejects {settings}: {type(exc).__name__}: {exc}. "
                f"Fix the settings, or mark the config {EXPERIMENTAL_HEADER!r} with what it needs."
            )
        checked += 1
    if not checked:
        pytest.skip("no solver used by the maintained configs is installed")
