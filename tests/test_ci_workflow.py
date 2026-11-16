"""Keep .github/workflows/tests.yml in step with how the suite is organised.

Tests marked ``slow`` take minutes (most read every member of the SDPLIB
archive) and run in one job only, the coverage job, so the four matrix jobs
stay well inside the per-test timeout and the slow tests still run, under
coverage, on every pull request. Under coverage tracing they take two to
three times longer, so each carries its own per-test timeout above the
300 s default. ``tools/`` is linted with the rest of the code.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = yaml.safe_load((REPO / ".github" / "workflows" / "tests.yml").read_text())
DEFAULT_TIMEOUT = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]["timeout"]


def _runs(job: str) -> list[str]:
    return [step["run"] for step in WORKFLOW["jobs"][job]["steps"] if "run" in step]


def _pytest_marker_expressions(job: str) -> list[str]:
    commands = [run for run in _runs(job) if "pytest" in run]
    assert commands, f"{job} runs no pytest"
    return [match for run in commands for match in re.findall(r'-m "([^"]+)"', run)]


def test_matrix_jobs_skip_network_and_slow_tests():
    assert _pytest_marker_expressions("tests") == ["not network and not slow"]


def test_coverage_job_still_runs_the_slow_tests():
    assert _pytest_marker_expressions("coverage") == ["not network"]


def test_jobs_with_every_extra_require_every_solver():
    for job in ("tests", "coverage"):
        env = WORKFLOW["jobs"][job].get("env") or {}
        assert env.get("BENCH_REQUIRE_ALL_SOLVERS") == "1", job


def test_slow_tests_carry_a_longer_timeout():
    for path in sorted((REPO / "tests").glob("test_*.py")):
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            if line.strip() != "@pytest.mark.slow":
                continue
            start = i
            while start > 0 and lines[start - 1].lstrip().startswith("@"):
                start -= 1
            end = i
            while lines[end + 1].lstrip().startswith("@"):
                end += 1
            block = "\n".join(lines[start:end + 1])
            match = re.search(r"@pytest\.mark\.timeout\((\d+)\)", block)
            assert match and int(match.group(1)) > DEFAULT_TIMEOUT, f"{path.name}:{i + 1}"


def test_lint_covers_tools():
    ruff = [run for run in _runs("lint") if "ruff check" in run]
    assert ruff and all("tools" in run.split() for run in ruff)


def _extra(name: str) -> list:
    from packaging.requirements import Requirement

    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    return [Requirement(text) for text in extras[name]]


def test_pdlp_extras_exclude_only_the_highspy_releases_that_clash_with_ortools():
    # highspy 1.15.x ships a separate libhighs whose symbols clash with
    # OR-Tools 9.15 in one process; 1.14 and 1.16 do not.
    from packaging.version import Version

    for extra in ("pdlp", "all"):
        (highspy,) = [req for req in _extra(extra) if req.name == "highspy"]
        allowed = {v: highspy.specifier.contains(Version(v)) for v in ("1.14.0", "1.15.0", "1.15.1", "1.16.0")}
        assert allowed == {"1.14.0": True, "1.15.0": False, "1.15.1": False, "1.16.0": True}, extra


def test_proxqp_and_sdpa_install_on_python_314():
    # Both publish cp314 wheels (proxsuite >= 0.8.0 none for Windows,
    # sdpa-python >= 0.2.3).
    for extra, package in (("proxqp", "proxsuite"), ("sdpa", "sdpa-python"), ("all", "proxsuite"),
                           ("all", "sdpa-python")):
        for python in ("3.11", "3.14"):
            env = {"python_version": python, "sys_platform": "linux"}
            selected = [req for req in _extra(extra) if req.name == package
                        and (req.marker is None or req.marker.evaluate(env))]
            assert len(selected) == 1, (extra, package, python)


def test_all_extra_repeats_the_solver_extras_constraints():
    all_extra = {str(req) for req in _extra("all")}
    for extra in ("scs", "proxqp", "sdpa", "pdlp", "piqp", "cpo3", "qtqp"):
        missing = {str(req) for req in _extra(extra)} - all_extra
        assert not missing, (extra, missing)


def test_type_check_runs_on_the_oldest_and_newest_supported_python():
    # numpy and scipy drop old Pythons, so the stubs mypy sees depend on the
    # Python it runs on: check both ends of the supported range.
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    tested = WORKFLOW["jobs"]["tests"]["strategy"]["matrix"]["python-version"]
    type_check = WORKFLOW["jobs"]["type-check"]
    cells = type_check["strategy"]["matrix"]["include"]
    versions = [cell["python-version"] for cell in cells]
    assert versions == [pyproject["tool"]["mypy"]["python_version"], max(tested, key=lambda v: tuple(map(int, v.split("."))))]
    # Branch protection may require the check by its original name.
    assert type_check["name"] == "${{ matrix.check-name }}"
    assert [cell["check-name"] for cell in cells] == ["Type check", f"Type check - Python {versions[1]}"]
    setup = [step for step in type_check["steps"] if step.get("uses", "").startswith("actions/setup-python")]
    assert setup[0]["with"]["python-version"] == "${{ matrix.python-version }}"


def test_only_pull_request_runs_cancel_each_other():
    # A push to master must not cancel the weekly scheduled run (which alone
    # runs the dataset-download smoke), nor the other way round.
    concurrency = WORKFLOW["concurrency"]
    assert "github.event_name" in concurrency["group"]
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_the_open_source_solver_list_covers_every_registered_solver():
    # BENCH_REQUIRE_ALL_SOLVERS fails a skip only for the solvers conftest.py
    # lists; a new open-source adapter must be added there.
    from conftest import COMMERCIAL_SOLVERS, OPEN_SOURCE_SOLVER_PACKAGES

    from solver_benchmarks.solvers import list_solvers

    assert COMMERCIAL_SOLVERS <= set(list_solvers())
    assert set(OPEN_SOURCE_SOLVER_PACKAGES) == set(list_solvers()) - COMMERCIAL_SOLVERS
    all_extra = {req.name for req in _extra("all")}
    assert {package for package, _ in OPEN_SOURCE_SOLVER_PACKAGES.values()} <= all_extra
    assert not {"gurobipy", "Mosek", "cplex"} & all_extra


def _ci_pythons(job: str) -> list[str]:
    setup = [step for step in WORKFLOW["jobs"][job]["steps"]
             if step.get("uses", "").startswith("actions/setup-python")]
    version = setup[0]["with"]["python-version"]
    if version == "${{ matrix.python-version }}":
        return WORKFLOW["jobs"][job]["strategy"]["matrix"]["python-version"]
    return [version]


def test_every_open_source_solver_is_required_in_the_jobs_that_require_them():
    # The jobs install the `all` extra on Linux: every open-source solver is
    # installed there on every tested Python, so each one's skips fail.
    from conftest import OPEN_SOURCE_SOLVER_PACKAGES, solvers_installed_by_all_extra
    from packaging.markers import default_environment

    for job in ("tests", "coverage"):
        assert WORKFLOW["jobs"][job]["runs-on"] == "ubuntu-latest"
        assert any('".[test,all]"' in run for run in _runs(job)), job
        for python in _ci_pythons(job):
            env = {**default_environment(), "python_version": python,
                   "python_full_version": f"{python}.0", "sys_platform": "linux",
                   "platform_system": "Linux", "os_name": "posix"}
            assert solvers_installed_by_all_extra(tuple(env.items())) == set(OPEN_SOURCE_SOLVER_PACKAGES), (job, python)


def test_a_solver_the_all_extra_leaves_out_is_not_required():
    # proxsuite has no Windows wheel on Python 3.14, so the extra skips it there.
    from conftest import OPEN_SOURCE_SOLVER_PACKAGES, solvers_installed_by_all_extra
    from packaging.markers import default_environment

    env = {**default_environment(), "python_version": "3.14", "python_full_version": "3.14.0",
           "sys_platform": "win32", "platform_system": "Windows", "os_name": "nt"}
    assert solvers_installed_by_all_extra(tuple(env.items())) == set(OPEN_SOURCE_SOLVER_PACKAGES) - {"proxqp"}



def test_coverage_job_measures_the_package_and_the_readers():
    (command,) = [run for run in _runs("coverage") if "python -m pytest" in run]
    assert {"--cov=solver_benchmarks", "--cov=problem_classes"} <= set(command.split())
    gate = re.search(r"--cov-fail-under=(\d+)", command)
    # Both packages measure about 93%; keep the gate a few points under that.
    assert gate and int(gate.group(1)) >= 90, command
