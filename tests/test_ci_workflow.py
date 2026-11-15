"""Keep .github/workflows/tests.yml in step with how the suite is organised.

Tests marked ``slow`` take minutes (they read every member of the SDPLIB
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
