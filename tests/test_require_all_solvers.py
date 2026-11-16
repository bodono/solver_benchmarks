"""BENCH_REQUIRE_ALL_SOLVERS turns a skip for a missing open-source solver
into a failure, suite-wide (the hook in conftest.py)."""

from __future__ import annotations

import sys

import conftest
import pytest
from conftest import missing_solver_in_skip, require_all_solvers

ALL_OPEN_SOURCE = frozenset(conftest.OPEN_SOURCE_SOLVER_PACKAGES)


@pytest.mark.parametrize("value, expected", [
    (None, False), ("", False), ("0", False), ("false", False), ("False", False),
    ("NO", False), ("off", False), (" Off ", False),
    ("1", True), ("true", True), ("yes", True), ("on", True), ("anything", True),
])
def test_require_all_solvers_reads_the_variable(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("BENCH_REQUIRE_ALL_SOLVERS", raising=False)
    else:
        monkeypatch.setenv("BENCH_REQUIRE_ALL_SOLVERS", value)
    assert require_all_solvers() is expected


@pytest.mark.parametrize("reason, solver", [
    ("could not import 'ecos': No module named 'ecos'", "ecos"),
    ("could not import 'cvxopt.solvers': No module named 'cvxopt'", "cvxopt"),
    ("could not import 'ortools.pdlp.solvers_pb2': No module named 'ortools'", "pdlp"),
    ("could not import 'sdpap': No module named 'sdpap'", "sdpa"),
    ("could not import 'proxsuite': No module named 'proxsuite'", "proxqp"),
    ("could not import 'highspy': No module named 'highspy'", "highs"),
    ("scs not installed", "scs"),
    ("proxqp not installed", "proxqp"),
    ("clarabel is not installed", "clarabel"),
    ("ortools is not installed", "pdlp"),
    ("piqp is not installed (or does not import)", "piqp"),
    ("this OSQP build never stopped at 'solved inaccurate' here", "osqp"),
])
def test_a_skip_for_a_missing_open_source_solver_names_it(reason, solver):
    assert missing_solver_in_skip(reason, ALL_OPEN_SOURCE) == solver


@pytest.mark.parametrize("reason", [
    # commercial solvers
    "could not import 'mosek': No module named 'mosek'",
    "gurobi not installed",
    "cuopt is not installed (or does not import)",
    # not a solver
    "could not import 'h5py': No module named 'h5py'",
    "could not import 'no_such_module': No module named 'no_such_module'",
    # data, platform and solver-build skips
    "bundled SDPLIB archive is not present",
    "MPC benchmark data not downloaded",
    "POSIX-only: relies on process groups",
    "this ProxSuite build solves afiro (optimal)",
    "this SCS build has OpenMP threads",
    "qdldl is installed",
    "no solver used by the maintained configs is installed",
])
def test_other_skips_stay_skips(reason):
    assert missing_solver_in_skip(reason, ALL_OPEN_SOURCE) is None


def test_a_solver_the_all_extra_does_not_install_here_stays_a_skip():
    without_proxqp = ALL_OPEN_SOURCE - {"proxqp"}
    assert missing_solver_in_skip("proxqp not installed", without_proxqp) is None
    assert missing_solver_in_skip("could not import 'proxsuite': x", without_proxqp) is None


_THROWAWAY_TESTS = """
import pytest

def test_needs_ecos():
    pytest.importorskip("ecos")

def test_needs_mosek():
    pytest.importorskip("mosek")

def test_needs_data():
    pytest.skip("bundled SDPLIB archive is not present")
"""


@pytest.mark.parametrize("value", ["1", "0"])
def test_the_hook_fails_a_skip_for_a_missing_solver(pytester, monkeypatch, value):
    # ecos is installed here; make it fail to import as a missing package does.
    monkeypatch.setitem(sys.modules, "ecos", None)
    monkeypatch.setitem(sys.modules, "mosek", None)
    monkeypatch.setenv("BENCH_REQUIRE_ALL_SOLVERS", value)
    assert "ecos" in conftest.solvers_installed_by_all_extra()
    pytester.makepyfile(test_throwaway=_THROWAWAY_TESTS)

    result = pytester.runpytest("-rfs", plugins=[conftest])

    if value == "1":
        result.assert_outcomes(failed=1, skipped=2)
        result.stdout.fnmatch_lines([
            "ecos is required: BENCH_REQUIRE_ALL_SOLVERS is set*could not import 'ecos'*",
            "FAILED test_throwaway.py::test_needs_ecos*",
        ])
    else:
        result.assert_outcomes(skipped=3)


def _module_level_importorskips(tree) -> list[int]:
    """Lines of importorskip calls that run when the module is imported
    (outside any function body)."""
    import ast

    lines = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "importorskip":
                lines.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return lines


def test_no_test_module_importorskips_at_import_time():
    # The hook sees the skips of running tests, not those raised while a
    # module is collected, so a module-level importorskip would stay a skip
    # with BENCH_REQUIRE_ALL_SOLVERS on.
    import ast

    assert _module_level_importorskips(ast.parse("import pytest\nscs = pytest.importorskip('scs')\n")) == [2]
    assert _module_level_importorskips(ast.parse("def test():\n    pytest.importorskip('scs')\n")) == []
    found = {
        path.name: lines
        for path in sorted((conftest.REPO_ROOT / "tests").glob("test_*.py"))
        if (lines := _module_level_importorskips(ast.parse(path.read_text())))
    }
    assert found == {}
