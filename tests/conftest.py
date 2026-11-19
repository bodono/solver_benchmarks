"""Shared pytest configuration for the solver_benchmarks test suite."""

from __future__ import annotations

import functools
import os
import re
import tomllib
from collections.abc import Collection
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

pytest_plugins = ["pytester"]

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


# --- BENCH_REQUIRE_ALL_SOLVERS ------------------------------------------------
#
# CI installs the `all` extra and sets BENCH_REQUIRE_ALL_SOLVERS=1: there an
# open-source solver that is not installed, or does not import, is a packaging
# or import regression, so a test that skips for that reason fails instead.

REQUIRE_ALL_SOLVERS_ENV = "BENCH_REQUIRE_ALL_SOLVERS"

# Solvers the `all` extra does not install (they need a license, or for cuOpt
# an NVIDIA GPU), so their tests always skip where they are missing.
COMMERCIAL_SOLVERS = frozenset({"cplex", "cuopt", "gurobi", "mosek"})

# Every registered open-source solver: the distribution the `all` extra installs
# for it, and the top-level modules its tests import.
OPEN_SOURCE_SOLVER_PACKAGES: dict[str, tuple[str, tuple[str, ...]]] = {
    "clarabel": ("clarabel", ("clarabel",)),
    "cpo3": ("cpo3", ("cpo3",)),
    "cvxopt": ("cvxopt", ("cvxopt",)),
    "ecos": ("ecos", ("ecos",)),
    "highs": ("highspy", ("highspy",)),
    "osqp": ("osqp", ("osqp",)),
    "pdlp": ("ortools", ("ortools",)),
    "piqp": ("piqp", ("piqp",)),
    "proxqp": ("proxsuite", ("proxsuite",)),
    "qtqp": ("qtqp", ("qtqp",)),
    "scs": ("scs", ("scs",)),
    "sdpa": ("sdpa-python", ("sdpap",)),
}


def require_all_solvers() -> bool:
    """Whether BENCH_REQUIRE_ALL_SOLVERS is on: anything but unset, empty,
    ``0``, ``false``, ``no`` or ``off`` (in any case)."""
    value = os.environ.get(REQUIRE_ALL_SOLVERS_ENV, "").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


@functools.cache
def solvers_installed_by_all_extra(environment: tuple[tuple[str, str], ...] | None = None) -> frozenset[str]:
    """The open-source solvers the `all` extra of pyproject.toml installs on
    this Python and platform (or in ``environment``, the items of a marker
    environment).

    Some requirements carry environment markers (proxsuite has no Windows
    wheel on Python 3.14), so a solver the extra leaves out here is not
    required here."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    installed = {
        canonicalize_name(requirement.name)
        for requirement in map(Requirement, pyproject["project"]["optional-dependencies"]["all"])
        if requirement.marker is None
        or requirement.marker.evaluate(None if environment is None else dict(environment))
    }
    return frozenset(
        solver
        for solver, (distribution, _modules) in OPEN_SOURCE_SOLVER_PACKAGES.items()
        if canonicalize_name(distribution) in installed
    )


def _solver_aliases() -> dict[str, str]:
    aliases = {}
    for solver, (distribution, modules) in OPEN_SOURCE_SOLVER_PACKAGES.items():
        for name in (solver, distribution, *modules):
            aliases[name.lower()] = solver
    return aliases


_SOLVER_ALIASES = _solver_aliases()
_MISSING_SOLVER_REASONS = (
    # pytest.importorskip("<module>[.submodule]")
    re.compile(r"^could not import '(?P<name>[\w-]+)[\w.]*'"),
    # pytest.skip("<solver> not installed"), "<package> is not installed (...)"
    re.compile(r"^(?P<name>[\w-]+) (?:is )?not installed\b"),
    # pytest.skip("this OSQP build never stopped at ...")
    re.compile(r"^this (?P<name>[\w-]+) build never stopped at\b"),
)


def missing_solver_in_skip(reason: str, required: Collection[str]) -> str | None:
    """The solver in ``required`` whose absence (or failure to import, or
    build that never reaches a tested status) ``reason`` reports, else None.

    Skips for commercial solvers, absent data, the platform and anything
    else return None."""
    for pattern in _MISSING_SOLVER_REASONS:
        match = pattern.match(reason.strip())
        if match is None:
            continue
        solver = _SOLVER_ALIASES.get(match.group("name").lower())
        if solver is not None and solver in required:
            return solver
    return None


def skip_or_fail_missing_solver(solver: str) -> None:
    """Skip the calling test because ``solver`` is not installed (or does not
    import); with BENCH_REQUIRE_ALL_SOLVERS on, fail instead where the `all`
    extra installs it."""
    message = f"{solver} is not installed (or does not import)"
    if require_all_solvers() and solver in solvers_installed_by_all_extra():
        pytest.fail(f"{message}, and {REQUIRE_ALL_SOLVERS_ENV} is set")
    pytest.skip(message)


# The old-style hook wrapper works with every pytest the `test` extra allows.
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (
        report.skipped
        and call.excinfo is not None
        and call.excinfo.errisinstance(pytest.skip.Exception)
        and require_all_solvers()
    ):
        reason = str(call.excinfo.value.msg)
        solver = missing_solver_in_skip(reason, solvers_installed_by_all_extra())
        if solver is not None:
            report.outcome = "failed"
            report.longrepr = (
                f"{solver} is required: {REQUIRE_ALL_SOLVERS_ENV} is set and the `all` extra "
                f"installs {solver} on this Python, but the test skipped: {reason}"
            )


# --- rows that pass the default KKT verification ------------------------------
#
# load_results verifies statuses by default (analysis/verify.py): a claimed
# optimum without residuals becomes optimal_inaccurate. Tests of analysis
# output that write hand-made rows give their claimed optima residuals that
# pass, and a requested tolerance, so the rows mean what they say.

PASSING_KKT = {"form": "qp", "primal_res_rel": 1.0e-10, "dual_res_rel": 1.0e-10, "duality_gap_rel": 1.0e-10}
PASSING_REQUESTED_TOLERANCE = {"optimality": 1.0e-6, "certificate": 1.0e-6}


def with_passing_kkt(row: dict) -> dict:
    """``row`` with residuals that pass verification, when it claims
    ``optimal`` and records no KKT residuals of its own (nested or as
    flattened ``kkt.*`` keys); any other row is returned as it is."""
    if row.get("status") != "optimal" or any(key == "kkt" or str(key).startswith("kkt.") for key in row):
        return row
    metadata = dict(row["metadata"]) if isinstance(row.get("metadata"), dict) else {}
    metadata.setdefault("requested_tolerance", dict(PASSING_REQUESTED_TOLERANCE))
    return {**row, "kkt": dict(PASSING_KKT), "metadata": metadata}


# --- a stand-in for the worker subprocess -------------------------------------


class FakeWorker:
    """A stand-in for ``runner._run_subprocess``: reads the worker payload,
    records it in ``calls`` and writes ``worker_result.json`` as a worker that
    solved the problem would.

    Every result takes the fields of ``outcome`` (status ``optimal`` by
    default), then those of ``outcomes[problem]``. Keyword arguments that
    ``_run_subprocess`` takes are accepted and ignored.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.outcome: dict = {"status": "optimal", "run_time_seconds": 0.01}
        self.outcomes: dict[str, dict] = {}

    @property
    def problems(self) -> list[str]:
        return [payload["problem"] for payload in self.calls]

    def __call__(self, cmd, *, stdout_path=None, stderr_path=None, **_ignored):
        import json
        from types import SimpleNamespace

        payload = json.loads(Path(cmd[-1]).read_text())
        self.calls.append(payload)
        if stdout_path is not None:
            stdout_path.write_text(f"attempt {len(self.calls)}\n")
        if stderr_path is not None:
            stderr_path.write_text("")
        self.write_result(payload)
        return SimpleNamespace(returncode=0, stderr="", timed_out=False)

    def write_result(self, payload: dict, **fields) -> None:
        """Write the worker result for ``payload`` (``fields`` override)."""
        import json

        from solver_benchmarks.core.result import ProblemResult

        values = {**self.outcome, **self.outcomes.get(payload["problem"], {}), **fields}
        values.setdefault("objective_value", 0.0 if values["status"] == "optimal" else None)
        values.setdefault("iterations", 1)
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            artifact_dir=payload["artifacts_dir"],
            **values,
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))


@pytest.fixture
def fake_worker(monkeypatch) -> FakeWorker:
    """Replace the worker subprocess with a :class:`FakeWorker`."""
    from solver_benchmarks.core import runner

    worker = FakeWorker()
    monkeypatch.setattr(runner, "_run_subprocess", worker)
    return worker


def sdplib_archive_dataset(repo_root: Path, tmp_path: Path, **options):
    """The SDPLIB dataset over the bundled archive, extracting into tmp_path,
    not the source tree (the archive is linked, not copied)."""
    from solver_benchmarks.datasets import get_dataset

    archive = repo_root / "problem_classes/sdplib_data/sdplib.tar"
    if not archive.exists():
        pytest.skip("bundled SDPLIB archive is not present")
    folder = tmp_path / "problem_classes" / "sdplib_data"
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "sdplib.tar").exists():
        (folder / "sdplib.tar").symlink_to(archive)
    return get_dataset("sdplib")(repo_root=repo_root, data_root=tmp_path / "problem_classes", **options)


@pytest.fixture(scope="session")
def sdplib_archive_members(tmp_path_factory) -> dict[str, dict]:
    """One pass over the bundled SDPLIB archive, shared by the slow tests
    that check every member.

    Per member (by name): ``blocks``, its block record; and, unless its
    original SDPLIB file is bundled beside the archive (the damaged members
    the loader does not read), ``shape`` (A's shape and the lengths of b and
    q) when it reads, or ``error`` when it does not. Each member is copied to
    a scratch file, read and removed, so the pass costs one decompression of
    the archive and never holds a member in memory (the largest is 395 MB).
    """
    import shutil
    import tarfile

    from solver_benchmarks.transforms.sdplib import read_sdplib_block_sizes, read_sdplib_jld2

    archive = REPO_ROOT / "problem_classes/sdplib_data/sdplib.tar"
    if not archive.exists():
        pytest.skip("bundled SDPLIB archive is not present")
    scratch = tmp_path_factory.mktemp("sdplib_members") / "member.jld2"
    members: dict[str, dict] = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.name.endswith(".jld2"):
                continue
            name = Path(member.name).stem
            source = tar.extractfile(member)
            assert source is not None, member.name
            with source, open(scratch, "wb") as sink:
                shutil.copyfileobj(source, sink)
            entry: dict = {"blocks": read_sdplib_block_sizes(scratch)}
            if not (archive.parent / f"{name}.dat-s.gz").exists():
                try:
                    problem = read_sdplib_jld2(scratch)
                except Exception as exc:  # noqa: BLE001 - the tests list every failure
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    entry["shape"] = (problem["A"].shape, problem["b"].size, problem["q"].size)
            members[name] = entry
            scratch.unlink()
    return members


# --- a stand-in for cuopt.linear_programming ----------------------------------


class FakeCuOpt:
    """A fake ``cuopt.linear_programming`` (cuOpt needs an NVIDIA GPU).

    It records what the adapter hands cuOpt (``model_calls``: the ``set_*``
    calls on the data model; ``parameters`` and ``optimality_tolerance``: the
    solver settings) and returns a solution built from ``outcome``. Like
    cuOpt, ``set_parameter`` rejects a name it does not know. The conventions
    it follows were checked by hand on a GPU with ``probe_cuopt`` in
    ``tools/bench_modal.py``; if a cuOpt release changes them, update this
    fake to match the probe's output.
    """

    KNOWN_PARAMETERS = frozenset({
        "time_limit",
        "iteration_limit",
        "num_cpu_threads",
        "log_to_console",
        "method",
        "absolute_primal_tolerance",
        "absolute_dual_tolerance",
        "absolute_gap_tolerance",
        "relative_primal_tolerance",
        "relative_dual_tolerance",
        "relative_gap_tolerance",
    })

    def __init__(self) -> None:
        import enum
        import types

        import numpy as np

        self.outcome: dict = {"status": "Optimal", "x": [], "y": [], "objective": 0.0,
                              "iterations": np.int64(17), "solve_time": 0.25}
        self.model_calls: dict[str, tuple] = {}
        self.parameters: dict[str, object] = {}
        self.optimality_tolerance: float | None = None
        fake = self
        # cuOpt reports its termination status as an enum member.
        termination = enum.Enum("PDLPTerminationStatus", [
            "Optimal", "PrimalInfeasible", "DualInfeasible", "TimeLimit", "IterationLimit",
            "NumericalError", "FeasibleFound", "UnboundedOrInfeasible",
        ])

        class DataModel:
            def __getattr__(self, name):
                if not name.startswith("set_"):
                    raise AttributeError(name)

                def record(*args):
                    fake.model_calls[name] = tuple(np.asarray(arg).copy() for arg in args)

                return record

        class SolverSettings:
            def set_optimality_tolerance(self, value):
                fake.optimality_tolerance = value

            def set_parameter(self, key, value):
                if key not in fake.KNOWN_PARAMETERS:
                    raise ValueError(f"Invalid parameter {key}")
                fake.parameters[key] = value

        def solve(model, settings):
            outcome = fake.outcome
            return types.SimpleNamespace(
                get_termination_status=lambda: termination[outcome["status"]],
                get_solve_time=lambda: outcome["solve_time"],
                get_primal_objective=lambda: outcome["objective"],
                get_dual_objective=lambda: outcome["objective"],
                get_lp_stats=lambda: {"nb_iterations": outcome["iterations"]},
                get_primal_solution=lambda: outcome["x"],
                get_dual_solution=lambda: outcome["y"],
            )

        self.module = types.SimpleNamespace(
            data_model=types.SimpleNamespace(DataModel=DataModel),
            solver=types.SimpleNamespace(Solve=solve),
            solver_settings=types.SimpleNamespace(SolverSettings=SolverSettings),
        )

    def returns(self, status: str, **outcome) -> FakeCuOpt:
        """Make the next solves end with ``status`` (and ``x``, ``y``, ...)."""
        self.outcome.update(status=status, **outcome)
        return self


@pytest.fixture
def fake_cuopt(monkeypatch) -> FakeCuOpt:
    """Install a :class:`FakeCuOpt` as ``cuopt.linear_programming``."""
    import sys
    import types

    fake = FakeCuOpt()
    package = types.ModuleType("cuopt")
    package.linear_programming = fake.module
    monkeypatch.setitem(sys.modules, "cuopt", package)
    monkeypatch.setitem(sys.modules, "cuopt.linear_programming", fake.module)
    return fake
