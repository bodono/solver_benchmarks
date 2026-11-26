"""The warm-up before a solve's timed region (core/warmup.py).

The tiny problem is checked for every form and cone type against every
installed solver (feasible, bounded, its documented optimum); the warm-up's
parts, its failures and its record through fake adapters; the worker's
ordering and switch; and the runner's handling of a worker that dies or
hangs in the warm-up. The GPU part has its own tests (test_gpu_warmup.py).
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status, warmup
from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.problem import CONE, QP, ProblemData, ProblemSpec, cone_row_count
from solver_benchmarks.core.result import ProblemResult, SolverResult
from solver_benchmarks.solvers import get_solver, list_solvers
from solver_benchmarks.solvers.base import SolverAdapter, WarmupModules
from solver_benchmarks.solvers.registry import SOLVERS
from solver_benchmarks.worker import run_payload

# --- the tiny problem ---------------------------------------------------------

ROOT2, ROOT3 = math.sqrt(2.0), math.sqrt(3.0)
# The optimum of each block of tiny_cone, without and with P = I
# (q'x and q'x + x'x / 2 at the block's optimal x).
BLOCK_OPTIMA = {
    "z": (1.0, 1.5),
    "f": (1.0, 1.5),
    "l": (-1.0, -0.5),
    "box": (-1.0, -0.5),
    "q": (-ROOT3, -ROOT3 + 0.5),
    "s": (-1.5, -1.5 + 0.375),
    "cs": (-ROOT2, -ROOT2 + 0.5),
    "ep": (0.0, 0.0),
    "ed": (-1.0, -0.5),
    "p+": (-1.0, -0.5),
    "p-": (-2.0, -0.5),  # |x| <= 2 in the dual power cone of 0.5; with P, x = -1 inside
}
CONES = {
    "z": {"z": 3},
    "f": {"f": 2},
    "l": {"l": 5},
    "box": {"bl": [0.0, -1.0], "bu": [1.0, 2.0]},
    "q": {"q": [3, 7]},
    "s": {"s": [5, 2]},
    "cs": {"cs": [3]},
    "ep": {"ep": 4},
    "ed": {"ed": 2},
    "p+": {"p": [0.3, 0.7]},
    "p-": {"p": [-0.4]},
}
ALL_CONES = {
    "z": 2, "l": 3, "bl": [0.0], "bu": [1.0], "q": [3], "s": [4], "cs": [2], "ep": 1, "ed": 1,
    "p": [0.3, -0.4, 0.5],
}
ALL_BLOCKS = ["z", "l", "box", "q", "s", "cs", "ep", "ed", "p+", "p-"]


def _cone_problem(cone: dict, p_kind: str = "none", *, seed: int = 0) -> ProblemData:
    """A problem with ``cone``'s structure and arbitrary values (its rows
    are random: only the structure may reach the tiny problem)."""
    rng = np.random.default_rng(seed)
    m, n = cone_row_count(cone), 4
    data = {
        "A": sp.csc_matrix(rng.normal(size=(m, n))),
        "b": rng.normal(size=m),
        "q": rng.normal(size=n),
        "r": 0.0,
        "cone": cone,
        "n": n,
        "m": m,
    }
    if p_kind == "nonzero":
        data["P"] = sp.csc_matrix(np.diag(rng.uniform(1, 2, size=n)))
    elif p_kind == "zero":
        data["P"] = sp.csc_matrix((n, n))
    return ProblemData("dataset", "real", CONE, data)


def _qp_problem(has_p: bool, *, seed: int = 0) -> ProblemData:
    rng = np.random.default_rng(seed)
    n, m = 5, 3
    p = sp.csc_matrix(np.diag(rng.uniform(1, 2, size=n))) if has_p else sp.csc_matrix((n, n))
    data = {
        "P": p, "q": rng.normal(size=n), "r": 0.0, "A": sp.csc_matrix(rng.normal(size=(m, n))),
        "l": -np.ones(m), "u": np.ones(m), "n": n, "m": m,
    }
    return ProblemData("dataset", "real", QP, data)


@pytest.mark.parametrize("has_p", [False, True])
def test_the_tiny_qp_has_the_real_qps_objective_kind(has_p):
    tiny = warmup.tiny_problem(_qp_problem(has_p))

    assert tiny.kind == QP and tiny.dataset_id == "warmup"
    assert (tiny.qp["P"].nnz > 0) == has_p
    # Every kind of row bound: equality, two-sided, above only, below only.
    l, u = tiny.qp["l"], tiny.qp["u"]
    assert (l == u).any() and np.isinf(l).any() and np.isinf(u).any()
    assert warmup.describe(tiny) == {"kind": QP, "n": 2, "m": 6, "p": "nonzero" if has_p else "zero"}


@pytest.mark.parametrize("name", sorted(CONES))
@pytest.mark.parametrize("p_kind", ["none", "zero", "nonzero"])
def test_the_tiny_cone_problem_has_one_small_block_of_each_cone_type(name, p_kind):
    tiny = warmup.tiny_problem(_cone_problem(CONES[name], p_kind))

    cone = tiny.cone["cone"]
    expected_keys = {"box": ["bl", "bu"]}.get(name, [name.rstrip("+-")])
    assert list(cone) == expected_keys
    assert warmup._p_kind(tiny.cone.get("P")) == p_kind
    if name == "p+":
        assert cone["p"] == [0.5]
    if name == "p-":
        assert cone["p"] == [-0.5]


def test_the_tiny_problem_follows_the_structure_not_the_values():
    first = warmup.tiny_problem(_cone_problem(ALL_CONES, "nonzero", seed=1))
    second = warmup.tiny_problem(_cone_problem(ALL_CONES, "nonzero", seed=2))

    for key in ("b", "q"):
        np.testing.assert_array_equal(first.cone[key], second.cone[key])
    assert (first.cone["A"] != second.cone["A"]).nnz == 0
    assert first.cone["cone"] == second.cone["cone"]
    # Every cone type, in row order; one primal and one dual power block.
    assert list(first.cone["cone"]) == ["z", "l", "bl", "bu", "q", "s", "cs", "ep", "ed", "p"]
    assert first.cone["cone"]["p"] == [0.5, -0.5]


@pytest.mark.parametrize("has_p", [False, True])
def test_the_tiny_qp_follows_the_structure_not_the_values(has_p):
    first = _qp_problem(has_p, seed=1)
    second = _qp_problem(has_p, seed=2)
    second.qp["q"] = -np.abs(second.qp["q"])  # every sign of the data flipped one way
    first.qp["q"] = np.abs(first.qp["q"])
    second.qp["l"], second.qp["u"] = -np.full(3, np.inf), np.full(3, 7.0)

    tiny_first, tiny_second = warmup.tiny_problem(first), warmup.tiny_problem(second)

    for key in ("q", "l", "u"):
        np.testing.assert_array_equal(tiny_first.qp[key], tiny_second.qp[key])
    for key in ("A", "P"):
        assert (tiny_first.qp[key] != tiny_second.qp[key]).nnz == 0


def test_empty_blocks_are_left_out_and_unknown_cones_are_refused():
    tiny = warmup.tiny_cone({"z": 0, "l": 2, "q": [], "s": [], "p": []})
    assert tiny["cone"] == {"l": 2}
    with pytest.raises(warmup.TinyProblemUnsupported, match="cone key 'xyz'"):
        warmup.tiny_cone({"l": 1, "xyz": 3})
    with pytest.raises(warmup.TinyProblemUnsupported, match="no block"):
        warmup.tiny_cone({"z": 0})


def _expected_optimum(blocks: list[str], with_p: bool) -> float:
    return sum(BLOCK_OPTIMA[block][1 if with_p else 0] for block in blocks)


def _cases():
    cases = [("qp", False, None), ("qp", True, None)]
    for name in sorted(CONES):
        for p_kind in ("none", "zero", "nonzero"):
            cases.append(("cone", p_kind, name))
    for p_kind in ("none", "nonzero"):
        cases.append(("cone", p_kind, "all"))
    return cases


@pytest.mark.parametrize("solver_name", list_solvers())
def test_every_installed_solver_solves_the_tiny_problems_it_supports(solver_name, tmp_path):
    """The tiny problem is feasible and bounded with the documented optimum:
    every installed solver that takes it solves it to that optimum. A solver
    that skips a cone type skips the real problem with it too."""
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} is not installed")
    solved = 0
    for kind, p, name in _cases():
        if kind not in adapter_cls.supported_problem_kinds:
            continue
        if kind == "qp":
            tiny = warmup.tiny_problem(_qp_problem(p))
            expected = 1.5 if p else 1.0
        else:
            cone = ALL_CONES if name == "all" else CONES[name]
            tiny = warmup.tiny_problem(_cone_problem(cone, p))
            blocks = ALL_BLOCKS if name == "all" else [name]
            expected = _expected_optimum(blocks, p == "nonzero")
        with tempfile.TemporaryDirectory(dir=tmp_path) as artifacts:
            result = adapter_cls({"time_limit": 60}).solve(tiny, Path(artifacts))
        if result.status == status.SKIPPED_UNSUPPORTED:
            continue
        assert result.status == status.OPTIMAL, (kind, p, name, result.status, result.info)
        assert result.objective_value == pytest.approx(expected, abs=1e-3, rel=1e-3), (kind, p, name)
        solved += 1
    assert solved, f"{solver_name} solved no tiny problem"


# --- the warm-up's parts, with a fake adapter ---------------------------------


class _Fake(SolverAdapter):
    """Records each solve with the settings it got; ``behaviour`` maps a
    problem's dataset ("warmup" for the tiny problem) to what it does."""

    solver_name = "warmup_fake"
    supported_problem_kinds = {QP, CONE}
    solver_modules = ("json",)
    calls: list = []
    behaviour: dict = {}
    gpu = False

    @classmethod
    def requested_tolerance(cls, settings, **kwargs):
        from solver_benchmarks.solvers.base import loosest_tolerance

        return loosest_tolerance({}, {"eps": 1e-4})

    def warmup_modules(self):
        return WarmupModules(modules=self.solver_modules, gpu=self.gpu)

    def solve(self, problem, artifacts_dir: Path) -> SolverResult:
        self.calls.append((problem.dataset_id, dict(self.settings), artifacts_dir))
        action = self.behaviour.get(problem.dataset_id, {})
        if action.get("print"):
            print(action["print"], flush=True)
        if action.get("sleep"):
            time.sleep(action["sleep"])
        if action.get("write"):
            (artifacts_dir / action["write"]).write_text("written")
        if action.get("import"):
            __import__(action["import"])
        if action.get("raise"):
            raise RuntimeError(action["raise"])
        start = time.perf_counter()
        return SolverResult(
            status=action.get("status", status.OPTIMAL), objective_value=0.0, iterations=3,
            run_time_seconds=time.perf_counter() - start, info=action.get("info", {}),
        )


@pytest.fixture
def fake(monkeypatch):
    class Fake(_Fake):
        calls: list = []
        behaviour: dict = {}

    monkeypatch.setitem(SOLVERS, "warmup_fake", Fake)
    return Fake


def _lp() -> ProblemData:
    return _qp_problem(False)


def test_run_imports_the_modules_then_solves_the_tiny_problem_with_the_solves_settings(fake, tmp_path):
    state = tmp_path / warmup.STATE_FILE
    solver = fake({"threads": 4, "eps_abs": 1e-6, "verbose": True})

    record = warmup.run(solver, _lp(), state)

    (call,) = fake.calls
    assert call[0] == "warmup"
    assert call[1] == {"threads": 4, "eps_abs": 1e-6, "verbose": False}  # only output changes
    # The timed solve's own settings are untouched.
    assert solver.settings == {"threads": 4, "eps_abs": 1e-6, "verbose": True}
    assert record["complete"] is True and record["notes"] == []
    assert record["modules"] == ["json"] and record["import_seconds"] >= 0
    assert record["gpu"] is False and record["gpu_seconds"] is None
    assert record["tiny_problem"] == {"kind": QP, "n": 2, "m": 6, "p": "zero"}
    assert (record["tiny_status"], record["tiny_iterations"]) == (status.OPTIMAL, 3)
    assert record["seconds"] >= record["tiny_solve_seconds"] >= record["tiny_run_time_seconds"] >= 0
    assert warmup.read_state(state) == {"state": warmup.DONE, "record": json.loads(json.dumps(record))}
    assert warmup.finished_seconds(state) == pytest.approx(record["seconds"])


def test_the_tiny_solve_writes_nothing_where_the_run_keeps_its_files(fake, tmp_path, capfd):
    fake.behaviour = {"warmup": {"write": "trace.jsonl", "print": "tiny solve output"}}
    run_dir = tmp_path / "artifacts"
    run_dir.mkdir()

    record = warmup.run(fake({}), _lp(), run_dir / warmup.STATE_FILE)

    assert record["complete"] is True
    (call,) = fake.calls
    assert call[2] != run_dir and not call[2].exists()  # its own directory, deleted
    assert sorted(path.name for path in run_dir.iterdir()) == [warmup.STATE_FILE]
    # Its standard output went to the null device, not the solve's stdout.log.
    assert "tiny solve output" not in capfd.readouterr().out
    print("after the warm-up")
    assert "after the warm-up" in capfd.readouterr().out


def test_the_modules_the_tiny_solve_imports_are_recorded(fake, tmp_path, monkeypatch):
    (tmp_path / "warmup_lazy_backend.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "warmup_lazy_backend", raising=False)
    fake.behaviour = {"warmup": {"import": "warmup_lazy_backend"}}

    record = warmup.run(fake({}), _lp())

    assert record["tiny_solve_imported"] == ["warmup_lazy_backend"]


@pytest.mark.parametrize(
    "behaviour, note",
    [
        ({"raise": "no licence"}, "tiny solve raised RuntimeError: no licence"),
        ({"status": status.SOLVER_ERROR, "info": {"error": "bad pivot"}}, "tiny solve ended solver_error: bad pivot"),
        ({"status": status.SKIPPED_UNSUPPORTED, "info": {"reason": "no cone ed"}},
         "tiny solve ended skipped_unsupported: no cone ed"),
        ({"status": status.TIME_LIMIT}, "tiny solve ended time_limit"),
    ],
)
def test_a_failed_tiny_solve_is_a_note_and_the_row_is_not_fully_warmed(fake, behaviour, note):
    fake.behaviour = {"warmup": behaviour}

    record = warmup.run(fake({}), _lp())

    assert record["complete"] is False
    assert record["notes"] == [note]


def test_a_module_that_does_not_import_is_a_note(fake):
    fake.solver_modules = ("solver_benchmarks_no_such_module",)

    record = warmup.run(fake({}), _lp())

    assert record["complete"] is False
    assert record["notes"][0].startswith("import solver_benchmarks_no_such_module: ModuleNotFoundError")
    assert record["tiny_status"] == status.OPTIMAL  # the other parts still ran


def test_a_cone_the_tiny_problem_cannot_mirror_is_a_note(fake):
    problem = ProblemData("dataset", "real", CONE, {
        "A": sp.csc_matrix(np.ones((2, 1))), "b": np.ones(2), "q": np.ones(1), "cone": {"l": 1, "xyz": 1},
    })

    record = warmup.run(fake({}), problem)

    assert record["complete"] is False and fake.calls == []
    assert record["notes"] == ["no tiny problem: cone key 'xyz'"]


def test_an_adapter_without_the_hooks_is_warmed_by_the_tiny_solve(tmp_path):
    class Bare:
        def __init__(self, settings):
            self.settings = settings

        def solve(self, problem, artifacts_dir):
            return SolverResult(status=status.OPTIMAL, run_time_seconds=0.0)

    record = warmup.run(Bare({}), _lp())

    assert record["tiny_status"] == status.OPTIMAL
    assert record["complete"] is False and record["notes"][0].startswith("warmup_modules raised AttributeError")


def test_state_file_reading_is_forgiving(tmp_path):
    path = tmp_path / warmup.STATE_FILE
    assert warmup.read_state(path) is None and warmup.finished_seconds(path) is None
    path.write_text("{not json")
    assert warmup.read_state(path) is None
    warmup.write_state(path, {"state": warmup.RUNNING, "part": "tiny_solve"})
    assert warmup.finished_seconds(path) is None
    warmup.write_state(path, {"state": warmup.DONE, "record": {"seconds": "x"}})
    assert warmup.finished_seconds(path) is None
    warmup.write_state(path, {"state": warmup.DONE, "record": {"seconds": 0.25}})
    assert warmup.finished_seconds(path) == 0.25


def test_the_state_file_names_the_part_that_is_running(fake, tmp_path, monkeypatch):
    """What a worker killed in each part leaves for the runner: the state
    the runner reads (``runner._warmup_failure``) is "running" and the part,
    written before the part starts, and "done" only once all have run."""
    state = tmp_path / warmup.STATE_FILE
    seen: dict[str, object] = {}

    class Watching(fake):
        def warmup_modules(self):
            seen["imports"] = warmup.read_state(state)
            return WarmupModules(modules=("json",), gpu=True)

        def solve(self, problem, artifacts_dir):
            seen["tiny_solve"] = warmup.read_state(state)
            return super().solve(problem, artifacts_dir)

    def gpu_part(modules):
        seen["gpu"] = warmup.read_state(state)
        return {"seconds": 0.0, "cuda_context_seconds": 0.0, "cublas_libraries": [], "cublas_seconds": None,
                "notes": []}

    monkeypatch.setattr(warmup.gpu_warmup, "warm_up", gpu_part)
    monkeypatch.setattr(warmup.gpu_warmup, "mapped_libraries", lambda prefix: [])

    record = warmup.run(Watching({}), _lp(), state)

    assert seen == {
        "imports": {"state": warmup.RUNNING, "part": "imports"},
        "gpu": {"state": warmup.RUNNING, "part": "gpu"},
        "tiny_solve": {"state": warmup.RUNNING, "part": "tiny_solve"},
    }
    assert record["complete"] is True
    assert warmup.read_state(state)["state"] == warmup.DONE


def test_a_tiny_problem_that_cannot_be_built_never_fails_the_run(fake, monkeypatch):
    def broken(problem):
        raise KeyError("A")

    monkeypatch.setattr(warmup, "tiny_problem", broken)

    record = warmup.run(fake({}), _lp())

    assert record["complete"] is False and fake.calls == []
    assert record["notes"] == ["no tiny problem: KeyError: 'A'"]


def test_what_the_tiny_solve_kept_of_stdout_still_works_after_it(fake, tmp_path, capfd):
    """A module that keeps the ``sys.stdout`` it saw in the tiny solve writes
    to the null device afterwards, never to a closed file."""
    kept = []

    class Keeping(fake):
        def solve(self, problem, artifacts_dir):
            kept.append(sys.stdout)
            return super().solve(problem, artifacts_dir)

    warmup.run(Keeping({}), _lp())

    print("from the tiny solve's stdout, later", file=kept[0], flush=True)
    assert "later" not in capfd.readouterr().out


# --- the worker ----------------------------------------------------------------


def _payload(repo_root, tmp_path, **extra) -> dict:
    return {
        "run_id": "run", "dataset": "synthetic_qp", "problem": "one_variable_lp", "problem_kind": QP,
        "repo_root": str(repo_root),
        "solver": {"id": "fake", "solver": "warmup_fake", "settings": {}},
        "artifacts_dir": str(tmp_path / "artifacts"),
        **extra,
    }


def test_the_worker_warms_up_before_the_timed_solve_and_records_it(fake, repo_root, tmp_path):
    tiny_seconds = 0.2
    fake.behaviour = {"warmup": {"sleep": tiny_seconds}}

    record = run_payload(_payload(repo_root, tmp_path, warmup=True)).to_record()

    assert [call[0] for call in fake.calls] == ["warmup", "synthetic_qp"]
    warmup_record = record["metadata"]["warmup"]
    assert warmup_record["enabled"] is True and warmup_record["complete"] is True
    assert warmup_record["tiny_solve_seconds"] >= tiny_seconds
    assert record["run_time_seconds"] < tiny_seconds  # the timer excludes the warm-up
    assert warmup_record["imported_during_solve"] == []
    assert record["status"] == status.OPTIMAL
    # The runner's view of it: the state file in the solve's artifacts directory.
    state = warmup.read_state(tmp_path / "artifacts" / warmup.STATE_FILE)
    assert state["state"] == warmup.DONE and state["record"]["complete"] is True


def test_the_timed_solve_gets_the_settings_as_configured(fake, repo_root, tmp_path):
    payload = _payload(repo_root, tmp_path, warmup=True)
    payload["solver"]["settings"] = {"verbose": True, "threads": 2}

    run_payload(payload)

    tiny, real = fake.calls
    assert (tiny[0], tiny[1]) == ("warmup", {"verbose": False, "threads": 2})
    assert (real[0], real[1]) == ("synthetic_qp", {"verbose": True, "threads": 2})


def test_the_warm_up_is_on_when_the_payload_does_not_say(fake, repo_root, tmp_path):
    record = run_payload(_payload(repo_root, tmp_path)).to_record()

    assert [call[0] for call in fake.calls] == ["warmup", "synthetic_qp"]
    assert record["metadata"]["warmup"]["enabled"] is True


def test_the_switch_turns_the_warm_up_off_and_the_row_says_so(fake, repo_root, tmp_path, monkeypatch):
    (tmp_path / "warmup_cold_module.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "warmup_cold_module", raising=False)
    fake.behaviour = {"synthetic_qp": {"import": "warmup_cold_module"}}

    record = run_payload(_payload(repo_root, tmp_path, warmup=False)).to_record()

    assert [call[0] for call in fake.calls] == ["synthetic_qp"]
    assert record["metadata"]["warmup"] == {
        "enabled": False, "complete": False, "notes": [], "imported_during_solve": ["warmup_cold_module"],
    }
    assert not (tmp_path / "artifacts" / warmup.STATE_FILE).exists()


def test_a_failing_warm_up_never_fails_the_solve(fake, repo_root, tmp_path):
    fake.behaviour = {"warmup": {"raise": "boom"}}

    record = run_payload(_payload(repo_root, tmp_path)).to_record()

    assert record["status"] == status.OPTIMAL
    assert record["metadata"]["warmup"]["complete"] is False
    assert record["metadata"]["warmup"]["notes"] == ["tiny solve raised RuntimeError: boom"]


def test_a_rerun_after_a_warm_up_failure_skips_the_warm_up_and_says_why(fake, repo_root, tmp_path):
    payload = _payload(repo_root, tmp_path, warmup=True, warmup_failure="killed by SIGSEGV")

    record = run_payload(payload).to_record()

    assert [call[0] for call in fake.calls] == ["synthetic_qp"]
    assert record["metadata"]["warmup"] == {
        "enabled": True, "complete": False, "rerun_without_warmup": True, "notes": ["killed by SIGSEGV"],
        "imported_during_solve": [],
    }


def test_scs_warm_up_end_to_end(repo_root, tmp_path, capfd):
    """The real SCS adapter: the tiny solve is quiet, writes none of the run's
    files, and leaves the timed solve nothing to import."""
    if not get_solver("scs").is_available():
        pytest.skip("scs is not installed")
    payload = _payload(repo_root, tmp_path, warmup=True)
    payload["solver"] = {
        "id": "scs", "solver": "scs", "settings": {"verbose": True, "log_csv_filename": True, "time_limit": 60},
    }

    record = run_payload(payload).to_record()

    assert record["status"] == status.OPTIMAL
    warmup_record = record["metadata"]["warmup"]
    assert warmup_record["complete"] is True, warmup_record["notes"]
    assert warmup_record["tiny_status"] == status.OPTIMAL
    assert warmup_record["imported_during_solve"] == []
    # Only the real solve's files: SCS's CSV log and the worker's trace.
    assert sorted(path.name for path in (tmp_path / "artifacts").iterdir()) == [
        "scs_trace.csv", "trace.jsonl", warmup.STATE_FILE,
    ]
    # SCS prints its banner once per verbose solve: the real one only.
    assert capfd.readouterr().out.count("SCS v") == 1


# --- the runner ------------------------------------------------------------------


def _run_one(tmp_path, repo_root, worker, monkeypatch, *, warmup_on=True, timeout=None):
    from solver_benchmarks.core import runner
    from solver_benchmarks.core.storage import ResultStore

    config = parse_run_config({
        "run": {"dataset": "synthetic_qp", "warmup": warmup_on, "timeout_seconds": timeout},
        "solvers": [{"id": "fake", "solver": "warmup_fake"}],
    })
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    monkeypatch.setattr(runner, "_run_subprocess", worker)
    return runner._run_one(
        store, config, repo_root, config.datasets[0],
        ProblemSpec(dataset_id="synthetic_qp", name="one_variable_lp", kind=QP), config.solvers[0],
    )


class _ScriptedWorker:
    """Plays one worker per call: ``steps`` say, in turn, which warm-up state
    it leaves and how it ends."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.payloads: list[dict] = []
        self.extra_time = None

    def __call__(self, cmd, *, stdout_path, stderr_path, extra_time=None, **_ignored):
        payload = json.loads(Path(cmd[-1]).read_text())
        self.payloads.append(payload)
        self.extra_time = extra_time
        step = self.steps.pop(0)
        stdout_path.write_text("")
        stderr_path.write_text(step.get("stderr", ""))
        artifacts = Path(payload["artifacts_dir"])
        if "state" in step:
            warmup.write_state(artifacts / warmup.STATE_FILE, step["state"])
        if step.get("result"):
            record = ProblemResult(
                run_id=payload["run_id"], dataset=payload["dataset"], problem=payload["problem"],
                problem_kind=payload["problem_kind"], solver_id="fake", solver="warmup_fake",
                status=status.OPTIMAL, objective_value=0.0, iterations=1, run_time_seconds=0.01,
                artifact_dir=str(artifacts), metadata={"warmup": {"enabled": True}},
            ).to_record()
            (artifacts / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(
            returncode=step.get("returncode", 0), stderr=step.get("stderr", ""),
            timed_out=step.get("timed_out", False), extra_seconds=step.get("extra_seconds", 0.0),
        )


RUNNING_TINY = {"state": warmup.RUNNING, "part": "tiny_solve"}


def test_a_worker_killed_in_the_warm_up_is_run_again_without_it(fake, tmp_path, repo_root, monkeypatch):
    worker = _ScriptedWorker([
        {"state": RUNNING_TINY, "returncode": -11, "stderr": "Fatal Python error: Segmentation fault"},
        {"result": True},
    ])

    result = _run_one(tmp_path, repo_root, worker, monkeypatch)

    assert result.status == status.OPTIMAL
    artifacts = Path(result.artifact_dir)
    # The first worker's output is kept.
    assert (artifacts / "stderr.warmup_failure.log").read_text() == "Fatal Python error: Segmentation fault"
    first, second = worker.payloads
    assert first["warmup"] is True and "warmup_failure" not in first
    failure = second["warmup_failure"]
    assert failure.startswith("the worker ended in the warm-up (tiny_solve): Worker killed by SIGSEGV")


def test_a_worker_that_hangs_in_the_warm_up_is_run_again_without_it(fake, tmp_path, repo_root, monkeypatch):
    worker = _ScriptedWorker([
        {"state": {"state": warmup.RUNNING, "part": "gpu"}, "timed_out": True, "returncode": -15},
        {"result": True},
    ])

    result = _run_one(tmp_path, repo_root, worker, monkeypatch, timeout=5)

    assert result.status == status.OPTIMAL
    assert worker.payloads[1]["warmup_failure"] == (
        "the worker was still in the warm-up (gpu) at its timeout and was killed"
    )


def test_a_rerun_that_fails_too_is_a_worker_error_that_names_both(fake, tmp_path, repo_root, monkeypatch):
    worker = _ScriptedWorker([
        {"state": RUNNING_TINY, "returncode": -11},
        {"returncode": -6, "stderr": "abort"},
    ])

    result = _run_one(tmp_path, repo_root, worker, monkeypatch)

    assert result.status == status.WORKER_ERROR
    assert "SIGABRT" in result.error and "a first worker ended in the warm-up" in result.error
    assert result.metadata["warmup"]["rerun_without_warmup"] is True


def test_a_solve_that_crashes_after_the_warm_up_is_not_run_again(fake, tmp_path, repo_root, monkeypatch):
    done = {"state": warmup.DONE, "record": {"enabled": True, "complete": True, "seconds": 0.3}}
    worker = _ScriptedWorker([{"state": done, "returncode": -11}])

    result = _run_one(tmp_path, repo_root, worker, monkeypatch)

    assert len(worker.payloads) == 1
    assert result.status == status.WORKER_ERROR
    assert result.metadata["warmup"] == done["record"]


def test_a_timeout_after_the_warm_up_keeps_its_record_and_says_how_long_was_added(fake, tmp_path, repo_root, monkeypatch):
    done = {"state": warmup.DONE, "record": {"enabled": True, "complete": True, "seconds": 0.3}}
    worker = _ScriptedWorker([{"state": done, "timed_out": True, "returncode": -15, "extra_seconds": 0.3}])

    result = _run_one(tmp_path, repo_root, worker, monkeypatch, timeout=5)

    assert len(worker.payloads) == 1
    assert result.status == status.TIME_LIMIT
    assert "plus the warm-up's 0.3 s" in result.error
    assert result.metadata["warmup"] == done["record"]


def test_the_runner_gives_the_worker_the_warm_ups_time(fake, tmp_path, repo_root, monkeypatch):
    worker = _ScriptedWorker([{"result": True}])
    _run_one(tmp_path, repo_root, worker, monkeypatch, timeout=5)
    state = Path(worker.payloads[0]["artifacts_dir"]) / warmup.STATE_FILE
    assert worker.extra_time() is None  # no state: nothing added
    warmup.write_state(state, {"state": warmup.DONE, "record": {"seconds": 0.75}})
    assert worker.extra_time() == 0.75


def test_a_run_without_the_warm_up_tells_the_worker(fake, tmp_path, repo_root, monkeypatch):
    worker = _ScriptedWorker([{"state": RUNNING_TINY, "returncode": -11}])

    result = _run_one(tmp_path, repo_root, worker, monkeypatch, warmup_on=False)

    # A stale state cannot make the runner retry a run that has no warm-up:
    # the worker of such a run writes none (this one is scripted to).
    assert worker.payloads[0]["warmup"] is False
    assert result.status == status.WORKER_ERROR


def test_the_timeout_is_extended_by_the_finished_warm_up(tmp_path, repo_root):
    """The real subprocess wait: a worker past its timeout runs on for the
    time its warm-up took, and is killed after that."""
    from solver_benchmarks.core.runner import _run_subprocess

    def run(sleep, extra):
        return _run_subprocess(
            [sys.executable, "-c", f"import time; time.sleep({sleep})"],
            cwd=tmp_path, timeout=0.5, stdout_path=tmp_path / "out.log", stderr_path=tmp_path / "err.log",
            stream_output=False, extra_time=lambda: extra,
        )

    extended = run(1.0, 3.0)
    assert not extended.timed_out and extended.returncode == 0 and extended.extra_seconds == 3.0
    killed = run(2.0, None)
    assert killed.timed_out and killed.extra_seconds == 0.0


# --- the runner with real workers ------------------------------------------------------

# An adapter that dies (SIGSEGV) or hangs in the warm-up's tiny solve and
# solves the real problem at once, registered in every worker by a
# sitecustomize on PYTHONPATH (as test_worker_lifecycle.py's solvers are).
_CRASHY = textwrap.dedent(
    """
    import os, signal, time
    from solver_benchmarks.core.result import SolverResult
    from solver_benchmarks.solvers import registry
    from solver_benchmarks.solvers.base import SolverAdapter, loosest_tolerance

    class Crashy(SolverAdapter):
        solver_name = "warmup_crashy"
        supported_problem_kinds = {"qp", "cone"}

        @classmethod
        def requested_tolerance(cls, settings, **kwargs):
            return loosest_tolerance({}, {"eps": 1e-4})

        def solve(self, problem, artifacts_dir):
            if problem.dataset_id == "warmup":
                if self.settings["mode"] == "segv":
                    os.kill(os.getpid(), signal.SIGSEGV)
                time.sleep(600)
            return SolverResult(
                status="optimal", objective_value=0.0, iterations=1, run_time_seconds=0.001,
                info={"settings": dict(self.settings)},
            )

    registry.SOLVERS["warmup_crashy"] = Crashy
    """
)


@pytest.fixture
def crashy(tmp_path, monkeypatch):
    """Register ``warmup_crashy`` here and in every worker this test starts."""
    import importlib.util

    site = tmp_path / "crashy_site"
    site.mkdir()
    (site / "bench_warmup_crashy.py").write_text(_CRASHY)
    (site / "sitecustomize.py").write_text(
        "import os\nif os.environ.get('BENCH_TEST_WARMUP_CRASHY'):\n    import bench_warmup_crashy\n"
    )
    monkeypatch.setitem(SOLVERS, "warmup_crashy", None)
    spec = importlib.util.spec_from_file_location("bench_warmup_crashy_parent", site / "bench_warmup_crashy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(SOLVERS, "warmup_crashy", module.Crashy)
    path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(site) + (os.pathsep + path if path else ""))
    monkeypatch.setenv("BENCH_TEST_WARMUP_CRASHY", "1")


def _run_crashy(tmp_path, mode: str, timeout: float):
    from solver_benchmarks.core.runner import run_benchmark

    config = parse_run_config({
        "run": {"dataset": "synthetic_qp", "include": ["one_variable_lp"], "timeout_seconds": timeout},
        "solvers": [{"id": "crashy", "solver": "warmup_crashy", "settings": {"mode": mode}}],
    })
    store = run_benchmark(config, run_dir=tmp_path / "run")
    rows = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines() if line.strip()]
    events = [json.loads(line) for line in store.events_path.read_text().splitlines() if line.strip()]
    return rows, events


def _assert_rerun_cold(rows, events, failure: str):
    (row,) = rows
    assert row["status"] == status.OPTIMAL
    record = row["metadata"]["warmup"]
    assert record["enabled"] is True and record["complete"] is False
    assert record["rerun_without_warmup"] is True
    assert record["notes"][0].startswith(failure), record["notes"]
    assert row["info"]["settings"] == {"mode": row["info"]["settings"]["mode"]}  # as configured
    artifacts = Path(row["artifact_dir"])
    assert (artifacts / "stdout.warmup_failure.log").exists()
    assert (artifacts / "stderr.warmup_failure.log").exists()
    # The run says, at its end, that the row is cold.
    (warning,) = [event for event in events if "not fully warmed" in event["message"]]
    assert warning["level"] == "warning" and warning["warmup_counts"] == {warmup.RERUN_COLD: 1}


@pytest.mark.skipif(sys.platform == "win32", reason="no SIGSEGV by os.kill")
def test_a_real_worker_killed_in_the_tiny_solve_is_run_again_cold(crashy, tmp_path):
    """The real worker, state file and _run_subprocess: a segfault in the
    tiny solve gives the solve's row, re-run without the warm-up."""
    rows, events = _run_crashy(tmp_path, "segv", timeout=60)

    _assert_rerun_cold(rows, events, "the worker ended in the warm-up (tiny_solve): Worker killed by SIGSEGV")


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_a_real_worker_hung_in_the_tiny_solve_is_run_again_cold(crashy, tmp_path):
    """As above for a hang: killed at the worker's deadline (the time limit
    plus the 5 s minimum grace), then re-run without the warm-up."""
    began = time.monotonic()
    rows, events = _run_crashy(tmp_path, "hang", timeout=1)

    _assert_rerun_cold(rows, events, "the worker was still in the warm-up (tiny_solve) at its timeout")
    assert time.monotonic() - began >= 6.0


# --- not fully warmed rows: the runner's and the analysis's warnings, and resume -----


def test_timing_categories_and_the_warning():
    assert warmup.timing_category(None) == warmup.NO_RECORD
    assert warmup.timing_category({"enabled": True, "complete": True}) == warmup.WARMED
    assert warmup.timing_category(warmup.off_record()) == warmup.COLD_BY_REQUEST
    assert warmup.timing_category(warmup.rerun_record("x")) == warmup.RERUN_COLD
    assert warmup.timing_category({"enabled": True, "complete": False}) == warmup.INCOMPLETE

    assert warmup.not_warmed_warning({warmup.WARMED: 4}) is None
    assert warmup.not_warmed_warning({}) is None
    message = warmup.not_warmed_warning({warmup.WARMED: 4, warmup.NO_RECORD: 2, warmup.COLD_BY_REQUEST: 1})
    assert message.startswith("3 of 7 timed rows were not fully warmed up before their timed region (")
    assert "1 recorded with run.warmup: false" in message and "2 with no warm-up record" in message
    assert "the other 4 are warmed, so these rows mix the two" in message
    assert "mix" not in warmup.not_warmed_warning({warmup.COLD_BY_REQUEST: 3})


def _row(name: str, warmup_record, *, run_time=0.5, row_status=status.OPTIMAL) -> dict:
    metadata = {} if warmup_record is None else {"warmup": warmup_record}
    return {
        "run_id": "r", "dataset": "synthetic_qp", "problem": name, "problem_kind": QP, "solver_id": "fake",
        "solver": "warmup_fake", "status": row_status, "objective_value": 0.0, "iterations": 1,
        "run_time_seconds": run_time, "metadata": metadata,
    }


def test_the_analysis_counts_timed_rows_that_were_not_fully_warmed():
    import pandas as pd

    from solver_benchmarks.analysis.warmup_check import warmup_counts, warmup_warnings

    rows = [
        _row("a", {"enabled": True, "complete": True, "notes": []}),
        _row("b", {"enabled": True, "complete": True, "notes": []}),
        _row("c", warmup.off_record()),
        _row("d", warmup.rerun_record("killed")),
        _row("e", {"enabled": True, "complete": False, "notes": ["no tiny problem"]}),
        _row("f", None),
        _row("g", None, run_time=None, row_status=status.WORKER_ERROR),  # no time: not counted
        _row("h", None, run_time=0.0, row_status=status.SKIPPED_UNSUPPORTED),  # a skip: not counted
    ]
    frame = pd.json_normalize(rows)

    assert warmup_counts(frame) == {
        warmup.WARMED: 2, warmup.COLD_BY_REQUEST: 1, warmup.RERUN_COLD: 1, warmup.INCOMPLETE: 1,
        warmup.NO_RECORD: 1,
    }
    (message,) = warmup_warnings(frame)
    assert message.startswith("4 of 6 timed rows were not fully warmed up")
    assert warmup_warnings(pd.json_normalize(rows[:2])) == []
    # Rows recorded before the warm-up existed have no metadata.warmup columns at all.
    assert warmup_counts(pd.json_normalize([_row("x", None)])) == {warmup.NO_RECORD: 1}


def test_the_report_and_the_cli_say_whether_rows_were_warmed(tmp_path):
    from click.testing import CliRunner

    from solver_benchmarks.analysis.markdown_report import _render_warmup_block
    from solver_benchmarks.cli import main
    from solver_benchmarks.core.storage import read_results_jsonl

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [_row("a", {"enabled": True, "complete": True, "notes": []}), _row("b", warmup.off_record())]
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    block = _render_warmup_block(read_results_jsonl(run_dir / "results.jsonl"))
    assert block[0] == "## Warm-up" and block[2].startswith("Warning: 1 of 2 timed rows were not fully warmed")
    warmed = read_results_jsonl(run_dir / "results.jsonl").iloc[:1]
    assert _render_warmup_block(warmed)[2] == (
        "All 1 timed rows were warmed up before their timed region (`metadata.warmup.complete`)."
    )

    result = CliRunner().invoke(main, ["summary", str(run_dir), "--unverified"])
    assert result.exit_code == 0, result.output
    assert "warning: 1 of 2 timed rows were not fully warmed up" in result.output


def _resume_dir(tmp_path, *, manifest_config: dict | None, rows: list[dict]) -> Path:
    run_dir = tmp_path / "old_run"
    run_dir.mkdir()
    if manifest_config is not None:
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "old", "config": manifest_config}))
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run_dir


@pytest.mark.parametrize(
    "manifest_config, rows, refused_with, allowed_with",
    [
        # A manifest from before the warm-up existed: its rows are cold.
        ({"name": "old"}, [_row("a", None)], True, False),
        ({"name": "old", "warmup": True}, [_row("a", {"enabled": True, "complete": True})], False, True),
        ({"name": "old", "warmup": False}, [_row("a", warmup.off_record())], True, False),
        # No manifest: the rows' records decide; rows without one say nothing.
        (None, [_row("a", warmup.off_record())], True, False),
        (None, [_row("a", None)], None, None),
    ],
)
def test_a_resume_may_not_change_the_warm_up_switch(tmp_path, manifest_config, rows, refused_with, allowed_with):
    from solver_benchmarks.core.runner import refuse_to_change_warmup

    run_dir = _resume_dir(tmp_path, manifest_config=manifest_config, rows=rows)
    manifest = json.loads((run_dir / "manifest.json").read_text()) if manifest_config is not None else None
    if refused_with is None:
        for value in (True, False):
            refuse_to_change_warmup(run_dir, _config(warmup=value, resume=True), manifest)
        return
    with pytest.raises(RuntimeError, match=f"recorded with run.warmup: {str(allowed_with).lower()}"):
        refuse_to_change_warmup(run_dir, _config(warmup=refused_with, resume=True), manifest)
    refuse_to_change_warmup(run_dir, _config(warmup=allowed_with, resume=True), manifest)


def test_a_refused_resume_leaves_the_run_directory_as_it_was(fake, tmp_path, repo_root):
    from solver_benchmarks.core.runner import run_benchmark

    run_dir = _resume_dir(tmp_path, manifest_config={"name": "old", "warmup": False}, rows=[
        _row("one_variable_lp", warmup.off_record()),
    ])
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    config = parse_run_config({
        "run": {"dataset": "synthetic_qp", "resume": True},
        "solvers": [{"id": "fake", "solver": "warmup_fake"}],
    })

    with pytest.raises(RuntimeError, match="Resume with the same run.warmup, or use a new --run-dir"):
        run_benchmark(config, run_dir=run_dir, repo_root=repo_root)

    after = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.name != ".bench.lock"}
    assert after == before  # the manifest was not rewritten with the refused setting


def test_the_progress_reporter_warns_at_the_end_about_cold_rows(tmp_path, capsys):
    from solver_benchmarks.core.runner import _ProgressReporter
    from solver_benchmarks.core.storage import ResultStore

    store = ResultStore.create(_config(), run_dir=tmp_path / "run")
    progress = _ProgressReporter(
        store=store, stream_output=True, total_expected=3, already_complete=0, skipped_during_planning=0,
        queued=3, parallelism=1,
    )
    for name, record in (("a", {"enabled": True, "complete": True}), ("b", None), ("c", warmup.off_record())):
        progress.record_result(ProblemResult(**_row(name, record)))
    progress.emit_final()

    events = [json.loads(line) for line in store.events_path.read_text().splitlines()]
    (warning,) = [event for event in events if event["level"] == "warning"]
    assert warning["message"].startswith("In this run, 2 of 3 timed rows were not fully warmed up")
    assert warning["warmup_counts"] == {warmup.WARMED: 1, warmup.NO_RECORD: 1, warmup.COLD_BY_REQUEST: 1}
    assert "[bench] warning: In this run, 2 of 3" in capsys.readouterr().err


# --- the run config ----------------------------------------------------------------


def _config(**run):
    return parse_run_config({"run": {"dataset": "synthetic_qp", **run}, "solvers": [{"id": "s", "solver": "scs"}]})


def test_the_warm_up_is_on_by_default_and_recorded_in_the_manifest():
    assert _config().warmup is True
    assert _config(warmup=False).warmup is False
    assert _config(warmup=False).to_manifest()["warmup"] is False
    # A timing switch, like parallelism: it does not change the config hash.
    assert _config(warmup=False).config_hash == _config().config_hash


def test_the_warm_up_switch_must_be_a_boolean():
    assert _config(warmup="off").warmup is False  # the config's usual boolean spellings
    with pytest.raises(ValueError, match="run.warmup"):
        _config(warmup="sometimes")


# --- adapter hooks ------------------------------------------------------------------


def test_scs_warm_up_settings_write_no_file_of_the_run():
    adapter = get_solver("scs")({
        "log_csv_filename": "/runs/x/trace.csv", "write_data_filename": "/runs/x/data", "eps_abs": 1e-5,
        "linear_solver": "cudss",
    })
    assert adapter.warmup_settings() == {
        "log_csv_filename": True, "eps_abs": 1e-5, "linear_solver": "cudss", "verbose": False,
    }
    assert get_solver("scs")({"log_csv_filename": False}).warmup_settings() == {
        "log_csv_filename": False, "verbose": False,
    }


def test_highs_gurobi_and_cuopt_warm_up_settings_drop_their_file_settings():
    highs = get_solver("highs")({
        "log_file": "h.log", "write_solution_to_file": True, "write_basis_file": "b.bas",
        "write_presolved_model_file": "p.mps", "write_iis_model_file": "i.lp", "read_basis_file": "in.bas",
        "threads": 2,
    })
    assert highs.warmup_settings() == {"threads": 2, "verbose": False}
    gurobi = get_solver("gurobi")({"LogFile": "g.log", "resultfile": "x.sol", "Record": 1, "Method": 2})
    assert gurobi.warmup_settings() == {"Method": 2, "verbose": False}
    cuopt = get_solver("cuopt")({"log_file": "c.log", "solution_file": "c.sol", "user_problem_file": "c.mps",
                                 "pdlp_solver_mode": 2})
    assert cuopt.warmup_settings() == {"pdlp_solver_mode": 2, "verbose": False}


def test_highs_keeps_its_thread_pool_while_the_thread_count_stays(monkeypatch):
    """The real solve reuses the pool the tiny solve created (same settings),
    so the pool's creation is not timed; another count resets it."""
    from solver_benchmarks.solvers import highs_adapter

    resets = []

    class FakeHighs:
        @staticmethod
        def resetGlobalScheduler(blocking):
            resets.append(blocking)

    monkeypatch.setattr(highs_adapter, "_POOL_THREADS", None)
    for threads in (4, 4, 1, 1, 0, 4):
        highs_adapter._reset_pool_if_resized(FakeHighs(), threads)

    assert len(resets) == 4  # the first solve, then each change of count
    assert all(resets)


@pytest.mark.parametrize("name", list_solvers())
def test_every_adapter_keeps_its_settings_for_the_tiny_solve(name):
    settings = {"threads": 3, "time_limit": 7.0, "max_iter": 50}
    assert get_solver(name)(settings).warmup_settings() == {**settings, "verbose": False}
