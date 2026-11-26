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
import sys
import tempfile
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


def test_highs_and_gurobi_warm_up_settings_drop_their_output_files():
    highs = get_solver("highs")({"log_file": "h.log", "write_solution_to_file": True, "threads": 2})
    assert highs.warmup_settings() == {"threads": 2, "verbose": False}
    gurobi = get_solver("gurobi")({"LogFile": "g.log", "resultfile": "x.sol", "Method": 2})
    assert gurobi.warmup_settings() == {"Method": 2, "verbose": False}


@pytest.mark.parametrize("name", list_solvers())
def test_every_adapter_keeps_its_settings_for_the_tiny_solve(name):
    settings = {"threads": 3, "time_limit": 7.0, "max_iter": 50}
    assert get_solver(name)(settings).warmup_settings() == {**settings, "verbose": False}
