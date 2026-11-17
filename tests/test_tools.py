"""Tests for the campaign scripts under ``tools/``.

``tools/`` is not a package and nothing in the harness imports it, so these
tests load the scripts by path. They pin the parts that must agree with the
harness: the figures score outcomes exactly as ``solver_benchmarks.analysis``
does, the "largest quartile" is sized from problem metadata rather than from
whichever rows happen to carry it, and every campaign config the generator
writes is one the harness can load. ``tools/bench_modal.py`` imports Modal,
which is not installed here, so it is loaded with a stand-in for ``modal``.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solver_benchmarks.analysis.profiles import shifted_geomean
from solver_benchmarks.core import status
from solver_benchmarks.core.config import load_run_config

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"_tools_{name}", TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def make_figures():
    return _load_tool("make_figures")


@pytest.fixture(scope="module")
def make_campaign():
    return _load_tool("make_campaign")


def _row(problem, solver_id, row_status, seconds, **extra):
    return {"dataset": "netlib_infeasible", "problem": problem, "solver_id": solver_id,
            "status": row_status, "run_time_seconds": seconds, **extra}


def _final_fractions(profile: pd.DataFrame) -> dict[str, float]:
    return profile.iloc[-1].drop("tau").to_dict()


def test_profile_scores_known_infeasible_problems_like_the_harness(make_figures):
    # On problems known to be infeasible the correct answer is a certificate;
    # a claim of optimality is wrong. The profile must agree with the geomean
    # bars drawn next to it, which go through the harness outcome helpers.
    expected = {"metadata.expected_status": status.EXPECT_INFEASIBLE}
    df = pd.DataFrame([
        row
        for problem in ("bgdbg1", "itest2")
        for row in (
            _row(problem, "scs_cpu_1e-4", status.PRIMAL_INFEASIBLE, 0.5, **expected),
            _row(problem, "osqp_1e-4", status.OPTIMAL, 0.2, **expected),
        )
    ])

    fractions = _final_fractions(make_figures.full_denominator_profile(df))

    assert fractions == {"osqp_1e-4": 0.0, "scs_cpu_1e-4": 1.0}
    bars = shifted_geomean(df, metric="run_time_seconds", max_value=make_figures.failure_penalty("infeas"))
    solved = dict(zip(bars["solver_id"], bars["success_count"]))
    assert {sid: fractions[sid] * 2 for sid in fractions} == solved


def test_profile_rejects_a_certificate_the_harness_found_invalid(make_figures):
    expected = {"metadata.expected_status": status.EXPECT_INFEASIBLE}
    df = pd.DataFrame([
        _row("bgdbg1", "scs_cpu_1e-4", status.PRIMAL_INFEASIBLE, 0.5, **{**expected, "kkt.valid": True}),
        _row("bgdbg1", "clarabel_1e-4", status.PRIMAL_INFEASIBLE, 0.3, **{**expected, "kkt.valid": False}),
    ])

    fractions = _final_fractions(make_figures.full_denominator_profile(df))

    assert fractions == {"clarabel_1e-4": 0.0, "scs_cpu_1e-4": 1.0}


def test_profile_keeps_unsolved_and_unattempted_problems_in_the_denominator(make_figures):
    df = pd.DataFrame([
        {"dataset": "lp", "problem": "a", "solver_id": "scs_cpu_1e-4", "status": status.OPTIMAL,
         "run_time_seconds": 1.0},
        {"dataset": "lp", "problem": "a", "solver_id": "highs_1e-4", "status": status.OPTIMAL,
         "run_time_seconds": 2.0},
        # Every solver failed on b; c has no highs row at all.
        {"dataset": "lp", "problem": "b", "solver_id": "scs_cpu_1e-4", "status": status.TIME_LIMIT,
         "run_time_seconds": 360.0},
        {"dataset": "lp", "problem": "b", "solver_id": "highs_1e-4", "status": status.SOLVER_ERROR,
         "run_time_seconds": 1.0},
        {"dataset": "lp", "problem": "c", "solver_id": "scs_cpu_1e-4", "status": status.OPTIMAL,
         "run_time_seconds": 1.0},
    ])

    profile = make_figures.full_denominator_profile(df)

    assert _final_fractions(profile) == pytest.approx({"scs_cpu_1e-4": 2 / 3, "highs_1e-4": 1 / 3})
    assert profile["tau"].iloc[0] == pytest.approx(1.0)
    assert profile["tau"].iloc[-1] == pytest.approx(make_figures.TAU_MAX)


def _sized_rows(known: int, unknown: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for i in range(1, known + 1):
        for sid in ("scs_cpu_1e-4", "clarabel_1e-4"):
            rows.append({"dataset": "lp", "problem": f"small{i}", "solver_id": sid,
                         "status": status.OPTIMAL, "run_time_seconds": 1.0,
                         "metadata.nnz_a": 1000 * i, "metadata.nnz_p": 0})
    # Rows the runner writes itself (harness timeout, crashed worker) carry
    # no dimensions; when every solver ended that way the results alone
    # cannot size the problem.
    for name in unknown:
        for sid in ("scs_cpu_1e-4", "clarabel_1e-4"):
            rows.append({"dataset": "lp", "problem": name, "solver_id": sid,
                         "status": status.TIME_LIMIT, "run_time_seconds": 360.0,
                         "metadata.nnz_a": np.nan, "metadata.nnz_p": np.nan})
    return pd.DataFrame(rows)


def test_largest_quartile_sizes_unsized_problems_through_the_lookup(make_figures):
    df = _sized_rows(6, ("huge1", "huge2"))
    looked_up = []

    def lookup(dataset, problem):
        looked_up.append((dataset, problem))
        return 10_000_000

    sizes = make_figures.problem_sizes(df, size_lookup=lookup)
    big = make_figures.largest_quartile(df, sizes)

    assert sorted(looked_up) == [("lp", "huge1"), ("lp", "huge2")]
    assert sorted(big["problem"].unique()) == ["huge1", "huge2"]


def test_largest_quartile_reports_unknown_sizes_instead_of_counting_them_as_zero(make_figures, capsys):
    df = _sized_rows(4, ("huge1", "huge2", "huge3", "huge4"))

    big = make_figures.largest_quartile(df, make_figures.problem_sizes(df))

    # Zeros for the four unsized problems would pull the cutoff down to
    # 2250 and admit small3; the cutoff is taken over known sizes only.
    assert sorted(big["problem"].unique()) == ["small4"]
    err = capsys.readouterr().err
    assert "huge1" in err and "huge4" in err


def test_problem_sizes_use_any_row_that_recorded_dimensions(make_figures):
    # A solver that returned a failure status still went through the
    # worker, which records the problem's dimensions whatever status the
    # solver reported.
    df = pd.DataFrame([
        {"dataset": "lp", "problem": "p", "solver_id": "a", "status": status.SOLVER_ERROR,
         "run_time_seconds": 1.0, "metadata.nnz_a": 40, "metadata.nnz_p": 2},
        {"dataset": "lp", "problem": "p", "solver_id": "b", "status": status.TIME_LIMIT,
         "run_time_seconds": 360.0},
        {"dataset": "cone", "problem": "q", "solver_id": "a", "status": status.OPTIMAL,
         "run_time_seconds": 1.0, "metadata.nnz_a": 7},
    ])

    def lookup(dataset, problem):  # pragma: no cover - must not be needed
        raise AssertionError(f"{dataset}/{problem} should be sized from its rows")

    sizes = make_figures.problem_sizes(df, size_lookup=lookup)

    assert sizes.to_dict() == {("cone", "q"): 7.0, ("lp", "p"): 42.0}


def test_registry_size_lookup_loads_problems_named_in_the_manifest(make_figures, tmp_path, repo_root):
    manifest = {"config": {"datasets": [{"id": "qp_alias", "name": "synthetic_qp"}]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    lookup = make_figures.registry_size_lookup(tmp_path, repo_root=repo_root)

    assert lookup("qp_alias", "one_variable_lp") == 1
    assert lookup("not_in_manifest", "one_variable_lp") is None


def test_make_figures_main_writes_figures_and_summary(make_figures, tmp_path, monkeypatch):
    run_dir = tmp_path / "lp_run"
    run_dir.mkdir()
    rows = _sized_rows(4, ())
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in rows.to_dict("records"):
            handle.write(json.dumps(record) + "\n")
    out_dir = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", str(out_dir), f"lp={run_dir}"])

    make_figures.main()

    for name in ("lp_1e-4_profile.png", "lp_1e-4_geomean.png", "lp_1e-4_profile_largest.png",
                 "geomean_summary.csv"):
        assert (out_dir / name).exists(), name
    summary = pd.read_csv(out_dir / "geomean_summary.csv")
    assert set(summary["subset"]) == {"all", "largest"}


def test_make_campaign_writes_configs_the_harness_can_load(make_campaign, tmp_path, monkeypatch):
    out_dir = tmp_path / "campaign"
    monkeypatch.setattr(sys, "argv", ["make_campaign.py", str(out_dir),
                                      "--families", "qp,lp,sdp,lpbig", "--tols", "1e-4"])

    make_campaign.main()

    shards = json.loads((out_dir / "campaign.json").read_text())
    assert shards
    for shard in shards:
        config = load_run_config(shard["config"])
        assert [solver.id for solver in config.solvers] == [shard["solver_id"]]
        assert config.parallelism == 1


def test_bench_modal_compiles():
    path = TOOLS / "bench_modal.py"
    compile(path.read_text(), str(path), "exec")


def _load_bench_modal(monkeypatch, **environ):
    """bench_modal with a stand-in for the ``modal`` package (images,
    volumes and decorators become mocks; plain functions stay real), loaded
    with only the image variables in ``environ`` set."""
    from unittest import mock

    for name in ("SCS_PYTHON_REF", "BENCH_MODAL_PINS", "QTQP_REPO", "CPO3_REPO"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "modal", mock.MagicMock(name="modal"))
    module = _load_tool("bench_modal")
    # The host record runs nvidia-smi, dpkg-query and git; here they would go
    # through the tests' fake Popen.
    module._command_output = lambda args: ""
    return module


@pytest.fixture
def bench_modal(monkeypatch):
    return _load_bench_modal(monkeypatch)


def _dockerignored(path: str, patterns: list[str]) -> bool:
    # Modal's ignore patterns follow .dockerignore: a pattern matches a path
    # or any of its parent directories; ``**`` matches any number of parts.
    from fnmatch import fnmatchcase

    parts = path.split("/")
    for pattern in patterns:
        pattern_parts = pattern.split("/")
        if pattern_parts[0] == "**":
            tail = pattern_parts[1:]
            windows = [parts[i:i + len(tail)] for i in range(len(parts) - len(tail) + 1)]
            if any(all(fnmatchcase(p, q) for p, q in zip(w, tail)) for w in windows):
                return True
            continue
        head = parts[:len(pattern_parts)]
        if len(head) == len(pattern_parts) and all(fnmatchcase(p, q) for p, q in zip(head, pattern_parts)):
            return True
    return False


def test_bench_modal_images_carry_the_reader_code_but_not_the_data(bench_modal):
    # The readers (problem_classes/*.py) must come from the checkout baked
    # into the image, not from whatever copy sits on the data volume.
    repo = TOOLS.parent
    for module in sorted((repo / "problem_classes").glob("*.py")):
        assert not _dockerignored(f"problem_classes/{module.name}", bench_modal.REPO_IGNORE), module.name
    folders = [path for path in (repo / "problem_classes").iterdir() if path.is_dir() and path.name != "__pycache__"]
    assert folders
    for folder in folders:
        for path in [folder, *sorted(folder.rglob("*"))[:5]]:
            relative = path.relative_to(repo).as_posix()
            assert _dockerignored(relative, bench_modal.REPO_IGNORE), relative


def test_bench_modal_runs_from_the_image_checkout_with_data_from_the_volume(bench_modal, tmp_path, monkeypatch):
    calls = []

    class FakeProcess:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            calls.append((cmd, kwargs))
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise bench_modal.subprocess.TimeoutExpired("bench", timeout)
            return 0

        def poll(self):
            self.waits += 1
            return None if self.waits == 1 else 0

    monkeypatch.setattr(bench_modal.subprocess, "Popen", FakeProcess)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(bench_modal, "Path", lambda p: tmp_path / str(p).lstrip("/"))
    commits = []
    monkeypatch.setattr(bench_modal.results_vol, "commit", lambda: commits.append(1))
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)

    result = bench_modal._run("camp", "lp_netlib_scs_cpu_1e-4", "run: {}\n")

    (cmd, kwargs), = calls
    options = dict(zip(cmd, cmd[1:]))
    assert options.get("--repo-root") == "/root/repo"
    assert options.get("--data-root") == "/data/problem_classes"
    assert kwargs["cwd"] == "/root/repo"
    assert "/data" not in kwargs["env"].get("PYTHONPATH", "").split(os.pathsep)
    # One commit while the shard ran, one at the end, and no fixed sleep.
    assert len(commits) == 2
    assert sleeps == []
    assert result["exit"] == 0


@pytest.mark.parametrize("exported_omp", [None, ""], ids=["unset", "empty"])
def test_bench_modal_thread_settings_reach_the_workers_unchanged(bench_modal, tmp_path, monkeypatch, exported_omp):
    """bench_modal sets the OpenMP, MKL, OpenBLAS and Rayon thread counts for
    the shard's `bench run` to the container's reserved cores; the runner then
    leaves the BLAS / OpenMP variables alone, so a worker sees exactly what
    bench_modal set (no OpenBLAS cap of 1 next to OpenMP's 4)."""
    from solver_benchmarks.core import runner

    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    for name in thread_vars:
        monkeypatch.delenv(name, raising=False)
    if exported_omp is not None:
        monkeypatch.setenv("OMP_NUM_THREADS", exported_omp)
    calls = []

    class FakeProcess:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            calls.append(kwargs)

        def wait(self, timeout=None):
            return 0

    real_popen = bench_modal.subprocess.Popen
    monkeypatch.setattr(bench_modal.subprocess, "Popen", FakeProcess)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(bench_modal, "Path", lambda p: tmp_path / str(p).lstrip("/"))
    monkeypatch.setattr(bench_modal.results_vol, "commit", lambda: None)
    bench_modal._run("camp", "shard", "run: {}\n")
    # The same subprocess module starts the worker below.
    monkeypatch.setattr(bench_modal.subprocess, "Popen", real_popen)
    (kwargs,) = calls
    shard_env = kwargs["env"]
    assert {name: shard_env[name] for name in bench_modal.THREAD_VARS} == dict.fromkeys(bench_modal.THREAD_VARS, "4")

    # The shard's runner, with that environment, starts a worker.
    for name, value in shard_env.items():
        monkeypatch.setenv(name, value)
    script = tmp_path / "show_threads.py"
    script.write_text(
        "import json, os\n"
        f"print(json.dumps({{name: os.environ.get(name) for name in {thread_vars!r}}}))\n"
    )
    result = runner._run_subprocess(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout=60.0,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        stream_output=False,
        env=runner.worker_thread_environ({"threads": 2}),
    )
    assert result.returncode == 0
    seen = json.loads((tmp_path / "stdout.log").read_text())
    assert seen == {name: shard_env.get(name) for name in thread_vars}
    assert seen["OMP_NUM_THREADS"] == "4" and seen["OPENBLAS_NUM_THREADS"] == "4"


def test_bench_modal_installs_the_highspy_the_pdlp_extra_allows(bench_modal):
    import tomllib

    extras = tomllib.loads((TOOLS.parent / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    (pdlp_highspy,) = [req for req in extras["pdlp"] if req.startswith("highspy")]
    assert bench_modal.HIGHSPY == pdlp_highspy
    source = (TOOLS / "bench_modal.py").read_text()
    assert source.count('"highspy') == 1, "every image should use HIGHSPY"


def _fake_shard_process(bench_modal, tmp_path, monkeypatch) -> list[dict]:
    """Run _run against tmp_path with a shard process that exits at once; returns its Popen kwargs."""
    calls = []

    class FakeProcess:
        returncode = 0

        def __init__(self, cmd, **kwargs):
            calls.append(kwargs)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(bench_modal.subprocess, "Popen", FakeProcess)
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(bench_modal, "Path", lambda p: tmp_path / str(p).lstrip("/"))
    monkeypatch.setattr(bench_modal.results_vol, "commit", lambda: None)
    return calls


@pytest.mark.parametrize(("kind", "threads", "expected"), [("cpu", 0, "4"), ("gpu", 0, "8"), ("gpu", 4, "4")])
def test_bench_modal_threads_are_the_flag_or_the_reserved_cores(bench_modal, tmp_path, monkeypatch, kind,
                                                                threads, expected):
    calls = _fake_shard_process(bench_modal, tmp_path, monkeypatch)
    controls = json.dumps({"threads": threads, "require_cpu": "", "require_gpu": ""})

    result = bench_modal._run_kind(kind, "camp", "shard", "run: {}\n", controls)

    assert result["exit"] == 0
    (kwargs,) = calls
    assert {name: kwargs["env"][name] for name in bench_modal.THREAD_VARS} == dict.fromkeys(
        bench_modal.THREAD_VARS, expected)
    # host.txt: one block per container, appended, in the lines the campaign checks read.
    host = (tmp_path / "results" / "camp" / "shard" / "host.txt").read_text().splitlines()
    assert host[0].startswith("=== container start ") and " rows_before=0 host=" in host[0]
    assert f"threads {expected}" in host
    assert any(line.startswith("scs-python ") and " scs_source " in line for line in host)
    assert host[-1].startswith("=== container end ") and host[-1].endswith("exit=0 rows_after=0")


def test_bench_modal_wrong_host_does_nothing_and_stops_taking_inputs(bench_modal, monkeypatch):
    from unittest import mock

    def no_process(*args, **kwargs):
        raise AssertionError("a wrong-host container must not start the shard")

    monkeypatch.setattr(bench_modal.subprocess, "Popen", no_process)
    monkeypatch.setitem(sys.modules, "modal.experimental", mock.MagicMock())
    monkeypatch.setattr(bench_modal, "_cpu_family_model", lambda: "25/17")
    monkeypatch.setattr(bench_modal, "_gpu_name", lambda: "NVIDIA A100 80GB PCIe")
    monkeypatch.setenv("MODAL_TASK_ID", "ta-1")

    result = bench_modal._run("camp", "shard", "run: {}\n", require_cpu="25/1")

    assert result == {"name": "shard", "exit": 75, "rows": 0, "seconds": 0.0, "wrong_host": "25/17", "task": "ta-1"}
    sys.modules["modal"].experimental.stop_fetching_inputs.assert_called_once()
    assert bench_modal._host_ok("", "") == (True, "25/17")
    assert bench_modal._host_ok("25/17", "") == (True, "25/17")
    assert bench_modal._host_ok("", "NVIDIA A100 80GB PCIe|25/17") == (True, "NVIDIA A100 80GB PCIe|25/17")
    assert bench_modal._host_ok("", "NVIDIA A100 80GB PCIe")[0]
    assert not bench_modal._host_ok("", "NVIDIA A100-SXM4-80GB|25/17")[0]
    assert not bench_modal._host_ok("", "NVIDIA A100 80GB PCIe|25/1")[0]


def test_bench_modal_caps_stay_within_the_approved_ceilings(bench_modal):
    caps = bench_modal.HARD_CAPS
    assert caps["cpu"] <= 32 and caps["cpu_big"] <= 16
    assert bench_modal.GPU_LIMIT <= 10 and all(caps[k] <= bench_modal.GPU_LIMIT for k in bench_modal.GPU_KINDS)
    assert set(caps) == set(bench_modal.MAX_IN_FLIGHT) == set(bench_modal.CPUS) == set(bench_modal.RATES_PER_HOUR)
    assert bench_modal.CPU_KINDS | bench_modal.GPU_KINDS == set(caps)
    assert all(bench_modal.MAX_IN_FLIGHT[k] <= caps[k] for k in caps)
    # Each shard function's max_containers is its kind's ceiling.
    limits = sorted(c.kwargs["max_containers"] for c in bench_modal.app.function.call_args_list
                    if "max_containers" in c.kwargs)
    assert limits == sorted(caps.values())

    assert bench_modal._parse_max_in_flight("") == bench_modal.MAX_IN_FLIGHT
    raised = bench_modal._parse_max_in_flight(" cpu=32, cpu_big=16,gpu=9 ")
    assert raised == {**bench_modal.MAX_IN_FLIGHT, "cpu": 32, "cpu_big": 16, "gpu": 9}
    for bad in ("cpu=33", "cpu_big=17", "gpu=11", "gpu=0", "tpu=1", "cpu", "cpu=x"):
        with pytest.raises(SystemExit):
            bench_modal._parse_max_in_flight(bad)


def test_bench_modal_shards_get_their_kind_and_run_controls(bench_modal):
    def shards():
        return [
            {"name": "lp_scs_cpu", "gpu": False},
            {"name": "lp_scs_gpu", "gpu": True},
            {"name": "lp_pdlp", "gpu": False, "runner": "pdlp"},
            {"name": "lp_cuopt", "gpu": True, "runner": "cuopt", "require_gpu": ""},
            {"name": "sdp_big", "gpu": False, "require_cpu": "25/17"},
        ]

    out = {s["name"]: s for s in bench_modal._prepare_shards(
        shards(), big="sdp_big", threads=4, require_cpu="25/1", require_gpu="NVIDIA A100-SXM4-80GB|25/1")}

    assert {n: s["kind"] for n, s in out.items()} == {
        "lp_scs_cpu": "cpu", "lp_scs_gpu": "gpu", "lp_pdlp": "pdlp", "lp_cuopt": "cuopt", "sdp_big": "cpu_big"}
    controls = {n: (s["controls"]["require_cpu"], s["controls"]["require_gpu"]) for n, s in out.items()}
    assert controls == {
        "lp_scs_cpu": ("25/1", ""),
        "lp_scs_gpu": ("", "NVIDIA A100-SXM4-80GB|25/1"),
        "lp_pdlp": ("25/1", ""),
        "lp_cuopt": ("", ""),  # the shard's own (empty) requirement wins
        "sdp_big": ("25/17", ""),
    }
    assert {s["controls"]["threads"] for s in out.values()} == {4}
    with pytest.raises(SystemExit):
        bench_modal._prepare_shards([{"name": "s", "gpu": False, "require_gpu": "NVIDIA A100-SXM4-80GB"}])
    with pytest.raises(SystemExit):
        bench_modal._prepare_shards(shards(), threads=-1)
    with pytest.raises(SystemExit):
        bench_modal._prepare_shards([{"name": "s", "gpu": False, "runner": "tpu"}])
    pinned = [{"name": "s", "gpu": False, "scs_python_ref": "13a25e84"}]
    with pytest.raises(SystemExit):
        bench_modal._prepare_shards(pinned)
    with pytest.raises(SystemExit):
        bench_modal._prepare_shards(pinned, scs_python_ref="91df3fcd59757ba65e416cdf820880b4026415d7")
    assert bench_modal._prepare_shards(pinned, scs_python_ref="13a25e84830de2549792df16161532624f576920")


class _FakeCall:
    def __init__(self, results, active, peak):
        self.results, self.active, self.peak = results, active, peak
        self.polls = 0

    def get(self, timeout=None):
        self.polls += 1
        if self.polls == 1:
            raise TimeoutError
        self.active.discard(self)
        return self.results.pop(0)

    def cancel(self):
        self.active.discard(self)


class _FakeRunner:
    """A runner kind whose calls finish on their second poll with the next scripted result for the shard."""

    def __init__(self, scripts, active, peak):
        self.scripts, self.active, self.peak = scripts, active, peak
        self.spawned = []

    def spawn(self, campaign, name, config_yaml, controls):
        self.spawned.append((campaign, name, json.loads(controls)))
        script = self.scripts.get(name) or [{"name": name, "exit": 0, "rows": 1}]
        call = _FakeCall([script.pop(0) if len(script) > 1 else script[0]], self.active, self.peak)
        self.active.add(call)
        self.peak[0] = max(self.peak[0], len(self.active))
        return call


def _orchestrate(bench_modal, tmp_path, monkeypatch, shards, scripts, **kwargs):
    monkeypatch.setattr(bench_modal.time, "sleep", lambda seconds: None)
    for shard in shards:
        shard.setdefault("config", str(tmp_path / f"{shard['name']}.yaml"))
        Path(shard["config"]).write_text("run: {}\n")
    active, peak = set(), [0]
    runners = {kind: _FakeRunner(scripts, active, peak) for kind in bench_modal.CPUS}
    kwargs = {"campaign": "camp", "budget_usd": 1e6, "caps": bench_modal.MAX_IN_FLIGHT, **kwargs}
    summary = bench_modal._orchestrate(shards, runners, **kwargs)
    return summary, runners, peak[0]


def test_bench_modal_wrong_host_shards_are_queued_again_per_distinct_container(bench_modal, tmp_path, monkeypatch):
    def wrong(name, task):
        return {"name": name, "exit": 75, "rows": 0, "wrong_host": "25/17", "task": task}

    scripts = {
        # one container rejects the shard twice (one try), a second once, then a Milan host runs it
        "a": [wrong("a", "t1"), wrong("a", "t1"), wrong("a", "t2"), {"name": "a", "exit": 0, "rows": 3}],
        "b": [wrong("b", f"u{i}") for i in range(10)],
    }
    shards = bench_modal._prepare_shards([{"name": "a", "gpu": False}, {"name": "b", "gpu": False, "campaign": "other"}],
                                         require_cpu="25/1")

    summary, runners, _ = _orchestrate(bench_modal, tmp_path, monkeypatch, shards, scripts, max_host_tries=3)

    assert summary == {"done": 1, "failed": 1, "pending": 0, "spent": pytest.approx(summary["spent"])}
    spawned = runners["cpu"].spawned
    assert [n for _, n, _ in spawned].count("a") == 4
    assert [n for _, n, _ in spawned].count("b") == 3  # gives up after three distinct containers
    assert {c for c, n, _ in spawned if n == "b"} == {"other"}  # a shard's own campaign
    assert all(controls["require_cpu"] == "25/1" for _, _, controls in spawned)


def test_bench_modal_a_container_that_keeps_taking_the_shard_cannot_loop(bench_modal, tmp_path, monkeypatch):
    # The same container rejecting the requeued shard over and over adds no try;
    # the shard still ends after WRONG_HOST_RESULTS_PER_TRY * max_host_tries results.
    wrong = {"name": "a", "exit": 75, "rows": 0, "wrong_host": "25/17", "task": "stuck"}
    shards = bench_modal._prepare_shards([{"name": "a", "gpu": False}], require_cpu="25/1")

    summary, runners, _ = _orchestrate(bench_modal, tmp_path, monkeypatch, shards, {"a": [wrong] * 1000},
                                       max_host_tries=2)

    assert summary["failed"] == 1 and summary["done"] == 0
    assert len(runners["cpu"].spawned) == bench_modal.WRONG_HOST_RESULTS_PER_TRY * 2


@pytest.mark.parametrize("shard, require_cpu, require_gpu", [
    ({"name": "s", "gpu": False}, "zen3", ""),
    ({"name": "s", "gpu": False, "require_cpu": "25-1"}, "", ""),
    ({"name": "s", "gpu": True}, "", "NVIDIA A100-SXM4-80GB|milan"),
    ({"name": "s", "gpu": True}, "", "|25/1"),
    ({"name": "s", "gpu": True, "require_cpu": "25/1"}, "", "NVIDIA A100-SXM4-80GB"),
])
def test_bench_modal_refuses_malformed_host_requirements(bench_modal, shard, require_cpu, require_gpu):
    with pytest.raises(SystemExit):
        bench_modal._prepare_shards([shard], require_cpu=require_cpu, require_gpu=require_gpu)


def test_bench_modal_gpu_kinds_together_stay_within_the_gpu_limit(bench_modal, tmp_path, monkeypatch):
    shards = bench_modal._prepare_shards(
        [{"name": f"g{i}", "gpu": True} for i in range(8)]
        + [{"name": f"c{i}", "gpu": True, "runner": "cuopt"} for i in range(8)])
    caps = bench_modal._parse_max_in_flight("gpu=10,cuopt=10")

    summary, _, peak = _orchestrate(bench_modal, tmp_path, monkeypatch, shards, {}, caps=caps)

    assert summary["done"] == 16 and summary["failed"] == 0
    assert peak == bench_modal.GPU_LIMIT


def _write_pins(tmp_path, data) -> str:
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_bench_modal_pins_apply_by_package_name(monkeypatch, tmp_path):
    pins = {"pins": {"cpu": {"osqp": "1.1.3", "HighsPy": "1.15.1", "numpy": "2.5.3"},
                     "qtqp_gpu": {"nvmath_python": "<0.9", "qtqp": "0.0.7"}},
            "scs_python_ref": "13a25e84830de2549792df16161532624f576920"}
    bench_modal = _load_bench_modal(monkeypatch, BENCH_MODAL_PINS=_write_pins(tmp_path, pins))

    assert bench_modal.SCS_PYTHON_REF == "13a25e84830de2549792df16161532624f576920"
    assert bench_modal._pinned("cpu", ["osqp", bench_modal.HIGHSPY, "numpy", "scipy"]) == [
        "osqp==1.1.3", "highspy==1.15.1", "numpy==2.5.3", "scipy"]
    assert bench_modal._pinned("qtqp_gpu", ["nvmath-python[cu12]", "qtqp>=0.0.7"]) == [
        "nvmath-python[cu12]<0.9", "qtqp==0.0.7"]
    assert bench_modal._pinned("pdlp", ["osqp"]) == ["osqp"]


@pytest.mark.parametrize("data", [
    {"pins": {"cpu": {"no-such-solver": "1.0"}}},  # no image installs it
    {"pins": {"cuopt": {"osqp": "1.1.3"}}},  # not in that image
    {"pins": {"tpu": {"osqp": "1.1.3"}}},
    {"pins": {"cpu": {"osqp": ""}}},
    {"pin": {}},
    {"scs_python_ref": "91df3fcd59757ba65e416cdf820880b4026415d7"},  # differs from SCS_PYTHON_REF below
])
def test_bench_modal_refuses_pins_it_cannot_apply(monkeypatch, tmp_path, data):
    with pytest.raises(SystemExit):
        _load_bench_modal(monkeypatch, BENCH_MODAL_PINS=_write_pins(tmp_path, data), SCS_PYTHON_REF="13a25e84830de2549792df16161532624f576920")


@pytest.mark.parametrize("ref", ["13a25e84", "master", "13A25E84830DE2549792DF16161532624F576920"])
def test_bench_modal_refuses_an_scs_python_ref_that_is_not_a_full_sha(monkeypatch, tmp_path, ref):
    # git fetch origin <short sha> fails in the image build, and a branch name
    # would let Modal's layer cache reuse an old build.
    with pytest.raises(SystemExit, match="full 40-character"):
        _load_bench_modal(monkeypatch, SCS_PYTHON_REF=ref)
    with pytest.raises(SystemExit, match="full 40-character"):
        _load_bench_modal(monkeypatch, BENCH_MODAL_PINS=_write_pins(tmp_path, {"scs_python_ref": ref}))


def test_bench_modal_refuses_an_scs_pin_when_scs_is_built_from_source(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="scs pin"):
        _load_bench_modal(monkeypatch, BENCH_MODAL_PINS=_write_pins(tmp_path, {"pins": {"cpu": {"scs": "3.3.2"}}}),
                          SCS_PYTHON_REF="13a25e84830de2549792df16161532624f576920")
    # The cuOpt image installs PyPI's scs whatever the ref is, so a pin there applies.
    bench_modal = _load_bench_modal(monkeypatch, SCS_PYTHON_REF="13a25e84830de2549792df16161532624f576920",
                                    BENCH_MODAL_PINS=_write_pins(tmp_path, {"pins": {"cuopt": {"scs": "3.3.2"}}}))
    assert bench_modal.PINS == {"cuopt": {"scs": "3.3.2"}}


def test_bench_modal_refuses_a_local_checkout_that_is_not_there(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="CPO3_REPO"):
        _load_bench_modal(monkeypatch, CPO3_REPO=str(tmp_path / "nonexistent"))


@pytest.mark.parametrize("pins_file", sorted((TOOLS / "modal_pins").glob("*.json")), ids=lambda p: p.stem)
def test_bench_modal_pin_files_apply_to_the_images(monkeypatch, pins_file):
    bench_modal = _load_bench_modal(monkeypatch, BENCH_MODAL_PINS=str(pins_file))
    assert bench_modal.PINS


def test_make_figures_main_survives_a_family_with_no_sized_problem(make_figures, tmp_path, monkeypatch):
    run_dir = tmp_path / "qp_run"
    run_dir.mkdir()
    rows = _sized_rows(0, ("huge1", "huge2"))
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in rows.to_dict("records"):
            handle.write(json.dumps(record) + "\n")
    out_dir = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", str(out_dir), f"qp={run_dir}"])

    make_figures.main()

    assert (out_dir / "qp_1e-4_profile.png").exists()
    assert not (out_dir / "qp_1e-4_profile_largest.png").exists()


@pytest.mark.parametrize("skip_lookup", [False, True], ids=["lookup", "no-size-lookup"])
def test_make_figures_no_size_lookup_leaves_unsized_problems_out(make_figures, tmp_path, monkeypatch, capsys,
                                                                 skip_lookup):
    # Loading a problem to size it can mean parsing a multi-GB MPS file (the
    # Mittelmann LP set); --no-size-lookup skips that and leaves such problems
    # out of the largest quartile, naming them on stderr.
    run_dir = tmp_path / "lpbig_run"
    run_dir.mkdir()
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in _sized_rows(4, ("huge1",)).to_dict("records"):
            handle.write(json.dumps(record) + "\n")
    looked_up = []

    def fake_registry_size_lookup(run_dir, repo_root=None):
        def lookup(dataset, problem):
            looked_up.append((dataset, problem))
            return 10_000_000
        return lookup

    monkeypatch.setattr(make_figures, "registry_size_lookup", fake_registry_size_lookup)
    out_dir = tmp_path / "figs"
    argv = ["make_figures.py", str(out_dir), f"lp={run_dir}"] + (["--no-size-lookup"] if skip_lookup else [])
    monkeypatch.setattr(sys, "argv", argv)

    make_figures.main()

    captured = capsys.readouterr()
    if skip_lookup:
        assert looked_up == []
        assert "huge1" in captured.err
        # small4 alone: the cutoff is taken over small1..small4.
        assert "lp 1e-4: 5 problems, largest quartile 1" in captured.out
    else:
        assert looked_up == [("lp", "huge1")]
        assert "lp 1e-4: 5 problems, largest quartile 2" in captured.out  # huge1 and small4


def _load_test_module(name: str):
    spec = importlib.util.spec_from_file_location(f"_tests_{name}", Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True


# Modules behind the campaign's platform-specific linear-solver backends.
PLATFORM_BACKEND_MODULES = {("qtqp", "pardiso"): "pymklpardiso"}


def _write_run(run_dir: Path, rows: list[dict]) -> Path:
    run_dir.mkdir(parents=True)
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in rows:
            handle.write(json.dumps(record) + "\n")
    return run_dir


def _sized(row: dict) -> dict:
    return {"metadata.nnz_a": 10, "metadata.nnz_p": 0, **row}


def test_make_campaign_families_are_the_families_make_figures_plots(make_campaign, make_figures):
    assert set(make_campaign.FAMILIES) == set(make_figures.FAMILY_LIMIT)
    # The lp family's figures leave out the Mittelmann set (it is lpbig's),
    # so the lp family must not spend shards on it.
    for family, dropped in make_figures.FAMILY_DROP_DATASETS.items():
        ids = {dataset["id"] for dataset in make_campaign.FAMILIES[family]}
        assert not ids & dropped, family
    infeas = make_campaign.FAMILIES["infeas"]
    assert [dataset["dataset_options"] for dataset in infeas] == [{"subset": "infeasible"}]


def test_make_campaign_limits_match_the_figures_limits(make_campaign, make_figures, tmp_path, monkeypatch):
    out_dir = tmp_path / "campaign"
    families = ",".join(make_campaign.FAMILIES)
    monkeypatch.setattr(sys, "argv", ["make_campaign.py", str(out_dir), "--families", families, "--tols", "1e-4"])

    make_campaign.main()

    for shard in json.loads((out_dir / "campaign.json").read_text()):
        config = load_run_config(shard["config"])
        assert config.timeout_seconds - 60.0 == make_figures.FAMILY_LIMIT[shard["family"]], shard["name"]


@pytest.mark.parametrize("tol", [1e-4, 1e-6, 5e-5, 1.5e-5])
def test_make_figures_reads_the_tolerance_tag_of_every_campaign_solver_id(make_campaign, make_figures, tol):
    for family in make_campaign.FAMILIES:
        for variant in make_campaign.solver_variants(family, tol, 300.0):
            tag = make_figures.solver_tol(variant["id"])
            assert tag is not None and float(tag) == tol, variant["id"]
            base = make_figures.base_solver(variant["id"])
            assert base in make_figures.SOLVER_LABELS, variant["id"]


def test_make_figures_fails_loudly_on_rows_without_a_tolerance_tag(make_figures, tmp_path, monkeypatch, capsys):
    bad = _write_run(tmp_path / "bad", [
        _sized({"dataset": "lp", "problem": "a", "solver_id": "clarabel", "status": status.OPTIMAL,
                "run_time_seconds": 1.0}),
    ])
    good = _write_run(tmp_path / "good", [
        _sized({"dataset": "qp", "problem": "a", "solver_id": "clarabel_1e-4", "status": status.OPTIMAL,
                "run_time_seconds": 1.0}),
    ])
    out_dir = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", str(out_dir), f"lp={bad}", f"qp={good}",
                                      "--no-size-lookup"])

    with pytest.raises(SystemExit) as excinfo:
        make_figures.main()

    assert excinfo.value.code not in (0, None)
    assert "tolerance tag" in str(excinfo.value.code) and "clarabel" in str(excinfo.value.code)
    # The families that could be plotted are still written.
    summary = pd.read_csv(out_dir / "geomean_summary.csv")
    assert set(summary["family"]) == {"qp"}


def test_make_figures_names_untagged_solver_ids_it_leaves_out(make_figures, tmp_path, monkeypatch, capsys):
    run_dir = _write_run(tmp_path / "run", [
        _sized({"dataset": "lp", "problem": "a", "solver_id": "clarabel_1e-4", "status": status.OPTIMAL,
                "run_time_seconds": 1.0}),
        _sized({"dataset": "lp", "problem": "a", "solver_id": "scs_default", "status": status.OPTIMAL,
                "run_time_seconds": 1.0}),
    ])
    out_dir = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", str(out_dir), f"lp={run_dir}", "--no-size-lookup"])

    make_figures.main()

    assert "scs_default" in capsys.readouterr().err
    summary = pd.read_csv(out_dir / "geomean_summary.csv")
    assert set(summary["solver_id"]) == {"clarabel_1e-4"}


def test_make_figures_skips_a_family_whose_rows_are_all_filtered_out(make_figures, tmp_path, monkeypatch, capsys):
    # An old lp-family archive that holds only the Mittelmann shard, which
    # the lp family's figures drop.
    run_dir = _write_run(tmp_path / "run", [
        _sized({"dataset": "mittelmann", "problem": "qap15", "solver_id": "clarabel_1e-4",
                "status": status.OPTIMAL, "run_time_seconds": 1.0}),
    ])
    out_dir = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", str(out_dir), f"lp={run_dir}", "--no-size-lookup"])

    make_figures.main()

    assert "skipped" in capsys.readouterr().out
    assert not (out_dir / "geomean_summary.csv").exists()


def test_make_figures_counts_a_solve_past_the_nominal_limit_as_a_failure(make_figures, tmp_path, monkeypatch):
    # PIQP ignores its time limit and runs until the harness kills it at the
    # limit + 60 s; Clarabel stops itself at the limit. A PIQP solve that ends
    # 30 s past the limit must not be scored as a success.
    limit = make_figures.FAMILY_LIMIT["qp"]
    run_dir = _write_run(tmp_path / "run", [
        _sized({"dataset": "qp", "problem": "a", "solver_id": "piqp_1e-4", "status": status.OPTIMAL,
                "run_time_seconds": limit + 30.0}),
        _sized({"dataset": "qp", "problem": "a", "solver_id": "clarabel_1e-4", "status": status.TIME_LIMIT,
                "run_time_seconds": limit}),
        _sized({"dataset": "qp", "problem": "b", "solver_id": "piqp_1e-4", "status": status.OPTIMAL,
                "run_time_seconds": 1.0}),
        _sized({"dataset": "qp", "problem": "b", "solver_id": "clarabel_1e-4", "status": status.OPTIMAL,
                "run_time_seconds": 1.0}),
    ])
    out_dir = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", str(out_dir), f"qp={run_dir}", "--no-size-lookup"])

    make_figures.main()

    summary = pd.read_csv(out_dir / "geomean_summary.csv")
    solved = summary[summary["subset"] == "all"].set_index("solver_id")["success_count"].to_dict()
    assert solved == {"piqp_1e-4": 1, "clarabel_1e-4": 1}


def test_make_figures_use_run_marks_the_substituted_run(make_figures, tmp_path, monkeypatch):
    rows = []
    for problem in ("a", "b"):
        rows += [
            _sized({"dataset": "lp", "problem": problem, "solver_id": "clarabel_1e-4", "status": status.OPTIMAL,
                    "run_time_seconds": 10.0}),
            _sized({"dataset": "lp", "problem": problem, "solver_id": "clarabel_1e-6", "status": status.OPTIMAL,
                    "run_time_seconds": 1.0}),
            _sized({"dataset": "lp", "problem": problem, "solver_id": "highs_1e-4", "status": status.OPTIMAL,
                    "run_time_seconds": 2.0}),
        ]
    run_dir = _write_run(tmp_path / "run", rows)
    out_dir = tmp_path / "figs"
    monkeypatch.setattr(sys, "argv", ["make_figures.py", str(out_dir), f"lp={run_dir}", "--tol-tag", "1e-4",
                                      "--use-run", "clarabel=1e-6", "--no-size-lookup"])

    make_figures.main()

    summary = pd.read_csv(out_dir / "geomean_summary.csv", dtype={"source_run": str})
    rows = summary[summary["subset"] == "all"].set_index("solver_id")
    assert rows.loc["clarabel_1e-4", "label"] == "Clarabel (1e-6)"
    assert rows.loc["clarabel_1e-4", "source_run"] == "1e-6"
    assert rows.loc["clarabel_1e-4", "run_time_seconds"] == pytest.approx(1.0)
    assert rows.loc["highs_1e-4", "label"] == "HiGHS"
    assert rows.loc["highs_1e-4", "source_run"] == "1e-4"


@pytest.mark.parametrize("family", ["qp", "lp", "sdp", "lpbig", "infeas"])
def test_campaign_solver_settings_are_accepted_by_the_installed_adapters(make_campaign, repo_root, tmp_path,
                                                                          family):
    # load_run_config only parses the YAML; this runs each campaign variant
    # (CPU ones, on solvers installed here) so the adapters accept its settings.
    from solver_benchmarks.datasets import get_dataset
    from solver_benchmarks.solvers import get_solver

    if family == "sdp":
        problem = get_dataset("synthetic_cone")(repo_root=repo_root).load_problem("one_variable_cone_lp")
    else:
        name = "one_variable_eq" if family == "qp" else "one_variable_lp"
        problem = get_dataset("synthetic_qp")(repo_root=repo_root).load_problem(name)
    portable = _load_test_module("test_bundled_configs").PORTABLE_BACKENDS
    ran = []
    for variant in make_campaign.solver_variants(family, 1e-4, 300.0):
        solver_cls = get_solver(variant["solver"])
        if variant["gpu"] or not solver_cls.is_available():
            continue
        backend = variant["settings"].get("linear_solver")
        if backend not in portable.get(variant["solver"], {backend}):
            # A backend the extras install on some platforms only (QTQP's
            # pardiso through py-mkl-pardiso): run it where it is installed.
            module = PLATFORM_BACKEND_MODULES.get((variant["solver"], backend))
            if module is None or not _importable(module):
                continue
        out = tmp_path / variant["id"]
        out.mkdir()
        result = solver_cls(dict(variant["settings"])).solve(problem, out)
        assert result.status == status.OPTIMAL, (variant["id"], result.status, result.info)
        ran.append(variant["id"])
    if not ran:
        pytest.skip(f"no CPU solver of the {family} campaign is installed")
