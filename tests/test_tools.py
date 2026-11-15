"""Tests for the campaign scripts under ``tools/``.

``tools/`` is not a package and nothing in the harness imports it, so these
tests load the scripts by path. They pin the parts that must agree with the
harness: the figures score outcomes exactly as ``solver_benchmarks.analysis``
does, the "largest quartile" is sized from problem metadata rather than from
whichever rows happen to carry it, and every campaign config the generator
writes is one the harness can load. ``tools/bench_modal.py`` imports Modal, so
it is only compiled.
"""

from __future__ import annotations

import importlib.util
import json
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
    bars = shifted_geomean(make_figures.with_missing_as_failures(df), metric="run_time_seconds",
                           max_value=make_figures.failure_penalty("infeas"))
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
