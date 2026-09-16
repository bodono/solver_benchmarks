import json
import math

import pandas as pd
import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.markdown_report import write_run_report
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.tables import completion_summary, expected_results, missing_results
from solver_benchmarks.cli import main


def test_geomean_charges_solver_for_unattempted_observed_problem():
    frame = pd.DataFrame([
        {"problem": "easy", "solver_id": "a", "status": "optimal", "run_time_seconds": 1.0},
        {"problem": "hard", "solver_id": "a", "status": "optimal", "run_time_seconds": 100.0},
        {"problem": "easy", "solver_id": "b", "status": "optimal", "run_time_seconds": 1.0},
    ])
    gm = shifted_geomean(frame).set_index("solver_id")
    assert gm.loc["b", "run_time_seconds"] == pytest.approx(math.sqrt(11 * 1010) - 10)
    assert gm.loc["b", "failure_count"] == 1
    assert gm.loc["a", "failure_count"] == 0
    success_only = shifted_geomean(frame, penalize_failures=False).set_index("solver_id")
    assert success_only.loc["b", "run_time_seconds"] == pytest.approx(1)
    assert success_only.loc["b", "failure_count"] == 1


def test_explicit_universe_covers_completely_absent_problems_and_solvers():
    expected = pd.DataFrame([
        {"dataset": d, "problem": "same-name", "solver_id": s}
        for d in ["first", "second"] for s in ["a", "b"]
    ])
    observed = pd.DataFrame([
        {"dataset": "first", "problem": "same-name", "solver_id": "a", "status": "optimal", "run_time_seconds": 1.0},
        {"dataset": "first", "problem": "same-name", "solver_id": "a", "status": "worker_error", "run_time_seconds": 0.1},
    ])
    gm = shifted_geomean(observed, expected=expected).set_index("solver_id")
    assert gm.loc["a", "success_count"] == 1
    assert gm.loc["a", "failure_count"] == 1
    assert gm.loc["b", "failure_count"] == 2
    assert gm.loc["b", "run_time_seconds"] == pytest.approx(1000)
    profile = performance_profile(observed, expected=expected, n_tau=3)
    assert (profile["a"] == 0.5).all()
    assert (profile["b"] == 0).all()
    empty = shifted_geomean(pd.DataFrame(), expected=expected)
    assert empty["failure_count"].tolist() == [2, 2]
    assert empty["run_time_seconds"].tolist() == pytest.approx([1000, 1000])


def _incomplete_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = {
        "datasets": [
            {"name": "synthetic_qp", "id": "first", "include": ["one_variable_eq", "one_variable_lp", "excluded"], "exclude": ["excluded"]},
            {"name": "synthetic_qp", "id": "second", "include": ["one_variable_eq"]},
        ],
        "solvers": [{"id": s, "solver": "scs", "settings": {}} for s in ["a", "b"]],
    }
    (run / "manifest.json").write_text(json.dumps({"config": config}))
    observed = {"dataset": "first", "problem": "one_variable_eq", "solver_id": "a", "status": "optimal", "run_time_seconds": 1.0}
    (run / "results.jsonl").write_text(json.dumps(observed) + "\n")
    return run


def test_run_commands_and_report_charge_completely_missing_solves(tmp_path):
    run = _incomplete_run(tmp_path)
    expected = expected_results(run)
    assert len(expected) == 6
    assert "excluded" not in set(expected["problem"])
    cli = CliRunner()
    result = cli.invoke(main, ["geomean", str(run), "--repo-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    gm = pd.read_csv(run / "shifted_geomean_run_time_seconds.csv").set_index("solver_id")
    assert gm.loc["a", "failure_count"] == 2
    assert gm.loc["b", "failure_count"] == 3
    result = cli.invoke(main, ["profile", str(run), "--repo-root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    prof = pd.read_csv(run / "performance_profile_run_time_seconds.csv")
    assert prof["a"].iloc[-1] == pytest.approx(1 / 3)
    assert (prof["b"] == 0).all()
    write_run_report(run, repo_root=tmp_path)
    report = run / "report"
    report_gm = pd.read_csv(report / "shifted_geomean_run_time_seconds.csv").set_index("solver_id")
    pd.testing.assert_frame_equal(gm.sort_index(), report_gm.sort_index())
    headline = pd.read_csv(report / "headline_solver_metrics.csv").set_index("solver_id")
    assert headline.loc["b", "completed"] == 0
    assert headline.loc["b", "penalized_shifted_geomean_run_time_seconds"] == pytest.approx(1000)
    for name, penalty in [("first", math.sqrt(11 * 1010) - 10), ("second", 1000)]:
        subset = pd.read_csv(report / "by_dataset" / name / "headline_solver_metrics.csv").set_index("solver_id")
        assert subset.loc["a", "penalized_shifted_geomean_run_time_seconds"] == pytest.approx(penalty)
        assert subset.loc["b", "completed"] == 0
    # Actual stored rows/completion are not inflated by comparison padding.
    completion = pd.read_csv(report / "completion.csv")
    assert completion["completed"].sum() == 1
    assert len((run / "results.jsonl").read_text().splitlines()) == 1
    assert (report / "shifted_geomean_run_time_seconds.png").is_file()


def test_recorded_solver_outside_manifest_uses_same_problem_universe():
    expected = pd.DataFrame([
        {"problem": "easy", "solver_id": "a"},
        {"problem": "hard", "solver_id": "a"},
    ])
    observed = pd.DataFrame([
        {"problem": "easy", "solver_id": "a", "status": "optimal", "run_time_seconds": 1.0},
        {"problem": "hard", "solver_id": "a", "status": "optimal", "run_time_seconds": 100.0},
        {"problem": "easy", "solver_id": "previous-solver", "status": "optimal", "run_time_seconds": 1.0},
    ])
    gm = shifted_geomean(observed, expected=expected).set_index("solver_id")
    assert gm.loc["previous-solver", "failure_count"] == 1
    assert gm.loc["previous-solver", "run_time_seconds"] == pytest.approx(math.sqrt(11 * 1010) - 10)


@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_valid_successful_retry_beats_invalid_success_metric(invalid):
    observed = pd.DataFrame([
        {"problem": "p", "solver_id": "a", "status": "optimal", "run_time_seconds": invalid},
        {"problem": "p", "solver_id": "a", "status": "optimal", "run_time_seconds": 2.0},
        {"problem": "p", "solver_id": "b", "status": "optimal", "run_time_seconds": 4.0},
    ])
    gm = shifted_geomean(observed).set_index("solver_id")
    assert gm.loc["a", "run_time_seconds"] == pytest.approx(2)
    assert gm.loc["a", "success_count"] == 1
    assert (performance_profile(observed, n_tau=3)["a"] == 1).all()


@pytest.mark.parametrize("metric", ["run_time_seconds", "iterations", "kkt.primal_res_rel"])
@pytest.mark.parametrize("penalty", [None, 1.0])
def test_geomean_failure_penalty_is_at_least_every_success(metric, penalty):
    # Exceed every built-in default, exercising the same invariant in all units.
    frame = pd.DataFrame([
        {"problem": p, "solver_id": "all", "status": "optimal", metric: 2e6}
        for p in ("p", "q")
    ] + [
        {"problem": "p", "solver_id": "some", "status": "optimal", metric: 1.5e6},
        {"problem": "q", "solver_id": "some", "status": "time_limit", metric: 1},
    ])
    # Missing "none" is charged on both problems even without recorded attempts.
    expected = pd.DataFrame([{"problem": p, "solver_id": "none"} for p in ("p", "q")])
    gm = shifted_geomean(frame, metric=metric, max_value=penalty, expected=expected).set_index("solver_id")
    assert (gm["max_value"] == 2e6).all()
    assert gm.loc["none", metric] == pytest.approx(2e6)
    assert gm.loc["all", metric] <= gm.loc["none", metric]
    assert gm.loc["some", metric] < gm.loc["none", metric]
    assert gm.loc["none", "failure_count"] == 2
    explicit = shifted_geomean(frame, metric=metric, max_value=3e6, expected=expected).set_index("solver_id")
    assert (explicit["max_value"] == 3e6).all()
    assert explicit.loc["none", metric] == pytest.approx(3e6)


def test_expected_includes_are_intersected_with_available_listing(tmp_path):
    run = _incomplete_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"]["datasets"][0]["include"].append("typo")
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert "typo" not in set(expected_results(run)["problem"])
    assert "typo" not in set(missing_results(run)["problem"])
    assert completion_summary(run)["expected"].sum() == 6


def test_missing_dataset_includes_fall_back_but_empty_staged_listing_does_not(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = {"datasets": [{"id": "lp", "name": "netlib", "include": ["planned", "excluded"],
                            "exclude": ["excluded"]}], "solvers": [{"id": "s"}]}
    (run / "manifest.json").write_text(json.dumps({"config": config}))
    expected = expected_results(run, repo_root=tmp_path)
    assert expected.to_dict("records") == [{"dataset": "lp", "problem": "planned", "solver_id": "s"}]
    # Different manifest path avoids the existing manifest-keyed listing cache.
    staged = tmp_path / "staged_run"
    staged.mkdir()
    (staged / "manifest.json").write_text(json.dumps({"config": config}))
    (tmp_path / "problem_classes" / "netlib_data" / "feasible").mkdir(parents=True)
    assert expected_results(staged, repo_root=tmp_path).empty


def test_merged_expected_groups_keep_filters_and_solver_associations(tmp_path):
    run = tmp_path / "merged"
    run.mkdir()
    # The display config's union is deliberately broader than the true source
    # selections, as can happen for mixed include/exclude filters in a merge.
    dataset = {"id": "qp", "name": "synthetic_qp"}
    group_a = {"datasets": [{**dataset, "exclude": ["one_variable_lp"]}], "solvers": ["a"]}
    group_b = {"datasets": [{**dataset, "include": ["one_variable_eq"]}], "solvers": ["b"]}
    manifest = {"config": {"datasets": [dataset], "solvers": [{"id": s} for s in ("a", "b")]},
                "derived": {"selections": [group_a, group_b, group_a]}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    expected = expected_results(run)
    assert expected.to_dict("records") == [
        {"dataset": "qp", "problem": "one_variable_eq", "solver_id": "a"},
        {"dataset": "qp", "problem": "one_variable_eq", "solver_id": "b"},
    ]
    group_b["datasets"] = [{**dataset, "include": ["one_variable_lp"]}]
    (run / "manifest.json").write_text(json.dumps(manifest))
    expected = expected_results(run)
    assert len(expected) == 2
    assert expected.iloc[1].to_dict() == {"dataset": "qp", "problem": "one_variable_lp", "solver_id": "b"}
    # Comparisons intentionally use the common problem population, while
    # planned-job completion counts preserve the associations above.
    gm = shifted_geomean(pd.DataFrame(), expected=expected)
    assert gm["failure_count"].tolist() == [2, 2]
