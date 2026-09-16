import json
import math

import pandas as pd
import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.markdown_report import write_run_report
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.tables import expected_results
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
            {"name": "synthetic_qp", "id": "first", "include": ["one_variable_eq", "never_recorded", "excluded"], "exclude": ["excluded"]},
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
