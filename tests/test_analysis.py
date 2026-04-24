import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.profiles import (
    DEFAULT_FAILURE_PENALTY,
    performance_profile,
    shifted_geomean,
)
from solver_benchmarks.analysis.reports import (
    completion_summary,
    failures_with_successful_alternatives,
    failure_rates,
    objective_spreads,
    pairwise_speedups,
    performance_ratio_matrix,
    problem_solver_comparison,
    slowest_solves,
    solver_metrics,
    solver_problem_tables,
    status_matrix,
)
from solver_benchmarks.analysis.report import write_run_report
from solver_benchmarks.cli import main


def _analysis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "problem": "one_variable_eq",
                "solver_id": "solver_a",
                "status": "optimal",
                "run_time_seconds": 1.0,
                "iterations": 10,
                "objective_value": 1.0,
            },
            {
                "problem": "one_variable_eq",
                "solver_id": "solver_b",
                "status": "optimal",
                "run_time_seconds": 2.0,
                "iterations": 20,
                "objective_value": 1.01,
            },
            {
                "problem": "one_variable_lp",
                "solver_id": "solver_a",
                "status": "time_limit",
                "run_time_seconds": 9.0,
                "iterations": 90,
                "objective_value": None,
            },
            {
                "problem": "one_variable_lp",
                "solver_id": "solver_b",
                "status": "optimal",
                "run_time_seconds": 4.0,
                "iterations": 40,
                "objective_value": 4.0,
            },
        ]
    )


def test_performance_profile_penalizes_failed_solves():
    profile = performance_profile(_analysis_frame(), max_value=100.0, n_tau=3)

    assert profile["tau"].tolist() == pytest.approx([1.0, 100.0, 10000.0])
    assert profile["solver_a"].tolist() == pytest.approx([0.5, 1.0, 1.0])
    assert profile["solver_b"].tolist() == pytest.approx([0.5, 1.0, 1.0])


def test_shifted_geomean_penalizes_failed_solves():
    geomean = shifted_geomean(_analysis_frame(), max_value=100.0, shift=0.0)
    values = dict(zip(geomean["solver_id"], geomean["run_time_seconds"]))

    assert values["solver_a"] == pytest.approx(10.0)
    assert values["solver_b"] == pytest.approx((2.0 * 4.0) ** 0.5)
    assert set(geomean["mode"]) == {"penalized"}
    assert geomean.set_index("solver_id").loc["solver_a", "failure_count"] == 1


def test_default_failure_penalty_is_one_thousand_seconds():
    geomean = shifted_geomean(_analysis_frame(), shift=0.0)
    values = dict(zip(geomean["solver_id"], geomean["run_time_seconds"]))

    assert DEFAULT_FAILURE_PENALTY == 1000.0
    assert values["solver_a"] == pytest.approx((1.0 * 1000.0) ** 0.5)
    assert set(geomean["max_value"]) == {1000.0}


def test_shifted_geomean_can_use_successful_solves_only():
    geomean = shifted_geomean(
        _analysis_frame(),
        shift=0.0,
        penalize_failures=False,
    )
    values = dict(zip(geomean["solver_id"], geomean["run_time_seconds"]))

    assert values["solver_a"] == pytest.approx(1.0)
    assert values["solver_b"] == pytest.approx((2.0 * 4.0) ** 0.5)
    assert set(geomean["mode"]) == {"success_only"}


def test_failure_rates_count_only_accurate_optimal_as_success():
    frame = pd.concat(
        [
            _analysis_frame(),
            pd.DataFrame(
                [
                    {
                        "problem": "extra_problem",
                        "solver_id": "solver_b",
                        "status": "optimal_inaccurate",
                        "run_time_seconds": 0.5,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    rates = failure_rates(frame)
    by_solver = rates.set_index("solver_id")

    assert by_solver.loc["solver_a", "total"] == 2
    assert by_solver.loc["solver_a", "success_count"] == 1
    assert by_solver.loc["solver_a", "failure_rate"] == pytest.approx(0.5)
    assert by_solver.loc["solver_b", "total"] == 3
    assert by_solver.loc["solver_b", "success_count"] == 2
    assert by_solver.loc["solver_b", "failure_rate"] == pytest.approx(1 / 3)
    assert "optimal_inaccurate=1" in by_solver.loc["solver_b", "statuses"]


def test_solver_metrics_include_runtime_and_iteration_aggregates():
    metrics = solver_metrics(_analysis_frame())
    by_solver = metrics.set_index("solver_id")

    assert by_solver.loc["solver_a", "completed"] == 2
    assert by_solver.loc["solver_a", "success_count"] == 1
    assert by_solver.loc["solver_a", "run_time_total_seconds"] == pytest.approx(10.0)
    assert by_solver.loc["solver_a", "run_time_mean_seconds"] == pytest.approx(5.0)
    assert by_solver.loc["solver_a", "run_time_median_seconds"] == pytest.approx(5.0)
    assert by_solver.loc["solver_a", "run_time_max_seconds"] == pytest.approx(9.0)
    assert by_solver.loc["solver_a", "iterations_total"] == pytest.approx(100.0)
    assert by_solver.loc["solver_a", "iterations_mean"] == pytest.approx(50.0)
    assert by_solver.loc["solver_a", "iterations_median"] == pytest.approx(50.0)
    assert by_solver.loc["solver_a", "iterations_max"] == pytest.approx(90.0)


def test_pairwise_and_outlier_reports():
    frame = _analysis_frame()
    speedups = pairwise_speedups(frame)
    objective = objective_spreads(frame)
    slowest = slowest_solves(frame)
    alternatives = failures_with_successful_alternatives(frame)
    matrix = status_matrix(frame)
    comparison = problem_solver_comparison(frame)
    solver_tables = solver_problem_tables(frame)
    ratios = performance_ratio_matrix(frame)

    assert speedups.loc[0, "solver_a"] == "solver_a"
    assert speedups.loc[0, "solver_b"] == "solver_b"
    assert speedups.loc[0, "common_successes"] == 1
    assert speedups.loc[0, "a_wins"] == 1
    assert objective.loc[0, "problem"] == "one_variable_eq"
    assert objective.loc[0, "relative_spread"] == pytest.approx(0.01 / 1.005)
    assert slowest.iloc[0]["problem"] == "one_variable_lp"
    assert alternatives.loc[0, "problem"] == "one_variable_lp"
    assert alternatives.loc[0, "best_success_solver"] == "solver_b"
    assert matrix.loc["one_variable_eq", "solver_a"] == "optimal"
    assert "solver_a__run_time_seconds" in comparison.columns
    assert "solver_b__status" in comparison.columns
    assert comparison.loc[comparison["problem"] == "one_variable_eq", "solver_a__iterations"].iloc[0] == 10
    assert set(solver_tables) == {"solver_a", "solver_b"}
    assert "iterations" in solver_tables["solver_a"].columns
    assert ratios.loc["one_variable_eq", "solver_a"] == pytest.approx(1.0)
    assert ratios.loc["one_variable_eq", "solver_b"] == pytest.approx(2.0)


def test_inaccurate_statuses_are_not_successful_by_default():
    frame = pd.DataFrame(
        [
            {
                "problem": "p1",
                "solver_id": "accurate",
                "status": "optimal",
                "run_time_seconds": 1.0,
            },
            {
                "problem": "p1",
                "solver_id": "inaccurate",
                "status": "optimal_inaccurate",
                "run_time_seconds": 0.1,
            },
        ]
    )

    profile = performance_profile(frame, max_value=100.0, n_tau=3)
    geomean = shifted_geomean(frame, max_value=100.0, shift=0.0)
    values = dict(zip(geomean["solver_id"], geomean["run_time_seconds"]))

    assert profile["accurate"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert profile["inaccurate"].tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert values["accurate"] == pytest.approx(1.0)
    assert values["inaccurate"] == pytest.approx(100.0)


def test_load_summary_and_cli_analysis_commands(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run",
        "config": {
            "dataset": "synthetic_qp",
            "dataset_options": {},
            "include": ["one_variable_eq", "one_variable_lp"],
            "exclude": [],
            "solvers": [
                {"id": "solver_a", "solver": "scs", "settings": {}},
                {"id": "solver_b", "solver": "scs", "settings": {}},
            ],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    records = _analysis_frame().to_dict("records")
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    loaded = load_results(run_dir)
    summary = solver_summary(run_dir)

    assert len(loaded) == 4
    assert set(summary["solver_id"]) == {"solver_a", "solver_b"}
    completion = completion_summary(run_dir, loaded, repo_root=Path.cwd())
    assert completion["missing"].tolist() == [0, 0]

    runner = CliRunner()
    summary_result = runner.invoke(main, ["summary", str(run_dir)])
    assert summary_result.exit_code == 0
    assert "solver_a" in summary_result.output
    assert "time_limit" in summary_result.output
    assert "Solver metrics" in summary_result.output
    assert "run_time_total_seconds" in summary_result.output
    assert "iterations_total" in summary_result.output
    assert "Completion" in summary_result.output

    profile_result = runner.invoke(main, ["profile", str(run_dir), "--metric", "run_time_seconds"])
    assert profile_result.exit_code == 0
    assert (run_dir / "performance_profile_run_time_seconds.csv").exists()

    geomean_result = runner.invoke(main, ["geomean", str(run_dir), "--metric", "run_time_seconds"])
    assert geomean_result.exit_code == 0
    assert (run_dir / "shifted_geomean_run_time_seconds.csv").exists()
    geomean_frame = pd.read_csv(run_dir / "shifted_geomean_run_time_seconds.csv")
    assert "mode" in geomean_frame.columns
    assert "failure_count" in geomean_frame.columns
    assert set(geomean_frame["max_value"]) == {1000.0}

    geomean_success_result = runner.invoke(
        main,
        ["geomean", str(run_dir), "--metric", "run_time_seconds", "--success-only"],
    )
    assert geomean_success_result.exit_code == 0
    assert (run_dir / "shifted_geomean_run_time_seconds_success_only.csv").exists()

    failures_result = runner.invoke(main, ["failures", str(run_dir)])
    assert failures_result.exit_code == 0
    assert "failure_rate" in failures_result.output
    assert "solver_a" in failures_result.output

    missing_result = runner.invoke(main, ["missing", str(run_dir)])
    assert missing_result.exit_code == 0
    assert "No missing results." in missing_result.output

    plot_result = runner.invoke(main, ["plot", str(run_dir), "--metric", "run_time_seconds"])
    assert plot_result.exit_code == 0
    assert (run_dir / "performance_profile_run_time_seconds.png").exists()
    assert (run_dir / "shifted_geomean_run_time_seconds.png").exists()
    assert (run_dir / "failure_rates.png").exists()
    assert (run_dir / "cactus_run_time_seconds.png").exists()
    assert (run_dir / "pairwise_scatter_run_time_seconds.png").exists()
    assert (run_dir / "performance_ratio_heatmap_run_time_seconds.png").exists()
    assert (run_dir / "status_heatmap.png").exists()

    report_dir = tmp_path / "report"
    report_result = runner.invoke(
        main,
        ["report", str(run_dir), "--metric", "run_time_seconds", "--output-dir", str(report_dir)],
    )
    assert report_result.exit_code == 0
    assert (report_dir / "README.md").exists()
    assert (report_dir / "solver_metrics.csv").exists()
    assert (report_dir / "pairwise_speedups_run_time_seconds.csv").exists()
    assert (report_dir / "performance_ratios_run_time_seconds.csv").exists()
    assert (report_dir / "problem_solver_comparison.csv").exists()
    assert (report_dir / "objective_spreads.csv").exists()
    assert (report_dir / "solver_problem_tables" / "solver_a.csv").exists()
    assert (report_dir / "status_heatmap.png").exists()

    direct_report_dir = tmp_path / "direct_report"
    outputs = write_run_report(run_dir, output_dir=direct_report_dir, repo_root=Path.cwd())
    assert outputs
    assert (direct_report_dir / "slowest_solves_run_time_seconds.csv").exists()
