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
    claimed_optimal_kkt_thresholds,
    completion_summary,
    difficulty_scaling,
    failures_with_successful_alternatives,
    failure_rates,
    objective_spreads,
    pairwise_speedups,
    performance_ratio_matrix,
    problem_solver_comparison,
    setup_solve_breakdown,
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
    assert (report_dir / "index.md").exists()
    assert (report_dir / "README.md").exists()
    assert (report_dir / "solver_metrics.csv").exists()
    assert (report_dir / "pairwise_speedups_run_time_seconds.csv").exists()
    assert (report_dir / "performance_ratios_run_time_seconds.csv").exists()
    assert (report_dir / "problem_solver_comparison.csv").exists()
    assert (report_dir / "objective_spreads.csv").exists()
    assert (report_dir / "solver_problem_tables" / "solver_a.csv").exists()
    assert (report_dir / "status_heatmap.png").exists()
    report_markdown = (report_dir / "index.md").read_text()
    assert "# Benchmark Report" in report_markdown
    assert "## Run Overview" in report_markdown
    assert "## Solver Metrics" in report_markdown
    assert "## Performance Plots" in report_markdown
    assert "![Dolan-More Performance Profile](performance_profile_run_time_seconds.png)" in report_markdown
    assert "[Cross-solver comparison](problem_solver_comparison.csv)" in report_markdown
    assert (report_dir / "README.md").read_text() == report_markdown

    direct_report_dir = tmp_path / "direct_report"
    outputs = write_run_report(run_dir, output_dir=direct_report_dir, repo_root=Path.cwd())
    assert outputs
    assert direct_report_dir / "index.md" in outputs
    assert (direct_report_dir / "slowest_solves_run_time_seconds.csv").exists()


def test_claimed_optimal_kkt_thresholds_buckets_worst_residual():
    frame = pd.DataFrame(
        [
            # solver_a: two claimed-optimal, one well-converged, one loose
            {
                "problem": "p1",
                "solver_id": "solver_a",
                "status": "optimal",
                "kkt.primal_res_rel": 1.0e-9,
                "kkt.dual_res_rel": 2.0e-10,
                "kkt.duality_gap_rel": 5.0e-12,
            },
            {
                "problem": "p2",
                "solver_id": "solver_a",
                "status": "optimal",
                "kkt.primal_res_rel": 1.0e-3,
                "kkt.dual_res_rel": 1.0e-9,
                "kkt.duality_gap_rel": 1.0e-9,
            },
            # solver_b: claimed optimal but residuals above the loosest threshold
            {
                "problem": "p1",
                "solver_id": "solver_b",
                "status": "optimal",
                "kkt.primal_res_rel": 5.0,
                "kkt.dual_res_rel": 1.0e-9,
                "kkt.duality_gap_rel": 1.0e-9,
            },
            # solver_b: claimed optimal but residuals missing entirely
            {
                "problem": "p2",
                "solver_id": "solver_b",
                "status": "optimal",
                "kkt.primal_res_rel": None,
                "kkt.dual_res_rel": None,
                "kkt.duality_gap_rel": None,
            },
            # non-success rows are excluded
            {
                "problem": "p3",
                "solver_id": "solver_a",
                "status": "time_limit",
                "kkt.primal_res_rel": 1.0,
                "kkt.dual_res_rel": 1.0,
                "kkt.duality_gap_rel": 1.0,
            },
        ]
    )
    table = claimed_optimal_kkt_thresholds(frame).set_index("solver_id")

    assert table.loc["solver_a", "claimed_optimal"] == 2
    assert table.loc["solver_a", "with_residuals"] == 2
    assert table.loc["solver_a", "missing_residuals"] == 0
    assert table.loc["solver_a", "count_le_1e-08"] == 1
    assert table.loc["solver_a", "count_le_1e-02"] == 2
    assert table.loc["solver_a", "count_above_max"] == 0
    assert table.loc["solver_a", "worst_max"] == pytest.approx(1.0e-3)

    assert table.loc["solver_b", "claimed_optimal"] == 2
    assert table.loc["solver_b", "with_residuals"] == 1
    assert table.loc["solver_b", "missing_residuals"] == 1
    assert table.loc["solver_b", "count_le_1e-02"] == 0
    assert table.loc["solver_b", "count_above_max"] == 1


def test_difficulty_scaling_buckets_problems_by_size():
    frame = pd.DataFrame(
        [
            {
                "problem": f"p{idx}",
                "solver_id": "solver_a",
                "status": "optimal",
                "run_time_seconds": float(idx),
                "metadata.n": float(idx),
            }
            for idx in range(1, 9)
        ]
    )
    table = difficulty_scaling(frame, bin_count=4)

    assert set(table["solver_id"]) == {"solver_a"}
    assert sorted(table["size_bin"].tolist()) == [0, 1, 2, 3]
    # Each equal-population bin holds 2 problems with all-success solves
    assert table["problem_count"].tolist() == [2, 2, 2, 2]
    assert table["success_count"].tolist() == [2, 2, 2, 2]
    # Smallest bucket median runtime should be smaller than the largest
    bins = table.set_index("size_bin")
    assert bins.loc[0, "median_time"] < bins.loc[3, "median_time"]


def test_difficulty_scaling_handles_missing_size_field():
    frame = _analysis_frame()
    assert difficulty_scaling(frame).empty


def test_setup_solve_breakdown_reports_phases_when_available():
    frame = pd.DataFrame(
        [
            {
                "problem": "p1",
                "solver_id": "split",
                "status": "optimal",
                "setup_time_seconds": 1.0,
                "solve_time_seconds": 3.0,
            },
            {
                "problem": "p2",
                "solver_id": "split",
                "status": "optimal",
                "setup_time_seconds": 2.0,
                "solve_time_seconds": 6.0,
            },
            {
                "problem": "p1",
                "solver_id": "no_split",
                "status": "optimal",
                "setup_time_seconds": None,
                "solve_time_seconds": None,
            },
            {
                "problem": "p2",
                "solver_id": "split",
                "status": "time_limit",
                "setup_time_seconds": 0.5,
                "solve_time_seconds": 99.0,
            },
        ]
    )
    table = setup_solve_breakdown(frame).set_index("solver_id")

    assert table.loc["split", "with_breakdown"] == 2
    assert table.loc["split", "setup_median"] == pytest.approx(1.5)
    assert table.loc["split", "solve_median"] == pytest.approx(4.5)
    assert table.loc["split", "total_median"] == pytest.approx(6.0)
    assert table.loc["split", "setup_share_median"] == pytest.approx(0.25)
    assert table.loc["no_split", "with_breakdown"] == 0
    assert pd.isna(table.loc["no_split", "setup_median"])


def test_setup_solve_breakdown_empty_without_columns():
    assert setup_solve_breakdown(_analysis_frame()).empty


def test_report_includes_new_analysis_sections(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run",
        "config": {
            "dataset": "synthetic_qp",
            "include": ["one_variable_eq", "one_variable_lp"],
            "solvers": [
                {"id": "solver_a", "solver": "scs", "settings": {}},
                {"id": "solver_b", "solver": "scs", "settings": {}},
            ],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    records = []
    for idx, entry in enumerate(_analysis_frame().to_dict("records")):
        entry = dict(entry)
        entry["metadata.n"] = float(10 + idx * 50)
        entry["setup_time_seconds"] = 0.1 * (idx + 1)
        entry["solve_time_seconds"] = 0.2 * (idx + 1)
        if entry["status"] == "optimal":
            entry["kkt.primal_res_rel"] = 1.0e-9
            entry["kkt.dual_res_rel"] = 2.0e-10
            entry["kkt.duality_gap_rel"] = 5.0e-12
        records.append(entry)
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    report_dir = tmp_path / "report"
    write_run_report(run_dir, output_dir=report_dir, repo_root=Path.cwd())

    assert (report_dir / "claimed_optimal_kkt_thresholds.csv").exists()
    assert (report_dir / "difficulty_scaling_run_time_seconds.csv").exists()
    assert (report_dir / "setup_solve_breakdown.csv").exists()
    assert (report_dir / "difficulty_scaling_run_time_seconds.png").exists()
    assert (report_dir / "setup_solve_breakdown.png").exists()

    markdown = (report_dir / "index.md").read_text()
    assert "## Setup vs Solve Time" in markdown
    assert "## Difficulty Scaling" in markdown
    assert "## Claimed-Optimal KKT Thresholds" in markdown
    assert "![Difficulty Scaling](difficulty_scaling_run_time_seconds.png)" in markdown
    assert "![Setup vs Solve Time](setup_solve_breakdown.png)" in markdown


def test_kkt_plots_match_markdown_report_filenames(tmp_path: Path):
    # Regression: ``_write_kkt_residual_boxplot`` previously saved to
    # ``kkt_residuals.png`` while the markdown report linked
    # ``kkt_residual_boxplot.png``, so the boxplot silently never appeared
    # in the rendered report. Pin every KKT plot's filename to what the
    # markdown writer expects.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run",
        "config": {
            "dataset": "synthetic_qp",
            "include": ["one_variable_eq", "one_variable_lp"],
            "solvers": [
                {"id": "solver_a", "solver": "scs", "settings": {}},
                {"id": "solver_b", "solver": "scs", "settings": {}},
            ],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    records = []
    for entry in _analysis_frame().to_dict("records"):
        entry = dict(entry)
        if entry["status"] == "optimal":
            entry["kkt.primal_res_rel"] = 1.0e-8
            entry["kkt.dual_res_rel"] = 2.0e-8
            entry["kkt.comp_slack"] = 3.0e-9
            entry["kkt.duality_gap_rel"] = 4.0e-9
        records.append(entry)
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    report_dir = tmp_path / "report"
    write_run_report(run_dir, output_dir=report_dir, repo_root=Path.cwd())
    markdown = (report_dir / "index.md").read_text()
    for filename, alt_text in [
        ("kkt_residual_boxplot.png", "KKT Residual Boxplot"),
        ("kkt_residual_heatmap.png", "KKT Residual Heatmap"),
        ("kkt_accuracy_profile.png", "KKT Accuracy Profile"),
    ]:
        assert (report_dir / filename).exists(), f"missing plot file {filename}"
        assert f"![{alt_text}]({filename})" in markdown, f"markdown missing {filename}"


def _multi_dataset_records(dataset_a: str, dataset_b: str) -> list[dict]:
    base = []
    for problem, solver, dataset, status_str, run_time, obj in [
        ("p1", "solver_a", dataset_a, "optimal", 1.0, 1.0),
        ("p1", "solver_b", dataset_a, "optimal", 2.0, 1.0),
        ("p1", "solver_a", dataset_b, "optimal", 0.5, 0.5),
        ("p1", "solver_b", dataset_b, "time_limit", 9.0, None),
        ("p2", "solver_a", dataset_b, "optimal", 0.8, 1.5),
        ("p2", "solver_b", dataset_b, "optimal", 0.6, 1.5),
    ]:
        base.append(
            {
                "problem": problem,
                "solver_id": solver,
                "dataset": dataset,
                "status": status_str,
                "run_time_seconds": run_time,
                "iterations": 10,
                "objective_value": obj,
            }
        )
    return base


def _register_fake_datasets(monkeypatch, dataset_a: str, dataset_b: str):
    from solver_benchmarks.core.problem import QP, ProblemSpec
    from solver_benchmarks.datasets import registry as dataset_registry

    class _FakeA:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id=dataset_a, name="p1", kind=QP)]

    class _FakeB:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [
                ProblemSpec(dataset_id=dataset_b, name="p1", kind=QP),
                ProblemSpec(dataset_id=dataset_b, name="p2", kind=QP),
            ]

    monkeypatch.setitem(dataset_registry.DATASETS, dataset_a, _FakeA)
    monkeypatch.setitem(dataset_registry.DATASETS, dataset_b, _FakeB)


def test_completion_summary_reports_per_dataset_rows(monkeypatch, tmp_path: Path):
    _register_fake_datasets(monkeypatch, "ds_a", "ds_b")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run",
        "config": {
            "datasets": [
                {
                    "name": "ds_a",
                    "dataset_options": {},
                    "include": ["p1"],
                    "exclude": [],
                },
                {
                    "name": "ds_b",
                    "dataset_options": {},
                    "include": ["p1", "p2"],
                    "exclude": [],
                },
            ],
            "solvers": [
                {"id": "solver_a", "solver": "scs", "settings": {}},
                {"id": "solver_b", "solver": "scs", "settings": {}},
            ],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in _multi_dataset_records("ds_a", "ds_b"):
            handle.write(json.dumps(record) + "\n")

    completion = completion_summary(run_dir, load_results(run_dir), repo_root=Path.cwd())
    by_pair = completion.set_index(["solver_id", "dataset"])

    assert by_pair.loc[("solver_a", "ds_a"), "expected"] == 1
    assert by_pair.loc[("solver_a", "ds_a"), "missing"] == 0
    assert by_pair.loc[("solver_a", "ds_b"), "expected"] == 2
    assert by_pair.loc[("solver_a", "ds_b"), "completed"] == 2
    assert by_pair.loc[("solver_b", "ds_b"), "expected"] == 2
    # solver_b only completed p1 and p2 in ds_b (one row each); none missing.
    assert by_pair.loc[("solver_b", "ds_b"), "missing"] == 0


def test_report_includes_per_dataset_breakdown(monkeypatch, tmp_path: Path):
    _register_fake_datasets(monkeypatch, "ds_a", "ds_b")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run",
        "config": {
            "datasets": [
                {
                    "name": "ds_a",
                    "dataset_options": {},
                    "include": ["p1"],
                    "exclude": [],
                },
                {
                    "name": "ds_b",
                    "dataset_options": {},
                    "include": ["p1", "p2"],
                    "exclude": [],
                },
            ],
            "solvers": [
                {"id": "solver_a", "solver": "scs", "settings": {}},
                {"id": "solver_b", "solver": "scs", "settings": {}},
            ],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in _multi_dataset_records("ds_a", "ds_b"):
            handle.write(json.dumps(record) + "\n")

    report_dir = tmp_path / "report"
    write_run_report(run_dir, output_dir=report_dir, repo_root=Path.cwd())
    markdown = (report_dir / "index.md").read_text()

    assert "## By Dataset" in markdown
    # Per-dataset h3 sections should exist for each dataset.
    assert "ds_a" in markdown and "ds_b" in markdown
    # The Run Overview should list both datasets.
    assert "Datasets:" in markdown


def test_report_omits_per_dataset_breakdown_for_single_dataset(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run",
        "config": {
            "dataset": "synthetic_qp",
            "include": ["one_variable_eq", "one_variable_lp"],
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

    report_dir = tmp_path / "report"
    write_run_report(run_dir, output_dir=report_dir, repo_root=Path.cwd())
    markdown = (report_dir / "index.md").read_text()

    assert "## By Dataset" not in markdown
