import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.markdown_report import (
    _section_table,
    _sort_report_table,
    write_run_report,
)
from solver_benchmarks.analysis.profiles import (
    DEFAULT_FAILURE_PENALTY,
    performance_profile,
    shifted_geomean,
)
from solver_benchmarks.analysis.tables import (
    claimed_optimal_kkt_thresholds,
    completion_summary,
    difficulty_scaling,
    failure_rates,
    failures_with_successful_alternatives,
    missing_results,
    objective_spreads,
    pairwise_speedups,
    performance_ratio_matrix,
    problem_dimensions,
    problem_solver_comparison,
    setup_solve_breakdown,
    slowest_solves,
    solver_metrics,
    solver_problem_tables,
    status_matrix,
)
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
    # Pin tau_max so the logspace endpoints are deterministic; the
    # default now derives tau_max from ratios.max().
    profile = performance_profile(
        _analysis_frame(), max_value=100.0, n_tau=3, tau_max=10000.0
    )

    assert profile["tau"].tolist() == pytest.approx([1.0, 100.0, 10000.0])
    assert profile["solver_a"].tolist() == pytest.approx([0.5, 1.0, 1.0])
    assert profile["solver_b"].tolist() == pytest.approx([0.5, 1.0, 1.0])


def test_performance_profile_is_deterministic_under_duplicate_rows():
    """Duplicate (problem, solver_id) rows used to make pivot_table
    aggfunc='first' pick non-deterministically. Now we pre-deduplicate
    by best metric so the profile is stable."""
    frame = pd.DataFrame(
        [
            {"problem": "p", "solver_id": "a", "status": "optimal", "run_time_seconds": 1.0},
            # Worst-case duplicate appears first; dedup must keep the
            # better (lower) value so the profile is independent of row
            # order.
            {"problem": "p", "solver_id": "b", "status": "optimal", "run_time_seconds": 100.0},
            {"problem": "p", "solver_id": "b", "status": "optimal", "run_time_seconds": 2.0},
        ]
    )
    profile = performance_profile(frame, n_tau=4, tau_max=10000.0)
    # b's best run is 2s; ratio b/a = 2.0; b appears at tau >= 2.
    assert profile["b"].iloc[0] == 0.0  # at tau=1, b is not within 1x of a
    assert profile["b"].iloc[-1] == pytest.approx(1.0)


def test_performance_profile_drops_problems_where_every_solver_failed():
    """When no solver succeeded on a problem the row is undefined under
    Dolan-Moré; including it would inflate every curve at tau=1."""
    frame = pd.DataFrame(
        [
            {"problem": "ok", "solver_id": "a", "status": "optimal", "run_time_seconds": 1.0},
            {"problem": "ok", "solver_id": "b", "status": "optimal", "run_time_seconds": 2.0},
            {"problem": "doom", "solver_id": "a", "status": "solver_error", "run_time_seconds": None},
            {"problem": "doom", "solver_id": "b", "status": "solver_error", "run_time_seconds": None},
        ]
    )
    profile = performance_profile(frame, max_value=100.0, n_tau=3, tau_max=10000.0)
    # Only the "ok" problem contributes; a wins, so rho_a(1) = 1.0
    # and rho_b(1) = 0.0, rising to 1.0 at the b/a ratio of 2.
    assert profile["a"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert profile["b"].tolist()[0] == pytest.approx(0.0)
    assert profile["b"].tolist()[-1] == pytest.approx(1.0)


def test_performance_profile_default_tau_max_is_dynamic():
    """tau_max derives from ratios.max() so the right tail isn't clipped."""
    frame = pd.DataFrame(
        [
            {"problem": "p", "solver_id": "fast", "status": "optimal", "run_time_seconds": 1.0},
            {"problem": "p", "solver_id": "slow", "status": "optimal", "run_time_seconds": 5.0},
        ]
    )
    profile = performance_profile(frame, n_tau=4)
    # Maximum ratio is 5.0; tau_max should cover it (next power of 10
    # after log10(5+1) → 1e1).
    assert profile["tau"].iloc[-1] >= 5.0
    # Slow's curve must reach 1.0 within the plotted range.
    assert profile["slow"].iloc[-1] == pytest.approx(1.0)


def test_shifted_geomean_penalizes_failed_solves():
    geomean = shifted_geomean(_analysis_frame(), max_value=100.0, shift=0.0)
    values = dict(zip(geomean["solver_id"], geomean["run_time_seconds"]))

    assert values["solver_a"] == pytest.approx(10.0)
    assert values["solver_b"] == pytest.approx((2.0 * 4.0) ** 0.5)
    assert set(geomean["mode"]) == {"penalized"}
    assert geomean.set_index("solver_id").loc["solver_a", "failure_count"] == 1


def test_metric_defaults_dispatch_per_metric():
    """Per-metric defaults stop the run-time-style penalty from leaking
    into iterations / KKT residuals."""
    from solver_benchmarks.analysis.profiles import metric_defaults

    assert metric_defaults("run_time_seconds") == (1.0e3, 10.0)
    assert metric_defaults("iterations") == (1.0e6, 100.0)
    assert metric_defaults("kkt.primal_res_rel") == (1.0, 0.0)
    # Unknown metrics fall back to the run-time-style defaults.
    assert metric_defaults("unknown_column") == (1.0e3, 10.0)


def test_shifted_geomean_uses_metric_defaults_when_unspecified():
    """Calling shifted_geomean with metric=iterations (no shift kwarg)
    should pick the iterations defaults, not 10.0 seconds."""
    frame = pd.DataFrame(
        [
            {"solver_id": "a", "status": "optimal", "iterations": 100},
            {"solver_id": "a", "status": "optimal", "iterations": 200},
        ]
    )
    geomean = shifted_geomean(frame, metric="iterations")
    # max_value should be 1e6 (per metric_defaults("iterations"))
    assert geomean.iloc[0]["max_value"] == 1.0e6


def test_default_failure_penalty_is_one_thousand_seconds():
    geomean = shifted_geomean(_analysis_frame(), shift=0.0)
    values = dict(zip(geomean["solver_id"], geomean["run_time_seconds"]))

    assert DEFAULT_FAILURE_PENALTY == 1000.0
    assert values["solver_a"] == pytest.approx((1.0 * 1000.0) ** 0.5)
    assert set(geomean["max_value"]) == {1000.0}


def test_report_truncated_table_note_links_full_csv():
    table = pd.DataFrame(
        {
            "solver_id": [f"solver_{idx:02d}" for idx in range(25)],
            "run_time_seconds": list(range(25)),
        }
    )

    markdown = "\n".join(
        _section_table(
            "Long Table",
            table,
            max_rows=20,
            source_link="long_table.csv",
        )
    )

    assert "Showing first 20 rows" in markdown
    assert "See [full CSV](long_table.csv) for the full table." in markdown


def test_report_table_sorting_prioritizes_useful_extremes():
    solver_metrics_table = pd.DataFrame(
        [
            {
                "solver_id": "failed",
                "success_rate": 0.5,
                "failure_rate": 0.5,
                "run_time_median_seconds": 0.1,
                "run_time_total_seconds": 0.1,
            },
            {
                "solver_id": "slow",
                "success_rate": 1.0,
                "failure_rate": 0.0,
                "run_time_median_seconds": 10.0,
                "run_time_total_seconds": 10.0,
            },
            {
                "solver_id": "fast",
                "success_rate": 1.0,
                "failure_rate": 0.0,
                "run_time_median_seconds": 1.0,
                "run_time_total_seconds": 1.0,
            },
        ]
    )
    sorted_metrics = _sort_report_table(
        "solver_metrics.csv",
        solver_metrics_table,
        metric="run_time_seconds",
    )

    assert sorted_metrics["solver_id"].tolist() == ["fast", "slow", "failed"]

    slowest = _sort_report_table(
        "slowest_solves_run_time_seconds.csv",
        pd.DataFrame(
            [
                {"problem": "small", "run_time_seconds": 1.0},
                {"problem": "large", "run_time_seconds": 100.0},
                {"problem": "medium", "run_time_seconds": 10.0},
            ]
        ),
        metric="run_time_seconds",
    )
    assert slowest["problem"].tolist() == ["large", "medium", "small"]

    kkt = _sort_report_table(
        "kkt_summary.csv",
        pd.DataFrame(
            [
                {"solver_id": "good", "kkt_missing": 0, "primal_res_rel_max": 1.0e-8},
                {"solver_id": "bad", "kkt_missing": 0, "primal_res_rel_max": 1.0e-2},
            ]
        ),
        metric="run_time_seconds",
    )
    assert kkt["solver_id"].tolist() == ["bad", "good"]


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

    # Pin tau_max so the assertion targets are determined by the call,
    # not by the now-dynamic upper bound.
    profile = performance_profile(frame, max_value=100.0, n_tau=3, tau_max=10000.0)
    geomean = shifted_geomean(frame, max_value=100.0, shift=0.0)
    values = dict(zip(geomean["solver_id"], geomean["run_time_seconds"]))

    assert profile["accurate"].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert profile["inaccurate"].tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert values["accurate"] == pytest.approx(1.0)
    assert values["inaccurate"] == pytest.approx(100.0)


def test_load_summary_and_cli_analysis_commands(tmp_path: Path, repo_root: Path):
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
    (run_dir / "run_config.yaml").write_text(
        "run:\n  dataset: synthetic_qp\nsolvers:\n  - id: solver_a\n    solver: scs\n"
    )
    records = _analysis_frame().to_dict("records")
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    loaded = load_results(run_dir)
    summary = solver_summary(run_dir)

    assert len(loaded) == 4
    assert set(summary["solver_id"]) == {"solver_a", "solver_b"}
    completion = completion_summary(run_dir, loaded, repo_root=repo_root)
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
    assert "## Executive Summary" in report_markdown
    assert "### Run Scope" in report_markdown
    assert "## Headline Solver Performance" in report_markdown
    assert "## Software and Runtime" in report_markdown
    assert "## Solver Metrics" in report_markdown
    assert "## Performance Plots" in report_markdown
    assert (
        '<img src="performance_profile_run_time_seconds.png" '
        'alt="Dolan-More Performance Profile" width="680">'
    ) in report_markdown
    assert "### Exact Source Config" in report_markdown
    assert "```yaml\nrun:\n  dataset: synthetic_qp" in report_markdown
    assert "solver_benchmarks version" in report_markdown
    assert "[Cross-solver comparison](problem_solver_comparison.csv)" in report_markdown
    assert (report_dir / "README.md").read_text() == report_markdown

    direct_report_dir = tmp_path / "direct_report"
    outputs = write_run_report(run_dir, output_dir=direct_report_dir, repo_root=repo_root)
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


def test_report_includes_new_analysis_sections(tmp_path: Path, repo_root: Path):
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
    write_run_report(run_dir, output_dir=report_dir, repo_root=repo_root)

    assert (report_dir / "claimed_optimal_kkt_thresholds.csv").exists()
    assert (report_dir / "difficulty_scaling_run_time_seconds.csv").exists()
    assert (report_dir / "setup_solve_breakdown.csv").exists()
    assert (report_dir / "difficulty_scaling_run_time_seconds.png").exists()
    assert (report_dir / "setup_solve_breakdown.png").exists()

    markdown = (report_dir / "index.md").read_text()
    assert "## Setup vs Solve Time" in markdown
    assert "## Difficulty Scaling" in markdown
    assert "## Claimed-Optimal KKT Thresholds" in markdown
    assert (
        '<img src="difficulty_scaling_run_time_seconds.png" '
        'alt="Difficulty Scaling" width="680">'
    ) in markdown
    assert (
        '<img src="setup_solve_breakdown.png" '
        'alt="Setup vs Solve Time" width="680">'
    ) in markdown


def test_kkt_plots_match_markdown_report_filenames(tmp_path: Path, repo_root: Path):
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
    write_run_report(run_dir, output_dir=report_dir, repo_root=repo_root)
    markdown = (report_dir / "index.md").read_text()
    for filename, alt_text in [
        ("kkt_residual_boxplot.png", "KKT Residual Boxplot"),
        ("kkt_residual_heatmap.png", "KKT Residual Heatmap"),
        ("kkt_accuracy_profile.png", "KKT Accuracy Profile"),
    ]:
        assert (report_dir / filename).exists(), f"missing plot file {filename}"
        assert (
            f'<img src="{filename}" alt="{alt_text}" width="920">' in markdown
        ), f"markdown missing {filename}"


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


def test_completion_summary_reports_per_dataset_rows(monkeypatch, tmp_path: Path, repo_root: Path):
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

    completion = completion_summary(run_dir, load_results(run_dir), repo_root=repo_root)
    by_pair = completion.set_index(["solver_id", "dataset"])

    assert by_pair.loc[("solver_a", "ds_a"), "expected"] == 1
    assert by_pair.loc[("solver_a", "ds_a"), "missing"] == 0
    assert by_pair.loc[("solver_a", "ds_b"), "expected"] == 2
    assert by_pair.loc[("solver_a", "ds_b"), "completed"] == 2
    assert by_pair.loc[("solver_b", "ds_b"), "expected"] == 2
    # solver_b only completed p1 and p2 in ds_b (one row each); none missing.
    assert by_pair.loc[("solver_b", "ds_b"), "missing"] == 0


def test_completion_summary_honors_dataset_size_filter(monkeypatch, tmp_path: Path, repo_root: Path):
    from solver_benchmarks.core.problem import QP, ProblemSpec
    from solver_benchmarks.datasets import registry as dataset_registry

    class _SizedDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [
                ProblemSpec(
                    dataset_id="sized",
                    name="small",
                    kind=QP,
                    metadata={"size_bytes": 10},
                ),
                ProblemSpec(
                    dataset_id="sized",
                    name="large",
                    kind=QP,
                    metadata={"size_bytes": 1_500_001},
                ),
            ]

    monkeypatch.setitem(dataset_registry.DATASETS, "sized", _SizedDataset)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run",
        "config": {
            "dataset": "sized",
            "dataset_options": {"max_size_mb": 1.0},
            "include": [],
            "exclude": [],
            "solvers": [{"id": "solver", "solver": "scs", "settings": {}}],
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    (run_dir / "results.jsonl").write_text("")

    completion = completion_summary(run_dir, load_results(run_dir), repo_root=repo_root)
    missing = missing_results(run_dir, load_results(run_dir), repo_root=repo_root)

    assert completion.loc[0, "expected"] == 1
    assert missing["problem"].tolist() == ["small"]


def test_report_includes_per_dataset_breakdown(monkeypatch, tmp_path: Path, repo_root: Path):
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
    write_run_report(run_dir, output_dir=report_dir, repo_root=repo_root)
    markdown = (report_dir / "index.md").read_text()

    assert "## By Dataset" in markdown
    # Per-dataset h3 sections should exist for each dataset.
    assert "ds_a" in markdown and "ds_b" in markdown
    # The Run Scope should list both datasets.
    assert "| Datasets | ds_a, ds_b |" in markdown


def test_report_per_dataset_breakdown_uses_entry_id_not_registry_name(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    """Two entries with the same registry name but distinct ids must each
    surface as their own populated h3 section."""
    from solver_benchmarks.core.problem import QP, ProblemSpec
    from solver_benchmarks.datasets import registry as dataset_registry

    class _FakeNetlib:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="netlib", name="p1", kind=QP)]

    monkeypatch.setitem(dataset_registry.DATASETS, "netlib", _FakeNetlib)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "run_id": "run",
        "config": {
            "datasets": [
                {
                    "id": "netlib_feasible",
                    "name": "netlib",
                    "dataset_options": {},
                    "include": ["p1"],
                    "exclude": [],
                },
                {
                    "id": "netlib_infeasible",
                    "name": "netlib",
                    "dataset_options": {},
                    "include": ["p1"],
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
    records = [
        {
            "problem": "p1",
            "solver_id": "solver_a",
            "dataset": "netlib_feasible",
            "status": "optimal",
            "run_time_seconds": 1.0,
            "iterations": 10,
            "objective_value": 1.0,
        },
        {
            "problem": "p1",
            "solver_id": "solver_b",
            "dataset": "netlib_feasible",
            "status": "optimal",
            "run_time_seconds": 2.0,
            "iterations": 20,
            "objective_value": 1.0,
        },
        {
            "problem": "p1",
            "solver_id": "solver_a",
            "dataset": "netlib_infeasible",
            "status": "primal_infeasible",
            "run_time_seconds": 0.5,
            "iterations": 5,
            "objective_value": None,
        },
        {
            "problem": "p1",
            "solver_id": "solver_b",
            "dataset": "netlib_infeasible",
            "status": "primal_infeasible",
            "run_time_seconds": 0.7,
            "iterations": 7,
            "objective_value": None,
        },
    ]
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    report_dir = tmp_path / "report"
    write_run_report(run_dir, output_dir=report_dir, repo_root=repo_root)
    markdown = (report_dir / "index.md").read_text()

    assert "### netlib_feasible (netlib)" in markdown
    assert "### netlib_infeasible (netlib)" in markdown
    # The Run Scope "Problems with results" should count unique
    # (dataset, problem) pairs, not unique problem names.
    assert "| Problems with results | 2 |" in markdown
    # Both sections should be populated, not the empty fallback.
    feasible_idx = markdown.index("### netlib_feasible (netlib)")
    infeasible_idx = markdown.index("### netlib_infeasible (netlib)")
    feasible_block = markdown[feasible_idx:infeasible_idx]
    infeasible_block = markdown[infeasible_idx:]
    assert "No rows for this dataset." not in feasible_block
    assert "No rows for this dataset." not in infeasible_block
    # The Run Scope should label each entry with the id (registry name) form.
    assert "netlib_feasible (netlib), netlib_infeasible (netlib)" in markdown


def test_problem_keyed_tables_preserve_dataset_for_shared_names():
    frame = pd.DataFrame(
        [
            {
                "problem": "p1",
                "solver_id": "solver_a",
                "dataset": "ds_a",
                "status": "optimal",
                "run_time_seconds": 1.0,
                "iterations": 10,
                "objective_value": 1.0,
                "metadata.n": 5.0,
                "metadata.m": 3.0,
            },
            {
                "problem": "p1",
                "solver_id": "solver_a",
                "dataset": "ds_b",
                "status": "time_limit",
                "run_time_seconds": 9.0,
                "iterations": 90,
                "objective_value": None,
                "metadata.n": 200.0,
                "metadata.m": 100.0,
            },
            {
                "problem": "p1",
                "solver_id": "solver_b",
                "dataset": "ds_a",
                "status": "optimal",
                "run_time_seconds": 2.0,
                "iterations": 20,
                "objective_value": 1.01,
                "metadata.n": 5.0,
                "metadata.m": 3.0,
            },
        ]
    )

    dimensions = problem_dimensions(frame).set_index(["dataset", "problem"])
    # Same problem name in two datasets must be two separate rows, each with
    # the dimensions from its own dataset.
    assert dimensions.loc[("ds_a", "p1"), "n"] == pytest.approx(5.0)
    assert dimensions.loc[("ds_b", "p1"), "n"] == pytest.approx(200.0)

    comparison = problem_solver_comparison(frame)
    assert list(comparison.columns[:2]) == ["dataset", "problem"]
    by_pair = comparison.set_index(["dataset", "problem"])
    # solver_a status must reflect the dataset the row came from, not be
    # collapsed to whichever row happened to win drop_duplicates.
    assert by_pair.loc[("ds_a", "p1"), "solver_a__status"] == "optimal"
    assert by_pair.loc[("ds_b", "p1"), "solver_a__status"] == "time_limit"
    # solver_b only ran on ds_a; ds_b row should be missing for solver_b.
    assert by_pair.loc[("ds_a", "p1"), "solver_b__status"] == "optimal"
    assert pd.isna(by_pair.loc[("ds_b", "p1"), "solver_b__status"])

    matrix = status_matrix(frame)
    # Row index is (dataset, problem) when the column is present.
    assert matrix.index.names == ["dataset", "problem"]
    assert matrix.loc[("ds_a", "p1"), "solver_a"] == "optimal"
    assert matrix.loc[("ds_b", "p1"), "solver_a"] == "time_limit"

    tables = solver_problem_tables(frame)
    solver_a_table = tables["solver_a"].set_index(["dataset", "problem"])
    assert solver_a_table.loc[("ds_a", "p1"), "n"] == pytest.approx(5.0)
    assert solver_a_table.loc[("ds_b", "p1"), "n"] == pytest.approx(200.0)
    assert solver_a_table.loc[("ds_a", "p1"), "status"] == "optimal"
    assert solver_a_table.loc[("ds_b", "p1"), "status"] == "time_limit"


def test_profile_speedup_spread_ratio_separate_datasets_with_shared_problem_names():
    # Two datasets share problem name p1 but have different solver outcomes.
    # Without dataset-aware keying the pivots collapse them via aggfunc="first"
    # and: the profile counts only one row instead of two, the speedup
    # compares across datasets, the spread sees one solver per problem, and
    # the ratio matrix attaches a ratio from the wrong dataset.
    frame = pd.DataFrame(
        [
            # ds_a/p1: solver_a fast, solver_b slow
            {"problem": "p1", "dataset": "ds_a", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 1.0, "objective_value": 1.0},
            {"problem": "p1", "dataset": "ds_a", "solver_id": "solver_b",
             "status": "optimal", "run_time_seconds": 4.0, "objective_value": 1.0},
            # ds_b/p1: solver_a slow, solver_b fast (opposite of ds_a)
            {"problem": "p1", "dataset": "ds_b", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 8.0, "objective_value": 5.0},
            {"problem": "p1", "dataset": "ds_b", "solver_id": "solver_b",
             "status": "optimal", "run_time_seconds": 2.0, "objective_value": 5.5},
        ]
    )

    profile = performance_profile(frame, max_value=100.0, n_tau=3, tau_max=10000.0)
    # Two distinct (dataset, problem) rows, each solver wins one, so
    # rho(tau=1) = 0.5 for both. If keyed only on problem, aggfunc="first"
    # would keep one row and give 1.0 for the winner and 0.0 for the loser.
    assert profile["solver_a"].tolist() == pytest.approx([0.5, 1.0, 1.0])
    assert profile["solver_b"].tolist() == pytest.approx([0.5, 1.0, 1.0])

    speedups = pairwise_speedups(frame).iloc[0]
    # Two common (dataset, problem) successes, not one.
    assert speedups["common_successes"] == 2
    assert speedups["a_wins"] == 1
    assert speedups["b_wins"] == 1
    # Winner columns must encode the dataset so callers can locate the row.
    assert "biggest_a_win_dataset" in speedups
    assert speedups["biggest_a_win_dataset"] == "ds_a"
    assert speedups["biggest_a_win_problem"] == "p1"
    assert speedups["biggest_a_win_speedup"] == pytest.approx(4.0)
    assert speedups["biggest_b_win_dataset"] == "ds_b"
    assert speedups["biggest_b_win_problem"] == "p1"
    assert speedups["biggest_b_win_speedup"] == pytest.approx(4.0)

    spreads = objective_spreads(frame)
    # Two rows, one per (dataset, problem); with shared-name conflation the
    # second row would be silently dropped by aggfunc="first".
    by_pair = spreads.set_index(["dataset", "problem"])
    assert by_pair.loc[("ds_a", "p1"), "solver_count"] == 2
    assert by_pair.loc[("ds_b", "p1"), "solver_count"] == 2
    # Spreads must reflect each dataset's own objective values.
    assert by_pair.loc[("ds_a", "p1"), "absolute_spread"] == pytest.approx(0.0)
    assert by_pair.loc[("ds_b", "p1"), "absolute_spread"] == pytest.approx(0.5)

    ratios = performance_ratio_matrix(frame)
    # MultiIndex (dataset, problem) keeps the ratios from the wrong dataset
    # from being attached to the wrong row.
    assert ratios.index.names == ["dataset", "problem"]
    assert ratios.loc[("ds_a", "p1"), "solver_a"] == pytest.approx(1.0)
    assert ratios.loc[("ds_a", "p1"), "solver_b"] == pytest.approx(4.0)
    assert ratios.loc[("ds_b", "p1"), "solver_a"] == pytest.approx(4.0)
    assert ratios.loc[("ds_b", "p1"), "solver_b"] == pytest.approx(1.0)


def test_slowest_solves_includes_dataset_when_present():
    # Two datasets share problem name p1; without the dataset column the
    # report has ambiguous rows ("which p1 was the slow one?").
    frame = pd.DataFrame(
        [
            {"problem": "p1", "dataset": "ds_a", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 1.0,
             "iterations": 10, "objective_value": 1.0},
            {"problem": "p1", "dataset": "ds_b", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 9.0,
             "iterations": 90, "objective_value": 5.0},
        ]
    )
    table = slowest_solves(frame)
    assert "dataset" in table.columns
    # Output is sorted slowest-first and the slow row must be ds_b/p1.
    top = table.iloc[0]
    assert top["dataset"] == "ds_b"
    assert top["problem"] == "p1"
    assert top["run_time_seconds"] == pytest.approx(9.0)

    # Legacy frames without a dataset column keep their original shape.
    legacy = pd.DataFrame(
        [
            {"problem": "p1", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 1.0,
             "iterations": 10, "objective_value": 1.0},
        ]
    )
    legacy_table = slowest_solves(legacy)
    assert "dataset" not in legacy_table.columns


def test_failures_with_successful_alternatives_keys_on_dataset_and_problem():
    # ds_a/p1 has solver_a failing but solver_b succeeding — that's a real
    # alternative. ds_b/p1 has solver_a failing and solver_b also failing,
    # so there is NO valid alternative. If grouped by problem only, the
    # ds_a/p1 success would be incorrectly attributed to the ds_b/p1 failure.
    frame = pd.DataFrame(
        [
            {"problem": "p1", "dataset": "ds_a", "solver_id": "solver_a",
             "status": "time_limit", "run_time_seconds": 9.0, "objective_value": None},
            {"problem": "p1", "dataset": "ds_a", "solver_id": "solver_b",
             "status": "optimal", "run_time_seconds": 1.5, "objective_value": 1.0},
            {"problem": "p1", "dataset": "ds_b", "solver_id": "solver_a",
             "status": "time_limit", "run_time_seconds": 9.0, "objective_value": None},
            {"problem": "p1", "dataset": "ds_b", "solver_id": "solver_b",
             "status": "solver_error", "run_time_seconds": None, "objective_value": None},
        ]
    )
    table = failures_with_successful_alternatives(frame)
    rows = table.set_index(["dataset", "problem", "solver_id"])
    assert ("ds_a", "p1", "solver_a") in rows.index
    assert rows.loc[("ds_a", "p1", "solver_a"), "best_success_solver"] == "solver_b"
    # ds_b/p1 has no successful solver, so the failure must NOT appear here.
    assert ("ds_b", "p1", "solver_a") not in rows.index
    assert ("ds_b", "p1", "solver_b") not in rows.index


def test_cactus_plot_denominator_uses_unique_dataset_problem_pairs(tmp_path: Path, repo_root: Path):
    # Two datasets share name p1 → 2 distinct (dataset, problem) pairs.
    # If the denominator counted unique problem names only it would be 1,
    # which would push cactus fractions above 1.0.
    from solver_benchmarks.analysis.plots import _unique_problem_count, _write_cactus

    multi = pd.DataFrame(
        [
            {"problem": "p1", "dataset": "ds_a", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 1.0},
            {"problem": "p1", "dataset": "ds_b", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 2.0},
        ]
    )
    assert _unique_problem_count(multi) == 2

    legacy = pd.DataFrame(
        [
            {"problem": "p1", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 1.0},
            {"problem": "p2", "solver_id": "solver_a",
             "status": "optimal", "run_time_seconds": 2.0},
        ]
    )
    assert _unique_problem_count(legacy) == 2

    out_dir = tmp_path / "plots"
    out_dir.mkdir()
    path = _write_cactus(multi, out_dir, "run_time_seconds")
    assert path is not None and path.exists()


def test_legacy_module_paths_still_importable_with_deprecation_warning():
    """The PR 31 module rename ships compatibility shims so existing
    user code, notebooks, and scripts that import the old paths keep
    working — but each import emits a DeprecationWarning so users
    know to migrate."""
    import importlib
    import warnings as _warnings

    for legacy_name, expected_attr in (
        ("solver_benchmarks.analysis.report", "write_run_report"),
        ("solver_benchmarks.analysis.reports", "solver_metrics"),
    ):
        # Force a re-import so the warning fires even if the test
        # session imported the module earlier.
        import sys

        sys.modules.pop(legacy_name, None)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            module = importlib.import_module(legacy_name)
        deps = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert deps, f"Expected DeprecationWarning when importing {legacy_name}"
        # Re-exported public surface is intact.
        assert hasattr(module, expected_attr)


def test_cactus_plot_counts_zero_duration_successes(tmp_path: Path, repo_root: Path):
    """Reviewer-flagged regression: a strict ``> 0`` filter dropped
    zero-duration successes so the cactus curve undercounted them.
    With one solver and two successful problems at run times
    [0.0, 1.0] the curve must reach 1.0, not 0.5."""
    import numpy as np

    from solver_benchmarks.analysis.plots import _write_cactus

    frame = pd.DataFrame(
        [
            {"problem": "instant", "solver_id": "fast",
             "status": "optimal", "run_time_seconds": 0.0},
            {"problem": "slower", "solver_id": "fast",
             "status": "optimal", "run_time_seconds": 1.0},
        ]
    )
    out_dir = tmp_path / "plots"
    out_dir.mkdir()

    # The cactus plot itself is a PNG; invoke the writer and assert it
    # produces output. The contract that matters most for this fix is
    # internal — verify by patching ax.step to capture the y values
    # passed to it.
    captured: list[tuple] = []
    import matplotlib.axes as _axes

    real_step = _axes.Axes.step

    def capturing_step(self, x, y, *args, **kwargs):
        captured.append((np.asarray(x).copy(), np.asarray(y).copy()))
        return real_step(self, x, y, *args, **kwargs)

    try:
        _axes.Axes.step = capturing_step  # type: ignore[assignment]
        path = _write_cactus(frame, out_dir, "run_time_seconds")
    finally:
        _axes.Axes.step = real_step  # type: ignore[assignment]

    assert path is not None and path.exists()
    assert len(captured) == 1, "expected one solver curve to be drawn"
    x_values, y_values = captured[0]
    # Both successes must contribute one step each; the curve tops at
    # 1.0 for the single-solver / two-success case.
    assert len(x_values) == 2
    assert y_values[-1] == pytest.approx(1.0)


def test_report_omits_per_dataset_breakdown_for_single_dataset(tmp_path: Path, repo_root: Path):
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
    write_run_report(run_dir, output_dir=report_dir, repo_root=repo_root)
    markdown = (report_dir / "index.md").read_text()

    assert "## By Dataset" not in markdown
