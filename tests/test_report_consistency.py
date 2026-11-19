"""Metrics in one report agree with each other and are not distorted by edge values."""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import matplotlib.axes as mpl_axes
import numpy as np
import pandas as pd
import pytest
from conftest import with_passing_kkt

from solver_benchmarks.analysis.markdown_report import write_run_report
from solver_benchmarks.analysis.penalties import geomean_time_limits
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.tables import (
    claimed_optimal_kkt_thresholds,
    difficulty_scaling,
    failure_rates,
    kkt_certificate_summary,
    kkt_summary,
    objective_spreads,
    pairwise_speedups,
    performance_ratio_matrix,
    problem_dimensions,
    problem_solver_comparison,
    slowest_solves,
    solver_metrics,
    solver_problem_tables,
    status_matrix,
)


def _row(problem, solver, status="optimal", time=1.0, **extra):
    return {"dataset": "d", "problem": problem, "solver_id": solver, "status": status,
            "run_time_seconds": time, **extra}


def _write_run(tmp_path: Path, rows, *, timeout=60, solvers=None, include=None) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    solver_ids = solvers or sorted({r["solver_id"] for r in rows})
    problems = include or sorted({r["problem"] for r in rows})
    manifest = {"run_id": "run", "config": {
        "timeout_seconds": timeout,
        "datasets": [{"id": "d", "name": "synthetic_qp", "include": problems}],
        "solvers": [s if isinstance(s, dict) else {"id": s, "solver": "scs", "settings": {}} for s in solver_ids],
    }}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "results.jsonl").write_text("".join(json.dumps(with_passing_kkt(r)) + "\n" for r in rows))
    return run


# --- failure penalties honour per-solver time limits ---


def test_penalty_uses_the_largest_solver_time_limit(tmp_path):
    run = _write_run(tmp_path, [_row("p", "slow_ok")],
                     solvers=[{"id": "slow_ok", "solver": "scs", "timeout_seconds": 3600}, "fast"])
    assert geomean_time_limits(run, metric="run_time_seconds") == 3600
    frame = pd.DataFrame([
        _row("p1", "slow_ok", time=1000.0), _row("p2", "slow_ok", time=1000.0),
        _row("p1", "fast", "solver_error", 0.1), _row("p2", "fast", "solver_error", 0.1),
    ])
    gm = shifted_geomean(frame, timeout_seconds=3600).set_index("solver_id")["run_time_seconds"]
    assert gm["slow_ok"] < gm["fast"]


def test_penalty_with_only_solver_time_limits(tmp_path):
    solvers = [{"id": "a", "solver": "scs", "timeout_seconds": 300}, {"id": "b", "solver": "scs", "timeout_seconds": 120}]
    run = _write_run(tmp_path, [_row("p", "a")], timeout=None, solvers=solvers)
    assert geomean_time_limits(run, metric="run_time_seconds") == 300


def test_penalty_needs_a_limit_for_every_solver(tmp_path):
    solvers = [{"id": "a", "solver": "scs", "timeout_seconds": 300}, "unlimited"]
    run = _write_run(tmp_path, [_row("p", "a")], timeout=None, solvers=solvers)
    with pytest.raises(ValueError, match="--max-value"):
        geomean_time_limits(run, metric="run_time_seconds")


def test_geomean_warns_when_a_success_costs_more_than_a_failure(caplog):
    from solver_benchmarks.analysis import profiles

    profiles._warn_once.cache_clear()
    frame = pd.DataFrame([_row("p", "a", time=50.0), _row("p", "b", "time_limit", 10.0)])
    with caplog.at_level(logging.WARNING):
        # a report computes the same geomean several times; say it once
        shifted_geomean(frame, max_value=20.0)
        shifted_geomean(frame, max_value=20.0)
    assert caplog.text.count("a: a successful solve's run_time_seconds is at or above the failure penalty") == 1


# --- exact zeros ---


def test_kkt_geomean_is_not_collapsed_by_an_exact_zero():
    rows = [{"problem": f"p{i}", "solver_id": "a", "status": "optimal", "kkt.primal_res_rel": 1e-6} for i in range(99)]
    rows.append({"problem": "p99", "solver_id": "a", "status": "optimal", "kkt.primal_res_rel": 0.0})
    value = shifted_geomean(pd.DataFrame(rows), metric="kkt.primal_res_rel")["kkt.primal_res_rel"].iloc[0]
    assert 5e-7 < value < 1.5e-6


def test_ratio_tables_floor_like_the_profile():
    frame = pd.DataFrame([
        _row("p1", "a", iterations=0), _row("p1", "b", iterations=5),
        _row("p2", "a", iterations=3), _row("p2", "b", iterations=6),
    ])
    ratios = performance_ratio_matrix(frame, metric="iterations")
    assert ratios.loc[("d", "p1"), "a"] == 1.0
    assert ratios.loc[("d", "p1"), "b"] == 5.0
    speedups = pairwise_speedups(frame, metric="iterations").iloc[0]
    assert speedups["common_successes"] == 2
    profile = performance_profile(frame, metric="iterations")
    # one problem each at ratio 1 for a; b at ratios 5 and 2
    assert profile["a"].iloc[0] == 1.0


# --- one dedup rule for duplicate rows ---


def _retried():
    return pd.DataFrame([
        _row("p1", "a", time=1.0, iterations=10, objective_value=1.0),
        _row("p1", "b", "solver_error", 0.1, iterations=1),
        _row("p1", "b", "optimal", 1.5, iterations=12, objective_value=1.0),
        _row("p2", "a", time=2.0, iterations=20, objective_value=2.0),
        _row("p2", "b", time=2.5, iterations=25, objective_value=2.0),
    ])


def test_duplicate_rows_resolve_to_the_success_everywhere():
    frame = _retried()
    assert status_matrix(frame).loc[("d", "p1"), "b"] == "optimal"
    comparison = problem_solver_comparison(frame).set_index("problem")
    assert comparison.loc["p1", "b__status"] == "optimal"
    assert comparison.loc["p1", "b__run_time_seconds"] == 1.5
    metrics = solver_metrics(frame).set_index("solver_id")
    assert metrics.loc["b", "completed"] == 2 and metrics.loc["b", "failure_count"] == 0
    rates = failure_rates(frame).set_index("solver_id")
    assert rates.loc["b", "failure_count"] == 0 and rates.loc["b", "statuses"] == "optimal=2"
    geomean = shifted_geomean(frame, timeout_seconds=60).set_index("solver_id")
    assert geomean.loc["b", "failure_count"] == metrics.loc["b", "failure_count"]


def test_cactus_counts_each_problem_once(tmp_path):
    from solver_benchmarks.analysis.plots import _write_cactus

    frame = pd.DataFrame([_row("p1", "a", time=1.0), _row("p1", "a", time=1.2),
                          _row("p2", "a", time=2.0), _row("p2", "a", time=2.2)])
    captured = []
    real_step = mpl_axes.Axes.step

    def capturing_step(self, x, y, *args, **kwargs):
        captured.append(np.asarray(y).copy())
        return real_step(self, x, y, *args, **kwargs)

    mpl_axes.Axes.step = capturing_step  # type: ignore[method-assign]
    try:
        _write_cactus(frame, tmp_path, "run_time_seconds")
    finally:
        mpl_axes.Axes.step = real_step  # type: ignore[method-assign]
    assert captured[0].tolist() == [0.5, 1.0]


# --- unattempted solves and success-only statistics in the headline ---


def test_headline_counts_unattempted_solves_like_the_geomean(tmp_path, repo_root):
    problems = [f"p{i}" for i in range(4)]
    rows = [_row(p, "a") for p in problems] + [_row(p, "b") for p in problems[:2]]
    run = _write_run(tmp_path, rows, include=["one_variable_eq", "one_variable_lp"])
    # the manifest expects every observed problem for both solvers
    expected = pd.DataFrame([{"dataset": "d", "problem": p, "solver_id": s} for p in problems for s in "ab"])
    import solver_benchmarks.analysis.markdown_report as report

    original = report.expected_results
    report.expected_results = lambda *args, **kwargs: expected
    try:
        write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    finally:
        report.expected_results = original
    headline = pd.read_csv(tmp_path / "report" / "headline_solver_metrics.csv").set_index("solver_id")
    geomean = pd.read_csv(tmp_path / "report" / "shifted_geomean_run_time_seconds.csv").set_index("solver_id")
    assert headline.loc["b", "completed"] == 2
    assert headline.loc["b", "total"] == 4
    columns = list(headline.reset_index().columns)
    assert columns.index("total") + 1 == columns.index("completed")
    assert headline.loc["b", "success_count"] == geomean.loc["b", "success_count"] == 2
    assert headline.loc["b", "failure_count"] == geomean.loc["b", "failure_count"] == 2
    assert headline.loc["b", "success_rate"] == 0.5
    assert "not_attempted=2" in headline.loc["b", "statuses"]


def test_solver_metrics_total_is_what_the_rates_are_over():
    rows = [_row(f"p{i}", "a") for i in range(2)]
    expected = pd.DataFrame([{"dataset": "d", "problem": f"p{i}", "solver_id": "a"} for i in range(5)])
    metrics = solver_metrics(pd.DataFrame(rows), expected=expected)
    assert list(metrics.columns[:3]) == ["solver_id", "total", "completed"]
    row = metrics.set_index("solver_id").loc["a"]
    assert (row["total"], row["completed"], row["success_count"], row["failure_count"]) == (5, 2, 2, 3)
    assert row["success_count"] + row["failure_count"] == row["total"]
    assert row["success_rate"] == 2 / 5
    # Without planned solves the total is the recorded solves.
    assert solver_metrics(pd.DataFrame(rows)).set_index("solver_id").loc["a", "total"] == 2
    assert list(solver_metrics(pd.DataFrame()).columns[:3]) == ["solver_id", "total", "completed"]


def test_headline_statistics_cover_successes_only():
    rows = [_row(f"p{i}", "robust", time=2.0, iterations=20) for i in range(10)]
    rows += [_row(f"p{i}", "crashy", "solver_error", 0.01, iterations=0) for i in range(6)]
    rows += [_row(f"p{i}", "crashy", time=5.0, iterations=50) for i in range(6, 10)]
    metrics = solver_metrics(pd.DataFrame(rows)).set_index("solver_id")
    assert metrics.loc["crashy", "run_time_median_seconds"] == 5.0
    assert metrics.loc["crashy", "iterations_median"] == 50
    # totals still cover every attempt
    assert metrics.loc["crashy", "run_time_total_seconds"] == pytest.approx(6 * 0.01 + 4 * 5.0)


# --- a success without a value for the metric is not a failure ---


def test_solver_that_never_reports_the_metric_is_left_out(caplog):
    from solver_benchmarks.analysis import profiles

    profiles._warn_once.cache_clear()
    frame = pd.DataFrame([
        _row("p1", "scs", solve_time_seconds=0.5), _row("p2", "scs", solve_time_seconds=0.7),
        _row("p1", "clarabel"), _row("p2", "clarabel"),
        _row("p1", "broken", "solver_error"), _row("p2", "broken", "solver_error"),
    ])
    with caplog.at_level(logging.WARNING):
        geomean = shifted_geomean(frame, metric="solve_time_seconds", timeout_seconds=60)
        profile = performance_profile(frame, metric="solve_time_seconds")
    assert "clarabel" not in set(geomean["solver_id"])
    assert "clarabel" not in profile.columns
    assert "broken" in set(geomean["solver_id"]) and "broken" in profile.columns
    assert "clarabel" in caplog.text


# --- KKT tables and plots look at optimality claims ---


def test_kkt_summary_does_not_count_certificates_as_missing_residuals():
    kkt = {"kkt.primal_res_rel": 1e-9, "kkt.dual_res_rel": 1e-9, "kkt.duality_gap_rel": 1e-9}
    rows = [_row(f"p{i}", "a", **kkt) for i in range(3)]
    rows.append(_row("inf", "a", "primal_infeasible", **{"metadata.expected_status": "infeasible",
                                                          "kkt.Aty_rel": 1e-12, "kkt.valid": True}))
    summary = kkt_summary(pd.DataFrame(rows)).set_index("solver_id")
    assert summary.loc["a", "kkt_missing"] == 0
    assert summary.loc["a", "success_count"] == 3


def test_kkt_plots_keep_exact_zero_residuals(tmp_path):
    from solver_benchmarks.analysis import plots

    fields = ("kkt.primal_res_rel", "kkt.dual_res_rel", "kkt.comp_slack_rel", "kkt.duality_gap_rel")
    rows = [_row("p1", "zero", **{f: 0.0 for f in fields}), _row("p1", "other", **{f: 1e-8 for f in fields})]
    captured = []
    real_boxplot = mpl_axes.Axes.boxplot

    def capturing_boxplot(self, data, *args, **kwargs):
        captured.append(kwargs.get("tick_labels"))
        return real_boxplot(self, data, *args, **kwargs)

    mpl_axes.Axes.boxplot = capturing_boxplot  # type: ignore[method-assign]
    try:
        assert plots._write_kkt_residual_boxplot(pd.DataFrame(rows), tmp_path) is not None
    finally:
        mpl_axes.Axes.boxplot = real_boxplot  # type: ignore[method-assign]
    assert all("zero" in labels for labels in captured)
    captured_images = []
    real_imshow = mpl_axes.Axes.imshow

    def capturing_imshow(self, data, *args, **kwargs):
        captured_images.append(np.asarray(data, dtype=float))
        return real_imshow(self, data, *args, **kwargs)

    mpl_axes.Axes.imshow = capturing_imshow  # type: ignore[method-assign]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert plots._write_kkt_residual_heatmap(pd.DataFrame(rows), tmp_path) is not None
    finally:
        mpl_axes.Axes.imshow = real_imshow  # type: ignore[method-assign]
    assert all(np.isfinite(image).all() for image in captured_images)


def test_kkt_plots_use_scaled_complementarity_when_recorded(tmp_path):
    from solver_benchmarks.analysis import plots

    base = ["kkt.primal_res_rel", "kkt.dual_res_rel", "kkt.comp_slack", "kkt.duality_gap_rel"]
    labels = [label for _, label in plots._kkt_fields(pd.DataFrame(columns=[*base, "kkt.comp_slack_rel"]))]
    assert labels == ["primal_res_rel", "dual_res_rel", "comp_slack_rel", "duality_gap_rel"]
    # records written before comp_slack_rel existed keep the absolute value
    assert [label for _, label in plots._kkt_fields(pd.DataFrame(columns=base))][2] == "comp_slack"


def test_ratio_heatmap_avoids_deprecated_colormap_mutation(tmp_path):
    from solver_benchmarks.analysis import plots

    frame = pd.DataFrame([_row("p1", "a", time=1.0), _row("p1", "b", time=2.0), _row("p2", "a", "solver_error")])
    with warnings.catch_warnings():
        warnings.simplefilter("error", PendingDeprecationWarning)
        warnings.simplefilter("error", DeprecationWarning)
        assert plots._write_performance_ratio_heatmap(frame, tmp_path, "run_time_seconds") is not None


# --- report tables ---


def test_tables_carry_no_stray_index_column(tmp_path, repo_root):
    rows = [_row("p1", "a", time=3.0, objective_value=1.0), _row("p1", "b", time=1.0, objective_value=1.1),
            _row("p2", "a", time=1.0, objective_value=5.0), _row("p2", "b", time=2.0, objective_value=5.0),
            _row("p3", "a", time=2.0, objective_value=0.0), _row("p3", "b", time=5.0, objective_value=3.0)]
    assert isinstance(slowest_solves(pd.DataFrame(rows)).index, pd.RangeIndex)
    assert isinstance(objective_spreads(pd.DataFrame(rows)).index, pd.RangeIndex)
    run = _write_run(tmp_path, rows)
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    for name in ("slowest_solves_run_time_seconds.csv", "objective_spreads.csv"):
        header = (tmp_path / "report" / name).read_text().splitlines()[0]
        assert header.startswith("dataset,problem"), header
    assert "| index |" not in (tmp_path / "report" / "index.md").read_text()


def test_report_without_a_time_limit_still_writes_everything_else(tmp_path, repo_root):
    run = _write_run(tmp_path, [_row("p1", "a"), _row("p1", "b", "solver_error")], timeout=None)
    outputs = write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    names = {path.name for path in outputs}
    assert "index.md" in names
    assert "performance_profile_run_time_seconds.csv" in names
    assert "shifted_geomean_run_time_seconds_success_only.csv" in names
    assert "shifted_geomean_run_time_seconds.csv" not in names
    markdown = (tmp_path / "report" / "index.md").read_text()
    assert "--max-value" in markdown


def test_report_text_matches_the_scoring_rules(tmp_path, repo_root):
    rows = [_row("p1", "a", iterations=3, **{"metadata.n": 1}), _row("p2", "a", iterations=5, **{"metadata.n": 10}),
            _row("p1", "b", iterations=4, **{"metadata.n": 1}), _row("p2", "b", iterations=6, **{"metadata.n": 10})]
    run = _write_run(tmp_path, rows)
    write_run_report(run, metric="iterations", output_dir=tmp_path / "report", repo_root=repo_root)
    markdown = (tmp_path / "report" / "index.md").read_text()
    assert "only accurate `optimal` solves count as successes" not in markdown
    assert "Known outcomes" in markdown or "certificate" in markdown
    assert "median runtime" not in markdown
    scaling = pd.read_csv(tmp_path / "report" / "difficulty_scaling_iterations.csv")
    assert "median_iterations" in scaling.columns and "median_time" not in scaling.columns


def test_summary_and_failures_commands_count_unattempted_solves(tmp_path, repo_root, monkeypatch):
    """`bench summary` and `bench failures` agree with the report: a planned
    solve with no row is a not_attempted failure, not left out."""
    from click.testing import CliRunner

    import solver_benchmarks.cli as cli

    problems = ["p1", "p2", "p3", "p4"]
    rows = [_row(p, "a") for p in problems] + [_row(p, "b") for p in problems[:2]]
    run = _write_run(tmp_path, rows, include=problems)
    expected = pd.DataFrame([{"dataset": "d", "problem": p, "solver_id": s} for p in problems for s in "ab"])
    monkeypatch.setattr(cli, "expected_results", lambda *args, **kwargs: expected)
    runner = CliRunner()
    failures = runner.invoke(cli.main, ["failures", str(run), "--repo-root", str(repo_root)])
    assert failures.exit_code == 0, failures.output
    line_b = next(line for line in failures.output.splitlines() if line.split() and line.split()[0] == "b")
    assert "0.5" in line_b.split()
    summary = runner.invoke(cli.main, ["summary", str(run), "--repo-root", str(repo_root)])
    assert summary.exit_code == 0, summary.output
    metrics_block = summary.output.split("Status counts:")[0]
    line_b = next(line for line in metrics_block.splitlines() if line.split() and line.split()[0] == "b")
    assert "0.5" in line_b.split()


# --- difficulty scaling and dimensions keep solves that recorded no size ---


def _killed_on_large():
    """Eight problems, n = 100..800. Solver a times out on the four large
    ones; those rows carry no dimensions (a killed worker records the spec's
    metadata only), and a's row comes first in the file."""
    rows = []
    for i in range(1, 9):
        if i > 4:
            rows.append(_row(f"p{i}", "a", "time_limit", 60.0))
        else:
            rows.append(_row(f"p{i}", "a", time=1.0, **{"metadata.n": 100 * i}))
        rows.append(_row(f"p{i}", "b", time=2.0, **{"metadata.n": 100 * i}))
    return rows


def test_difficulty_scaling_counts_solves_that_recorded_no_size():
    table = difficulty_scaling(pd.DataFrame(_killed_on_large()), bin_count=2)
    table = table.set_index(["solver_id", "size_bin"])
    assert table.loc[("a", 0), "problem_count"] == 4 and table.loc[("a", 0), "success_count"] == 4
    assert table.loc[("a", 1), "problem_count"] == 4 and table.loc[("a", 1), "success_count"] == 0
    # the bins are over problems: both solvers see the same size ranges
    for solver in "ab":
        assert (table.loc[(solver, 0), "size_min"], table.loc[(solver, 0), "size_max"]) == (100.0, 400.0)
        assert (table.loc[(solver, 1), "size_min"], table.loc[(solver, 1), "size_max"]) == (500.0, 800.0)
    assert table.loc[("b", 1), "problem_count"] == 4


def test_difficulty_scaling_keeps_unsized_problems_in_an_unknown_size_row():
    rows = _killed_on_large() + [_row("p9", "a", "time_limit", 60.0), _row("p9", "b", "worker_error", None)]
    table = difficulty_scaling(pd.DataFrame(rows), bin_count=2)
    unknown = table[table["size_bin"].isna()].set_index("solver_id")
    assert unknown["problem_count"].to_dict() == {"a": 1, "b": 1}
    assert unknown["success_count"].to_dict() == {"a": 0, "b": 0}
    assert unknown["size_min"].isna().all() and unknown["size_max"].isna().all()
    assert table["problem_count"].sum() == 18


def test_problem_dimensions_use_any_row_that_recorded_them():
    frame = pd.DataFrame(_killed_on_large())
    dims = problem_dimensions(frame).set_index("problem")
    assert dims.loc[["p5", "p6", "p7", "p8"], "n"].tolist() == [500, 600, 700, 800]
    per_solver = solver_problem_tables(frame)
    assert per_solver["a"].set_index("problem").loc["p8", "n"] == 800
    assert per_solver["b"].set_index("problem").loc["p8", "n"] == 800


# --- one resolution of a duplicated solve in every view ---


def _two_successful_attempts(sloppy=5e-3):
    """Solver a has two optimal rows for p1: a slow accurate one first in the
    file, then a fast sloppy one. Every view must describe the fast one."""

    def row(solver, seconds, residual, objective):
        return _row("p1", solver, time=seconds, objective_value=objective, **{
            "kkt.primal_res_rel": residual, "kkt.dual_res_rel": residual, "kkt.duality_gap_rel": residual})

    return [row("a", 30.0, 1e-9, 1.000), row("a", 1.0, sloppy, 1.004), row("b", 10.0, 1e-9, 1.000)]


def test_duplicated_solve_is_the_same_attempt_in_every_table():
    frame = pd.DataFrame(_two_successful_attempts())
    assert shifted_geomean(frame, max_value=1000.0).set_index("solver_id").loc["a", "run_time_seconds"] == pytest.approx(1.0)
    assert solver_metrics(frame).set_index("solver_id").loc["a", "run_time_median_seconds"] == 1.0
    assert kkt_summary(frame).set_index("solver_id").loc["a", "primal_res_rel_max"] == 5e-3
    assert claimed_optimal_kkt_thresholds(frame).set_index("solver_id").loc["a", "worst_max"] == 5e-3
    comparison = problem_solver_comparison(frame).iloc[0]
    assert comparison["a__run_time_seconds"] == 1.0 and comparison["a__objective_value"] == 1.004
    spread = objective_spreads(frame).iloc[0]
    assert spread["objective_max"] == 1.004 and spread["solver_max"] == "a"


def test_views_resolve_failed_duplicates_by_the_metric_too():
    frame = pd.DataFrame([_row("p1", "a", "time_limit", 60.0), _row("p1", "a", "solver_error", 0.1),
                          _row("p1", "b")])
    assert status_matrix(frame).loc[("d", "p1"), "a"] == "solver_error"
    assert failure_rates(frame).set_index("solver_id").loc["a", "statuses"] == "solver_error=1"
    assert problem_solver_comparison(frame).iloc[0]["a__status"] == "solver_error"


def test_certificate_summary_counts_a_duplicated_solve_once():
    def cert(seconds, valid):
        return _row("p1", "a", "primal_infeasible", seconds, **{"kkt.certificate": "primal_infeasible",
                                                                 "kkt.valid": valid})

    summary = kkt_certificate_summary(pd.DataFrame([cert(2.0, False), cert(1.0, True)])).iloc[0]
    assert summary["infeasible_count"] == 1
    assert summary["cert_count"] == 1
    assert (summary["cert_valid"], summary["cert_invalid"]) == (1, 0)


def test_report_tables_describe_the_same_attempt(tmp_path, repo_root):
    # The report verifies the rows (SCS's default eps 1e-4, checked at 1e-3),
    # so the fast attempt is sloppy but still within that.
    run = _write_run(tmp_path, _two_successful_attempts(sloppy=5e-4))
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    report = tmp_path / "report"
    headline = pd.read_csv(report / "headline_solver_metrics.csv").set_index("solver_id")
    assert headline.loc["a", "run_time_median_seconds"] == 1.0
    kkt = pd.read_csv(report / "kkt_summary.csv").set_index("solver_id")
    assert kkt.loc["a", "primal_res_rel_max"] == 5e-4
    thresholds = pd.read_csv(report / "claimed_optimal_kkt_thresholds.csv").set_index("solver_id")
    assert thresholds.loc["a", "worst_max"] == 5e-4
    spreads = pd.read_csv(report / "objective_spreads.csv")
    assert spreads.loc[0, "objective_max"] == 1.004
    # status counts and slowest solves keep the rows as recorded
    counts = pd.read_csv(report / "status_counts.csv").set_index(["solver_id", "status"])
    assert counts.loc[("a", "optimal"), "count"] == 2
    slowest = pd.read_csv(report / "slowest_solves_run_time_seconds.csv")
    assert slowest["run_time_seconds"].tolist() == [30.0, 10.0, 1.0]


# --- the scoring helper column stays out of every written table ---


def test_report_scores_rows_once_and_writes_no_helper_column(tmp_path, repo_root, monkeypatch):
    from solver_benchmarks.analysis import outcomes

    rows = [_row(f"p{i}", s, "optimal" if (i + ord(s)) % 3 else "solver_error", 1.0 + i,
                 iterations=10 + i, objective_value=float(i), **{"metadata.n": i + 1, "kkt.primal_res_rel": 1e-9})
            for i in range(6) for s in "abc"]
    run = _write_run(tmp_path, rows)
    computed = []
    original = outcomes._compute_expected_outcomes
    monkeypatch.setattr(outcomes, "_compute_expected_outcomes",
                        lambda frame: computed.append(len(frame)) or original(frame))
    outputs = write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    # rows are scored about once, not once per table, plot and slice
    assert sum(computed) <= 2 * len(rows)
    for path in outputs:
        if path.suffix in {".csv", ".md"}:
            assert outcomes.OUTCOME_COLUMN not in path.read_text(), path


# --- a re-run of the report leaves no file of an earlier one under its names ---


def _manifest_run(tmp_path: Path, rows, datasets, solvers) -> Path:
    """A run whose expectation is the include lists (no dataset files under the repo root)."""
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    manifest = {"run_id": "run", "config": {
        "timeout_seconds": 60,
        "datasets": [{"id": d, "name": "maros_meszaros", "include": problems} for d, problems in datasets.items()],
        "solvers": [{"id": s, "solver": "scs", "settings": {}} for s in solvers],
    }}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "results.jsonl").write_text("".join(json.dumps(with_passing_kkt(r)) + "\n" for r in rows))
    return run


def _report_row(dataset, problem, solver, status="optimal", **extra):
    return {"dataset": dataset, "problem": problem, "solver_id": solver, "status": status,
            "run_time_seconds": 1.0, **extra}


def test_rerun_report_removes_tables_and_plots_it_no_longer_writes(tmp_path):
    row = _report_row
    empty_root = tmp_path / "repo"
    empty_root.mkdir()
    # interrupted: b never reached p2 and failed p1; sizes recorded
    first = [row("d", "p1", "a", **{"metadata.n": 1}), row("d", "p2", "a", **{"metadata.n": 2}),
             row("d", "p1", "b", "solver_error"), row("e", "q1", "a"), row("e", "q1", "b")]
    run = _manifest_run(tmp_path, first, {"d": ["p1", "p2"], "e": ["q1"]}, "ab")
    report = run / "report"
    write_run_report(run, repo_root=empty_root)
    unrelated = report / "notes.txt"
    unrelated.write_text("kept")
    owned = ["missing_results.csv", "failures_with_successful_alternatives_run_time_seconds.csv",
             "difficulty_scaling_run_time_seconds.csv", "difficulty_scaling_run_time_seconds.png",
             "by_dataset/e/headline_solver_metrics.csv", "by_dataset/e/kkt_summary.csv"]
    for name in owned:
        assert (report / name).exists(), name

    # rerun to completion with e excluded and no sizes recorded
    second = [row("d", "p1", "a"), row("d", "p2", "a"), row("d", "p1", "b"), row("d", "p2", "b")]
    (run / "results.jsonl").write_text("".join(json.dumps(with_passing_kkt(r)) + "\n" for r in second))
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"]["datasets"][1]["exclude"] = ["q1"]
    (run / "manifest.json").write_text(json.dumps(manifest))
    write_run_report(run, repo_root=empty_root)
    assert "Completion checks found no missing" in (report / "index.md").read_text()
    for name in owned:
        assert not (report / name).exists(), name
    assert unrelated.read_text() == "kept"
    assert (report / "by_dataset" / "d" / "headline_solver_metrics.csv").exists()


def test_rerun_report_removes_the_problem_table_of_a_solver_without_rows(tmp_path, repo_root):
    run = _write_run(tmp_path, [_row("p1", "a"), _row("p1", "c")], solvers=["a", "c"])
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    tables = tmp_path / "report" / "solver_problem_tables"
    assert (tables / "c.csv").exists()
    (run / "results.jsonl").write_text(json.dumps(with_passing_kkt(_row("p1", "a"))) + "\n")
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    assert (tables / "a.csv").exists() and not (tables / "c.csv").exists()


def test_sweep_variants_get_their_own_report_files(tmp_path):
    # The default sweep ids of scale: [1e-6, 1e+6]; distinct artifact directories.
    a, b = "scs__scale=1e-06", "scs__scale=1e+06"
    empty_root = tmp_path / "repo"
    empty_root.mkdir()
    rows = [_report_row(d, p, s, iterations=10 + k)
            for d in ("set=1e-06", "set=1e+06") for p in ("p1", "p2") for k, s in enumerate((a, b))]
    run = _manifest_run(tmp_path, rows, {"set=1e-06": ["p1", "p2"], "set=1e+06": ["p1", "p2"]}, [a, b])
    report = run / "report"
    # Files an earlier report wrote under the older name rule ('+' and '=' became '-').
    legacy = [report / "solver_problem_tables" / "scs__scale-1e-06.csv",
              report / "by_dataset" / "set-1e-06" / "headline_solver_metrics.csv"]
    for path in legacy:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n")
    write_run_report(run, repo_root=empty_root)
    tables = report / "solver_problem_tables"
    assert sorted(p.name for p in tables.glob("*.csv")) == [f"{b}.csv", f"{a}.csv"]
    assert set(pd.read_csv(tables / f"{a}.csv")["iterations"]) == {10}
    assert set(pd.read_csv(tables / f"{b}.csv")["iterations"]) == {11}
    markdown = (report / "index.md").read_text()
    for dataset in ("set=1e-06", "set=1e+06"):
        assert (report / "by_dataset" / dataset / "headline_solver_metrics.csv").exists()
        assert f"(by_dataset/{dataset}/headline_solver_metrics.csv)" in markdown
    for path in legacy:
        assert not path.exists(), path
    # the emptied folder of the older name goes too; a folder with other files stays
    assert not (report / "by_dataset" / "set-1e-06").exists()
    kept = report / "by_dataset" / "set-1e-06" / "notes.txt"
    kept.parent.mkdir()
    kept.write_text("mine")
    write_run_report(run, repo_root=empty_root)
    assert kept.read_text() == "mine"


def test_report_file_names_disambiguate_ids_that_share_a_slug():
    from solver_benchmarks.analysis.tables import report_file_names

    names = report_file_names(["a b", "a/b", "c", "C", "x=1e+06", "x=1e-06", "a-b~1"])
    assert names["x=1e+06"] == "x=1e+06" and names["x=1e-06"] == "x=1e-06"
    assert names["a-b~1"] == "a-b-1"
    # Ids whose names would coincide (case-insensitively, as on macOS and
    # Windows) all get the slug plus '~' and a hash of the id.
    for group in (("a b", "a/b"), ("c", "C")):
        for value in group:
            stem, _, digest = names[value].partition("~")
            assert stem.lower() == group[0].replace(" ", "-").lower() and len(digest) == 8
    assert len({name.lower() for name in names.values()}) == len(names)
    # Deterministic: the same names whatever the order.
    assert report_file_names(list(reversed(list(names)))) == names


def test_solvers_whose_ids_share_a_slug_get_separate_tables(tmp_path, repo_root):
    run = _write_run(tmp_path, [_row("p1", "a b", iterations=1), _row("p1", "a/b", iterations=2)])
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    tables = sorted((tmp_path / "report" / "solver_problem_tables").glob("*.csv"))
    assert len(tables) == 2
    assert sorted(int(pd.read_csv(path)["iterations"].iloc[0]) for path in tables) == [1, 2]
    # the artifact index says which id each hashed name belongs to
    markdown = (tmp_path / "report" / "index.md").read_text()
    for path in tables:
        solver = "a b" if pd.read_csv(path)["iterations"].iloc[0] == 1 else "a/b"
        assert f"(solver_problem_tables/{path.name}) (solver `{solver}`)" in markdown


def test_datasets_whose_ids_share_a_slug_are_labelled(tmp_path):
    from solver_benchmarks.analysis.tables import report_file_names

    empty_root = tmp_path / "repo"
    empty_root.mkdir()
    rows = [_report_row(d, "p1", "a") for d in ("x y", "x/y")]
    run = _manifest_run(tmp_path, rows, {"x y": ["p1"], "x/y": ["p1"]}, ["a"])
    write_run_report(run, repo_root=empty_root)
    markdown = (run / "report" / "index.md").read_text()
    for dataset, name in report_file_names(["x y", "x/y"]).items():
        assert "~" in name
        assert (run / "report" / "by_dataset" / name / "headline_solver_metrics.csv").exists()
        assert f"(by_dataset/{name}/headline_solver_metrics.csv) (dataset `{dataset}`)" in markdown


def test_unavailable_penalized_geomean_removes_the_earlier_one(tmp_path, repo_root):
    run = _write_run(tmp_path, [_row("p1", "a"), _row("p1", "b", "solver_error")])
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    for name in ("shifted_geomean_run_time_seconds.csv", "shifted_geomean_run_time_seconds.png"):
        assert (tmp_path / "report" / name).exists()
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["config"]["timeout_seconds"] = None
    (run / "manifest.json").write_text(json.dumps(manifest))
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    for name in ("shifted_geomean_run_time_seconds.csv", "shifted_geomean_run_time_seconds.png"):
        assert not (tmp_path / "report" / name).exists()


# --- every solver has its own line style, the same in every plot ---


def test_solver_styles_are_distinct_for_forty_solvers():
    from solver_benchmarks.analysis.plots import solver_styles

    ids = [f"s{i:02d}" for i in range(40)]
    styles = solver_styles(reversed(ids))
    assert list(styles) == ids
    assert len({(st["color"], st["linestyle"]) for st in styles.values()}) == 40
    assert len({(st["color"], st["marker"]) for st in styles.values()}) == 40
    # a solver's style depends on the set of ids, not on their order
    assert solver_styles(ids) == styles


def test_line_plots_give_each_solver_one_distinct_style(tmp_path, monkeypatch):
    import matplotlib.figure

    from solver_benchmarks.analysis import plots

    solvers = [f"solver_{i:02d}" for i in range(12)]
    rows = []
    for i, solver in enumerate(solvers):
        for j in range(6):
            row = _row(f"p{j}", solver, time=0.01 * (i + 1) * (j + 1), iterations=5, **{"metadata.n": 10 ** j})
            if i % 3 == 0:  # only some solvers have residuals: the KKT plot draws a subset
                row["kkt.primal_res_rel"] = 1e-9 * (j + 1)
            rows.append(row)
    frame = pd.DataFrame(rows)
    seen = {}
    original = matplotlib.figure.Figure.savefig

    def spy(self, path, *args, **kwargs):
        seen[Path(path).name] = [
            (line.get_label(), line.get_color(), line.get_linestyle(), line.get_marker())
            for ax in self.axes for line in ax.lines if not line.get_label().startswith("_")
        ]
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    run = _write_run(tmp_path, rows)
    plots.write_analysis_plots(run, output_dir=tmp_path / "plots", results=frame, expected=None,
                               geomean_options={"available": False})
    figures = ["performance_profile_run_time_seconds.png", "cactus_run_time_seconds.png",
               "kkt_accuracy_profile.png", "difficulty_scaling_run_time_seconds.png"]
    style_of = {}
    for name in figures:
        lines = seen[name]
        labels = {label for label, *_ in lines}
        assert labels == set(solvers if name != "kkt_accuracy_profile.png" else solvers[::3]), name
        assert len({(color, style) for _, color, style, _ in lines}) == len(lines), name
        for label, color, style, _marker in lines:
            assert style_of.setdefault(label, (color, style)) == (color, style), (name, label)
        if name.startswith("difficulty"):
            assert len({(color, marker) for _, color, _, marker in lines}) == len(lines)


def _profile_line_styles(monkeypatch):
    import matplotlib.colors
    import matplotlib.figure

    seen = {}
    original = matplotlib.figure.Figure.savefig

    def spy(self, path, *args, **kwargs):
        seen[Path(path).name] = [
            (line.get_label(), matplotlib.colors.to_hex(line.get_color()), line.get_linestyle())
            for ax in self.axes for line in ax.lines if not line.get_label().startswith("_")
        ]
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", spy)
    return seen


def test_planned_solvers_without_rows_get_their_own_style(tmp_path, monkeypatch):
    from solver_benchmarks.analysis import plots

    solvers = ["clarabel", "ecos", "highs", "osqp", "piqp", "scs"]
    rows = [_row(f"p{j}", s, time=0.1 * (i + 1) * (j + 1)) for i, s in enumerate(solvers[:3]) for j in range(4)]
    expected = pd.DataFrame([{"dataset": "d", "problem": f"p{j}", "solver_id": s} for s in solvers for j in range(4)])
    seen = _profile_line_styles(monkeypatch)
    run = _write_run(tmp_path, rows, solvers=solvers)
    plots.write_analysis_plots(run, output_dir=tmp_path / "plots", results=pd.DataFrame(rows),
                               expected=expected, geomean_options={"available": False})
    lines = seen["performance_profile_run_time_seconds.png"]
    assert {label for label, *_ in lines} == set(solvers)
    assert len({tuple(style) for _, *style in lines}) == len(solvers)
    styles = plots.solver_styles(solvers)
    assert all((color, ls) == (styles[label]["color"], styles[label]["linestyle"]) for label, color, ls in lines)
    # the profile on its own also styles planned solvers
    seen.clear()
    plots._write_performance_profile(pd.DataFrame(rows), tmp_path, "run_time_seconds", expected=expected)
    lines = seen["performance_profile_run_time_seconds.png"]
    assert len({tuple(style) for _, *style in lines}) == len(lines) == len(solvers)


def test_report_styles_a_configured_solver_without_rows(tmp_path, monkeypatch, repo_root):
    import solver_benchmarks.analysis.markdown_report as report

    solvers = ["clarabel", "osqp", "scs"]
    rows = [_row(f"p{j}", s, time=0.1 * (j + 1)) for s in ("osqp", "scs") for j in range(3)]
    expected = pd.DataFrame([{"dataset": "d", "problem": f"p{j}", "solver_id": s} for s in solvers for j in range(3)])
    monkeypatch.setattr(report, "expected_results", lambda *args, **kwargs: expected)
    seen = _profile_line_styles(monkeypatch)
    write_run_report(_write_run(tmp_path, rows, solvers=solvers), output_dir=tmp_path / "report", repo_root=repo_root)
    lines = seen["performance_profile_run_time_seconds.png"]
    assert {label for label, *_ in lines} == set(solvers)
    assert len({tuple(style) for _, *style in lines}) == 3


# --- the cactus plot counts like the profile ---


def _cactus_curves(frame, tmp_path, metric, **kwargs):
    from solver_benchmarks.analysis.plots import _write_cactus

    captured = {}
    real_step = mpl_axes.Axes.step

    def capturing_step(self, x, y, *args, label=None, **kw):
        captured[label] = (np.asarray(x).copy(), np.asarray(y).copy())
        return real_step(self, x, y, *args, label=label, **kw)

    mpl_axes.Axes.step = capturing_step  # type: ignore[method-assign]
    try:
        _write_cactus(frame, tmp_path, metric, **kwargs)
    finally:
        mpl_axes.Axes.step = real_step  # type: ignore[method-assign]
    return captured


def test_cactus_uses_the_profile_denominator_and_floor(tmp_path):
    frame = pd.DataFrame([_row("p1", "a", iterations=0), _row("p2", "a", iterations=12),
                          _row("p1", "b", iterations=9), _row("p2", "b", iterations=15)])
    expected = pd.DataFrame([{"dataset": "d", "problem": p, "solver_id": s}
                             for p in ("p1", "p2", "p3", "p4") for s in "ab"])
    # a presolve-only solve sits at the profile's floor of one iteration
    assert _cactus_curves(frame, tmp_path, "iterations")["a"][0].tolist() == [1.0, 12.0]
    # planned solves never recorded count in the denominator, as in the profile
    profile = performance_profile(frame, metric="iterations", expected=expected)
    curves = _cactus_curves(frame, tmp_path, "iterations", expected=expected)
    assert curves["a"][1][-1] == profile["a"].iloc[-1] == 0.5
    # metrics without a floor still draw a zero at 1e-12
    zero = pd.DataFrame([_row("p1", "a", **{"kkt.primal_res_rel": 0.0}),
                         _row("p2", "a", **{"kkt.primal_res_rel": 1e-6})])
    assert _cactus_curves(zero, tmp_path, "kkt.primal_res_rel")["a"][0].tolist() == [1e-12, 1e-6]


# --- every KKT residual gets the residual geomean defaults ---


def test_every_relative_kkt_residual_gets_residual_defaults():
    import scipy.sparse as sp

    from solver_benchmarks.analysis import derive, kkt
    from solver_benchmarks.analysis.profiles import KKT_SHIFT, metric_defaults

    one = sp.csc_matrix(np.array([[1.0]]))
    records = [
        kkt.qp_residuals(sp.csc_matrix((1, 1)), np.array([1.0]), one, np.array([0.0]), np.array([1.0]),
                         np.array([0.5]), np.array([0.0])),
        kkt.cone_residuals(sp.csc_matrix((1, 1)), np.array([-1.0]), one, np.array([1.0]), {"l": 1},
                           np.array([1.0]), np.array([1.0]), np.array([0.0])),
        kkt.cone_primal_infeasibility_cert(sp.csc_matrix(np.array([[1.0], [-1.0]])), np.array([0.0, -1.0]),
                                           {"z": 1, "l": 1}, np.array([1.0, 1.0])),
    ]
    emitted = {key for record in records for key, value in record.items()
               if isinstance(value, float) and (key.endswith("_rel") or key == "cert_res")}
    checked = set(derive.KKT_FIELDS) | set(derive.QP_FIELDS) | set(derive.CONE_FIELDS) | {"cert_res"}
    assert {"dual_bound_res_rel", "primal_cone_res_rel", "primal_eq_res_rel", "cert_res"} <= emitted | checked
    for field in sorted(emitted | checked | {"comp_slack"}):
        assert metric_defaults(f"kkt.{field}") == (1.0, KKT_SHIFT), field
    assert metric_defaults("run_time_seconds") == (1.0e3, 10.0)
    assert metric_defaults("iterations") == (1.0e6, 100.0)


# --- a derived run says so ---


def test_report_of_a_verified_merge_states_its_provenance(tmp_path, repo_root):
    from solver_benchmarks.analysis.derive import kkt_verify, merge_runs

    def shard(name, solver, rows):
        run = tmp_path / name
        run.mkdir()
        (run / "manifest.json").write_text(json.dumps({"run_id": name, "config": {
            "name": name, "timeout_seconds": 60,
            "datasets": [{"id": "d", "name": "synthetic_qp", "include": ["p1", "p2"]}],
            "solvers": [{"id": solver, "solver": "scs", "settings": {}}]}}))
        (run / "results.jsonl").write_text("".join(json.dumps(with_passing_kkt(r)) + "\n" for r in rows))
        (run / "run_config.yaml").write_text(f"run:\n  name: {name}\n")
        return run

    good = {"primal_res_rel": 1e-9, "dual_res_rel": 1e-9, "duality_gap_rel": 1e-9}
    loose = {"primal_res_rel": 1e-3, "dual_res_rel": 1e-9, "duality_gap_rel": 1e-9}
    m3 = shard("m3", "A", [_row("p1", "A", kkt=good), _row("p2", "A", kkt=loose)])
    m4 = shard("m4", "B", [_row("p1", "B", kkt=good)])
    merged = tmp_path / "merged"
    merge_runs([m3, m4], merged)
    write_run_report(merged, output_dir=tmp_path / "merge_report", repo_root=repo_root)
    merge_md = (tmp_path / "merge_report" / "index.md").read_text()
    assert "### Derived Run" in merge_md
    assert str(m3) in merge_md and str(m4) in merge_md
    assert "merge of 2 runs" in merge_md.split("## Run Scope")[0]
    assert "first merged run" in merge_md

    verified = tmp_path / "verified"
    kkt_verify(merged, verified, tol=1e-6, promote=True)
    write_run_report(verified, output_dir=tmp_path / "verified_report", repo_root=repo_root)
    markdown = (tmp_path / "verified_report" / "index.md").read_text()
    provenance = markdown.split("## Provenance")[1]
    assert "### Derived Run" in provenance
    assert "kkt_verify" in provenance and "1e-06" in provenance
    assert "| promote | True |" in provenance and "| missing_is_failure | True |" in provenance
    derived = provenance.split("### Derived Run")[1].split("\n### ")[0]
    assert "| A |" in derived and "| B |" in derived
    assert "demoted" in derived and "kept" in derived
    # the verified merge still lists the merged runs and labels the copied config
    assert str(m3) in derived and str(m4) in derived
    assert "first merged run" in provenance
    summary = markdown.split("## Run Scope")[0]
    assert "bench kkt-verify" in summary and "1e-06" in summary

    # the other order: shards verified separately, then merged
    v3, v4 = tmp_path / "v3", tmp_path / "v4"
    kkt_verify(m3, v3, tol=1e-6, promote=True)
    kkt_verify(m4, v4, tol=1e-4, missing_is_failure=False, promote=False)
    verified_first = tmp_path / "verified_first"
    merge_runs([v3, v4], verified_first)
    write_run_report(verified_first, output_dir=tmp_path / "vf_report", repo_root=repo_root)
    markdown = (tmp_path / "vf_report" / "index.md").read_text()
    summary = markdown.split("## Run Scope")[0]
    assert "merge of 2 runs" in summary
    assert "bench kkt-verify" in summary and "1e-06" in summary and "0.0001" in summary
    derived = markdown.split("### Derived Run")[1].split("\n### ")[0]
    merged_runs = derived.split("Merged runs")[1]
    row_v3 = next(line for line in merged_runs.splitlines() if str(v3) in line)
    assert "kkt-verify" in row_v3 and "1e-06" in row_v3 and "1 demoted" in row_v3
    row_v4 = next(line for line in merged_runs.splitlines() if str(v4) in line)
    assert "0.0001" in row_v4 and "without `--promote`" in row_v4


# --- a retried success wins even under a metric the solver does not record ---


def _failed_then_retried():
    """Solver plain records no ``solve_time_seconds``: p1 failed, then succeeded on retry."""
    return [
        _row("p1", "plain", "solver_error", 0.1),
        _row("p1", "plain", time=2.0),
        _row("p2", "plain", time=1.0),
        _row("p1", "timed", time=1.0, solve_time_seconds=0.5),
        _row("p2", "timed", time=1.0, solve_time_seconds=0.5),
    ]


def test_retried_success_without_the_metric_beats_the_failure():
    frame = pd.DataFrame(_failed_then_retried())
    assert status_matrix(frame, metric="solve_time_seconds").loc[("d", "p1"), "plain"] == "optimal"
    metrics = solver_metrics(frame, metric="solve_time_seconds").set_index("solver_id")
    assert metrics.loc["plain", "success_count"] == 2 and metrics.loc["plain", "failure_count"] == 0
    rates = failure_rates(frame, metric="solve_time_seconds").set_index("solver_id")
    assert rates.loc["plain", "statuses"] == "optimal=2"
    # a kkt metric the success did not record: same rule
    kkt = pd.DataFrame([_row("p1", "a", "solver_error", 0.1, **{"kkt.primal_res_rel": 1e-9}),
                        _row("p1", "a", time=2.0), _row("p1", "b", **{"kkt.primal_res_rel": 1e-9})])
    assert status_matrix(kkt, metric="kkt.primal_res_rel").loc[("d", "p1"), "a"] == "optimal"
    assert kkt_summary(kkt, metric="kkt.primal_res_rel").set_index("solver_id").loc["a", "success_count"] == 1
    # among successes, a usable value still wins over a missing one, then the best value
    both = pd.DataFrame([_row("p1", "a", time=3.0), _row("p1", "a", time=2.0, solve_time_seconds=0.9),
                         _row("p1", "a", time=1.0, solve_time_seconds=0.4)])
    comparison = problem_solver_comparison(both, metric="solve_time_seconds").iloc[0]
    assert comparison["a__run_time_seconds"] == 1.0


def test_report_with_an_unrecorded_metric_keeps_the_retried_success(tmp_path, repo_root):
    run = _write_run(tmp_path, _failed_then_retried())
    write_run_report(run, metric="solve_time_seconds", output_dir=tmp_path / "report", repo_root=repo_root)
    report = tmp_path / "report"
    headline = pd.read_csv(report / "headline_solver_metrics.csv").set_index("solver_id")
    assert headline.loc["plain", "success_count"] == 2
    matrix = pd.read_csv(report / "status_matrix.csv").set_index(["dataset", "problem"])
    assert matrix.loc[("d", "p1"), "plain"] == "optimal"


# --- the resolved frame keeps file order ---


def test_resolved_rows_keep_file_order():
    from solver_benchmarks.analysis.tables import one_row_per_solve

    frame = pd.DataFrame([_row("p1", "a", time=5.0), _row("p2", "a", time=1.0), _row("p3", "a", time=3.0),
                          _row("p2", "a", "solver_error", 0.1), _row("p1", "b", time=2.0)])
    resolved = one_row_per_solve(frame, metric="run_time_seconds")
    assert resolved.index.tolist() == [0, 1, 2, 4]


# --- the report sizes problems from every recorded row ---


def test_report_sizes_problems_from_discarded_duplicates(tmp_path, repo_root):
    # p3's only sized row is a failure that loses to the killed attempt (faster)
    rows = [_row(f"p{i}", s, time=1.0, **{"metadata.n": 10 * i}) for i in (1, 2) for s in "ab"]
    rows += [_row("p3", "a", "solver_error", 12.0, **{"metadata.n": 30}), _row("p3", "a", "time_limit", 10.0),
             _row("p3", "b", "time_limit", 10.0)]
    run = _write_run(tmp_path, rows)
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root)
    scaling = pd.read_csv(tmp_path / "report" / "difficulty_scaling_run_time_seconds.csv")
    assert scaling["size_bin"].notna().all()
    for solver in "ab":
        table = pd.read_csv(tmp_path / "report" / "solver_problem_tables" / f"{solver}.csv").set_index("problem")
        assert table.loc["p3", "n"] == 30
        assert table.loc["p3", "status"] == "time_limit"


# --- the plots of a report count and resolve like its tables ---


def test_report_plots_use_the_expected_problems_and_the_report_resolution(tmp_path, monkeypatch):
    empty_root = tmp_path / "repo"
    empty_root.mkdir()
    row = _report_row
    # interrupted: p3 and p4 never ran; a ran p1 twice, the attempt with fewer
    # iterations is the less accurate one (within the 1e-3 SCS's default eps
    # is verified at)
    def residuals(primal):
        return {"kkt.primal_res_rel": primal, "kkt.dual_res_rel": 1e-9, "kkt.duality_gap_rel": 1e-9}

    rows = [row("d", "p1", "a", iterations=50, **residuals(1e-9)),
            row("d", "p1", "a", iterations=5, **residuals(5e-4)),
            row("d", "p2", "a", iterations=7, **residuals(1e-9)),
            row("d", "p1", "b", iterations=9, **residuals(1e-9)),
            row("d", "p2", "b", iterations=9, **residuals(1e-9))]
    run = _manifest_run(tmp_path, rows, {"d": ["p1", "p2", "p3", "p4"]}, "ab")
    steps, boxes = {}, []
    real_step, real_boxplot = mpl_axes.Axes.step, mpl_axes.Axes.boxplot

    def capturing_step(self, x, y, *args, label=None, **kw):
        steps[label] = np.asarray(y).copy()
        return real_step(self, x, y, *args, label=label, **kw)

    def capturing_boxplot(self, data, *args, tick_labels=None, **kw):
        boxes.append(dict(zip(tick_labels, [np.asarray(d).copy() for d in data])))
        return real_boxplot(self, data, *args, tick_labels=tick_labels, **kw)

    monkeypatch.setattr(mpl_axes.Axes, "step", capturing_step)
    monkeypatch.setattr(mpl_axes.Axes, "boxplot", capturing_boxplot)
    write_run_report(run, metric="iterations", repo_root=empty_root)
    profile = pd.read_csv(run / "report" / "performance_profile_iterations.csv")
    assert steps["a"][-1] == profile["a"].iloc[-1] == 0.5
    kkt = pd.read_csv(run / "report" / "kkt_summary.csv").set_index("solver_id")
    assert kkt.loc["a", "primal_res_rel_max"] == 5e-4
    # the boxplot draws the attempt the tables describe (fewest iterations)
    assert sorted(boxes[0]["a"].tolist()) == sorted(np.log10([5e-4, 1e-9]).tolist())
