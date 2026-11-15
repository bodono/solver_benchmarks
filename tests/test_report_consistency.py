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

from solver_benchmarks.analysis.markdown_report import write_run_report
from solver_benchmarks.analysis.penalties import geomean_time_limits
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.tables import (
    failure_rates,
    kkt_summary,
    objective_spreads,
    pairwise_speedups,
    performance_ratio_matrix,
    problem_solver_comparison,
    slowest_solves,
    solver_metrics,
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
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
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
    frame = pd.DataFrame([_row("p", "a", time=50.0), _row("p", "b", "time_limit", 10.0)])
    with caplog.at_level(logging.WARNING):
        shifted_geomean(frame, max_value=20.0)
    assert "failure penalty" in caplog.text


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
    assert headline.loc["b", "success_count"] == geomean.loc["b", "success_count"] == 2
    assert headline.loc["b", "failure_count"] == geomean.loc["b", "failure_count"] == 2
    assert headline.loc["b", "success_rate"] == 0.5
    assert "not_attempted=2" in headline.loc["b", "statuses"]


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
