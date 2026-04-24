import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.cli import main


def _analysis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "problem": "p1",
                "solver_id": "solver_a",
                "status": "optimal",
                "run_time_seconds": 1.0,
            },
            {
                "problem": "p1",
                "solver_id": "solver_b",
                "status": "optimal",
                "run_time_seconds": 2.0,
            },
            {
                "problem": "p2",
                "solver_id": "solver_a",
                "status": "time_limit",
                "run_time_seconds": 9.0,
            },
            {
                "problem": "p2",
                "solver_id": "solver_b",
                "status": "optimal",
                "run_time_seconds": 4.0,
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


def test_load_summary_and_cli_analysis_commands(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = _analysis_frame().to_dict("records")
    with (run_dir / "results.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    loaded = load_results(run_dir)
    summary = solver_summary(run_dir)

    assert len(loaded) == 4
    assert set(summary["solver_id"]) == {"solver_a", "solver_b"}

    runner = CliRunner()
    summary_result = runner.invoke(main, ["summary", str(run_dir)])
    assert summary_result.exit_code == 0
    assert "solver_a" in summary_result.output
    assert "time_limit" in summary_result.output

    profile_result = runner.invoke(main, ["profile", str(run_dir), "--metric", "run_time_seconds"])
    assert profile_result.exit_code == 0
    assert (run_dir / "performance_profile_run_time_seconds.csv").exists()

    geomean_result = runner.invoke(main, ["geomean", str(run_dir), "--metric", "run_time_seconds"])
    assert geomean_result.exit_code == 0
    assert (run_dir / "shifted_geomean_run_time_seconds.csv").exists()
