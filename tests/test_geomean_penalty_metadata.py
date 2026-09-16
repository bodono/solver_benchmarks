"""Fixed manifest penalties through CLI, plots, and complete reports."""

import json

import pandas as pd
import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.penalties import geomean_time_limits
from solver_benchmarks.cli import main


def _manifest(dataset="qp", timeout=300):
    return {"config": {
        "datasets": [{"id": dataset, "name": "synthetic_qp", "include": ["one_variable_eq"]}],
        "timeout_seconds": timeout,
        "solvers": [{"id": "a", "solver": "scs", "settings": {}}],
    }}


def _merge(*sources):
    return {"config": {
        **sources[0]["config"],
        "datasets": [entry for source in sources for entry in source["config"]["datasets"]],
    }, "derived": {"kind": "merge", "source_manifests": list(sources)}}


def _write_run(tmp_path, manifest, datasets=("qp",)):
    run = tmp_path / "run"
    run.mkdir()
    if manifest is not None:
        (run / "manifest.json").write_text(json.dumps(manifest))
    rows = [{"dataset": dataset, "problem": "one_variable_eq", "solver_id": "a",
             "solver": "scs", "status": "time_limit", "run_time_seconds": 300,
             "iterations": 12} for dataset in datasets]
    (run / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run


def test_nested_source_manifests_recover_each_dataset_limit(tmp_path):
    manifest = _merge(_merge(_manifest("qp", 300), _manifest("lp", 300)),
                      _manifest("sdp", 900), _manifest("mittelmann", 1800))
    run = _write_run(tmp_path, manifest)
    assert geomean_time_limits(run, metric="run_time_seconds") == {
        "qp": 300, "lp": 300, "sdp": 900, "mittelmann": 1800,
    }


@pytest.mark.parametrize("timeout", [None, 0, -1, float("inf"), float("nan"), True])
def test_invalid_or_unlimited_manifest_requires_explicit_penalty(tmp_path, timeout):
    run = _write_run(tmp_path, _manifest(timeout=timeout))
    with pytest.raises(ValueError, match="--max-value"):
        geomean_time_limits(run, metric="run_time_seconds")
    assert geomean_time_limits(run, metric="run_time_seconds", max_value=17) is None


@pytest.mark.parametrize("kind", ["conflict", "lost_verified_merge", "missing_manifest"])
def test_ambiguous_or_missing_provenance_requires_override(tmp_path, kind):
    if kind == "conflict":
        manifest = _merge(_manifest(timeout=300), _manifest(timeout=900))
    elif kind == "lost_verified_merge":
        manifest = _manifest()
        manifest["derived"] = {"kind": "kkt_verify", "source": "/unavailable/merged",
                               "selections": [{"datasets": manifest["config"]["datasets"], "solvers": ["a"]}]}
    else:
        manifest = None
    run = _write_run(tmp_path, manifest)
    result = CliRunner().invoke(main, ["geomean", str(run)])
    assert result.exit_code != 0
    assert "--max-value" in result.output
    result = CliRunner().invoke(main, ["geomean", str(run), "--max-value", "17"])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(run / "shifted_geomean_run_time_seconds.csv")
    assert table["max_value"].tolist() == [17]
    assert table["run_time_seconds"].iloc[0] == pytest.approx(17)


def test_geomean_cli_uses_fixed_timeout_and_metric_specific_shift(tmp_path):
    run = _write_run(tmp_path, _manifest(timeout=300))
    result = CliRunner().invoke(main, ["geomean", str(run)])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(run / "shifted_geomean_run_time_seconds.csv")
    assert table["max_value"].tolist() == [900]
    assert table["run_time_seconds"].iloc[0] == pytest.approx(900)
    result = CliRunner().invoke(main, ["geomean", str(run), "--metric", "iterations"])
    assert result.exit_code == 0, result.output
    table = pd.read_csv(run / "shifted_geomean_iterations.csv")
    assert table["shift"].tolist() == [100]
    assert table["max_value"].tolist() == [1e6]


def test_success_only_geomean_needs_no_manifest_limit(tmp_path):
    run = _write_run(tmp_path, None)
    result = CliRunner().invoke(main, ["geomean", str(run), "--success-only"])
    assert result.exit_code == 0, result.output


def test_report_uses_dataset_penalties_in_tables_plots_and_methodology(tmp_path, monkeypatch):
    from solver_benchmarks.analysis import plots

    manifest = _merge(_manifest("qp", 300), _manifest("sdp", 900))
    run = _write_run(tmp_path, manifest, datasets=("qp", "sdp"))
    calls = []
    original = plots.shifted_geomean

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(plots, "shifted_geomean", capture)
    result = CliRunner().invoke(main, ["report", str(run)])
    assert result.exit_code == 0, result.output
    report = run / "report"
    pooled = pd.read_csv(report / "shifted_geomean_run_time_seconds.csv")
    assert pd.isna(pooled["max_value"].iloc[0])
    assert json.loads(pooled["max_value_by_dataset"].iloc[0]) == {"qp": 900, "sdp": 2700}
    assert pooled["run_time_seconds"].iloc[0] == pytest.approx((910 * 2710) ** 0.5 - 10)
    assert calls[0]["timeout_seconds"] == {"qp": 300, "sdp": 900}
    for dataset, penalty in (("qp", 900), ("sdp", 2700)):
        table = pd.read_csv(report / "by_dataset" / dataset / "headline_solver_metrics.csv")
        assert table["penalized_shifted_geomean_run_time_seconds"].iloc[0] == pytest.approx(penalty)
        table = pd.read_csv(report / "by_dataset" / dataset / "failure_penalties.csv")
        assert table["max_value"].tolist() == [penalty]
    methodology = pd.read_csv(report / "failure_penalties.csv").set_index("dataset")
    assert methodology["max_value"].to_dict() == {"qp": 900, "sdp": 2700}
    assert "max_value_by_dataset" in (report / "index.md").read_text()
    assert (report / "shifted_geomean_run_time_seconds.png").exists()


@pytest.mark.parametrize("command, metric, penalty", [("plot", "run_time_seconds", 17),
                                                       ("report", "run_time_seconds", 17),
                                                       ("report", "iterations", 25)])
def test_plot_and_report_overrides_are_exact_and_apply_only_to_selected_metric(
    tmp_path, monkeypatch, command, metric, penalty,
):
    from solver_benchmarks.analysis import plots

    run = _write_run(tmp_path, _manifest(timeout=None))
    original = plots.shifted_geomean
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(plots, "shifted_geomean", capture)
    result = CliRunner().invoke(main, [command, str(run), "--metric", metric, "--max-value", str(penalty)])
    assert result.exit_code == 0, result.output
    assert calls[0]["max_value"] == penalty
    assert calls[0]["timeout_seconds"] is None
    if command == "report":
        table = pd.read_csv(run / "report" / f"shifted_geomean_{metric}.csv")
        assert table["max_value"].tolist() == [penalty]
        assert table[metric].iloc[0] == pytest.approx(penalty)
        if metric != "iterations":
            iterations = pd.read_csv(run / "report" / "shifted_geomean_iterations.csv")
            assert iterations["max_value"].tolist() == [1e6]
