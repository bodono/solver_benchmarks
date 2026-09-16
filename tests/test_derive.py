import json

from solver_benchmarks.analysis.derive import kkt_verify, merge_runs


def _write_run(tmp_path, name, rows):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": name, "config": {"name": name}}))
    (run_dir / "run_config.yaml").write_text("run: {}\n")
    return run_dir


def _row(problem, solver_id, status="optimal", kkt=None):
    return {
        "dataset": "d",
        "problem": problem,
        "solver_id": solver_id,
        "status": status,
        "run_time_seconds": 1.0,
        "kkt": kkt,
        "metadata": {},
    }


def test_kkt_verify_demotes_inaccurate_and_missing(tmp_path):
    rows = [
        _row("p1", "a", kkt={"primal_res_rel": 1e-7, "dual_res_rel": 1e-8, "duality_gap_rel": 1e-9}),
        _row("p2", "a", kkt={"primal_res_rel": 1e-3, "dual_res_rel": 1e-8, "duality_gap_rel": 1e-9}),
        _row("p3", "a", kkt=None),
        _row("p1", "b", status="time_limit", kkt=None),
    ]
    src = _write_run(tmp_path, "src", rows)
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6)
    out_rows = [json.loads(l) for l in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    by_key = {(r["problem"], r["solver_id"]): r for r in out_rows}
    assert by_key[("p1", "a")]["status"] == "optimal"
    assert by_key[("p2", "a")]["status"] == "optimal_inaccurate"
    assert by_key[("p2", "a")]["metadata"]["kkt_verify"]["original_status"] == "optimal"
    assert by_key[("p3", "a")]["status"] == "optimal_inaccurate"
    assert by_key[("p1", "b")]["status"] == "time_limit"  # untouched
    assert summary["kept"] == {"a": 1}
    assert summary["demoted"] == {"a": 2}
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["derived"]["kind"] == "kkt_verify"
    assert manifest["derived"]["tol"] == 1e-6
    assert (tmp_path / "out" / "run_config.yaml").exists()


def test_kkt_verify_allow_missing_keeps_rows_without_residuals(tmp_path):
    src = _write_run(tmp_path, "src", [_row("p3", "a", kkt=None)])
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6, missing_is_failure=False)
    assert summary["kept"] == {"a": 1} and summary["demoted"] == {}
    assert summary["missing_residuals"] == {"a": 1}


def test_merge_runs_concatenates_and_reports_duplicates(tmp_path):
    a = _write_run(tmp_path, "a", [_row("p1", "s1"), _row("p2", "s1")])
    b = _write_run(tmp_path, "b", [_row("p1", "s2"), _row("p2", "s1")])
    summary = merge_runs([a, b], tmp_path / "merged")
    rows = (tmp_path / "merged" / "results.jsonl").read_text().splitlines()
    assert len(rows) == 4 and summary["rows"] == 4
    assert summary["duplicate_keys"] == 1
    manifest = json.loads((tmp_path / "merged" / "manifest.json").read_text())
    assert manifest["derived"]["kind"] == "merge"
    assert [s["rows"] for s in manifest["derived"]["sources"]] == [2, 2]


def test_kkt_verify_promote_accepts_accurate_inaccurate_rows(tmp_path):
    good = {"primal_res_rel": 1e-7, "dual_res_rel": 1e-8, "duality_gap_rel": 1e-9}
    bad = {"primal_res_rel": 1e-2, "dual_res_rel": 1e-8, "duality_gap_rel": 1e-9}
    rows = [
        _row("p1", "a", status="optimal_inaccurate", kkt=good),
        _row("p2", "a", status="max_iter_reached", kkt=bad),
        _row("p3", "a", status="time_limit", kkt=None),
    ]
    src = _write_run(tmp_path, "src", rows)
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6, promote=True)
    out_rows = {json.loads(l)["problem"]: json.loads(l) for l in (tmp_path / "out" / "results.jsonl").read_text().splitlines()}
    assert out_rows["p1"]["status"] == "optimal"
    assert out_rows["p1"]["metadata"]["kkt_verify"]["original_status"] == "optimal_inaccurate"
    assert out_rows["p2"]["status"] == "max_iter_reached"
    assert out_rows["p3"]["status"] == "time_limit"
    assert summary["promoted"] == {"a": 1}
    # without promote nothing changes
    kkt_verify(src, tmp_path / "out2", tol=1e-6)
    out2 = {json.loads(l)["problem"]: json.loads(l) for l in (tmp_path / "out2" / "results.jsonl").read_text().splitlines()}
    assert out2["p1"]["status"] == "optimal_inaccurate"
