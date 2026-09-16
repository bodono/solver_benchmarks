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


def _write_run_at(path, rows, config=None):
    import json
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "manifest.json").write_text(json.dumps({"config": config or {}}))


def test_output_overlapping_a_source_is_refused(tmp_path):
    import pytest

    from solver_benchmarks.analysis.derive import kkt_verify, merge_runs
    run = tmp_path / "run"
    _write_run_at(run, [{"dataset": "d", "problem": "p", "solver_id": "s", "status": "optimal",
                      "kkt": {"primal_res_rel": 0.0, "dual_res_rel": 0.0, "duality_gap_rel": 0.0}}])
    with pytest.raises(ValueError):
        kkt_verify(run, run, tol=1e-3, overwrite=True)
    with pytest.raises(ValueError):
        merge_runs([run], run / "nested", overwrite=True)
    assert (run / "results.jsonl").exists()  # nothing was deleted


def test_missing_or_nonfinite_residual_never_verifies():
    from solver_benchmarks.analysis.derive import worst_relative_residual
    ok = {"kkt": {"primal_res_rel": 1e-9, "dual_res_rel": 1e-9, "duality_gap_rel": 1e-9}}
    assert worst_relative_residual(ok) == 1e-9
    assert worst_relative_residual({"kkt": {"primal_res_rel": 0.0, "dual_res_rel": None, "duality_gap_rel": 0.0}}) is None
    assert worst_relative_residual({"kkt": {"primal_res_rel": float("nan"), "dual_res_rel": 0.0, "duality_gap_rel": 0.0}}) is None
    assert worst_relative_residual({"kkt": {"primal_res_rel": 0.0, "dual_res_rel": 0.0}}) is None


def test_cone_infeasible_point_is_not_verified():
    from solver_benchmarks.analysis.derive import worst_relative_residual
    # Ax + s = b holds but s is outside the cone: the cone residual must count.
    rec = {"kkt": {"form": "cone", "primal_res_rel": 0.0, "dual_res_rel": 0.0, "duality_gap_rel": 0.0,
                   "primal_cone_res": 1.0, "dual_cone_res": 0.0}}
    assert worst_relative_residual(rec) == 1.0
    assert worst_relative_residual({"kkt": {"form": "cone", "primal_res_rel": 0.0, "dual_res_rel": 0.0, "duality_gap_rel": 0.0}}) is None


def test_merge_unions_selections_and_keeps_manifests(tmp_path):
    import json

    from solver_benchmarks.analysis.derive import merge_runs
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_run_at(a, [{"dataset": "A", "problem": "p1", "solver_id": "s1", "status": "optimal"}],
               config={"datasets": [{"id": "A"}], "solvers": [{"id": "s1", "settings": {"eps": 1e-4}}]})
    _write_run_at(b, [{"dataset": "B", "problem": "p2", "solver_id": "s2", "status": "optimal"}],
               config={"datasets": [{"id": "B"}], "solvers": [{"id": "s2", "settings": {"eps": 1e-6}}]})
    out = tmp_path / "merged"
    merge_runs([a, b], out, overwrite=True)
    m = json.loads((out / "manifest.json").read_text())
    assert sorted(d["id"] for d in m["config"]["datasets"]) == ["A", "B"]
    assert sorted(s["id"] for s in m["config"]["solvers"]) == ["s1", "s2"]
    assert len(m["derived"]["source_manifests"]) == 2
