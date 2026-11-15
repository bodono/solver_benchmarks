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


def test_qp_dual_bound_residual_counts_when_recorded():
    from solver_benchmarks.analysis.derive import worst_relative_residual
    base = {"form": "qp", "primal_res_rel": 0.0, "dual_res_rel": 0.0, "duality_gap_rel": 0.0}
    assert worst_relative_residual({"kkt": {**base, "dual_bound_res_rel": 0.5}}) == 0.5
    assert worst_relative_residual({"kkt": {**base, "dual_bound_res_rel": float("nan")}}) is None
    # Records written before the residual existed are checked on the rest.
    assert worst_relative_residual({"kkt": base}) == 0.0


def test_kkt_verify_keeps_qp_records_without_dual_bound_residual(tmp_path):
    good = {"form": "qp", "primal_res_rel": 1e-9, "dual_res_rel": 1e-9, "duality_gap_rel": 1e-9}
    src = _write_run(tmp_path, "src", [
        _row("old", "a", kkt=good),
        _row("new", "a", kkt={**good, "dual_bound_res_rel": 0.5}),
    ])
    kkt_verify(src, tmp_path / "out", tol=1e-6)
    out_rows = [json.loads(l) for l in (tmp_path / "out" / "results.jsonl").read_text().splitlines()]
    by_problem = {r["problem"]: r for r in out_rows}
    assert by_problem["old"]["status"] == "optimal"
    assert by_problem["new"]["status"] == "optimal_inaccurate"


def test_merge_unions_selections_and_keeps_manifests(tmp_path):
    import json

    from solver_benchmarks.analysis.derive import merge_runs
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_run_at(a, [{"dataset": "A", "problem": "p1", "solver_id": "s1", "status": "optimal"}],
               config={"datasets": [{"id": "A", "name": "A"}], "solvers": [{"id": "s1", "settings": {"eps": 1e-4}}]})
    _write_run_at(b, [{"dataset": "B", "problem": "p2", "solver_id": "s2", "status": "optimal"}],
               config={"datasets": [{"id": "B", "name": "B"}], "solvers": [{"id": "s2", "settings": {"eps": 1e-6}}]})
    out = tmp_path / "merged"
    merge_runs([a, b], out, overwrite=True)
    m = json.loads((out / "manifest.json").read_text())
    assert sorted(d["id"] for d in m["config"]["datasets"]) == ["A", "B"]
    assert sorted(s["id"] for s in m["config"]["solvers"]) == ["s1", "s2"]
    assert len(m["derived"]["source_manifests"]) == 2


def test_failed_residual_demotes_even_with_allow_missing(tmp_path):
    # primal residual fails by six orders of magnitude; the missing dual one
    # must not hide that under --allow-missing
    rows = [_row("p1", "a", kkt={"primal_res_rel": 1.0, "dual_res_rel": None, "duality_gap_rel": 0.0}),
            _row("p2", "a", kkt={"primal_res_rel": 0.0, "dual_res_rel": None, "duality_gap_rel": 0.0})]
    src = _write_run(tmp_path, "src", rows)
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6, missing_is_failure=False)
    out_rows = {r["problem"]: r for r in map(json.loads, (tmp_path / "out" / "results.jsonl").read_text().splitlines())}
    assert out_rows["p1"]["status"] == "optimal_inaccurate"
    assert "missing" in out_rows["p1"]["metadata"]["kkt_verify"]["reason"]
    assert out_rows["p2"]["status"] == "optimal"  # only incomplete, tolerated
    assert summary["demoted"] == {"a": 1} and summary["kept"] == {"a": 1}
    # promotion still needs every residual
    src2 = _write_run(tmp_path, "src2", [_row("p3", "a", status="optimal_inaccurate",
                                               kkt={"primal_res_rel": 0.0, "dual_res_rel": None, "duality_gap_rel": 0.0})])
    kkt_verify(src2, tmp_path / "out2", tol=1e-6, missing_is_failure=False, promote=True)
    (row,) = map(json.loads, (tmp_path / "out2" / "results.jsonl").read_text().splitlines())
    assert row["status"] == "optimal_inaccurate"


def test_cone_distances_are_checked_relative_to_problem_scale():
    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.analysis.derive import worst_relative_residual
    from solver_benchmarks.analysis.kkt import cone_residuals

    # x >= 1 written as -x + s = -1, s in the nonnegative cone; the same point
    # with the constraint scaled by 1e6 has the same relative residuals.
    x, y = np.array([1.0]), np.array([1.0])
    for scale in (1.0, 1e6):
        s = np.array([-1e-7 * scale])  # slightly outside the cone, in absolute terms
        kkt = cone_residuals(sp.csc_matrix((1, 1)), np.array([1.0]), sp.csc_matrix([[-scale]]),
                             np.array([-scale]), {"l": 1}, x, y / scale, s)
        assert kkt["primal_cone_res"] == 1e-7 * scale
        assert kkt["primal_cone_res_rel"] < 1e-6
        assert worst_relative_residual({"kkt": kkt}) < 1e-6
    # a dual point outside K* must not pass because A and q are large: the LP
    # min M x s.t. M x + s = 0, s >= 0 is unbounded, and x = 0, s = 0, y = -1
    # has zero equality, stationarity and gap residuals for every M
    for m in (1.0, 1e12):
        kkt = cone_residuals(sp.csc_matrix((1, 1)), np.array([m]), sp.csc_matrix([[m]]), np.array([0.0]),
                             {"l": 1}, np.array([0.0]), np.array([-1.0]), np.array([0.0]))
        assert kkt["primal_res_rel"] == 0.0 and kkt["dual_res_rel"] == 0.0 and kkt["duality_gap_rel"] == 0.0
        assert kkt["dual_cone_res_rel"] == 0.5
        assert worst_relative_residual({"kkt": kkt}) == 0.5
    # records written before the relative distances existed fall back to the absolute ones
    old = {"kkt": {"form": "cone", "primal_res_rel": 0.0, "dual_res_rel": 0.0, "duality_gap_rel": 0.0,
                   "primal_cone_res": 0.5, "dual_cone_res": 0.0}}
    assert worst_relative_residual(old) == 0.5


def test_merge_preserves_each_shards_problem_selection(tmp_path):
    import json

    from solver_benchmarks.analysis.derive import merge_runs
    from solver_benchmarks.core.config import manifest_dataset_entries
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    _write_run_at(a, [], config={"datasets": [{"id": "A", "name": "A"}], "include": ["p1"],
                                 "solvers": [{"id": "s1"}]})
    _write_run_at(b, [], config={"datasets": [{"id": "A", "name": "A", "include": ["p2"], "exclude": ["p1"]}],
                                 "solvers": [{"id": "s1"}]})
    out = tmp_path / "merged"
    merge_runs([a, b], out, overwrite=True)
    m = json.loads((out / "manifest.json").read_text())
    (entry,) = manifest_dataset_entries(m["config"])
    assert entry["include"] == ["p1", "p2"] and entry["exclude"] == []
    assert "include" not in m["config"]
    # a shard with no include list expects every problem, so the union does too
    _write_run_at(c, [], config={"datasets": [{"id": "A", "name": "A"}], "solvers": [{"id": "s1"}]})
    merge_runs([a, c], tmp_path / "merged2", overwrite=True)
    (entry,) = manifest_dataset_entries(json.loads((tmp_path / "merged2" / "manifest.json").read_text())["config"])
    assert entry["include"] == []


def _synthetic_run(path, rows, *, datasets, solvers, **run_level):
    config = {"datasets": datasets, "solvers": [{"id": s, "solver": "scs", "settings": {}} for s in solvers], **run_level}
    _write_run_at(path, [{"dataset": "synthetic_qp", "problem": p, "solver_id": s, "status": "optimal",
                          "run_time_seconds": 1.0, "kkt": None, "metadata": {}} for p, s in rows], config=config)
    (path / "run_config.yaml").write_text("run: {}\n")


def test_merged_completion_does_not_cross_solvers_and_datasets(tmp_path):
    from solver_benchmarks.analysis.derive import merge_runs
    from solver_benchmarks.analysis.load import load_results
    from solver_benchmarks.analysis.tables import completion_summary, missing_results

    # shard a ran s1 on one problem, shard b ran s2 on another; both complete
    a, b = tmp_path / "a", tmp_path / "b"
    _synthetic_run(a, [("one_variable_eq", "s1")], datasets=[{"name": "synthetic_qp", "include": ["one_variable_eq"]}], solvers=["s1"])
    _synthetic_run(b, [("one_variable_lp", "s2")], datasets=[{"name": "synthetic_qp", "include": ["one_variable_lp"]}], solvers=["s2"])
    out = tmp_path / "merged"
    merge_runs([a, b], out, overwrite=True)
    completion = completion_summary(out, load_results(out))
    assert completion["complete"].all() and len(completion) == 2
    assert missing_results(out, load_results(out)).empty
    # nested merge and kkt-verify keep the selections
    c = tmp_path / "c"
    _synthetic_run(c, [("one_variable_eq", "s2")], datasets=[{"name": "synthetic_qp", "include": ["one_variable_eq"]}], solvers=["s2"])
    merge_runs([out, c], tmp_path / "merged2", overwrite=True)
    m2 = json.loads((tmp_path / "merged2" / "manifest.json").read_text())
    assert len(m2["derived"]["selections"]) == 3
    kkt_verify(tmp_path / "merged2", tmp_path / "verified", tol=1e-6, missing_is_failure=False)
    verified = json.loads((tmp_path / "verified" / "manifest.json").read_text())
    assert len(verified["derived"]["selections"]) == 3
    assert missing_results(tmp_path / "verified", load_results(tmp_path / "verified")).empty


def test_merged_completion_applies_each_shards_excludes_before_union(tmp_path):
    from solver_benchmarks.analysis.derive import merge_runs
    from solver_benchmarks.analysis.load import load_results
    from solver_benchmarks.analysis.tables import completion_summary
    from solver_benchmarks.datasets import get_dataset

    names = [p.name for p in get_dataset("synthetic_qp")().list_problems()]
    assert "one_variable_lp" in names
    # shard a: everything except the LP; shard b: only one_variable_eq (no rows yet)
    a, b = tmp_path / "a", tmp_path / "b"
    _synthetic_run(a, [(n, "s1") for n in names if n != "one_variable_lp"],
                   datasets=[{"name": "synthetic_qp"}], solvers=["s1"], exclude=["one_variable_lp"])
    _synthetic_run(b, [], datasets=[{"name": "synthetic_qp", "include": ["one_variable_eq"]}], solvers=["s1"])
    out = tmp_path / "merged"
    merge_runs([a, b], out, overwrite=True)
    (row,) = completion_summary(out, load_results(out)).to_dict("records")
    assert row["expected"] == len(names) - 1 and row["missing"] == 0 and row["complete"]


def test_verify_keeps_merge_provenance_including_source_manifests(tmp_path):
    from solver_benchmarks.analysis.derive import merge_runs

    a, b = tmp_path / "a", tmp_path / "b"
    _synthetic_run(a, [("one_variable_eq", "s1")], datasets=[{"name": "synthetic_qp", "include": ["one_variable_eq"]}],
                   solvers=["s1"], timeout_seconds=300)
    _synthetic_run(b, [("one_variable_lp", "s1")], datasets=[{"name": "synthetic_qp", "include": ["one_variable_lp"]}],
                   solvers=["s1"], timeout_seconds=900)
    merged = tmp_path / "merged"
    merge_runs([a, b], merged, overwrite=True)
    kkt_verify(merged, tmp_path / "v1", tol=1e-6, missing_is_failure=False)
    kkt_verify(tmp_path / "v1", tmp_path / "v2", tol=1e-6, missing_is_failure=False)
    for out in (tmp_path / "v1", tmp_path / "v2"):
        derived = json.loads((out / "manifest.json").read_text())["derived"]
        assert derived["kind"] == "kkt_verify"
        limits = sorted(m["config"]["timeout_seconds"] for m in derived["source_manifests"])
        assert limits == [300, 900]
        assert len(derived["selections"]) == 2
        assert derived["source_derived"]["kind"] in {"merge", "kkt_verify"}
    # the complete merge provenance is still reachable after two verifications
    v2 = json.loads((tmp_path / "v2" / "manifest.json").read_text())["derived"]
    assert v2["source_derived"]["source_derived"]["kind"] == "merge"


def test_merge_reports_problems_posed_in_different_forms(tmp_path):
    from solver_benchmarks.analysis.derive import merge_runs

    def row(problem, solver_id, rows_posed):
        base = _row(problem, solver_id)
        metadata = dict(base.get("metadata") or {})
        if rows_posed is not None:
            metadata["m"] = rows_posed
        return {**base, "metadata": metadata}

    # hinf1 as one merged block (105 rows) in one shard and with its declared
    # blocks (41 rows) in the other; truss1 the same in both; one skip row.
    a = _write_run(tmp_path, "a", [row("hinf1", "s1", 105), row("truss1", "s1", 19), row("skip", "s1", None)])
    b = _write_run(tmp_path, "b", [row("hinf1", "s2", 41), row("truss1", "s2", 19)])
    summary = merge_runs([a, b], tmp_path / "merged")
    assert summary["mixed_forms"] == ["d/hinf1"]
    manifest = json.loads((tmp_path / "merged" / "manifest.json").read_text())
    assert manifest["derived"]["mixed_forms"] == ["d/hinf1"]


def _read_rows(path):
    return {(r["problem"], r["solver_id"]): r for r in map(json.loads, (path / "results.jsonl").read_text().splitlines())}


def test_torn_final_line_is_skipped_by_verify_and_merge(tmp_path):
    good = {"primal_res_rel": 1e-9, "dual_res_rel": 1e-9, "duality_gap_rel": 1e-9}
    src = _write_run(tmp_path, "src", [_row("p1", "a", kkt=good)])
    with (src / "results.jsonl").open("a") as handle:
        handle.write('{"dataset": "d", "problem": "p2", "solver_id": "a", "sta')
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6)
    assert list(_read_rows(tmp_path / "out")) == [("p1", "a")]
    assert summary["skipped_lines"] == 1
    merged = merge_runs([src], tmp_path / "merged")
    assert merged["rows"] == 1 and merged["skipped_lines"] == 1


def test_failed_derivation_leaves_no_output_and_keeps_an_existing_one(tmp_path, monkeypatch):
    import pytest

    from solver_benchmarks.analysis import derive

    src = _write_run(tmp_path, "src", [_row("p1", "a", kkt=None)])
    out = tmp_path / "out"

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(derive, "_write_jsonl", boom)
    with pytest.raises(RuntimeError):
        kkt_verify(src, out, tol=1e-6)
    assert not out.exists()
    monkeypatch.undo()
    kkt_verify(src, out, tol=1e-6)
    monkeypatch.setattr(derive, "_write_jsonl", boom)
    with pytest.raises(RuntimeError):
        kkt_verify(src, out, tol=1e-6, overwrite=True)
    # the earlier output survives a failed overwrite
    assert ("p1", "a") in _read_rows(out)
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".out")] == []


def test_verify_records_the_fields_it_checked_and_the_failing_one(tmp_path):
    kkt = {"form": "cone", "primal_res_rel": 1e-9, "dual_res_rel": 1e-9, "duality_gap_rel": 1e-9,
           "primal_cone_res_rel": 0.0, "dual_cone_res_rel": 1e-3}
    src = _write_run(tmp_path, "src", [_row("p1", "a", kkt=kkt)])
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6, fields=("primal_res_rel",))
    record = _read_rows(tmp_path / "out")[("p1", "a")]["metadata"]["kkt_verify"]
    assert record["fields"] == ["primal_res_rel", "primal_cone_res_rel", "dual_cone_res_rel"]
    assert record["worst_field"] == "dual_cone_res_rel"
    assert "dual_cone_res_rel" in record["reason"]
    assert summary["fields"] == ["primal_res_rel"]
    assert summary["checked_fields"] == ["dual_cone_res_rel", "primal_cone_res_rel", "primal_res_rel"]


def test_verify_holds_certificates_to_the_tolerance(tmp_path):
    exp = {"metadata": {"expected_status": "infeasible"}}
    rows = [
        {**_row("tight", "a", status="primal_infeasible",
                kkt={"certificate": "primal_infeasible", "bty": -1.0, "Aty_inf": 1e-9, "dual_cone_res": 0.0,
                     "cert_res": 1e-9, "valid": True}), **exp},
        {**_row("loose", "a", status="primal_infeasible",
                kkt={"certificate": "primal_infeasible", "bty": -1.0, "Aty_inf": 1e-4, "dual_cone_res": 0.0,
                     "cert_res": 1e-4, "valid": True}), **exp},
        # written before cert_res existed: rebuilt from the absolute fields
        {**_row("old", "a", status="dual_infeasible",
                kkt={"certificate": "dual_infeasible", "qtx": -2.0, "Px_inf": 0.0, "Ax_cone_res": 1e-3}), **exp},
        {**_row("none", "a", status="primal_infeasible", kkt=None), **exp},
    ]
    src = _write_run(tmp_path, "src", rows)
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6)
    out = _read_rows(tmp_path / "out")
    assert out[("tight", "a")]["status"] == "primal_infeasible"
    assert out[("loose", "a")]["status"] == "primal_infeasible_inaccurate"
    assert out[("loose", "a")]["metadata"]["kkt_verify"]["worst_field"] == "cert_res"
    assert out[("old", "a")]["status"] == "dual_infeasible_inaccurate"
    assert out[("old", "a")]["metadata"]["kkt_verify"]["worst"] == 5e-4
    # no certificate recorded: nothing to hold against the solver
    assert out[("none", "a")]["status"] == "primal_infeasible"
    assert out[("none", "a")]["metadata"]["kkt_verify"]["reason"] == "no certificate recorded"
    assert summary["kept"] == {"a": 2} and summary["demoted"] == {"a": 2}
    assert summary["unverified_certificates"] == {"a": 1}
    assert summary["missing_residuals"] == {}


def test_verify_keeps_certificates_it_cannot_check_whatever_missing_is_failure(tmp_path):
    # HiGHS, PIQP, ProxQP, SDPA and the commercial adapters record no
    # certificate vector; that is no reason to fail their infeasibility claims.
    exp = {"metadata": {"expected_status": "infeasible"}}
    rows = [
        {**_row("p1", "highs", status="primal_infeasible", kkt=None), **exp},
        {**_row("p2", "highs", status="dual_infeasible", kkt={}), **exp},
        {**_row("p1", "scs", status="primal_infeasible",
                kkt={"certificate": "primal_infeasible", "bty": -1.0, "Aty_inf": 1e-9, "dual_cone_res": 0.0,
                     "cert_res": 1e-9, "cert_tol": 1e-3, "valid": True}), **exp},
    ]
    src = _write_run(tmp_path, "src", rows)
    for missing_is_failure, name in ((True, "strict"), (False, "lenient")):
        summary = kkt_verify(src, tmp_path / name, tol=1e-6, missing_is_failure=missing_is_failure)
        out = _read_rows(tmp_path / name)
        assert {key: row["status"] for key, row in out.items()} == {
            ("p1", "highs"): "primal_infeasible", ("p2", "highs"): "dual_infeasible", ("p1", "scs"): "primal_infeasible"}
        assert "kkt_verify" not in out[("p1", "scs")]["metadata"]
        assert summary["demoted"] == {}
        assert summary["kept"] == {"highs": 2, "scs": 1}
        assert summary["unverified_certificates"] == {"highs": 2}


def test_verify_checks_what_it_can_of_a_certificate_with_an_unsupported_cone(tmp_path):
    exp = {"metadata": {"expected_status": "infeasible"}}

    def cert_row(problem, kkt):
        return {**_row(problem, "a", status="primal_infeasible", kkt=kkt), **exp}

    # Current records: the cone distance is NaN, stored as null, and so is
    # cert_res unless a known residual already fails CERT_TOL.
    current = {"certificate": "primal_infeasible", "bty": -2.0, "dual_cone_res": None, "cert_res": None,
               "valid": None, "unsupported_cones": ["cs"]}
    # Older records reported the cone distance over the cones they could
    # check only, and no cert_res.
    older = {"certificate": "primal_infeasible", "bty": -2.0, "unsupported_cones": ["ep"]}
    rows = [
        cert_row("current_ok", {**current, "Aty_inf": 2e-9}),
        cert_row("current_bad", {**current, "Aty_inf": 2e-4}),
        # known residual fails CERT_TOL, so cert_res was recorded
        cert_row("current_cert", {**current, "Aty_inf": 4e-3, "cert_res": 2e-3, "valid": False}),
        cert_row("older_ok", {**older, "Aty_inf": 2e-9, "dual_cone_res": 0.0}),
        cert_row("older_bad", {**older, "Aty_inf": 2e-9, "dual_cone_res": 2e-4}),
    ]
    src = _write_run(tmp_path, "src", rows)
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6)
    out = _read_rows(tmp_path / "out")
    statuses = {problem: row["status"] for (problem, _), row in out.items()}
    assert statuses == {
        "current_ok": "primal_infeasible",
        "current_bad": "primal_infeasible_inaccurate",
        "current_cert": "primal_infeasible_inaccurate",
        "older_ok": "primal_infeasible",
        "older_bad": "primal_infeasible_inaccurate",
    }
    assert out[("current_ok", "a")]["metadata"]["kkt_verify"]["worst"] == 1e-9
    assert "cs" in out[("current_ok", "a")]["metadata"]["kkt_verify"]["reason"]
    assert "ep" in out[("older_ok", "a")]["metadata"]["kkt_verify"]["reason"]
    assert out[("older_bad", "a")]["metadata"]["kkt_verify"]["worst"] == 1e-4
    assert "unknown" in out[("current_bad", "a")]["metadata"]["kkt_verify"]["reason"]
    assert summary["unverified_certificates"] == {"a": 2}
    # A cone the harness cannot check is unknown, not passed, at any tolerance.
    loose = kkt_verify(src, tmp_path / "loose", tol=1e-2)
    assert loose["demoted"] == {} and loose["unverified_certificates"] == {"a": 5}


def test_verify_fails_a_certificate_whose_objective_term_is_not_negative(tmp_path):
    # cert_res is inf then, which the results file stores as null.
    kkt = {"certificate": "dual_infeasible", "qtx": 0.5, "Px_inf": 0.0, "Ax_cone_res": 0.0,
           "cert_res": None, "cert_tol": 1e-3, "valid": False}
    src = _write_run(tmp_path, "src", [_row("p1", "a", status="dual_infeasible", kkt=kkt)])
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6)
    row = _read_rows(tmp_path / "out")[("p1", "a")]
    assert row["status"] == "dual_infeasible_inaccurate"
    assert row["metadata"]["kkt_verify"]["reason"].startswith("certificate residual inf")
    assert summary["demoted"] == {"a": 1}


def test_kkt_verify_cli_warns_about_certificates_it_could_not_check(tmp_path):
    from click.testing import CliRunner

    from solver_benchmarks.cli import main

    rows = [_row("p1", "highs", status="primal_infeasible", kkt=None),
            _row("p2", "highs", status="primal_infeasible", kkt=None)]
    src = _write_run(tmp_path, "src", rows)
    result = CliRunner().invoke(main, ["kkt-verify", str(src), "--tol", "1e-6", "--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "highs\tkept=2\tdemoted=0" in result.stdout
    assert "certificate could not be checked: highs=2" in result.stderr


def test_derived_run_directories_follow_the_umask(tmp_path):
    import os

    src = _write_run(tmp_path, "src", [_row("p1", "a", kkt=None)])
    previous = os.umask(0o022)
    try:
        kkt_verify(src, tmp_path / "verified", tol=1e-6)
        merge_runs([src], tmp_path / "merged")
        kkt_verify(src, tmp_path / "verified", tol=1e-6, overwrite=True)
    finally:
        os.umask(previous)
    for name in ("verified", "merged"):
        assert (tmp_path / name).stat().st_mode & 0o777 == 0o755


def test_verify_does_not_demote_rows_whose_cones_could_not_be_checked(tmp_path):
    # Records written before exponential cones were projected carry NaN cone
    # distances and name the cone; the residuals that were computed pass.
    nan = float("nan")
    kkt = {"form": "cone", "primal_res_rel": 1e-12, "dual_res_rel": 1e-12, "duality_gap_rel": 1e-12,
           "primal_cone_res_rel": nan, "dual_cone_res_rel": nan, "unsupported_cones": ["ep"]}
    bad = {**kkt, "dual_res_rel": 1e-3}
    src = _write_run(tmp_path, "src", [_row("p1", "a", kkt=kkt), _row("p2", "a", kkt=bad)])
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6)
    out = _read_rows(tmp_path / "out")
    assert out[("p1", "a")]["status"] == "optimal"
    assert "ep" in out[("p1", "a")]["metadata"]["kkt_verify"]["reason"]
    assert out[("p2", "a")]["status"] == "optimal_inaccurate"
    assert summary["unverified_cones"] == {"a": 1}
    assert summary["kept"] == {"a": 1} and summary["demoted"] == {"a": 1}


_PASSING_KKT = {"primal_res_rel": 1e-9, "dual_res_rel": 1e-9, "duality_gap_rel": 1e-9}


def test_promote_never_turns_a_solve_returned_after_the_limit_into_optimal(tmp_path):
    """The runner records a conclusion reached after timeout_seconds as
    time_limit: the limit is every solver's budget. Its point passes the
    check by construction, and promoting it would give that solver more
    time than the others. A solver-reported time_limit within the limit
    whose point passes is still promoted."""
    from solver_benchmarks.core.result import ProblemResult
    from solver_benchmarks.core.runner import _apply_time_limit

    over = ProblemResult(
        run_id="r", dataset="d", problem="late", problem_kind="qp", solver_id="a", solver="a",
        status="optimal", objective_value=0.5, iterations=3, run_time_seconds=5.9, kkt=dict(_PASSING_KKT),
    )
    late_iters = ProblemResult(
        run_id="r", dataset="d", problem="late_iters", problem_kind="qp", solver_id="a", solver="a",
        status="max_iter_reached", objective_value=None, iterations=3, run_time_seconds=5.9,
        kkt=dict(_PASSING_KKT),
    )
    within = ProblemResult(
        run_id="r", dataset="d", problem="within", problem_kind="qp", solver_id="a", solver="a",
        status="time_limit", objective_value=None, iterations=3, run_time_seconds=0.99,
        kkt=dict(_PASSING_KKT),
    )
    for result in (over, late_iters, within):
        _apply_time_limit(result, 1.0)
    assert over.status == "time_limit" and over.objective_value is None
    src = _write_run(tmp_path, "src", [r.to_record() for r in (over, late_iters, within)])
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6, promote=True)
    rows = {r["problem"]: r for r in map(json.loads, (tmp_path / "out" / "results.jsonl").read_text().splitlines())}
    assert rows["late"]["status"] == "time_limit"
    assert rows["late_iters"]["status"] == "max_iter_reached"
    assert rows["within"]["status"] == "optimal"
    assert "time limit" in rows["late"]["metadata"]["kkt_verify"]["reason"]
    assert rows["late"]["metadata"]["kkt_verify"]["original_status"] == "time_limit"
    assert summary["promoted"] == {"a": 1}
    assert summary["over_time_limit"] == {"a": 2}

    from click.testing import CliRunner

    from solver_benchmarks.cli import main

    result = CliRunner().invoke(
        main, ["kkt-verify", str(src), "--tol", "1e-6", "--promote", "--output-dir", str(tmp_path / "cli")]
    )
    assert result.exit_code == 0, result.output
    assert "a\tkept=0\tdemoted=0\tpromoted=1\tno_residuals=0\tover_time_limit=2" in result.stdout


def test_promote_skips_rows_marked_by_an_older_runner(tmp_path):
    """Rows of runs recorded before the exceeded_time_limit marker carry
    only status_before_timeout; they are not promoted either."""
    row = _row("p1", "a", status="time_limit", kkt=dict(_PASSING_KKT))
    row["info"] = {"status_before_timeout": "optimal"}
    row["objective_value"] = 0.5
    src = _write_run(tmp_path, "src", [row])
    summary = kkt_verify(src, tmp_path / "out", tol=1e-6, promote=True)
    (out,) = map(json.loads, (tmp_path / "out" / "results.jsonl").read_text().splitlines())
    assert out["status"] == "time_limit"
    assert summary["promoted"] == {}
    assert summary["over_time_limit"] == {"a": 1}


def test_derive_commands_report_user_errors_without_a_traceback(tmp_path):
    from click.testing import CliRunner

    from solver_benchmarks.cli import main

    src = _write_run(tmp_path, "src", [_row("p1", "a", kkt=dict(_PASSING_KKT))])
    runner = CliRunner()
    assert runner.invoke(main, ["kkt-verify", str(src), "--tol", "1e-6"]).exit_code == 0
    cases = [
        (["kkt-verify", str(src), "--tol", "1e-6"], "--overwrite"),
        (["kkt-verify", str(src), "--tol", "1e-6", "--output-dir", str(src / "sub")], "overlaps"),
        (["merge", str(tmp_path / "src_kkt1e-06"), str(src)], "--overwrite"),
        (["merge", str(src / "sub"), str(src)], "overlaps"),
    ]
    for args, expected in cases:
        result = runner.invoke(main, args)
        assert result.exit_code == 1, (args, result.output)
        assert isinstance(result.exception, SystemExit), (args, result.exception)
        assert "Error:" in result.output and expected in result.output, (args, result.output)


def test_a_directory_that_is_not_a_run_is_refused(tmp_path):
    import pytest
    from click.testing import CliRunner

    from solver_benchmarks.cli import main

    empty = tmp_path / "not_a_run"
    empty.mkdir()
    with pytest.raises(ValueError, match="results.jsonl"):
        kkt_verify(empty, tmp_path / "out", tol=1e-6)
    with pytest.raises(ValueError, match="results.jsonl"):
        merge_runs([empty], tmp_path / "merged")
    assert not (tmp_path / "out").exists() and not (tmp_path / "merged").exists()
    result = CliRunner().invoke(main, ["kkt-verify", str(empty), "--tol", "1e-6"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "results.jsonl" in result.output
    assert not (tmp_path / "not_a_run_kkt1e-06").exists()


def test_a_run_that_planned_no_solves_can_be_merged(tmp_path):
    """A finished run whose filters left nothing to solve has a manifest but
    no results.jsonl; it is a run, not a wrong path."""
    zero = tmp_path / "zero"
    zero.mkdir()
    (zero / "manifest.json").write_text(json.dumps({"config": {}}))
    src = _write_run(tmp_path, "src", [_row("p1", "a", kkt=dict(_PASSING_KKT))])
    summary = merge_runs([zero, src], tmp_path / "merged")
    assert summary["rows"] == 1
