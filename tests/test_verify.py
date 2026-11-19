"""KKT verification is the default wherever results are read (analysis/verify.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from solver_benchmarks.analysis.derive import kkt_verify, merge_runs
from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.analysis.outcomes import success_mask
from solver_benchmarks.analysis.verify import (
    VERIFY_COLUMNS,
    VERIFY_SLACK,
    RequestedToleranceUnknown,
    VerifyInputError,
    VerifyPolicy,
    classify,
    row_info_from_record,
    verification_table,
)

# SCS at eps 1e-4: solutions are checked at 1e-3, certificates (eps_infeas
# left at SCS's 1e-7) at 1e-6.
SCS = {"id": "scs_1e-4", "solver": "scs", "settings": {"eps_abs": 1e-4, "eps_rel": 1e-4}}
TOL, CERT_TOL = 1e-3, 1e-6


def _kkt(worst=1e-9, **extra):
    return {"form": "qp", "primal_res_rel": worst, "dual_res_rel": 1e-12, "duality_gap_rel": 1e-12,
            "primal_obj_reported": 3.5, **extra}


def _cert(cert_res, kind="primal_infeasible", **extra):
    sign = {"bty": -1.0, "Aty_inf": cert_res, "dual_cone_res": 0.0} if kind == "primal_infeasible" else {
        "qtx": -1.0, "Px_inf": cert_res, "primal_cone_res": 0.0}
    return {"form": "cone", "certificate": kind, **sign, "cert_res": cert_res, "cert_tol": 1e-3,
            "valid": cert_res <= 1e-3, **extra}


def _row(problem, status, kkt=None, solver_id="scs_1e-4", **extra):
    row = {"run_id": "run", "dataset": "d", "problem": problem, "problem_kind": "qp", "solver_id": solver_id,
           "solver": "scs", "status": status, "run_time_seconds": 1.0,
           "objective_value": 3.5 if status in ("optimal", "optimal_inaccurate") else None,
           "kkt": kkt, "metadata": {"nnz_p": 0}, "info": {}}
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(row.get(key), dict):
            row[key] = {**row[key], **value}
        else:
            row[key] = value
    return row


def _write_run(path: Path, rows, solvers=(SCS,), run_id="run", manifest=True) -> Path:
    path.mkdir(parents=True)
    (path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if manifest:
        (path / "manifest.json").write_text(json.dumps({"run_id": run_id, "config": {
            "timeout_seconds": 60, "datasets": [{"id": "d", "name": "synthetic_qp", "include": []}],
            "solvers": list(solvers)}}))
    return path


def _rows():
    infeasible = {"metadata": {"expected_status": "infeasible"}}
    return [
        _row("ok", "optimal", _kkt(5e-4)),
        _row("loose", "optimal", _kkt(5e-3)),
        _row("bare", "optimal", None),
        _row("promote", "max_iter_reached", _kkt(1e-6)),
        _row("late", "time_limit", _kkt(1e-9), info={"exceeded_time_limit": True}),
        _row("stays", "optimal_inaccurate", _kkt(5e-2)),
        _row("cones", "optimal", _kkt(1e-9, form="cone", primal_cone_res_rel=None, dual_cone_res_rel=None,
                                       unsupported_cones=["cs"])),
        _row("cert_ok", "primal_infeasible", _cert(5e-7), **infeasible),
        _row("cert_bad", "primal_infeasible", _cert(5e-5), **infeasible),
        _row("cert_up", "primal_infeasible_inaccurate", _cert(1e-7), **infeasible),
        _row("cert_none", "dual_infeasible", None, **infeasible),
        _row("error", "solver_error", None),
    ]


EXPECTED = {
    "ok": ("optimal", "kept"),
    "loose": ("optimal_inaccurate", "demoted"),
    "bare": ("optimal_inaccurate", "missing_residuals"),
    "promote": ("optimal", "promoted"),
    "late": ("time_limit", "over_time_limit"),
    "stays": ("optimal_inaccurate", "kept"),
    "cones": ("optimal", "unverified_cones"),
    "cert_ok": ("primal_infeasible", "kept"),
    "cert_bad": ("primal_infeasible_inaccurate", "demoted"),
    "cert_up": ("primal_infeasible", "promoted"),
    "cert_none": ("dual_infeasible", "unverified_certificate"),
    "error": ("solver_error", "not_checked"),
}


def _by_problem(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("problem")


# --- the default load ---------------------------------------------------------


def test_load_results_verifies_at_ten_times_the_requested_tolerance(tmp_path):
    df = _by_problem(load_results(_write_run(tmp_path / "run", _rows())))
    assert {p: (df.loc[p, "status"], df.loc[p, "verify.verdict"]) for p in EXPECTED} == EXPECTED
    assert all(column in df.columns for column in VERIFY_COLUMNS)
    assert df.loc["loose", "status_reported"] == "optimal"
    assert df.loc["ok", "verify.tol"] == pytest.approx(TOL)
    assert df.loc["ok", "verify.requested_tol"] == pytest.approx(1e-4)
    assert df.loc["ok", "verify.requested_tol_source"] == "manifest"
    assert df.loc["cert_bad", "verify.cert_tol"] == pytest.approx(CERT_TOL)
    assert df.loc["cert_bad", "verify.requested_cert_tol"] == pytest.approx(1e-7)
    assert "5.000e-03" in df.loc["loose", "verify.reason"]
    assert df.loc["error", "verify.tol"] != df.loc["error", "verify.tol"]  # NaN: never needed one
    info = df.attrs["verification"]
    assert info["mode"] == "computed" and info["rule"] == "requested" and info["slack"] == VERIFY_SLACK


def test_a_promoted_stop_takes_its_point_objective(tmp_path):
    rows = [_row("promote", "max_iter_reached", _kkt(1e-6))]
    df = load_results(_write_run(tmp_path / "run", rows))
    assert df.loc[0, "status"] == "optimal"
    assert df.loc[0, "objective_value"] == pytest.approx(3.5)
    assert df.loc[0, "verify.objective_from"] == "kkt.primal_obj_reported"


def test_scoring_follows_the_verified_status(tmp_path):
    df = _by_problem(load_results(_write_run(tmp_path / "run", _rows())))
    solved = success_mask(df.reset_index()).set_axis(df.index)
    assert solved["ok"] and solved["promote"] and solved["cones"] and solved["cert_ok"] and solved["cert_up"]
    assert not (solved["loose"] or solved["bare"] or solved["late"] or solved["cert_bad"])


def test_a_certificate_passing_its_own_tolerance_succeeds_above_cert_tol(tmp_path):
    """A certificate verified at its own (here 1e-2) tolerance succeeds even
    though the solve-time check at kkt.CERT_TOL (1e-3) found it invalid;
    an unverified frame keeps that check."""
    infeasible = {"metadata": {"expected_status": "infeasible"}}
    loose_scs = {**SCS, "settings": {**SCS["settings"], "eps_infeas": 1e-3}}
    rows = [_row("p", "primal_infeasible", _cert(5e-3), **infeasible)]
    run = _write_run(tmp_path / "run", rows, solvers=(loose_scs,))
    verified = load_results(run)
    assert verified.loc[0, "kkt.valid"] is False or not verified.loc[0, "kkt.valid"]
    assert verified.loc[0, "verify.verdict"] == "kept"
    assert success_mask(verified).tolist() == [True]
    assert success_mask(load_results(run, verify=False)).tolist() == [False]


def test_missing_residuals_can_be_tolerated(tmp_path):
    run = _write_run(tmp_path / "run", [_row("bare", "optimal", None)])
    df = load_results(run, verify=VerifyPolicy(missing_is_failure=False))
    assert df.loc[0, "status"] == "optimal" and df.loc[0, "verify.verdict"] == "missing_residuals"


def test_without_promote_limit_stops_are_not_checked(tmp_path):
    run = _write_run(tmp_path / "run", [_row("promote", "max_iter_reached", _kkt(1e-6))])
    df = load_results(run, verify=VerifyPolicy(promote=False))
    assert df.loc[0, "status"] == "max_iter_reached" and df.loc[0, "verify.verdict"] == "not_checked"


def test_fixed_rule_holds_every_row_to_one_tolerance(tmp_path):
    df = _by_problem(load_results(_write_run(tmp_path / "run", _rows()),
                                  verify=VerifyPolicy(rule="fixed", tol=1e-2)))
    assert df.loc["loose", "status"] == "optimal"
    assert df.loc["cert_bad", "status"] == "primal_infeasible"  # 5e-5 <= 1e-2
    assert set(df["verify.rule"]) == {"fixed"}
    assert df.loc["ok", "verify.cert_tol"] == pytest.approx(1e-2)


def test_unverified_load_gives_the_reported_statuses(tmp_path):
    run = _write_run(tmp_path / "run", _rows())
    df = _by_problem(load_results(run, verify=False))
    assert df.loc["loose", "status"] == "optimal" and df.loc["promote", "status"] == "max_iter_reached"
    assert not any(column.startswith("verify.") for column in df.columns)
    assert (df["status_reported"] == df["status"]).all()
    assert df.attrs["verification"]["mode"] == "unverified"


def test_unverified_load_of_a_kkt_verify_copy_undoes_it(tmp_path):
    run = _write_run(tmp_path / "run", _rows())
    kkt_verify(run, tmp_path / "copy")
    copy = _by_problem(load_results(tmp_path / "copy", verify=False))
    source = _by_problem(load_results(run, verify=False))
    assert copy["status"].to_dict() == source["status"].to_dict()
    assert pd.isna(copy.loc["promote", "objective_value"])  # the promotion's objective is gone


# --- where the requested tolerance comes from --------------------------------


def test_the_row_record_beats_the_manifest_and_an_override_beats_both(tmp_path):
    rows = [_row("ok", "optimal", _kkt(5e-4), metadata={"requested_tolerance": {"optimality": 1e-5,
                                                                                  "certificate": None}})]
    run = _write_run(tmp_path / "run", rows)
    df = load_results(run)
    assert df.loc[0, "verify.requested_tol_source"] == "row"
    assert df.loc[0, "status"] == "optimal_inaccurate"  # 5e-4 > 1e-4
    overridden = load_results(run, verify=VerifyPolicy(requested_tol={"scs_1e-4": 1e-3}))
    assert overridden.loc[0, "verify.requested_tol_source"] == "override"
    assert overridden.loc[0, "status"] == "optimal"


def test_an_unknown_requested_tolerance_is_loud_and_names_the_remedies(tmp_path):
    run = _write_run(tmp_path / "run", [_row("ok", "optimal", _kkt(5e-4))], manifest=False)
    with pytest.raises(RequestedToleranceUnknown) as excinfo:
        load_results(run)
    message = str(excinfo.value)
    assert "scs_1e-4" in message and "no manifest.json" in message
    for remedy in ("--requested-tol", "--tol-from-solver-id", "--verify-tol", "--unverified"):
        assert remedy in message
    tagged = load_results(run, verify=VerifyPolicy(tol_from_solver_id=True))
    assert tagged.loc[0, "verify.requested_tol_source"] == "solver_id"
    assert tagged.loc[0, "verify.tol"] == pytest.approx(1e-3)


def test_rows_without_residuals_need_no_tolerance(tmp_path):
    rows = [_row("bare", "optimal", None), _row("error", "solver_error", None),
            _row("none", "primal_infeasible", None)]
    df = _by_problem(load_results(_write_run(tmp_path / "run", rows, manifest=False)))
    assert df.loc["bare", "status"] == "optimal_inaccurate"
    assert df.loc["none", "verify.verdict"] == "unverified_certificate"


def test_merged_shards_with_different_settings_resolve_by_run_id(tmp_path):
    loose = {**SCS, "settings": {"eps_abs": 1e-2, "eps_rel": 1e-2}}
    a = _write_run(tmp_path / "a", [_row("p", "optimal", _kkt(5e-2), run_id="a")], run_id="a")
    b = _write_run(tmp_path / "b", [_row("q", "optimal", _kkt(5e-2), run_id="b")], solvers=(loose,), run_id="b")
    merge_runs([a, b], tmp_path / "merged")
    df = _by_problem(load_results(tmp_path / "merged"))
    assert df.loc["p", "status"] == "optimal_inaccurate" and df.loc["q", "status"] == "optimal"


def test_conflicting_settings_without_a_run_id_are_loud(tmp_path):
    loose = {**SCS, "settings": {"eps_abs": 1e-2, "eps_rel": 1e-2}}
    a = _write_run(tmp_path / "a", [_row("p", "optimal", _kkt(5e-2), run_id=None)], run_id="a")
    b = _write_run(tmp_path / "b", [_row("q", "optimal", _kkt(5e-2), run_id=None)], solvers=(loose,), run_id="b")
    merge_runs([a, b], tmp_path / "merged")
    with pytest.raises(RequestedToleranceUnknown, match="different settings"):
        load_results(tmp_path / "merged")


def test_an_unregistered_solver_is_unknown(tmp_path):
    rows = [_row("ok", "optimal", _kkt(5e-4), solver_id="mine")]
    run = _write_run(tmp_path / "run", rows, solvers=({"id": "mine", "solver": "nosuch", "settings": {}},))
    with pytest.raises(RequestedToleranceUnknown, match="not registered"):
        load_results(run)


@pytest.mark.parametrize("bad", [{"rule": "fixed"}, {"rule": "sometimes"}, {"tol": 1e-3}, {"slack": 0}])
def test_bad_policies_are_refused(bad):
    with pytest.raises(VerifyInputError):
        VerifyPolicy(**bad)


# --- derived runs: idempotence -------------------------------------------------


def _jsonl(path: Path) -> str:
    return (path / "results.jsonl").read_text()


def test_kkt_verify_of_its_own_output_writes_the_same_rows(tmp_path):
    run = _write_run(tmp_path / "run", _rows())
    kkt_verify(run, tmp_path / "once")
    kkt_verify(tmp_path / "once", tmp_path / "twice")
    assert _jsonl(tmp_path / "once") == _jsonl(tmp_path / "twice")
    rows = [json.loads(line) for line in _jsonl(tmp_path / "twice").splitlines()]
    by_problem = {row["problem"]: row for row in rows}
    # the earliest reported status survives every pass
    assert by_problem["loose"]["metadata"]["kkt_verify"]["original_status"] == "optimal"
    assert by_problem["promote"]["metadata"]["kkt_verify"]["original_status"] == "max_iter_reached"
    assert "kkt_verify" not in by_problem["error"]["metadata"]  # not checked: no annotation


def test_a_kkt_verify_copy_loads_as_its_source_does(tmp_path):
    run = _write_run(tmp_path / "run", _rows())
    kkt_verify(run, tmp_path / "copy")
    columns = ["status", "status_reported", "objective_value", "verify.verdict", "verify.tol", "verify.cert_tol",
               "verify.requested_tol", "verify.worst"]
    recorded = load_results(tmp_path / "copy")
    assert recorded.attrs["verification"]["mode"] == "recorded"
    pd.testing.assert_frame_equal(_by_problem(recorded)[columns], _by_problem(load_results(run))[columns])
    reverified = load_results(tmp_path / "copy", verify=VerifyPolicy(rule="requested"))
    pd.testing.assert_frame_equal(_by_problem(reverified)[columns], _by_problem(load_results(run))[columns])


def test_a_fixed_tolerance_copy_is_used_as_recorded(tmp_path):
    run = _write_run(tmp_path / "run", _rows())
    kkt_verify(run, tmp_path / "copy", tol=1e-2)
    df = _by_problem(load_results(tmp_path / "copy"))
    assert df.loc["loose", "status"] == "optimal" and df.loc["loose", "verify.tol"] == pytest.approx(1e-2)
    assert df.attrs["verification"]["rule"] == "fixed"
    # --reverify checks it again by the default rule
    again = _by_problem(load_results(tmp_path / "copy", verify=VerifyPolicy(rule="requested")))
    assert again.loc["loose", "status"] == "optimal_inaccurate"


def test_a_copy_written_before_verdicts_were_recorded_is_read(tmp_path):
    """Older kkt-verify copies annotated only the rows they changed or kept
    with a note; the other rows are inferred."""
    rows = [
        _row("kept", "optimal", _kkt(1e-9)),
        _row("demoted", "optimal_inaccurate", _kkt(5e-2),
             metadata={"kkt_verify": {"tol": 1e-3, "original_status": "optimal", "reason": "worst ... > 1e-03"}}),
        _row("missing", "optimal_inaccurate", None,
             metadata={"kkt_verify": {"tol": 1e-3, "original_status": "optimal", "reason": "no KKT residuals recorded"}}),
        _row("promoted", "optimal", _kkt(1e-9),
             metadata={"kkt_verify": {"tol": 1e-3, "original_status": "max_iter_reached", "reason": "<="}}),
        _row("late", "time_limit", _kkt(1e-9), metadata={"kkt_verify": {
            "tol": 1e-3, "original_status": "time_limit", "reason": "returned after the time limit; not promoted"}}),
        _row("error", "solver_error", None),
    ]
    run = _write_run(tmp_path / "copy", rows)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["derived"] = {"kind": "kkt_verify", "tol": 1e-3, "promote": True}
    (run / "manifest.json").write_text(json.dumps(manifest))
    df = _by_problem(load_results(run))
    assert df["verify.verdict"].to_dict() == {"kept": "kept", "demoted": "demoted", "missing": "missing_residuals",
                                              "promoted": "promoted", "late": "over_time_limit",
                                              "error": "not_checked"}
    assert df.loc["promoted", "status_reported"] == "max_iter_reached"
    assert df.loc["kept", "verify.tol"] == pytest.approx(1e-3)


def test_a_merge_of_verified_copies_is_verified_again_with_a_warning(tmp_path):
    a = _write_run(tmp_path / "a", _rows()[:3], run_id="a")
    kkt_verify(a, tmp_path / "a_fixed", tol=1e-2)
    b = _write_run(tmp_path / "b", [_row("x", "optimal", _kkt(1e-9))], run_id="b")
    summary = merge_runs([tmp_path / "a_fixed", b], tmp_path / "merged")
    assert summary["mixed_verification"]
    df = _by_problem(load_results(tmp_path / "merged"))
    assert df.loc["loose", "status"] == "optimal_inaccurate"  # the default rule, not the copy's 1e-2
    warnings = df.attrs["verification"]["warnings"]
    assert any("--verify-tol 0.01" in warning for warning in warnings)


# --- one rule: the loader and kkt-verify agree, whatever the storage -----------


def test_verdicts_agree_between_parquet_jsonl_and_kkt_verify(tmp_path):
    from solver_benchmarks.core.storage import normalize_table_for_parquet, read_results_jsonl

    run = _write_run(tmp_path / "run", _rows())
    from_jsonl = _by_problem(load_results(run))
    normalize_table_for_parquet(read_results_jsonl(run / "results.jsonl")).to_parquet(
        run / "results.parquet", index=False)
    from_parquet = _by_problem(load_results(run))
    assert (run / "results.parquet").stat().st_mtime_ns >= (run / "results.jsonl").stat().st_mtime_ns
    columns = ["status", "verify.verdict", "verify.worst"]
    pd.testing.assert_frame_equal(from_parquet[columns], from_jsonl[columns])
    summary = kkt_verify(run, tmp_path / "copy")
    copy_rows = {json.loads(line)["problem"]: json.loads(line) for line in _jsonl(tmp_path / "copy").splitlines()}
    for problem, (expected_status, verdict) in EXPECTED.items():
        assert copy_rows[problem]["status"] == expected_status, problem
        annotation = copy_rows[problem]["metadata"].get("kkt_verify")
        assert (annotation or {}).get("verdict", "not_checked") == verdict, problem
    assert summary["tolerances"]["scs_1e-4"]["tol"] == [pytest.approx(TOL)]


def test_classify_asks_for_a_tolerance_only_to_compare():
    def no_tolerance(kind):
        raise AssertionError(f"asked for the {kind} tolerance")

    assert classify({"status": "optimal", "kkt": {}}, tolerance=no_tolerance).verdict == "missing_residuals"
    assert classify({"status": "primal_infeasible", "kkt": {}}, tolerance=no_tolerance).verdict == (
        "unverified_certificate")
    assert classify({"status": "solver_error"}, tolerance=no_tolerance).verdict == "not_checked"


def test_row_info_reads_what_resolution_needs():
    info = row_info_from_record(_row("p", "optimal", metadata={
        "nnz_p": 3, "runtime": {"solver_package_versions": {"qtqp": "0.0.6"}},
        "requested_tolerance": {"optimality": 1e-5, "certificate": 1e-6}}))
    assert (info.solver_id, info.run_id, info.is_lp) == ("scs_1e-4", "run", False)
    assert info.versions == (("qtqp", "0.0.6"),)
    assert (info.recorded_optimality, info.recorded_certificate) == (1e-5, 1e-6)


# --- consumers read verified statuses -------------------------------------------


def test_tables_count_verified_statuses(tmp_path):
    from solver_benchmarks.analysis.tables import failure_rates, solver_metrics

    df = load_results(_write_run(tmp_path / "run", _rows()))
    rates = failure_rates(df).set_index("solver_id")
    # solved: ok, promote and cones; on the infeasible problems cert_ok,
    # cert_up (promoted) and cert_none (kept unverified)
    assert rates.loc["scs_1e-4", "success_count"] == 6
    metrics = solver_metrics(df).set_index("solver_id")
    assert metrics.loc["scs_1e-4", "success_count"] == 6
    raw = failure_rates(load_results(tmp_path / "run", verify=False)).set_index("solver_id")
    # as reported: ok, loose, bare, cones; cert_ok, cert_bad (valid at 1e-3), cert_none
    assert raw.loc["scs_1e-4", "success_count"] == 7


def test_plots_draw_verified_statuses(tmp_path, monkeypatch):
    from solver_benchmarks.analysis import plots

    seen = []
    real = plots.performance_profile

    def capture(results, **kwargs):
        seen.append(results.copy())
        return real(results, **kwargs)

    monkeypatch.setattr(plots, "performance_profile", capture)
    run = _write_run(tmp_path / "run", _rows())
    plots.write_analysis_plots(run, output_dir=tmp_path / "plots", max_value=100.0)
    statuses = seen[0].set_index("problem")["status"]
    assert statuses["loose"] == "optimal_inaccurate" and statuses["promote"] == "optimal"


def test_verification_table_counts():
    frame = pd.DataFrame({
        "solver_id": ["a"] * 4,
        "status": ["optimal", "optimal_inaccurate", "optimal", "optimal_inaccurate"],
        "status_reported": ["optimal", "optimal", "max_iter_reached", "optimal"],
        "verify.verdict": ["kept", "demoted", "promoted", "missing_residuals"],
        "verify.tol": [1e-3] * 4,
    })
    row = verification_table(frame).iloc[0]
    assert (row["kept"], row["demoted"], row["promoted"], row["no_residuals"]) == (1, 2, 1, 1)
    assert row["tol"] == "0.001"
