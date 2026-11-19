"""KKT verification is the default wherever results are read (analysis/verify.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

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
    unverified_frame,
    verification_header,
    verification_info,
    verification_table,
    verify_frame,
)
from solver_benchmarks.cli import main

# SCS at eps 1e-4: solutions and certificates are checked at 1e-3.
SCS = {"id": "scs_1e-4", "solver": "scs", "settings": {"eps_abs": 1e-4, "eps_rel": 1e-4}}
TOL = 1e-3


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
        _row("cert_bad", "primal_infeasible", _cert(5e-3), **infeasible),
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
    "cert_none": ("dual_infeasible_inaccurate", "missing_residuals"),
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
    assert df.loc["cert_bad", "verify.tol"] == pytest.approx(TOL)  # certificates by the same tolerance
    assert df.loc["cert_ok", "verify.reason"] == "certificate residual 5.000e-07 <= 1.0e-03"
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
    assert not (solved["loose"] or solved["bare"] or solved["late"] or solved["cert_bad"] or solved["cert_none"])


def test_a_certificate_passing_its_own_tolerance_succeeds_above_cert_tol(tmp_path):
    """A certificate verified at its row's (here 1e-2) tolerance succeeds
    even though the solve-time check at kkt.CERT_TOL (1e-3) found it
    invalid; an unverified frame keeps that check."""
    infeasible = {"metadata": {"expected_status": "infeasible"}}
    loose_scs = {**SCS, "settings": {"eps_abs": 1e-3, "eps_rel": 1e-3}}
    rows = [_row("p", "primal_infeasible", _cert(5e-3), **infeasible)]
    run = _write_run(tmp_path / "run", rows, solvers=(loose_scs,))
    verified = load_results(run)
    assert verified.loc[0, "kkt.valid"] is False or not verified.loc[0, "kkt.valid"]
    assert verified.loc[0, "verify.verdict"] == "kept"
    assert success_mask(verified).tolist() == [True]
    assert success_mask(load_results(run, verify=False)).tolist() == [False]


def test_missing_residuals_can_be_tolerated(tmp_path):
    infeasible = {"metadata": {"expected_status": "infeasible"}}
    rows = [_row("bare", "optimal", None), _row("cert_none", "primal_infeasible", None, **infeasible)]
    run = _write_run(tmp_path / "run", rows)
    df = _by_problem(load_results(run, verify=VerifyPolicy(missing_is_failure=False)))
    assert df["status"].to_dict() == {"bare": "optimal", "cert_none": "primal_infeasible"}
    assert set(df["verify.verdict"]) == {"missing_residuals"}
    assert "2 claims without residuals or a certificate were kept unchecked" in verification_header(df)
    table = verification_table(df).iloc[0]
    assert (table["kept"], table["demoted"], table["no_residuals"], table["unverified"]) == (2, 0, 2, 2)
    default = _by_problem(load_results(run))
    assert default["status"].to_dict() == {"bare": "optimal_inaccurate", "cert_none": "primal_infeasible_inaccurate"}
    assert "counts as unsolved" in verification_header(default)


@pytest.mark.parametrize(
    "policy",
    [
        VerifyPolicy(),
        VerifyPolicy(slack=3.0, promote=False, missing_is_failure=False),
        VerifyPolicy(rule="fixed", tol=1e-2),
    ],
    ids=["default", "slack3_no_promote_allow_missing", "fixed"],
)
def test_a_frame_that_lost_its_attrs_is_described_from_its_columns(tmp_path, policy):
    """pandas drops attrs through some operations; the summary is then rebuilt
    from the verify.* columns, not assumed to be the default policy."""
    df = load_results(_write_run(tmp_path / "run", _rows()), verify=policy)
    recorded = df.attrs["verification"]
    df.attrs = {}
    rebuilt = verification_info(df)
    assert rebuilt["reconstructed"] is True
    for key in ("mode", "rule", "promote", "missing_is_failure", "counts"):
        assert rebuilt[key] == recorded[key], key
    if policy.rule == "fixed":
        assert rebuilt["tol"] == pytest.approx(policy.tol)
    else:
        assert rebuilt["slack"] == pytest.approx(policy.slack)
    header = verification_header(df)
    assert "rebuilt from the verify.* columns" in header
    assert ("count as solved" in header) == policy.promote
    assert ("--allow-missing" in header) == (not policy.missing_is_failure)


def test_a_rebuilt_summary_states_nothing_no_row_shows(tmp_path):
    """With no promotable stop and no claim without residuals, promotion and
    missing residuals changed nothing, so the rebuilt header does not name them."""
    df = load_results(_write_run(tmp_path / "run", [_row("ok", "optimal", _kkt(5e-4))]))
    df.attrs = {}
    info = verification_info(df)
    assert info["promote"] is None and info["missing_is_failure"] is None
    assert info["slack"] == pytest.approx(VERIFY_SLACK)
    header = verification_header(df)
    assert "count as solved" not in header and "no residuals" not in header
    assert f"{VERIFY_SLACK:g} x the tolerance its solver was asked for" in header


def test_without_promote_limit_stops_are_not_checked(tmp_path):
    run = _write_run(tmp_path / "run", [_row("promote", "max_iter_reached", _kkt(1e-6))])
    df = load_results(run, verify=VerifyPolicy(promote=False))
    assert df.loc[0, "status"] == "max_iter_reached" and df.loc[0, "verify.verdict"] == "not_checked"


def test_fixed_rule_holds_every_row_to_one_tolerance(tmp_path):
    df = _by_problem(load_results(_write_run(tmp_path / "run", _rows()),
                                  verify=VerifyPolicy(rule="fixed", tol=1e-2)))
    assert df.loc["loose", "status"] == "optimal"
    assert df.loc["cert_bad", "status"] == "primal_infeasible"  # 5e-3 <= 1e-2
    assert set(df["verify.rule"]) == {"fixed"}
    assert df.loc["cert_bad", "verify.tol"] == pytest.approx(1e-2)


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


@pytest.mark.parametrize("policy", [VerifyPolicy(rule="fixed", tol=1e-2), VerifyPolicy(rule="fixed", tol=1e-8),
                                    VerifyPolicy(promote=False)])
def test_a_verified_frame_is_verified_again_from_its_reported_statuses(tmp_path, policy):
    """make_figures verifies a series again at its figure's tolerance: that
    must start from what the solver reported, not from the first verdict
    (a demoted claim is not "promoted" back, a promotion's objective goes)."""
    run = _write_run(tmp_path / "run", _rows())
    manifest = json.loads((run / "manifest.json").read_text())
    columns = ["status", "status_reported", "objective_value", *VERIFY_COLUMNS]
    once = load_results(run)
    # the late rule of make_figures overwrites status; the report survives in status_reported
    once.loc[once["problem"] == "ok", "status"] = "time_limit"
    again = _by_problem(verify_frame(once, manifest, policy))[columns]
    direct = _by_problem(load_results(run, verify=policy))[columns]
    pd.testing.assert_frame_equal(again, direct)
    if policy.rule == "fixed" and policy.tol == 1e-2:
        assert again.loc["loose", "verify.verdict"] == "kept"  # reported optimal, passes 1e-2
    if policy.rule == "fixed" and policy.tol == 1e-8:
        assert again.loc["promote", "status"] == "max_iter_reached"
        assert pd.isna(again.loc["promote", "objective_value"])
    raw = _by_problem(unverified_frame(once))
    assert raw["status"].to_dict() == _by_problem(load_results(run, verify=False))["status"].to_dict()
    assert pd.isna(raw.loc["promote", "objective_value"])
    assert not any(column.startswith("verify.") for column in raw.columns)


# --- where the requested tolerance comes from --------------------------------


def test_the_row_record_beats_the_manifest_and_an_override_beats_both(tmp_path):
    rows = [_row("ok", "optimal", _kkt(5e-4), metadata={"requested_tolerance": {"optimality": 1e-5}})]
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
    assert df.loc["none", "status"] == "primal_infeasible_inaccurate"
    assert df.loc["none", "verify.verdict"] == "missing_residuals"


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


@pytest.mark.parametrize("bad", [
    {"rule": "fixed"}, {"rule": "sometimes"}, {"tol": 1e-3}, {"slack": 0},
    {"rule": "fixed", "tol": 1e-3, "requested_tol": {"a": 1e-4}},
    {"rule": "fixed", "tol": 1e-3, "tol_from_solver_id": True},
    {"rule": "fixed", "tol": 1e-3, "slack": 5.0},
    {"requested_tol": {"a": (1e-4, 1e-7)}},
])
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
    columns = ["status", "status_reported", "objective_value", "verify.verdict", "verify.tol",
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
    def no_tolerance():
        raise AssertionError("asked for the tolerance")

    assert classify({"status": "optimal", "kkt": {}}, tolerance=no_tolerance).verdict == "missing_residuals"
    claim = classify({"status": "primal_infeasible", "kkt": {}}, tolerance=no_tolerance)
    assert (claim.verdict, claim.status) == ("missing_residuals", "primal_infeasible_inaccurate")
    kept = classify({"status": "primal_infeasible", "kkt": {}}, tolerance=no_tolerance, missing_is_failure=False)
    assert (kept.verdict, kept.status) == ("missing_residuals", "primal_infeasible")
    assert classify({"status": "solver_error"}, tolerance=no_tolerance).verdict == "not_checked"


def test_a_certificate_on_cones_the_harness_cannot_project_is_kept_unverified():
    certificate = {"certificate": "primal_infeasible", "bty": -1.0, "Aty_inf": 1e-9, "dual_cone_res": None,
                   "unsupported_cones": ["cs"]}
    verdict = classify({"status": "primal_infeasible", "kkt": certificate}, tolerance=lambda: 1e-3)
    assert (verdict.verdict, verdict.status) == ("unverified_certificate", "primal_infeasible")


def test_row_info_reads_what_resolution_needs():
    info = row_info_from_record(_row("p", "optimal", metadata={
        "nnz_p": 3, "runtime": {"solver_package_versions": {"qtqp": "0.0.6"}},
        "requested_tolerance": {"optimality": 1e-5}}))
    assert (info.solver_id, info.run_id, info.is_lp) == ("scs_1e-4", "run", False)
    assert info.versions == (("qtqp", "0.0.6"),)
    assert info.recorded_optimality == 1e-5


# --- consumers read verified statuses -------------------------------------------


def test_tables_count_verified_statuses(tmp_path):
    from solver_benchmarks.analysis.tables import failure_rates, solver_metrics

    df = load_results(_write_run(tmp_path / "run", _rows()))
    rates = failure_rates(df).set_index("solver_id")
    # solved: ok, promote and cones; on the infeasible problems cert_ok and
    # cert_up (promoted); cert_none has nothing to check
    assert rates.loc["scs_1e-4", "success_count"] == 5
    metrics = solver_metrics(df).set_index("solver_id")
    assert metrics.loc["scs_1e-4", "success_count"] == 5
    raw = failure_rates(load_results(tmp_path / "run", verify=False)).set_index("solver_id")
    # as reported: ok, loose, bare, cones; cert_ok, cert_none (cert_bad is
    # invalid by the solve-time check at 1e-3)
    assert raw.loc["scs_1e-4", "success_count"] == 6


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


# --- the CLI ---------------------------------------------------------------------


def test_summary_prints_the_rule_and_the_verification_table(tmp_path, repo_root):
    run = _write_run(tmp_path / "run", _rows())
    result = CliRunner().invoke(main, ["summary", str(run), "--repo-root", str(repo_root)])
    assert result.exit_code == 0, result.output
    assert "Statuses are KKT-verified" in result.output
    verification = result.output.split("Verification:")[1]
    assert "scs_1e-4" in verification and "requested_tol" in verification
    unverified = CliRunner().invoke(main, ["summary", str(run), "--repo-root", str(repo_root), "--unverified"])
    assert "not KKT-verified" in unverified.output and "Verification:" not in unverified.output


def test_failures_take_the_verification_options(tmp_path, repo_root):
    run = _write_run(tmp_path / "run", _rows())

    def success(*options):
        result = CliRunner().invoke(main, ["failures", str(run), "--repo-root", str(repo_root), *options])
        assert result.exit_code == 0, result.output
        line = next(line for line in result.output.splitlines() if line.split()[:1] == ["scs_1e-4"])
        return int(line.split()[2])

    assert success() == 5
    assert success("--unverified") == 6
    assert success("--verify-tol", "1e-2") == 7  # loose and cert_bad pass; bare and cert_none have nothing to check
    assert success("--requested-tol", "scs_1e-4=1e-6") == 4  # ok fails at 1e-5
    assert success("--allow-missing") == 7  # bare and cert_none kept unchecked


def test_an_unknown_tolerance_is_a_click_error_with_the_remedies(tmp_path):
    run = _write_run(tmp_path / "run", _rows(), manifest=False)
    result = CliRunner().invoke(main, ["failures", str(run)])
    assert result.exit_code == 1
    assert "--requested-tol" in result.output and "--tol-from-solver-id" in result.output
    assert "Traceback" not in result.output
    tagged = CliRunner().invoke(main, ["failures", str(run), "--tol-from-solver-id"])
    assert tagged.exit_code == 0, tagged.output


@pytest.mark.parametrize("options", [["--unverified", "--verify-tol", "1e-3"], ["--unverified", "--reverify"],
                                     ["--unverified", "--allow-missing"],
                                     ["--verify-tol", "1e-3", "--reverify"],
                                     ["--verify-tol", "1e-3", "--requested-tol", "scs_1e-4=1e-4"],
                                     ["--verify-tol", "1e-3", "--tol-from-solver-id"],
                                     ["--requested-tol", "nonsense"],
                                     ["--requested-tol", "scs_1e-4=1e-4/1e-7"]])
def test_conflicting_verification_options_are_usage_errors(tmp_path, options):
    run = _write_run(tmp_path / "run", _rows())
    result = CliRunner().invoke(main, ["summary", str(run), *options])
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("options", [["--slack", "5"], ["--slack", "10"], ["--requested-tol", "scs_1e-4=1e-4"],
                                     ["--tol-from-solver-id"]])
def test_kkt_verify_refuses_options_a_fixed_tolerance_would_ignore(tmp_path, options):
    run = _write_run(tmp_path / "run", _rows())
    result = CliRunner().invoke(main, ["kkt-verify", str(run), "--tol", "1e-3", *options])
    assert result.exit_code == 2, result.output
    assert "would not be used" in result.output
    assert not (tmp_path / "run_kkt1e-03").exists()


def test_kkt_verify_defaults_to_the_requested_rule_and_says_so(tmp_path):
    run = _write_run(tmp_path / "run", _rows())
    result = CliRunner().invoke(main, ["kkt-verify", str(run)])
    assert result.exit_code == 0, result.output
    out_dir = tmp_path / "run_verified"
    assert result.output.strip().splitlines()[-1] == str(out_dir)
    line = next(line for line in result.output.splitlines() if line.startswith("scs_1e-4"))
    assert "promoted=2" in line and "requested=1e-04\t" in line and "check=1e-03" in line
    manifest = json.loads((out_dir / "manifest.json").read_text())["derived"]
    assert manifest["rule"] == "requested" and manifest["promote"] is True and manifest["annotates_all_rows"]
    fixed = CliRunner().invoke(main, ["kkt-verify", str(run), "--tol", "1e-3", "--no-promote"])
    assert fixed.exit_code == 0, fixed.output
    assert (tmp_path / "run_kkt1e-03").exists()
    # the scripts that ran it with --promote keep working
    assert CliRunner().invoke(main, ["kkt-verify", str(run), "--tol", "1e-3", "--promote", "--output-dir",
                                     str(tmp_path / "p")]).exit_code == 0


def test_kkt_verify_with_an_unknown_tolerance_writes_nothing(tmp_path):
    run = _write_run(tmp_path / "run", _rows(), manifest=False)
    result = CliRunner().invoke(main, ["kkt-verify", str(run)])
    assert result.exit_code == 1 and "--requested-tol" in result.output
    assert not (tmp_path / "run_verified").exists()


# --- the report ----------------------------------------------------------------


def test_report_states_and_tabulates_the_verification(tmp_path, repo_root):
    from solver_benchmarks.analysis.markdown_report import write_run_report

    run = _write_run(tmp_path / "run", _rows())
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root, max_value=100.0)
    report = tmp_path / "report"
    markdown = (report / "index.md").read_text()
    summary = markdown.split("## Verification")[0]
    assert "Statuses are KKT-verified" in summary
    section = markdown.split("## Verification")[1].split("\n## ")[0]
    assert "requested_tol" in section and "scs_1e-4" in section
    table = pd.read_csv(report / "verification.csv").set_index("solver_id")
    assert table.loc["scs_1e-4", "promoted"] == 2 and table.loc["scs_1e-4", "unverified"] == 1  # cones
    assert table.loc["scs_1e-4", "demoted"] == 4  # loose, bare and cert_none (nothing to check), cert_bad
    changes = pd.read_csv(report / "verification_changes.csv")
    assert ((changes["status_reported"] == "optimal") & (changes["status"] == "optimal_inaccurate")).any()
    counts = pd.read_csv(report / "status_counts.csv").set_index("status")["count"]
    assert counts["optimal"] == 3 and counts["optimal_inaccurate"] == 3
    # claims are what the solvers reported: the demoted ones are counted
    thresholds = pd.read_csv(report / "claimed_optimal_kkt_thresholds.csv").set_index("solver_id")
    assert thresholds.loc["scs_1e-4", "claimed_optimal"] == 4
    certificates = pd.read_csv(report / "kkt_certificate_summary.csv").set_index("solver_id")
    assert certificates.loc["scs_1e-4", "infeasible_count"] == 4
    assert certificates.loc["scs_1e-4", "cert_valid"] == 2  # kept and promoted
    assert certificates.loc["scs_1e-4", "cert_invalid"] == 1


def test_unverified_report_says_so(tmp_path, repo_root):
    from solver_benchmarks.analysis.markdown_report import write_run_report

    run = _write_run(tmp_path / "run", _rows())
    write_run_report(run, output_dir=tmp_path / "report", repo_root=repo_root, max_value=100.0, verify=False)
    markdown = (tmp_path / "report" / "index.md").read_text()
    assert "Statuses as the solvers reported them (not KKT-verified)." in markdown
    assert not (tmp_path / "report" / "verification.csv").exists()


# --- every analysis command takes the options and says how it verified ---------


def _scs_success(csv: Path, column: str) -> float:
    table = pd.read_csv(csv)
    return float(table.loc[table["solver_id"] == "scs_1e-4", column].iloc[0])


@pytest.mark.parametrize("command", ["profile", "geomean", "plot", "report"])
def test_file_writing_commands_print_the_rule_on_stderr(tmp_path, repo_root, command):
    run = _write_run(tmp_path / "run", _rows())
    extra = ["--output-dir", str(tmp_path / "out")] if command in ("plot", "report") else []
    if command in ("geomean", "plot", "report"):
        extra += ["--max-value", "100"]
    for options, says in (([], "Statuses are KKT-verified"),
                          (["--unverified"], "not KKT-verified"),
                          (["--verify-tol", "1e-2"], "fixed tolerance 0.01")):
        result = CliRunner().invoke(main, [command, str(run), "--repo-root", str(repo_root), *extra, *options])
        assert result.exit_code == 0, result.output
        assert says in result.stderr, (command, options)
        # stdout keeps only the paths written
        assert all(Path(line).exists() for line in result.stdout.splitlines()), result.stdout


def test_geomean_and_report_count_what_the_options_choose(tmp_path, repo_root):
    run = _write_run(tmp_path / "run", _rows())

    def geomean_successes(*options):
        result = CliRunner().invoke(main, ["geomean", str(run), "--repo-root", str(repo_root), "--max-value", "100",
                                           *options])
        assert result.exit_code == 0, result.output
        return _scs_success(Path(result.stdout.strip()), "success_count")

    assert geomean_successes() == 5
    assert geomean_successes("--unverified") == 6
    assert geomean_successes("--verify-tol", "1e-2") == 7
    assert geomean_successes("--allow-missing") == 7
    report = CliRunner().invoke(main, ["report", str(run), "--repo-root", str(repo_root), "--max-value", "100",
                                       "--output-dir", str(tmp_path / "report"), "--unverified"])
    assert report.exit_code == 0, report.output
    assert "not KKT-verified" in (tmp_path / "report" / "index.md").read_text()
    assert not (tmp_path / "report" / "verification.csv").exists()


def test_profile_counts_verified_statuses(tmp_path, repo_root):
    run = _write_run(tmp_path / "run", _rows())

    def solved_fraction(*options):
        result = CliRunner().invoke(main, ["profile", str(run), "--repo-root", str(repo_root), *options])
        assert result.exit_code == 0, result.output
        profile = pd.read_csv(Path(result.stdout.strip()))
        return float(profile["scs_1e-4"].max())

    # over the same planned solves: 5 verified successes against 6 reported
    assert solved_fraction() / solved_fraction("--unverified") == pytest.approx(5 / 6)


# --- what the header admits, and options a copy does not use --------------------


def test_the_header_counts_claims_on_cones_it_cannot_project(tmp_path):
    df = load_results(_write_run(tmp_path / "run", _rows()))
    header = verification_header(df)
    assert "1 claim passed every residual the harness can compute but lie on cones it cannot project" in header
    assert "counts as unsolved" in header


def test_a_kkt_verify_copy_warns_about_options_it_does_not_use(tmp_path):
    run = _write_run(tmp_path / "run", _rows())
    kkt_verify(run, tmp_path / "copy")
    df = load_results(tmp_path / "copy", verify=VerifyPolicy(tol_from_solver_id=True, missing_is_failure=False))
    (warning,) = df.attrs["verification"]["warnings"]
    assert "--tol-from-solver-id, --allow-missing are not applied" in warning and "--reverify" in warning
    assert df.attrs["verification"]["mode"] == "recorded"
    # --reverify applies them, and says the copy was verified otherwise
    again = load_results(tmp_path / "copy", verify=VerifyPolicy(rule="requested", missing_is_failure=False))
    (warning,) = again.attrs["verification"]["warnings"]
    assert "demoting claims without residuals" in warning
    assert _by_problem(again).loc["cert_none", "status"] == "dual_infeasible"


def test_a_merge_warns_about_copies_verified_without_promotion_or_with_allow_missing(tmp_path):
    a = _write_run(tmp_path / "a", _rows()[:4], run_id="a")
    kkt_verify(a, tmp_path / "a_plain", promote=False, missing_is_failure=False)
    merge_runs([tmp_path / "a_plain"], tmp_path / "merged")
    warnings = load_results(tmp_path / "merged").attrs["verification"]["warnings"]
    assert len(warnings) == 1
    assert "without promotion" in warnings[0] and "keeping claims without residuals" in warnings[0]
    df = _by_problem(load_results(tmp_path / "merged"))
    assert df.loc["promote", "status"] == "optimal" and df.loc["bare", "status"] == "optimal_inaccurate"
    same = load_results(tmp_path / "merged", verify=VerifyPolicy(promote=False, missing_is_failure=False))
    assert same.attrs["verification"]["warnings"] == []
