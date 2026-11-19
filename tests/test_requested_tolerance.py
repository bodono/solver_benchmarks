"""The tolerance each adapter's settings request (SolverAdapter.requested_tolerance).

KKT verification holds every row to VERIFY_SLACK times this, so it must be
what a solve with those settings is asked to meet: the adapter's own alias
translation applied, every knob it leaves unset at the library's default.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from solver_benchmarks.solvers import get_solver, list_solvers
from solver_benchmarks.solvers.base import RequestedTolerance, SolverAdapter, ToleranceUnknown

REPO_ROOT = Path(__file__).resolve().parent.parent

# What each adapter requests with empty settings.
DEFAULTS = {
    "clarabel": 1e-8,
    "cplex": 1e-6,
    "cpo3": 1e-8,
    "cuopt": 1e-4,
    "cvxopt": 1e-6,
    "ecos": 1e-8,
    "gurobi": 1e-6,
    "highs": 1e-7,
    "mosek": 1e-8,
    "osqp": 1e-3,
    "pdlp": 1e-6,
    "piqp": 1e-8,
    "proxqp": 1e-5,
    "qtqp": 1e-8,
    "scs": 1e-4,
    "sdpa": 1e-7,
}
# Adapters that refuse the shared eps aliases (no faithful translation).
REFUSE_EPS = {"cplex", "gurobi", "highs", "mosek"}


def _requested(solver, settings=None, **kwargs) -> RequestedTolerance:
    return get_solver(solver).requested_tolerance(settings or {}, **kwargs)


def test_every_registered_solver_has_an_expected_default():
    assert set(DEFAULTS) == set(list_solvers())


@pytest.mark.parametrize("solver", list_solvers())
def test_every_adapter_defines_requested_tolerance(solver):
    assert get_solver(solver).requested_tolerance.__func__ is not SolverAdapter.requested_tolerance.__func__


def test_the_base_adapter_refuses_rather_than_guess():
    class Bare(SolverAdapter):
        solver_name = "bare"
        supported_problem_kinds = {"qp"}

        def solve(self, problem, artifacts_dir):  # pragma: no cover - never solved
            raise NotImplementedError

    with pytest.raises(ToleranceUnknown, match="bare"):
        Bare.requested_tolerance({})


@pytest.mark.parametrize("solver", list_solvers())
def test_empty_settings_request_the_library_defaults(solver):
    tolerance = _requested(solver, problem_kind="qp", is_lp=True)
    assert tolerance.optimality == pytest.approx(DEFAULTS[solver])
    assert tolerance.defaults_used  # every knob came from a default


@pytest.mark.parametrize("solver", sorted(set(list_solvers()) - REFUSE_EPS))
def test_eps_aliases_set_the_request(solver):
    assert _requested(solver, {"eps": 3e-5}).optimality == pytest.approx(3e-5)
    # the loosest of the absolute and relative tolerances
    assert _requested(solver, {"eps_abs": 2e-5, "eps_rel": 4e-6}).optimality == pytest.approx(2e-5)
    assert _requested(solver, {"eps_abs": 4e-6, "eps_rel": 2e-5}).optimality == pytest.approx(2e-5)


@pytest.mark.parametrize("solver", sorted(REFUSE_EPS))
def test_solvers_that_refuse_eps_have_no_request_from_it(solver):
    with pytest.raises(ValueError, match="eps"):
        _requested(solver, {"eps": 1e-5})


@pytest.mark.parametrize(
    "solver, native",
    [
        ("scs", {"eps_abs": 1e-3, "eps_rel": 1e-3}),
        ("clarabel", {"tol_feas": 1e-3, "tol_gap_abs": 1e-3, "tol_gap_rel": 1e-3}),
        ("cpo3", {"tol_feas": 1e-3, "tol_gap_abs": 1e-3, "tol_gap_rel": 1e-3}),
        ("qtqp", {"tol_feas": 1e-3, "tol_gap_abs": 1e-3, "tol_gap_rel": 1e-3}),
        ("ecos", {"feastol": 1e-3, "abstol": 1e-3, "reltol": 1e-3}),
        ("cvxopt", {"abstol": 1e-3, "reltol": 1e-3, "feastol": 1e-3}),
        ("sdpa", {"epsilonStar": 1e-3, "epsilonDash": 1e-3}),
        ("cuopt", {"optimality_tolerance": 1e-3}),
    ],
)
def test_a_native_setting_wins_over_an_alias(solver, native):
    assert _requested(solver, {**native, "eps": 1e-6}).optimality == pytest.approx(1e-3)


def test_harness_settings_that_are_not_tolerances_are_ignored():
    common = {"verbose": True, "time_limit": 10, "threads": 1, "max_iter": 100}
    for solver in list_solvers():
        assert _requested(solver, common).optimality == pytest.approx(DEFAULTS[solver]), solver


def _make_campaign():
    spec = importlib.util.spec_from_file_location("make_campaign_for_tolerances", REPO_ROOT / "tools" / "make_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("family", ["qp", "lp", "sdp", "lpbig", "infeas"])
@pytest.mark.parametrize("tol", [1e-4, 5e-5, 1e-6])
def test_every_campaign_variant_requests_its_tag(family, tol):
    """make_campaign's nominal tolerance is what each variant asks for, so the
    default verification is 10 x the tag, the rule of the published figures."""
    campaign = _make_campaign()
    for variant in campaign.solver_variants(family, tol, 300.0):
        requested = _requested(variant["solver"], variant["settings"], problem_kind="qp", is_lp=family != "qp")
        assert requested.optimality == pytest.approx(tol), variant["id"]


@pytest.mark.parametrize("family", ["qp", "lp", "sdp", "lpbig", "infeas"])
def test_campaign_solutions_and_certificates_are_verified_at_ten_times_the_tag(family):
    """The default verification of a make_campaign run: every variant's
    rows, solutions and infeasibility certificates alike, are held to 10 x
    its tag (HiGHS below 1e-7 excepted), so one figure's claims are judged by
    one standard, the rule of the published figures."""
    from solver_benchmarks.analysis.verify import (
        VERIFY_SLACK,
        RowInfo,
        ToleranceResolver,
        VerifyPolicy,
        classify,
    )

    campaign = _make_campaign()
    tol = 1e-4
    variants = campaign.solver_variants(family, tol, 300.0)
    manifest = {"run_id": "run", "config": {"solvers": [
        {"id": v["id"], "solver": v["solver"], "settings": v["settings"]} for v in variants]}}
    resolver = ToleranceResolver(manifest, VerifyPolicy())
    check = VERIFY_SLACK * tol
    for variant in variants:
        resolved = resolver.resolve(RowInfo(solver_id=variant["id"], run_id="run", solver=variant["solver"],
                                            problem_kind="qp", is_lp=family != "qp"))
        assert resolved.tol == pytest.approx(check), variant["id"]
        for cert_res, verdict in ((0.5 * check, "kept"), (2.0 * check, "demoted")):
            certificate = {"certificate": "primal_infeasible", "cert_res": cert_res, "cert_tol": 1e-3}
            got = classify({"status": "primal_infeasible", "kkt": certificate}, tolerance=lambda r=resolved: r.tol)
            assert got.verdict == verdict, (variant["id"], cert_res)


@pytest.mark.parametrize(
    "solver, settings",
    [
        ("scs", {"eps_infeas": 1e-2}),
        ("clarabel", {"tol_infeas_rel": 1e-2, "tol_infeas_abs": 1e-2}),
        ("cpo3", {"tol_infeas_rel": 1e-2}),
        ("qtqp", {"tol_infeas_rel": 1e-2}),
        ("osqp", {"eps_prim_inf": 1e-2, "eps_dual_inf": 1e-2}),
        ("proxqp", {"eps_primal_inf": 1e-2, "eps_dual_inf": 1e-2}),
    ],
)
def test_infeasibility_tolerances_do_not_change_the_request(solver, settings):
    # Solvers measure certificates in units other than kkt.cert_res; a row's
    # certificates are held to the same tolerance as its solutions.
    assert _requested(solver, settings).optimality == pytest.approx(DEFAULTS[solver])


def test_highs_below_its_residual_defaults_requests_them():
    # make_campaign sets only HiGHS's feasibility tolerances; its residual
    # and optimality tolerances stay at 1e-7.
    campaign = _make_campaign()
    highs = next(v for v in campaign.solver_variants("lp", 1e-8, 300.0) if v["solver"] == "highs")
    assert _requested("highs", highs["settings"]).optimality == pytest.approx(1e-7)


def test_highs_kkt_tolerance_overrides_every_knob():
    settings = {"kkt_tolerance": 1e-5, "primal_feasibility_tolerance": 1e-3}
    assert _requested("highs", settings).optimality == pytest.approx(1e-5)
    # left at its default it changes nothing
    assert _requested("highs", {"kkt_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-3}).optimality == 1e-3


def test_highs_solver_choice_selects_its_optimality_tolerance():
    loose_ipm = {"ipm_optimality_tolerance": 1e-3}
    assert _requested("highs", {**loose_ipm, "solver": "ipm"}).optimality == pytest.approx(1e-3)
    assert _requested("highs", {**loose_ipm, "solver": "choose"}).optimality == pytest.approx(1e-3)
    assert _requested("highs", {**loose_ipm, "solver": "simplex"}).optimality == pytest.approx(1e-7)
    assert _requested("highs", {"pdlp_optimality_tolerance": 1e-4, "solver": "pdlp"}).optimality == 1e-4
    with pytest.raises(ToleranceUnknown, match="not one whose tolerances"):
        _requested("highs", {"solver": "barrier-of-the-future"})


def test_pdlp_parameters_text_is_read():
    pytest.importorskip("ortools.pdlp.solvers_pb2")
    detailed = _requested("pdlp", {
        "parameters_text": "termination_criteria { detailed_optimality_criteria { "
                           "eps_optimal_primal_residual_absolute: 1e-3 } eps_primal_infeasible: 1e-4 }",
    })
    assert detailed.optimality == pytest.approx(1e-3)
    # parameters_text is merged last, so it wins over eps
    simple = _requested("pdlp", {"eps": 1e-6, "parameters_text":
                                 "termination_criteria { simple_optimality_criteria { eps_optimal_relative: 1e-4 } }"})
    assert simple.optimality == pytest.approx(1e-4)


def test_pdlp_parameters_text_without_ortools_is_unknown(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_ortools(name, *args, **kwargs):
        if name.startswith(("ortools", "google.protobuf")):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ortools)
    with pytest.raises(ToleranceUnknown, match="parameters_text"):
        _requested("pdlp", {"parameters_text": "termination_criteria { eps_optimal_relative: 1e-4 }"})
    # a text that cannot change an optimality tolerance does not need OR-Tools
    assert _requested("pdlp", {"eps": 1e-5, "parameters_text": "num_threads: 2"}).optimality == 1e-5
    infeasibility_only = "termination_criteria { eps_primal_infeasible: 1e-4 }"
    assert _requested("pdlp", {"eps": 1e-5, "parameters_text": infeasibility_only}).optimality == 1e-5


def test_proxqp_duality_gap_check():
    # the adapter turns the gap check on and fills it from eps_abs / eps_rel
    assert _requested("proxqp", {"eps_abs": 1e-6, "eps_rel": 0.0}).optimality == pytest.approx(1e-6)
    assert _requested("proxqp", {"eps_abs": 1e-6, "eps_duality_gap_abs": 1e-3}).optimality == pytest.approx(1e-3)
    # with the check off, the gap tolerances are not requested
    off = {"eps_abs": 1e-6, "check_duality_gap": False, "eps_duality_gap_abs": 1e-3}
    assert _requested("proxqp", off).optimality == pytest.approx(1e-6)


def test_piqp_gap_tolerances_follow_eps_and_the_gap_check():
    assert _requested("piqp", {"eps_abs": 1e-5, "eps_rel": 1e-5}).optimality == pytest.approx(1e-5)
    assert _requested("piqp", {"eps": 1e-6, "eps_duality_gap_abs": 1e-4}).optimality == pytest.approx(1e-4)
    off = {"eps": 1e-6, "eps_duality_gap_abs": 1e-4, "check_duality_gap": False}
    assert _requested("piqp", off).optimality == pytest.approx(1e-6)


def test_qtqp_legacy_names():
    # atol / rtol supply the Clarabel-style names on qtqp 0.0.7+
    assert _requested("qtqp", {"atol": 1e-5, "rtol": 1e-6}).optimality == pytest.approx(1e-5)
    # before 0.0.7 they are the names, and their defaults are not recorded
    old = {"qtqp": "0.0.6"}
    assert _requested("qtqp", {"atol": 1e-5, "rtol": 1e-6}, versions=old).optimality == pytest.approx(1e-5)
    assert _requested("qtqp", {"eps": 1e-6}, versions=old).optimality == pytest.approx(1e-6)
    with pytest.raises(ToleranceUnknown, match="atol"):
        _requested("qtqp", {}, versions=old)
    assert _requested("qtqp", {}, versions={"qtqp": "0.0.7"}).optimality == pytest.approx(1e-8)


def test_scs_before_3_needs_its_tolerances_set():
    # SCS 2.x defaults are not recorded: a row that leaves them unset is unknown.
    assert _requested("scs", {"eps": 1e-5}, versions={"scs": "2.1.4"}).optimality == pytest.approx(1e-5)
    with pytest.raises(ToleranceUnknown, match="2.1.4"):
        _requested("scs", {}, versions={"scs": "2.1.4"})
    assert _requested("scs", {}, versions={"scs": "3.2.7"}).optimality == pytest.approx(1e-4)


def test_mosek_lp_and_qp_knobs():
    settings = {"MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5, "MSK_DPAR_INTPNT_QO_TOL_REL_GAP": 1e-4}
    assert _requested("mosek", settings, is_lp=True).optimality == pytest.approx(1e-5)
    assert _requested("mosek", settings, is_lp=False).optimality == pytest.approx(1e-4)
    # not knowing which, the looser set applies
    assert _requested("mosek", settings).optimality == pytest.approx(1e-4)


def test_gurobi_names_are_case_insensitive():
    assert _requested("gurobi", {"feasibilitytol": 1e-4}).optimality == pytest.approx(1e-4)


def test_cuopt_aliases_follow_the_order_solve_applies_them():
    assert _requested("cuopt", {"eps": 1e-6}).optimality == pytest.approx(1e-6)
    assert _requested("cuopt", {"eps": 1e-6, "eps_rel": 1e-5}).optimality == pytest.approx(1e-5)
    assert _requested("cuopt", {"eps": 1e-6, "absolute_gap_tolerance": 1e-3}).optimality == pytest.approx(1e-3)


@pytest.mark.parametrize("value", [-1.0, float("nan"), "loose", True])
def test_a_tolerance_that_is_not_a_number_is_unknown(value):
    with pytest.raises(ValueError):
        _requested("clarabel", {"tol_feas": value})


def test_record_form_has_one_numeric_field():
    record = _requested("scs", {"eps": 1e-5}).to_record()
    assert record == {"optimality": 1e-5, "knobs": ["eps_abs", "eps_rel"], "defaults_used": []}
    assert _requested("scs", {"eps_abs": 1e-5}).to_record()["defaults_used"] == ["eps_rel"]


# --- the defaults are the installed libraries' -------------------------------


def _skip_without(module):
    return pytest.importorskip(module)


def test_clarabel_defaults_match_the_library():
    clarabel = _skip_without("clarabel")
    from solver_benchmarks.solvers import clarabel_adapter

    settings = clarabel.DefaultSettings()
    for name, value in clarabel_adapter.CLARABEL_DEFAULT_TOLERANCES.items():
        assert getattr(settings, name) == pytest.approx(value), name


def test_osqp_defaults_match_the_library():
    osqp = _skip_without("osqp")
    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.solvers import osqp_adapter

    solver = osqp.OSQP()
    solver.setup(sp.csc_matrix((1, 1)), np.zeros(1), sp.csc_matrix(np.eye(1)), np.zeros(1), np.ones(1),
                 verbose=False)
    for name, value in osqp_adapter.OSQP_DEFAULT_TOLERANCES.items():
        assert getattr(solver.settings, name) == pytest.approx(value), name


def test_piqp_defaults_match_the_library():
    piqp = _skip_without("piqp")
    from solver_benchmarks.solvers import piqp_adapter

    settings = piqp.SparseSolver().settings
    for name, value in {**piqp_adapter.PIQP_DEFAULT_TOLERANCES, **piqp_adapter.PIQP_DEFAULT_GAP_TOLERANCES}.items():
        assert getattr(settings, name) == pytest.approx(value), name


def test_proxqp_defaults_match_the_library():
    proxsuite = _skip_without("proxsuite")
    from solver_benchmarks.solvers import proxqp_adapter

    settings = proxsuite.proxqp.Settings()
    assert settings.eps_abs == pytest.approx(proxqp_adapter._PROXQP_DEFAULT_EPS_ABS)
    assert settings.eps_rel == pytest.approx(proxqp_adapter._PROXQP_DEFAULT_EPS_REL)


def test_highs_defaults_match_the_library():
    highspy = _skip_without("highspy")
    from solver_benchmarks.solvers import highs_adapter

    highs = highspy.Highs()
    names = {**highs_adapter.HIGHS_DEFAULT_TOLERANCES, "kkt_tolerance": highs_adapter.HIGHS_DEFAULT_KKT_TOLERANCE}
    for extra in highs_adapter.HIGHS_SOLVER_TOLERANCES.values():
        names.update(extra)
    for name, value in names.items():
        found = highs.getOptionValue(name)
        # (status, value) in the highspy this was written with; some releases return the value alone
        default = found[1] if isinstance(found, tuple) else found
        assert default == pytest.approx(value), name


def test_pdlp_defaults_match_the_library():
    solvers_pb2 = pytest.importorskip("ortools.pdlp.solvers_pb2")
    from solver_benchmarks.solvers import pdlp_adapter

    criteria = solvers_pb2.PrimalDualHybridGradientParams().termination_criteria
    for name, value in pdlp_adapter.PDLP_DEFAULT_TOLERANCES.items():
        assert getattr(criteria, name) == pytest.approx(value), name


def test_qtqp_defaults_match_the_library():
    qtqp = _skip_without("qtqp")
    import inspect

    from solver_benchmarks.solvers import qtqp_adapter

    parameters = inspect.signature(qtqp.QTQP.solve).parameters
    if "tol_feas" not in parameters:
        pytest.skip("qtqp before 0.0.7 has no Clarabel-style tolerances")
    for name, value in qtqp_adapter.QTQP_DEFAULT_TOLERANCES.items():
        assert parameters[name].default == pytest.approx(value), name


def test_sdpa_defaults_match_the_library():
    sdpap = _skip_without("sdpap")
    from solver_benchmarks.solvers import sdpa_adapter

    defaults = sdpap.param.param() if callable(getattr(sdpap.param, "param", None)) else sdpap.param()
    for name, value in sdpa_adapter.SDPA_DEFAULT_TOLERANCES.items():
        assert defaults[name] == pytest.approx(value), name


def test_cvxopt_defaults_match_the_library():
    pytest.importorskip("cvxopt")
    import inspect
    import re

    import cvxopt.coneprog

    from solver_benchmarks.solvers import cvxopt_adapter

    source = inspect.getsource(cvxopt.coneprog)
    for name, value in cvxopt_adapter.CVXOPT_DEFAULT_TOLERANCES.items():
        found = {float(m) for m in re.findall(rf"options\.get\('{name}',\s*([0-9.e+-]+)\)", source)}
        assert found == {value}, name


# --- the worker records the requested tolerance ----------------------------------


def test_the_worker_records_the_requested_tolerance(tmp_path, repo_root):
    pytest.importorskip("scs")
    from solver_benchmarks.worker import run_payload

    payload = {
        "run_id": "run", "dataset": "synthetic_qp", "problem": "one_variable_lp", "repo_root": str(repo_root),
        "solver": {"id": "scs_test", "solver": "scs", "settings": {"eps": 1e-5}},
        "artifacts_dir": str(tmp_path / "artifacts"),
    }
    record = run_payload(payload).to_record()
    assert record["metadata"]["requested_tolerance"]["optimality"] == pytest.approx(1e-5)
    assert "certificate" not in record["metadata"]["requested_tolerance"]


def test_the_worker_records_why_a_tolerance_is_unknown(tmp_path, repo_root, monkeypatch):
    pytest.importorskip("scs")
    from solver_benchmarks.solvers.scs_adapter import SCSSolverAdapter
    from solver_benchmarks.worker import run_payload

    def unknown(*args, **kwargs):
        raise ValueError("cannot tell")

    monkeypatch.setattr(SCSSolverAdapter, "requested_tolerance", classmethod(unknown))
    payload = {
        "run_id": "run", "dataset": "synthetic_qp", "problem": "one_variable_lp", "repo_root": str(repo_root),
        "solver": {"id": "scs_test", "solver": "scs", "settings": {}},
        "artifacts_dir": str(tmp_path / "artifacts"),
    }
    record = run_payload(payload).to_record()
    assert record["status"] == "optimal"  # the solve is unaffected
    assert "cannot tell" in record["metadata"]["requested_tolerance"]["unknown"]
