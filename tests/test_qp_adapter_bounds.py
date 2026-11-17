"""How the QP adapters hand infinite bounds and free rows to their solvers.

The datasets write +-1e20 for an absent bound, sometimes with a few ULPs of
representation error, and the transform layer (``split_qp_bounds``) treats
every |bound| >= 1e19 as infinite. Each adapter must give its solver true
infinities by that same rule, so a solver is not scored differently
depending on how a loader spelled infinity, and rows that constrain nothing
(``l = -inf``, ``u = +inf``) are not handed to solvers that pay for them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.solvers import get_solver


def _solve(solver_name: str, qp: dict, tmp_path: Path, settings: dict | None = None):
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} not installed")
    artifacts = tmp_path / solver_name
    artifacts.mkdir(parents=True, exist_ok=True)
    adapter = adapter_cls({"verbose": False, **(settings or {})})
    return adapter.solve(ProblemData("test", "p", QP, qp), artifacts)


def _unbounded_lp(sentinel: float) -> dict:
    # min -x0 s.t. x0 - x1 <= 1, x1 >= 0: unbounded along x0 = x1 -> inf.
    return {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([-1.0, 0.0]),
        "A": sp.csc_matrix([[1.0, -1.0], [0.0, 1.0]]),
        "l": np.array([-sentinel, 0.0]),
        "u": np.array([1.0, sentinel]),
    }


@pytest.mark.parametrize("sentinel", [np.inf, 1e20, 9.999999999999998e19])
def test_osqp_detects_unboundedness_whatever_the_infinity_spelling(sentinel, tmp_path):
    # OSQP itself treats a bound as infinite only above 1e30 * 1e-4, so a raw
    # 1e20 reached it as a finite bound: it then ran to max_iter instead of
    # returning a dual infeasibility certificate.
    result = _solve("osqp", _unbounded_lp(sentinel), tmp_path)
    assert result.status == status.DUAL_INFEASIBLE, result.info
    assert result.kkt is not None
    assert result.kkt["certificate"] == "dual_infeasible"


def _qp_with_free_and_sentinel_rows(free: float, one_sided: float) -> dict:
    # min 0.5|x|^2 + q'x with one equality, one-sided rows written with a
    # sentinel, and two rows that constrain nothing.
    return {
        "P": sp.eye(3, format="csc"),
        "q": np.array([1.0, -2.0, 0.5]),
        "A": sp.csc_matrix(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [2.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 3.0, -1.0],
            ]
        ),
        "l": np.array([1.0, -free, -one_sided, -one_sided, -free]),
        "u": np.array([1.0, free, 0.5, 1.0, free]),
    }


_TIGHT = {
    "osqp": {"eps_abs": 1e-9, "eps_rel": 1e-9, "polish": True, "max_iter": 100000},
    "proxqp": {"eps_abs": 1e-9, "eps_rel": 1e-9},
    "scs": {"eps_abs": 1e-9, "eps_rel": 1e-9},
}


@pytest.mark.parametrize("solver_name", ["osqp", "piqp", "proxqp", "highs", "scs"])
@pytest.mark.parametrize(
    "free, one_sided",
    [(np.inf, np.inf), (1e20, 1e20), (9.999999999999998e19, 5e19)],
)
def test_qp_adapters_agree_on_infinite_bound_spellings(solver_name, free, one_sided, tmp_path):
    settings = _TIGHT.get(solver_name, {})
    reference = _solve(
        solver_name, _qp_with_free_and_sentinel_rows(np.inf, np.inf), tmp_path, settings
    )
    result = _solve(
        solver_name, _qp_with_free_and_sentinel_rows(free, one_sided), tmp_path, settings
    )
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(reference.objective_value, abs=1e-5)
    assert result.iterations == reference.iterations
    assert result.kkt is not None
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert result.kkt[field] < 1e-4


def test_osqp_does_not_receive_free_rows(tmp_path, monkeypatch):
    osqp = pytest.importorskip("osqp")
    received = {}
    real = osqp.OSQP

    class Recording(real):
        def setup(self, P, q, A, l, u, **settings):
            received.update(A=A.copy(), l=np.array(l), u=np.array(u))
            return super().setup(P, q, A, l, u, **settings)

    monkeypatch.setattr(osqp, "OSQP", Recording)
    qp = _qp_with_free_and_sentinel_rows(1e20, 1e20)
    result = _solve("osqp", qp, tmp_path, {"eps_abs": 1e-8, "eps_rel": 1e-8})
    assert received["A"].shape == (3, 3)
    assert received["l"].tolist() == [1.0, -np.inf, -np.inf]
    assert received["u"].tolist() == [1.0, 0.5, 1.0]
    assert result.status == status.OPTIMAL
    # Duals are reported for every original row; dropped rows carry zero.
    assert result.kkt["dual_bound_res_rel"] < 1e-8
    assert result.kkt["dual_res_rel"] < 1e-6


def test_osqp_with_only_free_rows(tmp_path):
    qp = {
        "P": sp.eye(2, format="csc"),
        "q": np.array([1.0, -1.0]),
        "A": sp.csc_matrix([[1.0, 1.0]]),
        "l": np.array([-1e20]),
        "u": np.array([1e20]),
    }
    result = _solve("osqp", qp, tmp_path, {"eps_abs": 1e-8, "eps_rel": 1e-8})
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(-1.0, abs=1e-6)
    assert result.kkt["dual_res_rel"] < 1e-6


@pytest.mark.parametrize("solver_name", ["piqp", "proxqp"])
def test_split_adapters_do_not_receive_free_rows(solver_name, tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module(f"solver_benchmarks.solvers.{solver_name}_adapter")
    real_split = module.split_qp_for_range_constraints
    shapes = []

    def recording_split(qp):
        out = real_split(qp)
        shapes.append(None if out[4] is None else out[4].shape)
        return out

    monkeypatch.setattr(module, "split_qp_for_range_constraints", recording_split)
    result = _solve(solver_name, _qp_with_free_and_sentinel_rows(1e20, 1e20), tmp_path)
    assert shapes == [(2, 3)]
    assert result.status == status.OPTIMAL
    assert result.kkt["dual_bound_res_rel"] < 1e-8


def test_scs_does_not_receive_free_rows(tmp_path, monkeypatch):
    scs = pytest.importorskip("scs")
    received = {}
    real = scs.solve

    def recording(data, cone, **settings):
        received.update(A=data["A"].copy(), cone=dict(cone))
        return real(data, cone, **settings)

    monkeypatch.setattr(scs, "solve", recording)
    qp = _qp_with_free_and_sentinel_rows(1e20, 1e20)
    result = _solve("scs", qp, tmp_path, {"eps_abs": 1e-8, "eps_rel": 1e-8})
    # One equality, the box cone's t row and two inequalities: the two free
    # rows are left out.
    assert received["A"].shape == (4, 3)
    assert received["cone"]["z"] == 1
    assert received["cone"]["bl"] == [-np.inf, -np.inf]
    assert received["cone"]["bu"] == [0.5, 1.0]
    assert result.status == status.OPTIMAL
    # Duals are reported for every original row; dropped rows carry zero.
    assert result.kkt["dual_bound_res_rel"] < 1e-8
    assert result.kkt["dual_res_rel"] < 1e-6


def test_scs_with_equalities_and_free_rows_only(tmp_path, monkeypatch):
    # Once the free rows are gone only the zero cone is left (no box cone).
    scs = pytest.importorskip("scs")
    cones = []
    real = scs.solve

    def recording(data, cone, **settings):
        cones.append(dict(cone))
        return real(data, cone, **settings)

    monkeypatch.setattr(scs, "solve", recording)
    qp = {
        "P": sp.eye(2, format="csc"),
        "q": np.array([1.0, -1.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([1.0, -1e20, -np.inf]),
        "u": np.array([1.0, 1e20, np.inf]),
    }
    result = _solve("scs", qp, tmp_path, {"eps_abs": 1e-8, "eps_rel": 1e-8})
    assert cones == [{"z": 1}]
    assert result.status == status.OPTIMAL
    # min 0.5|x|^2 + x0 - x1 s.t. x0 + x1 = 1: x = (-0.5, 1.5), objective -0.75.
    assert result.objective_value == pytest.approx(-0.75, abs=1e-6)
    assert result.kkt["dual_res_rel"] < 1e-6


def test_scs_with_only_free_rows(tmp_path):
    qp = {
        "P": sp.eye(2, format="csc"),
        "q": np.array([1.0, -1.0]),
        "A": sp.csc_matrix([[1.0, 1.0]]),
        "l": np.array([-1e20]),
        "u": np.array([1e20]),
    }
    result = _solve("scs", qp, tmp_path, {"eps_abs": 1e-8, "eps_rel": 1e-8})
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(-1.0, abs=1e-6)
    assert result.kkt["dual_res_rel"] < 1e-6


def test_highs_treats_near_sentinel_bounds_as_infinite(tmp_path, monkeypatch):
    highspy = pytest.importorskip("highspy")
    solver = highspy.Highs()
    monkeypatch.setattr(highspy, "Highs", lambda: solver)
    _solve("highs", _qp_with_free_and_sentinel_rows(9.999999999999998e19, 5e19), tmp_path)
    lp = solver.getLp()
    assert solver.getNumRow() == 3
    assert np.isneginf(np.asarray(lp.row_lower_)[1:]).all()


@pytest.mark.parametrize("free_bound", [np.inf, 1e20, 9.999999999999998e19])
def test_pdlp_reports_kkt_for_an_lp_with_a_free_row(free_bound, tmp_path):
    # PDLP leaves free rows out of its model, so its dual has one entry per
    # retained row; it must be scattered back to the original rows before the
    # KKT check (it used to raise inside a blanket except and leave kkt=None,
    # which kkt-verify then counted as a failure).
    qp = {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([1.0, 2.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [1.0, -1.0], [0.0, 1.0]]),
        "l": np.array([1.0, 0.0, -free_bound, 0.0]),
        "u": np.array([1.0, np.inf, free_bound, np.inf]),
    }
    result = _solve("pdlp", qp, tmp_path, {"eps_abs": 1e-8, "eps_rel": 1e-8})
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(1.0, abs=1e-6)
    assert result.kkt is not None
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert result.kkt[field] < 1e-6
    assert result.kkt["dual_bound_res_rel"] < 1e-8
