"""Infeasibility certificates of the adapters that run here, checked end to end.

Each installed adapter solves known infeasible and unbounded problems. The
certificate it records must pass the harness's verification at the row's
tolerance (``VERIFY_SLACK`` times what the settings request), and the same
solve with the certificate corrupted on its way out of the library (its sign
flipped, the likeliest conversion error, or that of its largest entry) must
fail it.
Gurobi, CPLEX, MOSEK and cuOpt are not installed here; their conversions are
pinned against fakes in test_commercial_adapters_fake.py and
test_mosek_adapter_fake.py.
"""

from __future__ import annotations

import time
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.analysis.verify import VERIFY_SLACK, classify
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.solvers import get_solver

INF = np.inf


def _qp(p, q, a, lower, upper) -> dict:
    a = sp.csc_matrix(np.asarray(a, dtype=float))
    n = a.shape[1]
    return {
        "P": sp.csc_matrix((n, n)) if p is None else sp.csc_matrix(np.asarray(p, dtype=float)),
        "q": np.asarray(q, dtype=float),
        "A": a,
        "l": np.asarray(lower, dtype=float),
        "u": np.asarray(upper, dtype=float),
        "n": n,
        "m": a.shape[0],
        "obj_type": "min",
    }


def _infeasible_lp() -> dict:
    # x1 + x2 <= 1, x1 >= 1, a free row (left out by HiGHS), 1 <= x2 <= 2:
    # the first row minus the second and fourth gives 0 <= -1.
    return _qp(None, [1.0, 1.0], [[1, 1], [1, 0], [5, 7], [0, 1]], [-INF, 1, -INF, 1], [1, INF, INF, 2])


def _infeasible_equality_lp() -> dict:
    # x1 + x2 = 3 with 0 <= x1 <= 1 and 0 <= x2 <= 1.
    return _qp(None, [0.0, 0.0], [[1, 1], [1, 0], [0, 1]], [3, 0, 0], [3, 1, 1])


def _infeasible_qp() -> dict:
    return _qp(np.eye(2), [1.0, 1.0], [[1, 1], [1, 0], [0, 1]], [-INF, 1, 1], [1, INF, INF])


def _unbounded_lp() -> dict:
    # min -x1 - x2 with x1 - x2 <= 1, x1, x2 >= 0 and a free row: ray (1, 1).
    return _qp(None, [-1.0, -1.0], [[1, -1], [1, 0], [0, 1], [1, 1]], [-INF, 0, 0, -INF], [1, INF, INF, INF])


def _unbounded_qp() -> dict:
    # min x1^2 / 2 - x2 with x2 >= 0, -1 <= x1 <= 1.
    return _qp(np.diag([1.0, 0.0]), [0.0, -1.0], [[0, 1], [1, 0]], [0, -1], [INF, 1])


def _netlib(name: str) -> ProblemData:
    from solver_benchmarks.datasets import get_dataset

    repo_root = Path(__file__).resolve().parent.parent
    return get_dataset("netlib")(repo_root=repo_root, subset="infeasible").load_problem(name)


def _problem(data) -> ProblemData:
    return data if isinstance(data, ProblemData) else ProblemData("test", "p", QP, data)


def _adapter(name: str, settings: dict):
    adapter_cls = get_solver(name)
    if not adapter_cls.is_available():
        pytest.skip(f"{name} not installed")
    return adapter_cls(settings)


def _verdict(name: str, settings: dict, result, *, is_lp: bool | None = None):
    """The verification of ``result`` by the default rule: ``VERIFY_SLACK``
    times the tolerance ``settings`` request."""
    requested = get_solver(name).requested_tolerance(settings, is_lp=is_lp).optimality
    tol = VERIFY_SLACK * requested
    return classify({"status": result.status, "kkt": result.kkt}, tolerance=lambda: tol)


def _flip(values):
    return -np.asarray(values, dtype=float)


def _negate_largest(values):
    values = np.array(values, dtype=float)
    values[np.argmax(np.abs(values))] *= -1.0
    return values


# A sign convention read the wrong way round, for the whole certificate or
# for one row (as a mistake on one kind of row, say range rows, would do).
CORRUPTIONS = {"sign flipped": _flip, "largest entry negated": _negate_largest}


# ---------------------------------------------------------------------------
# HiGHS: getDualRay / getPrimalRay


HIGHS_SETTINGS = {"verbose": False}

HIGHS_CLAIMS = [
    pytest.param(_infeasible_lp, status.PRIMAL_INFEASIBLE, id="lp-infeasible-free-and-range-rows"),
    pytest.param(_infeasible_equality_lp, status.PRIMAL_INFEASIBLE, id="lp-infeasible-equality"),
    pytest.param(_infeasible_qp, status.PRIMAL_INFEASIBLE, id="qp-infeasible"),
    pytest.param(_unbounded_lp, status.DUAL_INFEASIBLE, id="lp-unbounded"),
    # NETLIB: HiGHS proves these on the presolved LP, so the ray comes from
    # the re-solve getDualRay makes without presolve.
    pytest.param(lambda: _netlib("bgprtr"), status.PRIMAL_INFEASIBLE, id="netlib-bgprtr"),
    pytest.param(lambda: _netlib("woodinfe"), status.PRIMAL_INFEASIBLE, id="netlib-woodinfe"),
    pytest.param(lambda: _netlib("forest6"), status.PRIMAL_INFEASIBLE, id="netlib-forest6"),
]


@pytest.mark.parametrize("make, claim", HIGHS_CLAIMS)
def test_highs_certificate_passes_the_verification(make, claim, tmp_path):
    result = _adapter("highs", HIGHS_SETTINGS).solve(_problem(make()), tmp_path)
    assert result.status == claim
    assert result.kkt is not None and result.kkt["certificate"] == claim
    verdict = _verdict("highs", HIGHS_SETTINGS, result)
    assert (verdict.verdict, verdict.status) == ("kept", claim), verdict.reason
    assert result.info["certificate_seconds"] >= 0.0
    assert "certificate_unavailable" not in result.info


@pytest.mark.parametrize("corrupt", CORRUPTIONS.values(), ids=CORRUPTIONS.keys())
@pytest.mark.parametrize("make, claim", HIGHS_CLAIMS)
def test_highs_corrupted_certificate_fails_the_verification(make, claim, corrupt, monkeypatch, tmp_path):
    import highspy

    getter = "getDualRay" if claim == status.PRIMAL_INFEASIBLE else "getPrimalRay"
    real = getattr(highspy.Highs, getter)

    def corrupted(self):
        api_status, has_ray, values = real(self)
        return api_status, has_ray, corrupt(values)

    monkeypatch.setattr(highspy.Highs, getter, corrupted)
    result = _adapter("highs", HIGHS_SETTINGS).solve(_problem(make()), tmp_path)
    assert result.status == claim
    assert result.kkt is not None and result.kkt["certificate"] == claim
    verdict = _verdict("highs", HIGHS_SETTINGS, result)
    assert verdict.verdict == "demoted"
    assert verdict.status == f"{claim}_inaccurate"


def test_highs_counts_the_time_its_certificate_takes(monkeypatch, tmp_path):
    import highspy

    real = highspy.Highs.getDualRay

    def slow(self):
        time.sleep(0.2)
        return real(self)

    monkeypatch.setattr(highspy.Highs, "getDualRay", slow)
    result = _adapter("highs", HIGHS_SETTINGS).solve(_problem(_infeasible_lp()), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.info["certificate_seconds"] >= 0.2
    assert result.run_time_seconds >= result.info["certificate_seconds"]


def _no_ray(self):
    return None, False, np.zeros(0)


@pytest.mark.parametrize("make, claim, reason", [
    # HiGHS's QP solver gives no ray for an unbounded QP (highspy 1.15.1
    # and 1.16.0).
    (_unbounded_qp, status.DUAL_INFEASIBLE, "HiGHS returned no primal ray"),
    # Which LP claims get no dual ray depends on the HiGHS release (1.15.1:
    # none after interior point or PDLP; NETLIB cplex2, klein1 and qual in
    # 1.15.1 and 1.16.0), so this one is made to have none.
    (_infeasible_lp, status.PRIMAL_INFEASIBLE, "HiGHS returned no dual ray"),
])
def test_highs_claim_without_a_ray_is_not_verified(make, claim, reason, monkeypatch, tmp_path):
    import highspy

    if claim == status.PRIMAL_INFEASIBLE:
        monkeypatch.setattr(highspy.Highs, "getDualRay", _no_ray)
    result = _adapter("highs", HIGHS_SETTINGS).solve(_problem(make()), tmp_path)
    assert result.status == claim
    assert result.kkt is None
    assert result.info["certificate_unavailable"] == reason
    verdict = _verdict("highs", HIGHS_SETTINGS, result)
    assert (verdict.verdict, verdict.status) == ("missing_residuals", f"{claim}_inaccurate")


def test_highs_feasible_solve_does_not_look_for_a_certificate(tmp_path):
    lp = _qp(None, [-1.0, -1.0], [[1, 1], [1, 0], [0, 1]], [-INF, 0, 0], [1, INF, INF])
    result = _adapter("highs", HIGHS_SETTINGS).solve(_problem(lp), tmp_path)
    assert result.status == status.OPTIMAL
    assert "certificate_seconds" not in result.info
    assert "certificate" not in result.kkt


# ---------------------------------------------------------------------------
# PIQP, ProxQP and SDPA return their certificate in the solution vectors,
# which the adapters already record; these pin that and its sign.


PIQP_SETTINGS = {"verbose": False}


@pytest.mark.parametrize("make, claim", [
    # Most of PIQP's NETLIB certificates pass at ten times its default 1e-8
    # with piqp 0.6.4 but not with 0.7.0 (bgprtr 1.4e-6, itest2 1.2e-5);
    # these pass with both.
    pytest.param(lambda: _netlib("chemcom"), status.PRIMAL_INFEASIBLE, id="netlib-chemcom"),
    pytest.param(lambda: _netlib("mondou2"), status.PRIMAL_INFEASIBLE, id="netlib-mondou2"),
    pytest.param(_unbounded_lp, status.DUAL_INFEASIBLE, id="lp-unbounded"),
    pytest.param(_unbounded_qp, status.DUAL_INFEASIBLE, id="qp-unbounded"),
])
@pytest.mark.parametrize("corrupt", [None, *CORRUPTIONS.values()], ids=["as returned", *CORRUPTIONS])
def test_piqp_certificate_is_verified(make, claim, corrupt, monkeypatch, tmp_path):
    adapter = _adapter("piqp", PIQP_SETTINGS)
    if corrupt is not None:
        import piqp

        real = piqp.SparseSolver

        class Corrupted:
            """PIQP's solver, with the certificate in its result corrupted."""

            def __init__(self):
                self._solver = real()

            def __getattr__(self, name):
                return getattr(self._solver, name)

            @property
            def result(self):
                result = self._solver.result
                # The adapter's dual is z_u - z_l on inequality rows, y on
                # equality rows: corrupt that combination.
                m_eq, m_in = len(result.y), len(result.z_u)
                dual = corrupt(np.concatenate([result.y, result.z_u - result.z_l]))
                return types.SimpleNamespace(
                    x=corrupt(result.x) if claim == status.DUAL_INFEASIBLE else result.x,
                    y=dual[:m_eq] if claim == status.PRIMAL_INFEASIBLE else result.y,
                    z_u=(np.maximum(dual[m_eq:m_eq + m_in], 0.0)
                         if claim == status.PRIMAL_INFEASIBLE else result.z_u),
                    z_l=(np.maximum(-dual[m_eq:m_eq + m_in], 0.0)
                         if claim == status.PRIMAL_INFEASIBLE else result.z_l),
                    info=result.info,
                )

        monkeypatch.setattr(piqp, "SparseSolver", Corrupted)
    result = adapter.solve(_problem(make()), tmp_path)
    assert result.status == claim
    assert result.kkt is not None and result.kkt["certificate"] == claim
    verdict = _verdict("piqp", PIQP_SETTINGS, result)
    if corrupt is None:
        assert (verdict.verdict, verdict.status) == ("kept", claim), verdict.reason
    else:
        assert (verdict.verdict, verdict.status) == ("demoted", f"{claim}_inaccurate")


PROXQP_SETTINGS = {"verbose": False}


@pytest.mark.parametrize("corrupt", [None, *CORRUPTIONS.values()], ids=["as returned", *CORRUPTIONS])
def test_proxqp_certificate_is_verified(corrupt, monkeypatch, tmp_path):
    # ProxSuite writes its certificate (the last step of its iterates) into
    # x, y and z (proxqp/sparse/solver.hpp, "certificate of infeasibility").
    # It detects no unbounded problem here, so only a Farkas certificate is
    # checked.
    adapter = _adapter("proxqp", PROXQP_SETTINGS)
    if corrupt is not None:
        import proxsuite

        real = proxsuite.proxqp.sparse.solve

        def corrupted(*args, **kwargs):
            result = real(*args, **kwargs)
            dual = corrupt(np.concatenate([result.y, result.z]))
            return types.SimpleNamespace(
                x=result.x, y=dual[:len(result.y)], z=dual[len(result.y):], info=result.info,
            )

        corrupted.__doc__ = real.__doc__  # the adapter reads the accepted settings from it
        monkeypatch.setattr(proxsuite.proxqp.sparse, "solve", corrupted)
    result = adapter.solve(_problem(_infeasible_equality_lp()), tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.kkt is not None and result.kkt["certificate"] == status.PRIMAL_INFEASIBLE
    verdict = _verdict("proxqp", PROXQP_SETTINGS, result)
    if corrupt is None:
        assert (verdict.verdict, verdict.status) == ("kept", status.PRIMAL_INFEASIBLE), verdict.reason
    else:
        assert (verdict.verdict, verdict.status) == ("demoted", status.PRIMAL_INFEASIBLE_INACCURATE)


def _cone(q, a, b, cone) -> ProblemData:
    a = sp.csc_matrix(np.asarray(a, dtype=float))
    data = {"P": None, "q": np.asarray(q, dtype=float), "A": a, "b": np.asarray(b, dtype=float), "r": 0.0,
            "n": a.shape[1], "m": a.shape[0], "cone": cone, "obj_type": "min"}
    return ProblemData("test", "p", CONE, data)


SDPA_SETTINGS = {"verbose": False}


@pytest.mark.parametrize("problem, claim", [
    # x >= 2 and x <= 1 in cone form (A x + s = b, s >= 0).
    pytest.param(_cone([0.0], [[-1.0], [1.0]], [-2.0, 1.0], {"l": 2}), status.PRIMAL_INFEASIBLE, id="infeasible"),
    # min -x with x >= 0.
    pytest.param(_cone([-1.0], [[-1.0]], [0.0], {"l": 1}), status.DUAL_INFEASIBLE, id="unbounded"),
])
@pytest.mark.parametrize("corrupt", [None, _flip], ids=["as returned", "sign flipped"])
def test_sdpa_certificate_is_verified(problem, claim, corrupt, monkeypatch, tmp_path):
    adapter = _adapter("sdpa", SDPA_SETTINGS)
    if corrupt is not None:
        import sdpap

        real = sdpap.solve

        def corrupted(*args, **kwargs):
            x, y, *rest = real(*args, **kwargs)
            return (corrupt(np.asarray(x.toarray() if sp.issparse(x) else x).ravel()),
                    corrupt(np.asarray(y.toarray() if sp.issparse(y) else y).ravel()), *rest)

        monkeypatch.setattr(sdpap, "solve", corrupted)
    result = adapter.solve(problem, tmp_path)
    assert result.status == claim
    assert result.kkt is not None and result.kkt["certificate"] == claim
    verdict = _verdict("sdpa", SDPA_SETTINGS, result)
    if corrupt is None:
        assert (verdict.verdict, verdict.status) == ("kept", claim), verdict.reason
    else:
        assert (verdict.verdict, verdict.status) == ("demoted", f"{claim}_inaccurate")
