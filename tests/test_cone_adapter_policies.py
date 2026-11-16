"""Cross-adapter result policy for the conic adapters (SCS, Clarabel, ECOS,
CVXOPT, SDPA).

Every adapter reports the same kind of event the same way, so status
breakdowns, ``kkt-verify --promote`` and objective listings compare solvers
rather than adapters:

* a solve stopped by its iteration or time limit is ``max_iter_reached`` /
  ``time_limit`` (the solver's own wording stays in ``info``), never
  ``optimal_inaccurate`` or an ``*_infeasible_inaccurate`` guess;
* solution KKT residuals are recorded for every status in
  ``status.ANY_FEASIBLE`` whenever the solver returned a point;
* ``objective_value`` is reported only for ``optimal`` and
  ``optimal_inaccurate``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.solvers import get_solver


def _adapter(solver_name: str, settings: dict):
    adapter_cls = get_solver(solver_name)
    if not adapter_cls.is_available():
        pytest.skip(f"{solver_name} not installed")
    return adapter_cls({"verbose": False, **settings})


def _solve(solver_name: str, kind: str, data: dict, settings: dict, tmp_path: Path):
    adapter = _adapter(solver_name, settings)
    artifacts = tmp_path / solver_name
    artifacts.mkdir(parents=True, exist_ok=True)
    return adapter.solve(ProblemData("test", "p", kind, data), artifacts)


def _feasible_cone_lp(seed: int = 0, n: int = 60, m: int = 120) -> dict:
    """A random LP ``min c'x s.t. Ax + s = b, s >= 0`` with a strictly
    feasible primal point and a strictly feasible dual point."""
    rng = np.random.default_rng(seed)
    a = sp.random(m, n, density=0.3, random_state=seed + 1, format="csc")
    b = a @ rng.standard_normal(n) + np.abs(rng.standard_normal(m))
    c = -a.T @ np.abs(rng.standard_normal(m))
    return {"P": None, "A": a, "b": b, "q": c, "r": 0.0, "cone": {"l": m}}


def _feasible_qp(seed: int = 0, n: int = 80) -> dict:
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, n))
    return {
        "P": sp.csc_matrix(g.T @ g / n + np.eye(n)),
        "q": -np.linspace(1.0, 3.0, n),
        "A": sp.eye(n, format="csc"),
        "l": np.zeros(n),
        "u": np.ones(n),
    }


def _infeasible_cone_lp() -> dict:
    # x >= 2 and x <= 1 in cone form: [-x; x] + s = [-2; 1], s >= 0.
    return {
        "P": None,
        "A": sp.csc_matrix([[-1.0], [1.0]]),
        "b": np.array([-2.0, 1.0]),
        "q": np.array([1.0]),
        "r": 0.0,
        "cone": {"l": 2},
    }


def _infeasible_qp_form_lp() -> dict:
    return {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([1.0]),
        "A": sp.csc_matrix([[1.0], [1.0]]),
        "l": np.array([2.0, -np.inf]),
        "u": np.array([np.inf, 1.0]),
    }


def _assert_finite_solution_kkt(result) -> None:
    assert result.kkt is not None, result.info
    assert result.kkt.get("certificate") is None
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel"):
        assert np.isfinite(result.kkt[field]), (field, result.kkt)


# Settings that stop each adapter after very few iterations.
ITERATION_CAPS = {
    "scs": {"max_iters": 3},
    "clarabel": {"max_iter": 1, "presolve_enable": False},
    "ecos": {"max_iters": 1},
    "cvxopt": {"maxiters": 1},
    "sdpa": {"maxIteration": 1},
}


@pytest.mark.parametrize("solver_name", sorted(ITERATION_CAPS))
def test_iteration_limit_is_max_iter_with_kkt_and_no_objective(solver_name, tmp_path):
    result = _solve(
        solver_name, CONE, _feasible_cone_lp(), ITERATION_CAPS[solver_name], tmp_path
    )
    assert result.status == status.MAX_ITER_REACHED, (result.status, result.info)
    _assert_finite_solution_kkt(result)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "ecos", "cvxopt"])
def test_qp_iteration_limit_is_max_iter_with_kkt(solver_name, tmp_path):
    result = _solve(solver_name, QP, _feasible_qp(), ITERATION_CAPS[solver_name], tmp_path)
    assert result.status == status.MAX_ITER_REACHED, (result.status, result.info)
    _assert_finite_solution_kkt(result)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs", "clarabel"])
def test_time_limit_is_time_limit_with_kkt_and_no_objective(solver_name, tmp_path):
    settings = {"time_limit": 1e-6}
    if solver_name == "clarabel":
        settings["presolve_enable"] = False
    result = _solve(solver_name, CONE, _feasible_cone_lp(n=200, m=400), settings, tmp_path)
    assert result.status == status.TIME_LIMIT, (result.status, result.info)
    _assert_finite_solution_kkt(result)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "ecos", "cvxopt", "sdpa"])
def test_infeasible_cone_result_has_no_reported_objective(solver_name, tmp_path):
    result = _solve(solver_name, CONE, _infeasible_cone_lp(), {}, tmp_path)
    assert result.status in status.ANY_INFEASIBLE, (result.status, result.info)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "ecos", "cvxopt"])
def test_infeasible_qp_result_has_no_reported_objective(solver_name, tmp_path):
    result = _solve(solver_name, QP, _infeasible_qp_form_lp(), {}, tmp_path)
    assert result.status in status.ANY_INFEASIBLE, (result.status, result.info)
    assert result.objective_value is None


@pytest.mark.parametrize("solver_name", ["scs", "clarabel", "ecos", "cvxopt", "sdpa"])
def test_optimal_cone_result_reports_objective(solver_name, tmp_path):
    result = _solve(solver_name, CONE, _feasible_cone_lp(n=10, m=20), {}, tmp_path)
    assert result.status == status.OPTIMAL, (result.status, result.info)
    assert result.objective_value is not None
    assert result.objective_value == pytest.approx(result.kkt["primal_obj"], rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# SCS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info, expected",
    [
        ({"status_val": 2, "status": "solved (inaccurate - reached max_iters)"},
         status.MAX_ITER_REACHED),
        ({"status_val": 2, "status": "solved (inaccurate - reached time_limit_secs)"},
         status.TIME_LIMIT),
        ({"status_val": -6, "status": "unbounded (inaccurate - reached max_iters)"},
         status.MAX_ITER_REACHED),
        ({"status_val": -7, "status": "infeasible (inaccurate - reached time_limit_secs)"},
         status.TIME_LIMIT),
        ({"status_val": -7, "status": "infeasible (inaccurate - reached max_iters)"},
         status.MAX_ITER_REACHED),
        # Text missing: the configured limits decide.
        ({"status_val": 2, "status": "", "iter": 50}, status.MAX_ITER_REACHED),
        ({"status_val": 2, "status": "", "iter": 7, "solve_time": 20.0}, status.TIME_LIMIT),
        ({"status_val": 2, "status": "", "iter": 7, "solve_time": 1.0},
         status.OPTIMAL_INACCURATE),
        # A real solution or certificate is never relabelled.
        ({"status_val": 1, "status": "solved", "iter": 50}, status.OPTIMAL),
        ({"status_val": -2, "status": "infeasible", "iter": 50}, status.PRIMAL_INFEASIBLE),
    ],
)
def test_scs_limit_statuses(info, expected):
    from solver_benchmarks.solvers.scs_adapter import _map_scs_status

    assert _map_scs_status(info, max_iters=50, time_limit_secs=0.01) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("solved (inaccurate)", status.OPTIMAL_INACCURATE),
        ("infeasible (inaccurate)", status.PRIMAL_INFEASIBLE_INACCURATE),
        ("unbounded (inaccurate)", status.DUAL_INFEASIBLE_INACCURATE),
        ("solved (inaccurate - reached max_iters)", status.MAX_ITER_REACHED),
    ],
)
def test_scs_text_fallback_keeps_inaccurate_inaccurate(text, expected):
    from solver_benchmarks.solvers.scs_adapter import _map_scs_status

    assert _map_scs_status({"status": text}) == expected


def test_scs_limit_keeps_raw_status(tmp_path):
    result = _solve("scs", CONE, _feasible_cone_lp(), {"max_iters": 3}, tmp_path)
    assert result.status == status.MAX_ITER_REACHED
    assert "reached max_iters" in result.info["raw_status"]
    assert result.info["status_val"] == 2


def test_scs_translates_max_iter_and_eps(tmp_path):
    result = _solve(
        "scs", CONE, _feasible_cone_lp(), {"max_iter": 3, "eps": 1e-3}, tmp_path
    )
    assert result.status == status.MAX_ITER_REACHED
    assert result.iterations == 3
    assert result.info["settings_translated"] == {
        "max_iters": 3,
        "eps_abs": 1e-3,
        "eps_rel": 1e-3,
    }


def test_scs_native_settings_win_over_aliases(tmp_path):
    from solver_benchmarks.solvers.scs_adapter import translate_settings

    settings = {"max_iter": 3, "max_iters": 9, "eps": 1e-3, "eps_abs": 1e-7}
    translated = translate_settings(settings)
    assert settings == {"max_iters": 9, "eps_abs": 1e-7, "eps_rel": 1e-3}
    assert translated == {"eps_rel": 1e-3}


# ---------------------------------------------------------------------------
# Clarabel
# ---------------------------------------------------------------------------


def _exp_cone_problem() -> dict:
    """max sum_i log(x_i) s.t. sum_i x_i = 1, x_i <= 0.9, written with one
    exponential cone per term: log(x_i) >= u_i  <=>  (u_i, 1, x_i) in K_exp.

    Variables [u_1..u_3, x_1..x_3]; optimum x_i = 1/3, min -sum u = 3 log 3.
    """
    n = 3
    rows, b = [], []
    # zero cone: sum x = 1
    rows.append(np.concatenate([np.zeros(n), np.ones(n)]))
    b.append(1.0)
    # nonnegative: x_i <= 0.9  ->  x_i + s = 0.9
    for i in range(n):
        row = np.zeros(2 * n)
        row[n + i] = 1.0
        rows.append(row)
        b.append(0.9)
    # exp cones: s = (u_i, 1, x_i) = b - A z
    for i in range(n):
        r_u = np.zeros(2 * n)
        r_u[i] = -1.0
        r_x = np.zeros(2 * n)
        r_x[n + i] = -1.0
        rows.extend([r_u, np.zeros(2 * n), r_x])
        b.extend([0.0, 1.0, 0.0])
    return {
        "P": None,
        "A": sp.csc_matrix(np.array(rows)),
        "b": np.array(b),
        "q": np.concatenate([-np.ones(n), np.zeros(n)]),
        "r": 0.0,
        "cone": {"z": 1, "l": n, "ep": n},
    }


def _power_cone_problem(alpha: float) -> dict:
    """max z s.t. (2, 8, z) in K_pow(alpha): optimum 2^alpha 8^(1-alpha)."""
    return {
        "P": None,
        "A": sp.csc_matrix(np.array([[0.0], [0.0], [-1.0]])),
        "b": np.array([2.0, 8.0, 0.0]),
        "q": np.array([-1.0]),
        "r": 0.0,
        "cone": {"p": [alpha]},
    }


@pytest.mark.parametrize("solver_name", ["clarabel", "scs", "ecos"])
def test_exponential_cone_orientation_agrees_across_solvers(solver_name, tmp_path):
    settings = {"eps_abs": 1e-9, "eps_rel": 1e-9} if solver_name == "scs" else {}
    result = _solve(solver_name, CONE, _exp_cone_problem(), settings, tmp_path)
    assert result.status == status.OPTIMAL, (result.status, result.info)
    assert result.objective_value == pytest.approx(3.0 * np.log(3.0), abs=1e-5)
    # The harness projects onto exponential cones, so a wrong orientation
    # would show up as a cone distance in the residuals.
    assert "unsupported_cones" not in result.kkt
    assert result.kkt["primal_res_rel"] < 1e-6
    assert result.kkt["dual_res_rel"] < 1e-6


@pytest.mark.parametrize("alpha", [0.25, 0.6])
@pytest.mark.parametrize("solver_name", ["clarabel", "scs"])
def test_power_cone_orientation_agrees_across_solvers(solver_name, alpha, tmp_path):
    settings = {"eps_abs": 1e-9, "eps_rel": 1e-9} if solver_name == "scs" else {}
    result = _solve(solver_name, CONE, _power_cone_problem(alpha), settings, tmp_path)
    assert result.status == status.OPTIMAL, (result.status, result.info)
    assert result.objective_value == pytest.approx(-(2.0**alpha) * 8.0 ** (1 - alpha), abs=1e-5)
    assert "unsupported_cones" not in result.kkt
    assert result.kkt["primal_res_rel"] < 1e-6
    assert result.kkt["dual_res_rel"] < 1e-6


@pytest.mark.parametrize(
    "cone, reason",
    [({"p": [-0.5]}, "power cones with 0 < alpha < 1"), ({"ed": 1}, "cone key 'ed'")],
)
def test_clarabel_skips_dual_exponential_and_dual_power_cones(cone, reason, tmp_path):
    problem = {
        "P": None,
        "A": sp.csc_matrix(np.array([[0.0], [0.0], [-1.0]])),
        "b": np.array([2.0, 8.0, 0.0]),
        "q": np.array([-1.0]),
        "r": 0.0,
        "cone": cone,
    }
    result = _solve("clarabel", CONE, problem, {}, tmp_path)
    assert result.status == status.SKIPPED_UNSUPPORTED
    assert reason in result.info["reason"]


def test_clarabel_exp_cone_after_psd_block_keeps_rows_aligned(tmp_path):
    """An exponential cone after a PSD block: the PSD rows are permuted to
    Clarabel's layout and back, the exponential rows must not move."""
    exp = _exp_cone_problem()
    # Prepend a 2x2 PSD block on two fresh variables w = (W00, W11) with
    # W = diag(w) and minimise trace(W) = 0 at W = 0.
    n_old = exp["A"].shape[1]
    a_old = exp["A"].toarray()
    z_row, l_rows, e_rows = a_old[:1], a_old[1:4], a_old[4:]
    pad = np.zeros((a_old.shape[0], 2))
    psd = np.zeros((3, n_old + 2))
    psd[0, n_old] = -1.0
    psd[2, n_old + 1] = -1.0
    a = np.vstack([
        np.hstack([z_row, pad[:1]]),
        np.hstack([l_rows, pad[1:4]]),
        psd,
        np.hstack([e_rows, pad[4:]]),
    ])
    b = np.concatenate([exp["b"][:4], np.zeros(3), exp["b"][4:]])
    problem = {
        "P": None,
        "A": sp.csc_matrix(a),
        "b": b,
        "q": np.concatenate([exp["q"], [1.0, 1.0]]),
        "r": 0.0,
        "cone": {"z": 1, "l": 3, "s": [2], "ep": 3},
    }
    result = _solve("clarabel", CONE, problem, {}, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(3.0 * np.log(3.0), abs=1e-5)
    assert result.kkt["primal_res_rel"] < 1e-6
    assert result.kkt["dual_res_rel"] < 1e-6


def test_clarabel_tolerates_solver_without_get_info(tmp_path, monkeypatch):
    """clarabel < 0.11 (allowed by the extra) has no DefaultSolver.get_info."""
    clarabel = pytest.importorskip("clarabel")
    real_solver = clarabel.DefaultSolver

    class OldSolver:
        def __init__(self, *args):
            self._inner = real_solver(*args)

        def solve(self):
            return self._inner.solve()

    monkeypatch.setattr(clarabel, "DefaultSolver", OldSolver)
    result = _solve("clarabel", CONE, _feasible_cone_lp(n=10, m=20), {}, tmp_path)
    assert result.status == status.OPTIMAL
    assert "factorizations" not in result.info


def test_clarabel_translates_eps_aliases(tmp_path):
    result = _solve(
        "clarabel", CONE, _feasible_cone_lp(n=10, m=20), {"eps_abs": 1e-5, "eps_rel": 1e-4},
        tmp_path,
    )
    assert result.status == status.OPTIMAL
    assert result.info["settings_translated"] == {
        "tol_feas": 1e-5,
        "tol_gap_abs": 1e-5,
        "tol_gap_rel": 1e-4,
    }


def test_clarabel_native_tolerances_win_over_eps():
    from solver_benchmarks.solvers.clarabel_adapter import translate_settings

    settings = {"eps": 1e-3, "tol_gap_rel": 1e-9}
    translated = translate_settings(settings)
    assert settings == {"tol_feas": 1e-3, "tol_gap_abs": 1e-3, "tol_gap_rel": 1e-9}
    assert translated == {"tol_feas": 1e-3, "tol_gap_abs": 1e-3}


def test_clarabel_still_rejects_unknown_settings(tmp_path):
    with pytest.raises(ValueError, match="tol_fesa"):
        _solve("clarabel", CONE, _feasible_cone_lp(n=10, m=20), {"tol_fesa": 1e-3}, tmp_path)


_CBF_EXP = """VER
1

OBJSENSE
MAX

VAR
2 1
F 2

CON
4 2
L+ 1
EXP 3

OBJACOORD
1
0 1.0

ACOORD
3
0 1 -1.0
1 1 1.0
3 0 1.0

BCOORD
2
0 2.0
2 1.0
"""


@pytest.mark.parametrize("solver_name", ["clarabel", "scs", "ecos"])
def test_cblib_exponential_cone_solves_alike(solver_name, tmp_path):
    """max u s.t. x1 <= 2 and (x1, 1, u) in CBF's EXP cone, i.e. exp(u) <= x1:
    read through the CBLIB reader (which converts CBF's EXP to the canonical
    layout), every solver must find u = log 2."""
    from solver_benchmarks.datasets.cblib import read_cbf_cone_problem

    path = tmp_path / "exp.cbf"
    path.write_text(_CBF_EXP)
    cone_problem, _ = read_cbf_cone_problem(path)
    assert cone_problem["cone"] == {"l": 1, "ep": 1}
    result = _solve(solver_name, CONE, cone_problem, {}, tmp_path)
    assert result.status == status.OPTIMAL, (result.status, result.info)
    assert result.objective_value == pytest.approx(-np.log(2.0), abs=1e-5)


def test_clarabel_psd_conversion_is_exact_on_sdplib_control1(tmp_path, repo_root):
    """SDPLIB control1 (blocks of order 10 and 5, optimum 17.78463).

    With its default chordal decomposition Clarabel 0.11 reports Solved at
    18.0455 here: its own residuals are ~1e-10, but the point it hands back
    misses the original constraints (unscaled primal 2.7e-4, dual 3.7e-2;
    the harness records dual_res_rel ~1.9e-2). The adapter's PSD conversion
    is not the cause: SDPA's optimum maps through it to a feasible primal-
    dual pair of the converted problem, and Clarabel without chordal
    decomposition solves the converted problem to the published optimum.
    Scoring such a point is left to the analysis (kkt-verify).
    """
    from conftest import sdplib_archive_dataset

    problem = sdplib_archive_dataset(repo_root, tmp_path / "data").load_problem("control1")
    adapter = _adapter("clarabel", {"chordal_decomposition_enable": False})
    result = adapter.solve(problem, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(17.78463, abs=1e-4)
    assert result.kkt["primal_res_rel"] < 1e-6
    assert result.kkt["dual_res_rel"] < 1e-6
    assert result.kkt["primal_cone_res"] < 1e-6


# ---------------------------------------------------------------------------
# ECOS
# ---------------------------------------------------------------------------


def _box_qp(p: sp.spmatrix, seed: int = 0) -> dict:
    n = p.shape[0]
    rng = np.random.default_rng(seed)
    return {
        "P": sp.csc_matrix(p),
        "q": rng.standard_normal(n),
        "A": sp.eye(n, format="csc"),
        "l": -np.ones(n),
        "u": np.ones(n),
    }


def test_ecos_large_diagonal_p_stays_sparse(tmp_path, monkeypatch):
    """A diagonal P gives R = diag(sqrt(p_ii)) without densifying P: the SOC
    block has one entry per variable however large n is."""
    import scipy.linalg

    from solver_benchmarks.solvers.ecos_adapter import _qp_to_ecos_via_socp

    def no_dense_factorization(*args, **kwargs):
        raise AssertionError("a diagonal P must not be factored densely")

    n = 6000
    qp = _box_qp(sp.diags(np.linspace(0.0, 2.0, n)))
    with monkeypatch.context() as patch:
        patch.setattr(scipy.linalg, "cholesky", no_dense_factorization)
        patch.setattr(scipy.linalg, "eigh", no_dense_factorization)
        data, dims, _, state = _qp_to_ecos_via_socp(qp)
    assert dims["q"] == [n - 1 + 2]  # one zero diagonal entry dropped
    soc_rows = data["G"][-(n - 1 + 2):, :]
    assert soc_rows.nnz == 2 + (n - 1)  # (u, v) rows on t plus one entry per kept x_i
    result = _solve("ecos", QP, qp, {}, tmp_path)
    reference = _solve("clarabel", QP, qp, {}, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(reference.objective_value, rel=1e-6)


def test_ecos_factors_only_the_block_of_p_with_entries(tmp_path):
    from solver_benchmarks.solvers.ecos_adapter import _psd_square_root

    n = 5000
    block = np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    p = sp.lil_matrix((n, n))
    idx = [10, 2000, 4999]
    for a, i in enumerate(idx):
        for b, j in enumerate(idx):
            p[i, j] = block[a, b]
    r = _psd_square_root(sp.csc_matrix(p))
    assert r.shape == (3, n)
    assert set(np.unique(r.nonzero()[1])) <= set(idx)
    assert np.allclose((r.T @ r).toarray()[np.ix_(idx, idx)], block)
    result = _solve("ecos", QP, _box_qp(p), {}, tmp_path)
    reference = _solve("clarabel", QP, _box_qp(p), {}, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(reference.objective_value, rel=1e-6)


def test_ecos_refuses_a_large_dense_factor_up_front(tmp_path, monkeypatch):
    """A P whose off-diagonal entries couple more than ECOS_DENSE_FACTOR_MAX_DIM
    rows into one block is skipped with a reason instead of densified
    (path-graph Laplacian, n = 2500), before anything is factored."""
    import scipy.linalg

    from solver_benchmarks.solvers.ecos_adapter import ECOS_DENSE_FACTOR_MAX_DIM

    def no_dense_factorization(*args, **kwargs):
        raise AssertionError("an oversized block must be refused before factoring")

    monkeypatch.setattr(scipy.linalg, "cholesky", no_dense_factorization)
    monkeypatch.setattr(scipy.linalg, "eigh", no_dense_factorization)
    n = ECOS_DENSE_FACTOR_MAX_DIM + 500
    lap = sp.diags([-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
    result = _solve("ecos", QP, _box_qp(lap), {}, tmp_path)
    assert result.status == status.SKIPPED_UNSUPPORTED
    reason = result.info["reason"]
    assert str(ECOS_DENSE_FACTOR_MAX_DIM) in reason
    assert f"coupling {n} rows/columns in 1 connected block(s), the largest {n} x {n}" in reason


def _diagonal_plus_blocks(n: int, block_starts: list[int], block_size: int, seed: int = 0):
    """``diag(1..2)`` with dense SPD blocks (off-diagonal coupling) placed on
    the given leading indices; every other row has only its diagonal."""
    rng = np.random.default_rng(seed)
    p = sp.lil_matrix(sp.diags(np.linspace(1.0, 2.0, n)))
    for start in block_starts:
        g = rng.standard_normal((block_size, block_size))
        idx = np.arange(start, start + block_size)
        p[np.ix_(idx, idx)] = g.T @ g / block_size + np.eye(block_size)
    return sp.csc_matrix(p)


def test_ecos_factors_only_the_off_diagonal_block_of_a_mostly_diagonal_p(
    tmp_path, monkeypatch
):
    """Rows whose only entry is the diagonal take sqrt(p_ii) like a diagonal
    P; only the rows coupled by off-diagonal entries are factored densely.
    Maros-Meszaros MOSARQP1 (n = 2500, coupled block 10) and STCQP1/2
    (n = 4097, coupled block 1534) were refused while every row with an
    entry counted towards the dense limit."""
    import scipy.linalg

    from solver_benchmarks.solvers.ecos_adapter import (
        ECOS_DENSE_FACTOR_MAX_DIM,
        _psd_square_root,
        _qp_to_ecos_via_socp,
    )

    n = 3000
    assert n > ECOS_DENSE_FACTOR_MAX_DIM
    p = _diagonal_plus_blocks(n, [1234], 10)
    qp = _box_qp(p, seed=3)
    factored_shapes: list[tuple[int, ...]] = []
    real_cholesky = scipy.linalg.cholesky

    def recording_cholesky(a, *args, **kwargs):
        factored_shapes.append(np.shape(a))
        return real_cholesky(a, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(scipy.linalg, "cholesky", recording_cholesky)
        data, dims, _, state = _qp_to_ecos_via_socp(qp)
    assert factored_shapes == [(10, 10)]
    r = _psd_square_root(sp.csc_matrix(p))
    assert r.shape == (n, n)
    assert abs(r.T @ r - p).max() < 1e-12
    # n - 10 diagonal rows plus a 10 x 10 upper-triangular Cholesky factor.
    assert r.nnz == (n - 10) + 10 * 11 // 2
    result = _solve("ecos", QP, qp, {}, tmp_path)
    reference = _solve("clarabel", QP, qp, {}, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(reference.objective_value, rel=1e-6)


def test_ecos_factors_each_coupled_block_separately():
    """Separate off-diagonal blocks are factored one at a time, so R is block
    diagonal; the limit is on the entries of all dense factors together."""
    from solver_benchmarks.solvers.ecos_adapter import (
        ECOS_DENSE_FACTOR_MAX_DIM,
        _psd_square_root,
        _SOCPReformulationError,
    )

    n = 400
    starts = [0, 100, 250]
    p = _diagonal_plus_blocks(n, starts, 40)
    r = _psd_square_root(p)
    assert abs(r.T @ r - p).max() < 1e-12
    assert r.nnz == (n - 3 * 40) + 3 * (40 * 41 // 2)
    rows, cols = r.nonzero()
    for start in starts:
        block = (cols >= start) & (cols < start + 40)
        # A row touching one block touches no other column.
        assert set(cols[np.isin(rows, rows[block])]) <= set(range(start, start + 40))

    # Three blocks of 0.6 * limit each are fine one at a time but hold more
    # than limit^2 entries together.
    size = int(0.6 * ECOS_DENSE_FACTOR_MAX_DIM)
    big = sp.block_diag([sp.diags([-np.ones(size - 1), 2.0 * np.ones(size), -np.ones(size - 1)],
                                  [-1, 0, 1])] * 3, format="csc")
    with pytest.raises(_SOCPReformulationError, match="3 connected block"):
        _psd_square_root(big)


def test_ecos_rejects_indefinite_coupled_block_in_a_large_p():
    from solver_benchmarks.solvers.ecos_adapter import _psd_square_root

    n = 3000
    p = sp.lil_matrix(sp.eye(n))
    p[[5, 5, 6, 6], [5, 6, 5, 6]] = [1.0, 2.0, 2.0, 1.0]  # eigenvalues 3 and -1
    assert _psd_square_root(sp.csc_matrix(p)) is None


def test_ecos_times_the_socp_reformulation(tmp_path):
    result = _solve("ecos", QP, _feasible_qp(), {}, tmp_path)
    assert result.status == status.OPTIMAL
    reformulation = result.info["socp_reformulation_seconds"]
    assert reformulation > 0.0
    assert result.run_time_seconds >= reformulation


def test_ecos_translates_shared_aliases(tmp_path):
    result = _solve(
        "ecos", CONE, _feasible_cone_lp(), {"max_iter": 2, "eps_abs": 1e-6, "eps_rel": 1e-5},
        tmp_path,
    )
    assert result.status == status.MAX_ITER_REACHED
    assert result.iterations == 2
    assert result.info["settings_translated"] == {
        "max_iters": 2,
        "feastol": 1e-6,
        "abstol": 1e-6,
        "reltol": 1e-5,
    }
    assert result.info["raw_status"] == "Maximum number of iterations reached"


def test_ecos_native_settings_win_over_aliases():
    from solver_benchmarks.solvers.ecos_adapter import translate_settings

    settings = {"eps": 1e-4, "abstol": 1e-9, "max_iter": 5, "max_iters": 50}
    translated = translate_settings(settings)
    assert settings == {"abstol": 1e-9, "max_iters": 50, "feastol": 1e-4, "reltol": 1e-4}
    assert translated == {"feastol": 1e-4, "reltol": 1e-4}


def test_ecos_rejects_unknown_settings_by_name(tmp_path):
    from solver_benchmarks.solvers.base import SolverConfigError

    # ECOS's own TypeError for an unknown keyword is a configuration error.
    with pytest.raises(SolverConfigError, match="feastoll"):
        _solve("ecos", CONE, _feasible_cone_lp(), {"feastoll": 1e-6}, tmp_path)


# ---------------------------------------------------------------------------
# CVXOPT
# ---------------------------------------------------------------------------


def _unbounded_cone_lp() -> dict:
    # min -x s.t. x >= 0.
    return {
        "P": None,
        "A": sp.csc_matrix([[-1.0]]),
        "b": np.array([0.0]),
        "q": np.array([-1.0]),
        "r": 0.0,
        "cone": {"l": 1},
    }


def _infeasible_sdp() -> dict:
    # X in S^2_+ with X00 = -1: x = (X00, sqrt2 X10, X11), -x + s = 0.
    return {
        "P": None,
        "A": sp.csc_matrix(np.vstack([[1.0, 0.0, 0.0], -np.eye(3)])),
        "b": np.array([-1.0, 0.0, 0.0, 0.0]),
        "q": np.array([1.0, 0.0, 1.0]),
        "r": 0.0,
        "cone": {"z": 1, "s": [2]},
    }


def _hs52() -> dict:
    """Maros-Meszaros HS52: equality constraints only, singular P."""
    p = np.array(
        [
            [32.0, -8.0, 0.0, 0.0, 0.0],
            [-8.0, 4.0, 2.0, 0.0, 0.0],
            [0.0, 2.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0],
        ]
    )
    a = np.array(
        [[1.0, 3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, -2.0], [0.0, 1.0, 0.0, 0.0, -1.0]]
    )
    return {
        "P": sp.csc_matrix(p),
        "q": np.array([0.0, -4.0, -4.0, -2.0, -2.0]),
        "A": sp.csc_matrix(np.vstack([a, np.eye(5)])),
        "l": np.concatenate([np.zeros(3), -np.inf * np.ones(5)]),
        "u": np.concatenate([np.zeros(3), np.inf * np.ones(5)]),
    }


@pytest.mark.parametrize(
    "kind, data",
    [(CONE, _infeasible_cone_lp()), (QP, _infeasible_qp_form_lp()), (CONE, _infeasible_sdp())],
    ids=["cone-lp", "qp-form-lp", "sdp"],
)
def test_cvxopt_certifies_primal_infeasibility(kind, data, tmp_path):
    """Linear-objective problems go through conelp, which certifies
    infeasibility; coneqp raised 'domain error' (solver_error) or ran to the
    iteration cap on these."""
    result = _solve("cvxopt", kind, data, {}, tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE, (result.status, result.info)
    assert result.info["entry_point"] == "conelp"
    assert result.kkt["certificate"] == "primal_infeasible"
    assert result.kkt["valid"] is True


@pytest.mark.parametrize(
    "kind, data",
    [(CONE, _unbounded_cone_lp()),
     (QP, {"P": sp.csc_matrix((1, 1)), "q": np.array([-1.0]), "A": sp.csc_matrix([[1.0]]),
           "l": np.array([0.0]), "u": np.array([np.inf])})],
    ids=["cone", "qp-form"],
)
def test_cvxopt_certifies_dual_infeasibility(kind, data, tmp_path):
    result = _solve("cvxopt", kind, data, {}, tmp_path)
    assert result.status == status.DUAL_INFEASIBLE, (result.status, result.info)
    assert result.kkt["certificate"] == "dual_infeasible"
    assert result.kkt["qtx"] < 0


@pytest.mark.parametrize("settings", [{}, {"kktsolver": "qr"}], ids=["default", "qr"])
def test_cvxopt_discards_an_infeasibility_claim_at_the_starting_point(settings, tmp_path):
    """min -1e8 x s.t. 0 <= x <= 1: conelp's starting point x = 0.5 has
    objective -5e7, so its test (residual / -q'x <= feastol) calls the problem
    dual infeasible at iteration 0 (as for NETLIB agg / grow7). The adapter
    discards that claim and solves with coneqp (which has no 'qr' KKT
    solver, so a conelp-only choice falls back to coneqp's default)."""
    lp = {
        "P": None,
        "A": sp.csc_matrix([[-1.0], [1.0]]),
        "b": np.array([0.0, 1.0]),
        "q": np.array([-1e8]),
        "r": 0.0,
        "cone": {"l": 2},
    }
    result = _solve("cvxopt", CONE, lp, settings, tmp_path)
    assert result.info["conelp_discarded_claim"] == {
        "status": "dual infeasible",
        "iterations": 0,
    }
    assert result.info["entry_point"] == "coneqp"
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(-1e8, rel=1e-7)


def test_cvxopt_keeps_infeasibility_claims_after_the_starting_point(tmp_path):
    result = _solve("cvxopt", CONE, _infeasible_cone_lp(), {}, tmp_path)
    assert result.status == status.PRIMAL_INFEASIBLE
    assert result.iterations > 0
    assert "conelp_discarded_claim" not in result.info
    # coneqp, which cannot certify infeasibility, did not solve it either.
    assert result.info["coneqp_cross_check"]["status"] not in {
        status.OPTIMAL,
        status.OPTIMAL_INACCURATE,
    }


def _false_claim_after_iteration_0(monkeypatch):
    """Make conelp claim dual infeasibility at iteration 3 on any problem, as
    it does on badly scaled feasible NETLIB LPs (scagr25, gfrd-pnc, nesm)."""
    import cvxopt.solvers as cs

    real_conelp = cs.conelp

    def conelp(*args, **kwargs):
        raw = dict(real_conelp(*args, **kwargs))
        raw.update({"status": "dual infeasible", "iterations": 3})
        return raw

    monkeypatch.setattr(cs, "conelp", conelp)
    return cs


def test_cvxopt_overturns_a_later_claim_when_coneqp_solves_the_problem(monkeypatch, tmp_path):
    pytest.importorskip("cvxopt")
    _false_claim_after_iteration_0(monkeypatch)
    lp = {"P": None, "A": sp.csc_matrix([[-1.0], [1.0]]), "b": np.array([0.0, 1.0]),
          "q": np.array([-1.0]), "r": 0.0, "cone": {"l": 2}}
    result = _solve("cvxopt", CONE, lp, {}, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(-1.0, abs=1e-6)
    assert result.info["entry_point"] == "coneqp"
    assert result.info["conelp_discarded_claim"] == {"status": "dual infeasible", "iterations": 3}
    assert result.info["coneqp_cross_check"]["status"] == status.OPTIMAL


def test_cvxopt_keeps_a_later_claim_when_coneqp_fails(monkeypatch, tmp_path):
    pytest.importorskip("cvxopt")
    cs = _false_claim_after_iteration_0(monkeypatch)

    def coneqp(*args, **kwargs):
        raise ArithmeticError("domain error")

    monkeypatch.setattr(cs, "coneqp", coneqp)
    lp = {"P": None, "A": sp.csc_matrix([[-1.0], [1.0]]), "b": np.array([0.0, 1.0]),
          "q": np.array([-1.0]), "r": 0.0, "cone": {"l": 2}}
    result = _solve("cvxopt", CONE, lp, {}, tmp_path)
    assert result.status == status.DUAL_INFEASIBLE
    assert result.info["entry_point"] == "conelp"
    assert "conelp_discarded_claim" not in result.info
    assert result.info["coneqp_cross_check"]["status"] == status.SOLVER_ERROR


@pytest.mark.parametrize("settings, expected", [({}, "chol"), ({"kktsolver": "qr"}, "qr")])
def test_cvxopt_conelp_uses_the_sparse_kkt_solver_for_soc(settings, expected, tmp_path):
    """conelp defaults to the dense 'qr' KKT solver once there are second-
    order cones (an error after 490 s on DIMACS sched_50_50_scaled, 80 s with
    'chol'); the adapter passes coneqp's 'chol' unless kktsolver is set."""
    pytest.importorskip("cvxopt")
    import cvxopt.solvers as cs

    seen = []
    real_conelp = cs.conelp

    def conelp(*args, **kwargs):
        seen.append(kwargs.get("kktsolver"))
        return real_conelp(*args, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(cs, "conelp", conelp):
        # min t s.t. ||(1, 1)|| <= t  ->  t = sqrt(2)
        soc = {"P": None, "A": sp.csc_matrix([[-1.0], [0.0], [0.0]]), "b": np.array([0.0, 1.0, 1.0]),
               "q": np.array([1.0]), "r": 0.0, "cone": {"q": [3]}}
        result = _solve("cvxopt", CONE, soc, settings, tmp_path)
    assert seen == [expected]
    assert result.info["kktsolver"] == expected
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_cvxopt_conelp_keeps_qr_for_psd_only_problems(tmp_path):
    """With 'chol', conelp fails on SDPLIB control1 and hinf9, which its
    default 'qr' solves, so a PSD-only problem keeps conelp's default."""
    pytest.importorskip("cvxopt")
    # min X00 + 2 X11 s.t. trace(X) = 1, X in S^2_+ (canonical col-major
    # lower vec x = [X00, sqrt2 X10, X11]); optimum X = diag(1, 0), value 1.
    sdp = {
        "P": None,
        "A": sp.csc_matrix(np.vstack([np.array([[1.0, 0.0, 1.0]]), -np.eye(3)])),
        "b": np.array([1.0, 0.0, 0.0, 0.0]),
        "q": np.array([1.0, 0.0, 2.0]),
        "r": 0.0,
        "cone": {"z": 1, "s": [2]},
    }
    result = _solve("cvxopt", CONE, sdp, {}, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(1.0, abs=1e-6)
    assert result.info["entry_point"] == "conelp"
    assert "kktsolver" not in result.info


def test_ecos_adapter_imports_its_linear_algebra_at_module_load():
    """The runner starts a fresh worker per solve, so a first-use import
    inside the timed QP reformulation was charged to every ECOS solve
    (0.15 s on HS53, more than the solve)."""
    import subprocess
    import sys

    code = ("import sys; import solver_benchmarks.solvers.ecos_adapter; "
            "print('scipy.linalg' in sys.modules and 'scipy.sparse.csgraph' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"


def test_cvxopt_quadratic_objective_still_uses_coneqp(tmp_path):
    result = _solve("cvxopt", QP, _feasible_qp(), {}, tmp_path)
    assert result.status == status.OPTIMAL
    assert result.info["entry_point"] == "coneqp"


def test_cvxopt_equality_only_qp_is_solved_correctly(tmp_path):
    """No cone rows: coneqp solves the KKT system once and always says
    'optimal'. With CVXOPT's default chol2 factorization and a singular P
    that point violates Ax = b (objective 2.62 instead of 5.33 for HS52)."""
    result = _solve("cvxopt", QP, _hs52(), {}, tmp_path)
    reference = _solve("clarabel", QP, _hs52(), {}, tmp_path)
    assert reference.status == status.OPTIMAL
    assert result.status == status.OPTIMAL, result.info
    assert result.info["kktsolver"] == "ldl"
    assert result.objective_value == pytest.approx(reference.objective_value, abs=1e-6)
    assert result.kkt["primal_res_rel"] < 1e-9
    assert result.kkt["dual_res_rel"] < 1e-9


def _spy_on_coneqp(monkeypatch) -> dict:
    cs = pytest.importorskip("cvxopt.solvers")
    seen: dict = {}
    real_coneqp = cs.coneqp

    def spy(*args, **kwargs):
        seen["kktsolver"] = kwargs.get("kktsolver")
        return real_coneqp(*args, **kwargs)

    monkeypatch.setattr(cs, "coneqp", spy)
    return seen


def test_cvxopt_large_equality_only_qp_keeps_the_sparse_kkt_solver(tmp_path, monkeypatch):
    """ldl factors the KKT matrix [[P, A'], [A, 0]] densely (7.3 GB for
    Maros-Meszaros AUG2DC, n + p = 30200), so above CVXOPT_DENSE_LDL_MAX_DIM
    the adapter leaves CVXOPT's sparse default chol2 in place."""
    from solver_benchmarks.solvers.cvxopt_adapter import CVXOPT_DENSE_LDL_MAX_DIM

    n, m = 1800, 400
    assert n + m > CVXOPT_DENSE_LDL_MAX_DIM
    rng = np.random.default_rng(0)
    a = sp.random(m, n, density=0.01, random_state=1, format="csc") + sp.eye(m, n)
    b = a @ rng.standard_normal(n)
    qp = {
        "P": sp.diags(np.linspace(1.0, 2.0, n), format="csc"),
        "q": rng.standard_normal(n),
        "A": sp.csc_matrix(a),
        "l": b,
        "u": b,
    }
    seen = _spy_on_coneqp(monkeypatch)
    result = _solve("cvxopt", QP, qp, {}, tmp_path)
    assert seen["kktsolver"] is None
    assert "kktsolver" not in result.info
    reference = _solve("clarabel", QP, qp, {}, tmp_path)
    assert result.status == status.OPTIMAL, result.info
    assert result.objective_value == pytest.approx(reference.objective_value, rel=1e-7)


def test_cvxopt_equality_only_qp_at_the_limit_uses_ldl(tmp_path, monkeypatch):
    import solver_benchmarks.solvers.cvxopt_adapter as cvxopt_adapter

    seen = _spy_on_coneqp(monkeypatch)
    monkeypatch.setattr(cvxopt_adapter, "CVXOPT_DENSE_LDL_MAX_DIM", 8)  # HS52: n + p = 5 + 3
    result = _solve("cvxopt", QP, _hs52(), {}, tmp_path)
    assert seen["kktsolver"] == "ldl"
    assert result.status == status.OPTIMAL


def test_cvxopt_wrong_chol2_point_on_equality_only_qp_is_a_solver_error(tmp_path, monkeypatch):
    """Above the ldl limit a singular-P equality-only problem keeps chol2,
    whose 'optimal' point violates Ax = b; the residual check reports it as
    solver_error instead of optimal."""
    import solver_benchmarks.solvers.cvxopt_adapter as cvxopt_adapter

    seen = _spy_on_coneqp(monkeypatch)
    monkeypatch.setattr(cvxopt_adapter, "CVXOPT_DENSE_LDL_MAX_DIM", 7)  # HS52: n + p = 8
    result = _solve("cvxopt", QP, _hs52(), {}, tmp_path)
    assert seen["kktsolver"] is None
    assert result.info["raw_status"] == "optimal"
    assert result.status == status.SOLVER_ERROR, result.info
    assert result.objective_value is None


def test_cvxopt_rejects_unknown_settings_by_name(tmp_path):
    with pytest.raises(ValueError, match="tolerance"):
        _solve("cvxopt", CONE, _feasible_cone_lp(), {"tolerance": 1.0}, tmp_path)


def test_cvxopt_translates_shared_aliases(tmp_path):
    result = _solve(
        "cvxopt", CONE, _feasible_cone_lp(n=10, m=20),
        {"eps_abs": 1e-6, "eps_rel": 1e-5, "max_iter": 50},
        tmp_path,
    )
    assert result.status == status.OPTIMAL
    assert result.info["settings_translated"] == {
        "maxiters": 50,
        "abstol": 1e-6,
        "feastol": 1e-6,
        "reltol": 1e-5,
    }


def test_cvxopt_native_options_win_over_aliases():
    from solver_benchmarks.solvers.cvxopt_adapter import _resolve_options

    options, translated = _resolve_options({"eps": 1e-4, "feastol": 1e-9, "max_iter": 7})
    assert options == {"feastol": 1e-9, "maxiters": 7, "abstol": 1e-4, "reltol": 1e-4}
    assert translated == {"maxiters": 7, "abstol": 1e-4, "reltol": 1e-4}


def test_cvxopt_forwards_kktsolver_to_the_solver_call(tmp_path, monkeypatch):
    cs = pytest.importorskip("cvxopt.solvers")
    seen = {}
    real_conelp = cs.conelp

    def spy(*args, **kwargs):
        seen["kktsolver"] = kwargs.get("kktsolver")
        return real_conelp(*args, **kwargs)

    monkeypatch.setattr(cs, "conelp", spy)
    result = _solve("cvxopt", CONE, _feasible_cone_lp(n=10, m=20), {"kktsolver": "ldl"}, tmp_path)
    assert result.status == status.OPTIMAL
    assert seen["kktsolver"] == "ldl"


# ---------------------------------------------------------------------------
# SDPA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data, certificate",
    [(_infeasible_cone_lp(), "primal_infeasible"), (_infeasible_sdp(), "primal_infeasible"),
     (_unbounded_cone_lp(), "dual_infeasible")],
    ids=["lp", "sdp", "unbounded-lp"],
)
def test_sdpa_records_certificate_checks(data, certificate, tmp_path):
    """SDPA's infeasibility claims were recorded with kkt=None, so they were
    never checked; the y / x it returns are checked like SCS's and
    Clarabel's certificates."""
    result = _solve("sdpa", CONE, data, {}, tmp_path)
    assert result.status in status.ANY_INFEASIBLE, (result.status, result.info)
    assert result.kkt is not None
    assert result.kkt["certificate"] == certificate
    assert result.kkt["valid"] is True


def _sdpa_map(phase, *, primal=0.0, dual=0.0, gap=0.0, iteration=10, options=None, tol=1e-6):
    from solver_benchmarks.solvers.sdpa_adapter import _map_sdpa_status

    info = {"phasevalue": phase, "primalError": primal, "dualError": dual, "dualityGap": gap}
    return _map_sdpa_status(info, {"iteration": iteration}, options or {"maxIteration": 100}, tol)


def test_sdpa_missing_errors_are_unknown_not_failures():
    """sdpa-python sets primalError/dualError to None when its ARPACK check
    does not converge (PSD blocks of order ~40 and up): a converged pdOPT
    solve must not become optimal_inaccurate because of it."""
    assert _sdpa_map("pdOPT", primal=None, dual=None, gap=3.9e-8) == status.OPTIMAL
    assert _sdpa_map("pdOPT", primal=None, dual=1e-3, gap=3.9e-8) == status.OPTIMAL_INACCURATE
    assert _sdpa_map("pdOPT", primal=None, dual=None, gap=1e-2) == status.OPTIMAL_INACCURATE


def test_sdpa_default_tolerance_follows_requested_accuracy():
    from solver_benchmarks.solvers.sdpa_adapter import _default_optimality_tolerance

    assert _default_optimality_tolerance({}) == 1e-6
    assert _default_optimality_tolerance({"epsilonStar": 1e-8, "epsilonDash": 1e-8}) == 1e-6
    tol = _default_optimality_tolerance({"epsilonStar": 1e-4, "epsilonDash": 1e-4})
    assert tol == 1e-4
    assert _sdpa_map("pdOPT", gap=8.6e-6, tol=tol) == status.OPTIMAL


def test_sdpa_feasible_but_unconverged_at_the_cap_is_max_iter():
    assert _sdpa_map("pdFEAS", gap=1e-3, iteration=100) == status.MAX_ITER_REACHED
    assert _sdpa_map("pdFEAS", gap=1e-9, iteration=40) == status.OPTIMAL
    assert _sdpa_map("noINFO", iteration=100) == status.MAX_ITER_REACHED
    assert _sdpa_map("noINFO", iteration=40) == status.SOLVER_ERROR


def test_sdpa_options_defaults_and_translation(tmp_path):
    from solver_benchmarks.solvers.sdpa_adapter import (
        SDPA_LOWER_BOUND,
        SDPA_UPPER_BOUND,
        _sdpa_options,
    )

    info: dict = {}
    options = _sdpa_options({"threads": 2, "eps": 1e-5, "max_iter": 30}, tmp_path, info=info)
    assert options["numThreads"] == 2
    assert options["lowerBound"] == SDPA_LOWER_BOUND == -1e30
    assert options["upperBound"] == SDPA_UPPER_BOUND == 1e30
    assert options["epsilonStar"] == options["epsilonDash"] == 1e-5
    assert options["maxIteration"] == 30
    assert info["settings_translated"] == {
        "maxIteration": 30,
        "epsilonStar": 1e-5,
        "epsilonDash": 1e-5,
        "numThreads": 2,
    }
    assert "threads_ignored" not in info


@pytest.mark.parametrize("key, good, bad", [
    ("convMethod", "EQ", "eq"),
    ("domainMethod", "clique", "cliques"),
    ("frvMethod", "elimination", True),
    ("rangeMethod", "decomp", "none "),
])
def test_sdpa_method_parameters_take_only_values_sdpa_python_accepts(tmp_path, key, good, bad):
    # sdpa-python replaces any other value by its default with a printed note.
    from solver_benchmarks.solvers.base import SolverConfigError
    from solver_benchmarks.solvers.sdpa_adapter import _sdpa_options

    assert _sdpa_options({key: good}, tmp_path)[key] == good
    with pytest.raises(SolverConfigError, match=key):
        _sdpa_options({key: bad}, tmp_path)


def test_sdpa_defaults_to_one_thread_and_native_settings_win(tmp_path):
    from solver_benchmarks.solvers.sdpa_adapter import _sdpa_options

    options = _sdpa_options({}, tmp_path)
    assert options["numThreads"] == 1
    # sdpa-python resets an int-valued float parameter to its default,
    # so the adapter coerces it.
    options = _sdpa_options(
        {"threads": 4, "numThreads": 2, "lowerBound": -1000, "eps_abs": 1e-3,
         "epsilonStar": 1e-9},
        tmp_path,
    )
    assert options["numThreads"] == 2
    assert options["lowerBound"] == -1000.0 and isinstance(options["lowerBound"], float)
    assert options["epsilonStar"] == 1e-9


def test_sdpa_rejects_unknown_settings_by_name(tmp_path):
    with pytest.raises(ValueError, match="epsilonStarr"):
        _solve("sdpa", CONE, _feasible_cone_lp(n=10, m=20), {"epsilonStarr": 1e-3}, tmp_path)


def test_sdpa_solves_a_feasible_problem_with_a_large_objective(tmp_path, repo_root):
    """DIMACS hamming_7_5_6 (optimum 42.67) with q scaled by 1e4: with
    sdpa-python's default objective bounds (+-1e5) SDPA stopped with dUNBD
    and the adapter reported primal_infeasible."""
    if not (repo_root / "problem_classes" / "dimacs_data" / "hamming_7_5_6.mat.gz").exists():
        pytest.skip("DIMACS hamming_7_5_6 not present")
    from solver_benchmarks.datasets import get_dataset

    base = get_dataset("dimacs")().load_problem("hamming_7_5_6")
    data = dict(base.cone)
    data["q"] = np.asarray(data["q"]) * 1e4
    result = _solve("sdpa", CONE, data, {}, tmp_path)
    assert result.status == status.OPTIMAL, (result.status, result.info["phasevalue"])
    assert result.objective_value == pytest.approx(426666.67, rel=1e-6)


# ---------------------------------------------------------------------------
# Clarabel / ECOS: an "almost" verdict at the iteration limit is the limit
# ---------------------------------------------------------------------------


def _random_box_qp(seed: int = 0, n: int = 40, m: int = 30, lp: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, n))
    p = np.zeros((n, n)) if lp else g @ g.T / n + 0.1 * np.eye(n)
    a = np.vstack([rng.standard_normal((m, n)), np.eye(n)])
    x0 = rng.standard_normal(n)
    return {
        "P": sp.csc_matrix(p),
        "q": rng.standard_normal(n),
        "A": sp.csc_matrix(a),
        "l": np.concatenate([a[:m] @ x0 - rng.random(m), -5.0 * np.ones(n)]),
        "u": np.concatenate([a[:m] @ x0 + rng.random(m), 5.0 * np.ones(n)]),
    }


# Clarabel and ECOS relabel a stop at max_iter as AlmostSolved / "Close to
# optimal" when the last iterate meets their reduced tolerances.
_ALMOST = {"clarabel": "AlmostSolved", "ecos": "Close to optimal solution found"}


@pytest.mark.parametrize("lp", [False, True], ids=["qp", "lp"])
@pytest.mark.parametrize("solver_name", ["clarabel", "ecos"])
def test_almost_solved_at_the_iteration_cap_is_max_iter_reached(solver_name, lp, tmp_path):
    # Was optimal_inaccurate with an objective (Clarabel at max_iter 8-10 of
    # 11 needed, ECOS at max_iters 14-15 of 16), unlike SCS, QTQP, CVXOPT,
    # OSQP and PIQP at the same position.
    qp = _random_box_qp(lp=lp)
    needed = _solve(solver_name, QP, qp, {}, tmp_path).iterations
    almost_seen = 0
    for cap in range(max(1, needed - 4), needed):
        result = _solve(solver_name, QP, qp, {"max_iter": cap}, tmp_path)
        assert result.status != status.OPTIMAL_INACCURATE, (cap, result.info["raw_status"])
        if result.info["raw_status"] == _ALMOST[solver_name]:
            almost_seen += 1
            assert result.status == status.MAX_ITER_REACHED
            assert result.objective_value is None
            _assert_finite_solution_kkt(result)
    assert almost_seen, "no cap below convergence gave an 'almost' verdict"


def test_almost_verdicts_before_the_limit_stay_inaccurate():
    from solver_benchmarks.solvers.clarabel_adapter import _map_clarabel_status
    from solver_benchmarks.solvers.ecos_adapter import _map_ecos_status

    # Insufficient progress or numerical trouble: not a limit.
    assert _map_clarabel_status("AlmostSolved") == status.OPTIMAL_INACCURATE
    assert (
        _map_clarabel_status("AlmostPrimalInfeasible", hit_iteration_limit=True)
        == status.MAX_ITER_REACHED
    )
    assert _map_clarabel_status("AlmostSolved", hit_time_limit=True) == status.TIME_LIMIT
    assert _map_clarabel_status("Solved", hit_iteration_limit=True) == status.OPTIMAL
    assert _map_ecos_status({"exitFlag": 10, "iter": 12}, max_iters=100) == status.OPTIMAL_INACCURATE
    assert _map_ecos_status({"exitFlag": 11, "iter": 100}, max_iters=100) == status.MAX_ITER_REACHED
    assert _map_ecos_status({"exitFlag": 0, "iter": 100}, max_iters=100) == status.OPTIMAL


def test_ecos_close_to_optimal_nan_iterate_at_zero_iterations(tmp_path, monkeypatch):
    # ECOS at max_iters 0 can answer "Close to optimal" at iteration 0 with an
    # uninitialised (NaN) iterate; it was optimal_inaccurate with a KKT record
    # of NaNs. ECOS's output there varies, so the answer is fixed here.
    ecos = pytest.importorskip("ecos")
    real_solve = ecos.solve

    def nan_close_to_optimal(*args, **kwargs):
        raw = real_solve(*args, **kwargs)
        raw = dict(raw)
        raw["x"] = np.full_like(raw["x"], np.nan)
        raw["info"] = dict(raw["info"], exitFlag=10, iter=0,
                           infostring="Close to optimal solution found")
        return raw

    monkeypatch.setattr(ecos, "solve", nan_close_to_optimal)
    result = _solve("ecos", CONE, _feasible_cone_lp(n=5, m=10), {"max_iters": 0}, tmp_path)
    assert result.status == status.MAX_ITER_REACHED
    assert result.objective_value is None
    assert result.kkt is None


@pytest.mark.parametrize("settings", [
    {"tol_feas": -1.0},
    {"tol_feas": 0.0},
    {"tol_gap_abs": 0.0, "tol_gap_rel": 0.0},
    {"tol_infeas_abs": float("nan")},
    {"eps_abs": -1.0},
    {"eps": -1e-6},
])
def test_clarabel_rejects_tolerances_it_cannot_stop_on(settings, tmp_path):
    # Clarabel accepted these and never met its stopping test (AlmostSolved
    # or InsufficientProgress on every problem).
    from solver_benchmarks.solvers.base import SolverConfigError

    with pytest.raises(SolverConfigError, match="tol_"):
        _solve("clarabel", CONE, _feasible_cone_lp(), settings, tmp_path)


@pytest.mark.parametrize("settings", [
    {"eps_abs": 1e-6, "eps_rel": 0.0},
    {"tol_gap_abs": 0.0},
    {"tol_gap_rel": 0.0},
    {"tol_infeas_abs": 0.0},
])
def test_clarabel_accepts_a_single_zero_gap_or_infeasibility_tolerance(settings, tmp_path):
    # The gap test is absolute OR relative: one zero tolerance is a valid
    # absolute-only or relative-only test (the shared eps_rel: 0 shape).
    result = _solve("clarabel", CONE, _feasible_cone_lp(), settings, tmp_path)
    assert result.status == status.OPTIMAL


def test_clarabel_backend_missing_on_this_host_is_unavailable(tmp_path):
    # "Bad settings: Problem with mkl solver (not available)": a valid backend
    # whose library is missing is a broken environment, not a typo that
    # would stop the solver.
    from solver_benchmarks.solvers.base import SolverUnavailable

    try:
        result = _solve("clarabel", CONE, _feasible_cone_lp(), {"direct_solve_method": "mkl"}, tmp_path)
    except SolverUnavailable as exc:
        assert "not available" in str(exc)
    else:
        pytest.skip(f"MKL loads on this host ({result.status})")


def test_clarabel_enum_setting_typo_is_a_config_error(tmp_path):
    # Clarabel raised a bare Exception("Bad settings: ...") from DefaultSolver.
    from solver_benchmarks.solvers.base import SolverConfigError

    with pytest.raises(SolverConfigError, match="direct_solve_method"):
        _solve("clarabel", CONE, _feasible_cone_lp(), {"direct_solve_method": "qdldll"}, tmp_path)


def _integer_cone_lp() -> dict:
    # min x0 + x1 s.t. x0 + x1 >= 1, x >= 0 with an integer-typed A (scipy keeps
    # the dtype of an integer array).
    return {
        "P": None,
        "A": sp.csc_matrix(np.array([[-1, -1], [-1, 0], [0, -1]])),
        "b": np.array([-1.0, 0.0, 0.0]),
        "q": np.array([1.0, 1.0]),
        "r": 0.0,
        "cone": {"l": 3},
    }


@pytest.mark.parametrize("solver_name", ["scs", "ecos", "clarabel", "cvxopt", "cpo3"])
def test_integer_typed_cone_data_solves(solver_name, tmp_path):
    # SCS and ECOS refused int64 A with a TypeError the branch reported as
    # SolverConfigError "rejected the settings ['verbose']".
    result = _solve(solver_name, CONE, _integer_cone_lp(), {}, tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(1.0, abs=1e-5)


def test_scs_integer_typed_qp_hessian_solves(tmp_path):
    qp = _random_box_qp()
    qp["P"] = sp.csc_matrix(np.diag(np.arange(1, 41)))
    assert qp["P"].dtype.kind == "i"
    reference = _solve("clarabel", QP, qp, {}, tmp_path)
    result = _solve("scs", QP, qp, {"eps_abs": 1e-8, "eps_rel": 1e-8}, tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(reference.objective_value, rel=1e-5)


def test_ecos_zero_dimensional_second_order_cone_is_ignored(tmp_path):
    # ECOS refused q: [0] ("dims['q'] ought to be a list of positive integers").
    problem = _integer_cone_lp()
    problem["cone"] = {"l": 3, "q": [0]}
    result = _solve("ecos", CONE, problem, {}, tmp_path)
    assert result.status == status.OPTIMAL


def _soc_and_psd_cone_lp(cone: dict) -> dict:
    # Variables (t, a, b, c, d, X00, X10, X11): min t + c + X00 + 2 X11 with
    # X00 + X11 = 1, a >= 1, d >= 1, (t, a, b) and (c, d) second-order cones
    # and X PSD (col-major lower triangle). Optimum 3.
    a = np.zeros((11, 8))
    a[0, [5, 7]] = 1.0
    a[1, 1] = a[2, 4] = -1.0
    a[3:6, 0:3] = -np.eye(3)
    a[6:8, 3:5] = -np.eye(2)
    a[8:11, 5:8] = -np.eye(3)
    return {
        "P": None,
        "A": sp.csc_matrix(a),
        "b": np.array([1.0, -1.0, -1.0] + [0.0] * 8),
        "q": np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0]),
        "r": 0.0,
        "cone": cone,
    }


def test_cvxopt_zero_dimensional_cones_are_ignored(tmp_path):
    # CVXOPT refuses a second-order cone of dimension 0; such a cone, and an
    # empty PSD cone, has no rows.
    expected = _solve(
        "cvxopt", CONE, _soc_and_psd_cone_lp({"z": 1, "l": 2, "q": [3, 2], "s": [2]}), {}, tmp_path / "a"
    )
    result = _solve(
        "cvxopt", CONE, _soc_and_psd_cone_lp({"z": 1, "l": 2, "q": [3, 0, 2], "s": [0, 2]}), {}, tmp_path / "b"
    )
    assert expected.status == status.OPTIMAL
    assert expected.objective_value == pytest.approx(3.0, abs=1e-6)
    assert result.status == expected.status
    assert result.objective_value == pytest.approx(expected.objective_value, rel=1e-9)
    assert result.iterations == expected.iterations


def test_scs_accepts_a_cone_given_as_key_value_pairs(tmp_path):
    expected = _solve("scs", CONE, _integer_cone_lp(), {}, tmp_path / "dict")
    problem = _integer_cone_lp()
    problem["cone"] = [("l", 3)]
    result = _solve("scs", CONE, problem, {}, tmp_path / "pairs")
    assert expected.status == status.OPTIMAL
    assert result.status == expected.status
    assert result.objective_value == pytest.approx(expected.objective_value, rel=1e-9)
    assert result.iterations == expected.iterations


def test_scs_non_numeric_power_cone_exponent_is_a_solver_error(tmp_path):
    # SCS's TypeError ("must be real number, not str") read like a setting's.
    problem = _feasible_cone_lp(n=5, m=10)
    problem["A"] = sp.vstack([problem["A"], sp.csc_matrix((3, 5))], format="csc")
    problem["b"] = np.concatenate([problem["b"], [1.0, 1.0, 0.0]])
    problem["cone"] = {"l": 10, "p": ["a"]}
    result = _solve("scs", CONE, problem, {}, tmp_path)
    assert result.status == status.SOLVER_ERROR
    assert "exponent" in result.info["error"]


@pytest.mark.parametrize("solver_name, module_name", [("scs", "scs"), ("ecos", "ecos")])
def test_solver_type_error_about_data_is_a_solver_error(solver_name, module_name, tmp_path, monkeypatch):
    module = pytest.importorskip(module_name)

    def refuse(*args, **kwargs):
        raise TypeError("Ax must be a 1-D numpy array of floats")

    monkeypatch.setattr(module, "solve", refuse)
    result = _solve(solver_name, CONE, _feasible_cone_lp(n=5, m=10), {}, tmp_path)
    assert result.status == status.SOLVER_ERROR
    assert "Ax must be" in result.info["error"]


@pytest.mark.parametrize("solver_name, settings", [
    ("scs", {"acceleration_interval": "often"}),
    ("ecos", {"reltol": "tight"}),
    ("ecos", {"nitref": -1}),
])
def test_solver_type_error_about_a_setting_is_a_config_error(solver_name, settings, tmp_path):
    from solver_benchmarks.solvers.base import SolverConfigError

    with pytest.raises(SolverConfigError):
        _solve(solver_name, CONE, _feasible_cone_lp(n=5, m=10), settings, tmp_path)


def test_sdpa_problem_without_rows_is_a_solver_error(tmp_path):
    # sdpa-python's "Inconsistent Size" escaped as a worker_error.
    cone = {"P": None, "A": sp.csc_matrix((0, 2)), "b": np.zeros(0), "q": np.ones(2),
            "r": 0.0, "cone": {"l": 0}}
    result = _solve("sdpa", CONE, cone, {}, tmp_path)
    assert result.status == status.SOLVER_ERROR
    assert "Inconsistent Size" in result.info["error"]
    assert result.objective_value is None and result.kkt is None


@pytest.mark.parametrize("settings", [
    {"maxIteration": "many"},
    {"eps_abs": -1.0},
    {"optimality_tolerance": 0.0},
])
def test_sdpa_rejects_bad_values(settings, tmp_path):
    # sdpa-python resets a value of the wrong type to its default.
    from solver_benchmarks.solvers.base import SolverConfigError

    with pytest.raises(SolverConfigError):
        _solve("sdpa", CONE, _feasible_cone_lp(n=5, m=10), settings, tmp_path)


def test_sdpa_integer_parameters_follow_the_integer_rule(tmp_path):
    # numThreads -1 ends the process inside sdpa-python; numpy integers and
    # integral floats are the same count (master accepted them).
    from solver_benchmarks.solvers.base import SolverConfigError
    from solver_benchmarks.solvers.sdpa_adapter import _sdpa_options

    assert _sdpa_options({"maxIteration": np.int64(30)}, tmp_path)["maxIteration"] == 30
    assert _sdpa_options({"maxIteration": 30.0}, tmp_path)["maxIteration"] == 30
    # numThreads 0 works in sdpa-python (as on master); only negatives end it.
    assert _sdpa_options({"numThreads": 0}, tmp_path)["numThreads"] == 0
    for settings in ({"numThreads": -1}, {"maxIteration": 2.5}):
        with pytest.raises(SolverConfigError):
            _sdpa_options(settings, tmp_path)


def test_sdpa_threads_zero_leaves_sdpa_python_default(tmp_path):
    from solver_benchmarks.solvers.sdpa_adapter import _sdpa_options

    assert _sdpa_options({}, tmp_path)["numThreads"] == 1
    assert "numThreads" not in _sdpa_options({"threads": 0}, tmp_path)
    assert _sdpa_options({"threads": 3}, tmp_path)["numThreads"] == 3
