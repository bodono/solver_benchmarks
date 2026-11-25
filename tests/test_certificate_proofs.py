"""Infeasibility certificates must be proofs, or confirmed by the known outcome.

A small scaled residual (``kkt.cert_res``) is not a proof: a feasible problem
whose feasible points are all large has certificates with a residual as small
as one over their size (MIPLIB mas76: 2.57e-5, accepted at 1e-3 before this
check). These tests pin both sides of ``kkt.cert_proof`` (true certificates
pass, false ones fail, for every certificate type) and of the verification
rule built on it (``verify.classify`` with the problem's known outcome).

Most false certificates here miss being proofs by many orders of magnitude,
so those tests assert verdicts and signs, not tight numbers; the tests of the
rounding allowances and of residuals zero only to rounding sit at the
boundary on purpose.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.analysis.derive import kkt_verify
from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.analysis.verify import VerifyPolicy, classify
from solver_benchmarks.core import status

INF = np.inf


def _primal(A, l, u, y):
    return kkt.qp_primal_infeasibility_cert(sp.csr_matrix(np.asarray(A, dtype=float)), np.asarray(l, dtype=float),
                                            np.asarray(u, dtype=float), np.asarray(y, dtype=float))


def _dual(P, q, A, l, u, x):
    A = sp.csr_matrix(np.asarray(A, dtype=float))
    P = sp.csr_matrix((A.shape[1], A.shape[1])) if P is None else sp.csr_matrix(np.asarray(P, dtype=float))
    return kkt.qp_dual_infeasibility_cert(P, np.asarray(q, dtype=float), A, np.asarray(l, dtype=float),
                                          np.asarray(u, dtype=float), np.asarray(x, dtype=float))


# --- mas76: a bound of 1e12 and feasible points of size 4e4 -------------------


def _mas76_like(upper=1e12):
    """x0 in [0, upper] (the objective variable of MIPLIB mas76, bounded by
    1e12 there) and two binaries; x0 + x1 >= 4e4 and x0 + x2 >= 5e4. Every
    feasible point has x0 >= 4.9999e4."""
    A = [[1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    l = [4e4, 5e4, 0, 0, 0]
    u = [INF, INF, upper, 1, 1]
    return A, l, u


def test_a_false_certificate_with_a_small_residual_is_not_a_proof():
    # The best certificate the scaled residual can be shown: weight only on
    # the second row, support -1, residual 2e-5 on x0 and x2 (mas76's
    # single-precision SCS certificates are this one, at 2.57e-5).
    A, l, u = _mas76_like()
    record = _primal(A, l, u, [0.0, -1 / 5e4, 0.0, 0.0, 0.0])
    assert record["cert_res"] == pytest.approx(2e-5)
    assert record["valid"] is True  # what the scaled residual alone accepted
    assert record["cert_proof"] is False
    assert record["cert_proof_open"] == 0  # x0 has its bound; 1e12 is far too loose to pay with
    assert record["cert_proof_gap"] < -1e6


def test_the_same_problem_made_infeasible_has_a_proof():
    # x0 <= 3e4 contradicts x0 + x2 >= 5e4 with x2 <= 1: weights -1 on that
    # row and 1 on both bounds give A'y = 0 and support 3e4 + 1 - 5e4 < 0.
    A, l, u = _mas76_like(upper=3e4)
    record = _primal(A, l, u, [0.0, -1.0, 1.0, 0.0, 1.0])
    assert record["cert_proof"] is True
    assert record["cert_proof_gap"] > 0
    # The residual of the false certificate is now paid for by the bound.
    assert _primal(A, l, u, [0.0, -1 / 5e4, 0.0, 0.0, 0.0])["cert_proof"] is True


@pytest.mark.parametrize("column_scale, row_scale", [(1e5, 1.0), (1e-5, 1.0), (1.0, 1e-5), (1.0, 1e5)])
def test_the_proof_does_not_depend_on_units(column_scale, row_scale):
    """Expressing x0 in other units, or scaling the first row, changes the
    scaled residual by the same factor (and so its verdict at any tolerance)
    but not the proof's."""
    A, l, u = _mas76_like()
    y = np.array([0.0, -1 / 5e4, 0.0, 0.0, 0.0])
    A = np.asarray(A, dtype=float)
    A[:, 0] *= column_scale  # x0 = x0' * column_scale
    A[0] *= row_scale
    lower = np.array(l, dtype=float)
    lower[0] *= row_scale
    upper = np.array(u, dtype=float)
    scaled = _primal(A, lower, upper, y)
    assert scaled["cert_res"] == pytest.approx(2e-5 * max(column_scale, 1.0))
    assert scaled["cert_proof"] is False
    A_inf, l_inf, u_inf = _mas76_like(upper=3e4)
    A_inf = np.asarray(A_inf, dtype=float)
    A_inf[:, 0] *= column_scale
    A_inf[0] *= row_scale
    l_inf = np.asarray(l_inf, dtype=float)
    l_inf[0] *= row_scale
    assert _primal(A_inf, l_inf, u_inf, [0.0, -1.0, 1.0, 0.0, 1.0])["cert_proof"] is True


def test_scaling_a_feasible_problem_makes_the_scaled_residual_accept_but_not_the_proof():
    # x1 + x2 >= 1 with free x is feasible; its best certificate has residual
    # 1. In units of 1e-5 (x = 1e5 x') the same problem's is 1e-5.
    y = [-1.0]
    assert _primal([[1, 1]], [1], [INF], y)["cert_res"] == pytest.approx(1.0)
    small = _primal([[1e-5, 1e-5]], [1], [INF], y)
    assert small["cert_res"] == pytest.approx(1e-5) and small["valid"] is True
    assert small["cert_proof"] is False
    assert small["cert_proof_open"] == 2  # free variables: nothing pays for the residual


# --- other false certificates -------------------------------------------------


def test_a_huestis_like_certificate_on_variables_unbounded_above_is_not_a_proof():
    # sum x >= 1e8 with x >= 0 is feasible; y = -1e-8 on that row has support
    # -1 and residual 1e-8 on every column, which only lower bounds have.
    A = [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    record = _primal(A, [1e8, 0, 0, 0], [INF, INF, INF, INF], [-1e-8, 0, 0, 0])
    assert record["valid"] is True
    assert record["cert_proof"] is False
    assert record["cert_proof_open"] == 3


def test_a_powell20_like_nearly_cancelling_pair_of_rows_is_not_a_proof():
    # x1 - x2 >= 1 and -x1 + (1 + 1e-6) x2 >= 1: feasible (x2 >= 2e6), and
    # y = (-1, -1) leaves 1e-6 on x2 against a support of -2.
    record = _primal([[1, -1], [-1, 1 + 1e-6]], [1, 1], [INF, INF], [-1, -1])
    assert record["cert_res"] < 1e-6 and record["valid"] is True
    assert record["cert_proof"] is False


def test_a_wrong_signed_or_zero_certificate_is_not_a_proof():
    A, l, u = _mas76_like(upper=3e4)
    assert _primal(A, l, u, [0.0, 1.0, -1.0, 0.0, -1.0])["cert_proof"] is False
    assert _primal(A, l, u, [0.0, 0.0, 0.0, 0.0, 0.0])["cert_proof"] is False
    assert _primal(A, l, u, [0.0, np.nan, 1.0, 0.0, 1.0])["cert_proof"] is False


# --- true certificates ----------------------------------------------------------


def test_an_exact_certificate_with_free_variables_is_a_proof():
    # x1 + x2 <= 1 and x1 + x2 >= 2, x free.
    record = _primal([[1, 1], [1, 1]], [-INF, 2], [1, INF], [1, -1])
    assert record["cert_proof"] is True
    assert record["cert_proof_open"] == 0


def test_noise_on_bounded_variables_is_paid_for_by_their_bounds():
    # x1 + x2 >= 3 with x in [0, 1]^2: weights 1 on the upper bounds, noise of
    # 1e-12 on them (as a non-simplex solver leaves).
    A = [[1, 1], [1, 0], [0, 1]]
    l, u = [3, 0, 0], [INF, 1, 1]
    assert _primal(A, l, u, [-1, 1 + 1e-12, 1 - 1e-12])["cert_proof"] is True
    # With upper bounds only, noise that leaves r_1 < 0 is paid for by x1 <= 1;
    # noise that leaves r_1 > 0 needs a lower bound x1 does not have.
    upper_only = [3, -INF, -INF]
    assert _primal(A, upper_only, u, [-1, 1, 1])["cert_proof"] is True
    assert _primal(A, upper_only, u, [-1, 1 - 1e-12, 1])["cert_proof"] is True
    no_bound = _primal(A, upper_only, u, [-1, 1 + 1e-12, 1])
    assert no_bound["cert_proof"] is False and no_bound["cert_proof_open"] == 1


def _feasible(A, l, u, x):
    """Whether ``x`` satisfies ``l <= A x <= u`` exactly, in rationals."""
    for row, low, high in zip(A, l, u):
        value = sum(Fraction(a) * Fraction(v) for a, v in zip(row, x))
        if not (low == -INF or value >= Fraction(low)) or not (high == INF or value <= Fraction(high)):
            return False
    return True


# --- residuals that are zero only to rounding ------------------------------------


def test_a_residual_zero_only_to_rounding_is_not_taken_as_zero():
    """A certificate whose entries span more than 2^53 leaves a residual that
    is exactly nonzero but below the rounding error of evaluating it (about
    1e4 here), wherever rows imply an equality. Taking such a residual as
    zero accepted these false certificates on feasible problems with integer
    data; deciding its sign exactly rejects them."""
    # a + b <= 0 and -a - b <= 0 (together a + b = 0), a - b <= -1: feasible.
    A = [[1, 1], [-1, -1], [1, -1]]
    l, u = [-INF] * 3, [0, 0, -1]
    assert _feasible(A, l, u, [-0.5, 0.5])
    y = [1e20, 1e20, 1]  # A'y = (1, -1) exactly; the support is -1
    record = _primal(A, l, u, y)
    assert record["cert_proof"] is False and record["cert_proof_open"] == 2
    cone = kkt.cone_primal_infeasibility_cert(sp.csr_matrix(np.array(A, dtype=float)), np.array(u, dtype=float),
                                              {"l": 3}, np.array(y))
    assert cone["cert_proof"] is False and cone["cert_proof_open"] == 2
    # min -x3 with x1 - x2 + x3 <= 0 and x1 - x2 = 0: bounded (x3 <= 0), so
    # no direction proves it unbounded; (1e20, 1e20, 1) leaves A x = (1, 0).
    dual = _dual(None, [0, 0, -1], [[1, -1, 1], [1, -1, 0]], [-INF, 0], [0, 0], [1e20, 1e20, 1])
    assert dual["cert_proof"] is False and dual["cert_proof_open"] == 1
    # x - y >= 1 and -(1 + 2^-52) x + y >= 1 are feasible (only for x <= -2^53);
    # y = (-1, -1) leaves one unit in the last place on x. Likewise 0.3
    # against 0.1 + 0.2.
    near = [[1, -1], [-(1 + 2.0**-52), 1]]
    assert _feasible(near, [1, 1], [INF, INF], [-(2**54), -(2**54) - 1])
    assert _primal(near, [1, 1], [INF, INF], [-1, -1])["cert_proof"] is False
    decimal = [[0.3, -1], [-(0.1 + 0.2), 1]]
    assert _feasible(decimal, [1, 1], [INF, INF], [-(2**60), Fraction(0.3) * -(2**60) - 1])
    assert _primal(decimal, [1, 1], [INF, INF], [-1, -1])["cert_proof"] is False


def test_an_exact_certificate_with_a_wide_range_of_entries_is_a_proof():
    """The other side: exact cancellation that floating point cannot confirm
    (the residual's rounding error is about 1e4) is a proof."""
    # The problem above with a - b >= 0 added: infeasible. A'y = 0 exactly.
    A = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    record = _primal(A, [-INF] * 4, [0, 0, -1, 0], [1e20, 1e20, 1, 1])
    assert record["cert_proof"] is True and record["cert_proof_open"] == 0
    # With a >= 1 as well, a's residual is exactly zero, not merely within
    # its interval of about +-1e4 (which, times the bound, would lose the proof).
    record = _primal(A + [[1, 0]], [-INF] * 4 + [1], [0, 0, -1, 0, INF], [1e20, 1e20, 1, 1, 0])
    assert record["cert_proof"] is True
    # min -x3 with x1 - x2 <= 0 and -x1 + x2 <= 0, x3 free: unbounded.
    dual = _dual(None, [0, 0, -1], [[1, -1, 0], [-1, 1, 0]], [-INF, -INF], [0, 0], [1e20, 1e20, 1])
    assert dual["cert_proof"] is True


def test_exact_signs_match_rational_arithmetic(monkeypatch):
    """kkt._exact_signs against Fractions, on sums that cancel across the
    whole exponent range (subnormal to 1e300), in both orientations and with
    sums split across chunks."""
    rng = np.random.default_rng(0)

    def rational_signs(M, v, transpose):
        rows = M.toarray().T if transpose else M.toarray()
        totals = [sum((Fraction(float(a)) * Fraction(float(b)) for a, b in zip(row, v)), Fraction(0)) for row in rows]
        return [(total > 0) - (total < 0) for total in totals]

    for chunk in (2**16, 3):
        monkeypatch.setattr(kkt, "_EXACT_CHUNK", chunk)
        for trial in range(150):
            m, n = rng.integers(1, 9, 2)
            scale = rng.choice([5e-324, 1e-300, 1e-20, 1.0, 1e20, 1e300], size=(m, n))
            data = rng.integers(-3, 4, (m, n)) * scale * rng.choice([1.0, 0.1, 1 / 3], size=(m, n))
            data[rng.random((m, n)) < 0.3] = 0.0
            # Duplicated rows and columns make exact cancellation common.
            data[-1] = data[0]
            data[:, -1] = -data[:, 0]
            M = sp.csr_matrix(data)
            transpose = bool(trial % 2)
            v = rng.integers(-3, 4, m if transpose else n) * rng.choice([1e-200, 1.0, 1e16, 1e200], m if transpose else n)
            wanted = np.flatnonzero(rng.random(n if transpose else m) < 0.8)
            expected = np.array(rational_signs(M, v, transpose))[wanted]
            assert list(kkt._exact_signs(M, v, wanted, transpose=transpose)) == list(expected)


# --- the proof's own soundness conditions ------------------------------------------


def test_the_box_contains_every_feasible_value():
    # 3x <= 1 bounds x by 1/3, which rounds down (fl(1/3) < 1/3); 10x >= 1
    # by 1/10, which rounds up. The box must still contain them.
    lo, hi = kkt._singleton_box(sp.csr_matrix([[3.0], [10.0]]), np.array([-INF, 1.0]), np.array([1.0, INF]))
    assert Fraction(hi[0]) >= Fraction(1, 3) and Fraction(lo[0]) <= Fraction(1, 10)
    lo, hi = kkt._singleton_box(sp.csr_matrix([[-3.0], [-10.0]]), np.array([-1.0, -INF]), np.array([INF, -1.0]))
    assert Fraction(hi[0]) >= Fraction(1, 3) and Fraction(lo[0]) <= Fraction(1, 10)


def test_a_contradiction_within_its_rounding_error_is_not_a_proof():
    """The margin L - sigma must exceed the rounding error of sigma and of L,
    and be positive: each case here is a proof for any of those allowances
    dropped (or for a margin of zero accepted)."""
    A = sp.csr_matrix([[1.0]])
    lo, hi = np.array([1.0]), np.array([INF])

    def proof(y_hat, sigma, sigma_error):
        return kkt._primal_proof(A, np.array([y_hat]), sigma, sigma_error, lo, hi)["cert_proof"]

    # r = 0: L is zero, less its allowance of (n + 2) subnormals.
    least = -3 * np.finfo(float).smallest_subnormal
    assert proof(0.0, np.nextafter(least, -INF), 0.0) is True
    assert proof(0.0, least, 0.0) is False  # a margin of exactly zero
    assert proof(0.0, -1.0, 1.0) is False  # within sigma's rounding error
    # r = 1 on x >= 1: L = 1 less r's and L's rounding errors (about 1.1e-15).
    assert proof(1.0, 1.0 - 2e-15, 0.0) is True
    assert proof(1.0, 1.0 - 8e-16, 0.0) is False  # within L's rounding error


def test_weight_the_data_does_not_allow_is_not_used():
    # x >= 0 and x >= 1 with no upper bounds (feasible): y = (1, 0) puts its
    # weight on the absent upper bound of the first row, which the support
    # cannot use; read as a bound of 0 it would contradict x >= 1.
    assert _primal([[1], [1]], [0, 1], [INF, INF], [1, 0])["cert_proof"] is False
    # Cone form: x + z <= 5, x <= 1, z <= 1 (feasible), y = (-1, 0, 0) is not
    # in the dual cone; read as it is, it gives -x - z <= -5 against x, z <= 1.
    A = sp.csr_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    record = kkt.cone_primal_infeasibility_cert(A, np.array([5.0, 1.0, 1.0]), {"l": 3}, np.array([-1.0, 0.0, 0.0]))
    assert record["cert_proof"] is False


def test_bounds_of_1e18_or_more_count_for_the_dual_proof_and_not_for_the_primal_one():
    """The residuals read bounds of kkt._INF_BOUND (1e18) or more as absent;
    a solver may hold a row to one (HiGHS keeps bounds below 1e20). Each
    proof takes the reading that is safe for it."""
    # min -x with x <= 5e18 is bounded: the direction 1 is not a proof.
    assert _dual(None, [-1], [[1]], [-INF], [5e18], [1])["cert_proof"] is False
    assert _dual(None, [-1], [[1]], [-INF], [INF], [1])["cert_proof"] is True
    # x <= 5e18 and x >= 6e18 is infeasible if those are bounds and feasible if
    # they mean infinity: the primal proof does not use them.
    assert _primal([[1], [1]], [-INF, 6e18], [5e18, INF], [1, -1])["cert_proof"] is False
    assert _primal([[1], [1]], [-INF, 6e17], [5e17, INF], [1, -1])["cert_proof"] is True


# --- dual (unboundedness) certificates ------------------------------------------


def test_a_ray_of_an_unbounded_lp_is_a_proof():
    # min -x1 - x2 with x1 - x2 <= 1 and x >= 0: ray (1, 1).
    record = _dual(None, [-1, -1], [[1, -1], [1, 0], [0, 1]], [-INF, 0, 0], [1, INF, INF], [1, 1])
    assert record["cert_proof"] is True


def test_a_direction_of_a_bounded_lp_with_large_multipliers_is_not_a_proof():
    # min -x with 1e-8 x <= 1: bounded (x <= 1e8, multiplier 1e8). d = 1 has
    # q'd = -1 and violates the row's recession cone by 1e-8.
    record = _dual(None, [-1], [[1e-8]], [-INF], [1], [1])
    assert record["valid"] is True
    assert record["cert_proof"] is False and record["cert_proof_open"] == 1


def test_a_dual_certificate_with_curvature_left_is_not_a_proof():
    # min x1^2 / 2 - x2 with x2 <= 1: bounded; d = (1e-9, 1) has Pd = 1e-9.
    record = _dual([[1, 0], [0, 0]], [0, -1], [[0, 1]], [-INF], [1], [1e-9, 1])
    assert record["cert_proof"] is False
    unbounded = _dual([[1, 0], [0, 0]], [0, -1], [[0, 1]], [0], [INF], [0, 1])
    assert unbounded["cert_proof"] is True
    # min x1^2 / 2 - x1 with x2 >= 0: bounded; d = (1, 0) meets the row's
    # recession cone and has q'd = -1, and only its curvature Pd = (1, 0)
    # rules it out.
    curved = _dual([[1, 0], [0, 0]], [-1, 0], [[0, 1]], [0], [INF], [1, 0])
    assert curved["cert_proof"] is False and curved["cert_proof_open"] == 1


def test_a_dual_certificate_with_a_nonnegative_objective_is_not_a_proof():
    record = _dual(None, [1, 1], [[1, -1], [1, 0], [0, 1]], [-INF, 0, 0], [1, INF, INF], [1, 1])
    assert record["cert_proof"] is False


# --- cone form ----------------------------------------------------------------


def test_cone_form_certificates_on_linear_cones_are_proved():
    # x >= 2 and x <= 1 as A x + s = b, s >= 0.
    A = sp.csr_matrix([[-1.0], [1.0]])
    b = np.array([-2.0, 1.0])
    assert kkt.cone_primal_infeasibility_cert(A, b, {"l": 2}, np.array([1.0, 1.0]))["cert_proof"] is True
    assert kkt.cone_primal_infeasibility_cert(A, b, {"l": 2}, np.array([-1.0, 1.0]))["cert_proof"] is False
    # A zero-cone row states x = 3, against x <= 1.
    assert kkt.cone_primal_infeasibility_cert(
        A, np.array([-3.0, 1.0]), {"z": 1, "l": 1}, np.array([1.0, 1.0])
    )["cert_proof"] is True
    # min -x with x >= 0: -A d = d in K for d = 1.
    record = kkt.cone_dual_infeasibility_cert(sp.csr_matrix((1, 1)), np.array([-1.0]), sp.csr_matrix([[-1.0]]),
                                              {"l": 1}, np.array([1.0]))
    assert record["cert_proof"] is True
    record = kkt.cone_dual_infeasibility_cert(sp.csr_matrix((1, 1)), np.array([-1.0]), sp.csr_matrix([[-1.0]]),
                                              {"z": 1}, np.array([1.0]))
    assert record["cert_proof"] is False


def test_cone_form_false_certificate_on_large_feasible_points_is_not_a_proof():
    # x >= 1e8 (A x + s = b with A = -1, b = -1e8) and x >= 0... feasible;
    # y = 1e-8 on the first row: b'y = -1, A'y = -1e-8 with no upper bound.
    A = sp.csr_matrix([[-1.0], [-1.0]])
    record = kkt.cone_primal_infeasibility_cert(A, np.array([-1e8, 0.0]), {"l": 2}, np.array([1e-8, 0.0]))
    assert record["valid"] is True
    assert record["cert_proof"] is False


@pytest.mark.parametrize("cone, rows", [({"l": 1, "q": [3]}, 4), ({"s": [2]}, 3), ({"l": 1, "ep": 1}, 4),
                                        ({"l": 1, "p": [0.5]}, 4), ({"bu": [1.0, 1.0], "bl": [0.0, 0.0]}, 3)])
def test_cones_the_proof_does_not_cover_give_no_verdict(cone, rows):
    A = sp.csr_matrix(np.ones((rows, 1)))
    record = kkt.cone_primal_infeasibility_cert(A, np.ones(rows), cone, np.ones(rows))
    assert record["cert_proof"] is None and record["cert_proof_cones"]
    record = kkt.cone_dual_infeasibility_cert(sp.csr_matrix((1, 1)), np.array([-1.0]), A, cone, np.ones(1))
    assert record["cert_proof"] is None and record["cert_proof_cones"]


def test_empty_nonlinear_cones_do_not_stop_the_proof():
    A = sp.csr_matrix([[-1.0], [1.0]])
    cone = {"l": 2, "q": [], "s": [], "ep": 0, "p": []}
    record = kkt.cone_primal_infeasibility_cert(A, np.array([-2.0, 1.0]), cone, np.array([1.0, 1.0]))
    assert record["cert_proof"] is True


# --- the verification rule ----------------------------------------------------


def _claim(kind, cert_res, proof, expected=None, reported=None, **extra):
    kkt_record = {"certificate": kind, "cert_res": cert_res, "cert_tol": 1e-3, "valid": cert_res <= 1e-3,
                  **({"cert_proof": proof} if proof != "absent" else {}), **extra}
    view = {"status": reported or kind, "kkt": kkt_record, "expected_status": expected}
    return classify(view, tolerance=lambda: 1e-3)


P, D = status.PRIMAL_INFEASIBLE, status.DUAL_INFEASIBLE


@pytest.mark.parametrize("expected, kind, proof, verdict", [
    # Ruled out by the known outcome, whatever the residual or proof.
    (status.EXPECT_OPTIMAL, P, False, "demoted"),
    (status.EXPECT_OPTIMAL, D, False, "demoted"),
    (status.EXPECT_OPTIMAL, P, True, "demoted"),
    (status.EXPECT_UNBOUNDED, P, False, "demoted"),
    (status.EXPECT_UNBOUNDED, P, True, "demoted"),
    # Confirmed kinds: the residual decides.
    (status.EXPECT_INFEASIBLE, P, False, "kept"),
    (status.EXPECT_UNBOUNDED, D, False, "kept"),
    (status.EXPECT_PRIMAL_AND_DUAL_INFEASIBLE, P, False, "kept"),
    (status.EXPECT_PRIMAL_AND_DUAL_INFEASIBLE, D, False, "kept"),
    # A dual claim on a problem known only to be primal infeasible: a proof only.
    (status.EXPECT_INFEASIBLE, D, False, "demoted"),
    (status.EXPECT_INFEASIBLE, D, True, "kept"),
    # No known outcome: a proof only.
    (None, P, False, "demoted"),
    (None, D, False, "demoted"),
    (None, P, True, "kept"),
    (None, D, True, "kept"),
])
def test_a_certificate_is_accepted_only_when_confirmed(expected, kind, proof, verdict):
    # cert_res 1e-11: false certificates reach that (QPLIB 9002).
    judged = _claim(kind, 1e-11, proof, expected)
    assert judged.verdict == verdict, judged.reason
    assert judged.status == (kind if verdict == "kept" else f"{kind}_inaccurate")


def test_a_confirmed_kind_still_needs_its_residual_within_the_tolerance():
    assert _claim(P, 5e-3, False, status.EXPECT_INFEASIBLE).verdict == "demoted"
    # A proof needs no residual test: the bounds paid for it.
    assert _claim(P, 5e-3, True, status.EXPECT_INFEASIBLE).verdict == "kept"


def test_a_certificate_of_the_other_kind_does_not_support_a_claim():
    judged = _claim(D, 1e-11, True, status.EXPECT_PRIMAL_AND_DUAL_INFEASIBLE, reported=P)
    assert (judged.verdict, judged.status) == ("demoted", status.PRIMAL_INFEASIBLE_INACCURATE)
    assert "holds a dual_infeasible certificate" in judged.reason


def test_a_proof_the_known_outcome_rules_out_is_reported():
    judged = _claim(P, 1e-11, True, status.EXPECT_OPTIMAL)
    assert judged.verdict == "demoted"
    assert "rules out" in judged.reason and "label or the problem's loader is wrong" in judged.reason


def test_a_record_without_a_proof_needs_one_only_where_no_outcome_confirms_it():
    # Archived records have no cert_proof: kept where the key confirms the kind,
    # unverifiable (demoted unless missing residuals are tolerated) elsewhere.
    assert _claim(P, 1e-9, "absent", status.EXPECT_INFEASIBLE).verdict == "kept"
    missing = _claim(P, 1e-9, "absent")
    assert (missing.verdict, missing.status) == ("missing_residuals", "primal_infeasible_inaccurate")
    assert "holds no proof" in missing.reason
    view = {"status": P, "kkt": {"certificate": P, "cert_res": 1e-9, "valid": True}}
    tolerated = classify(view, tolerance=lambda: 1e-3, missing_is_failure=False)
    assert (tolerated.verdict, tolerated.status) == ("missing_residuals", P)


@pytest.mark.parametrize("kind, expected", [(P, None), (D, None), (D, status.EXPECT_INFEASIBLE)])
@pytest.mark.parametrize("cert_res", [0.5, INF])
def test_a_record_without_a_proof_whose_residual_fails_is_demoted_even_when_missing_is_tolerated(
    kind, expected, cert_res
):
    """Tolerating unverifiable claims keeps a certificate whose proof is
    unknown, not one whose residual is known to fail (the rule before proofs
    existed demoted it too)."""
    record = {"certificate": kind, "cert_res": None if cert_res == INF else cert_res, "valid": False}
    view = {"status": kind, "kkt": record, "expected_status": expected}
    for missing_is_failure in (True, False):
        judged = classify(view, tolerance=lambda: 1e-3, missing_is_failure=missing_is_failure)
        assert (judged.verdict, judged.status) == ("demoted", f"{kind}_inaccurate"), judged.reason
        assert "certificate residual" in judged.reason
        assert ("objective term is not negative" in judged.reason) == (cert_res == INF)


def test_the_reason_for_a_dual_proof_has_no_gap():
    assert _claim(D, 1e-11, True).reason == "the certificate is a proof"
    assert "proof gap 0.5" in _claim(P, 1e-11, True, cert_proof_gap=0.5).reason


def test_a_certificate_on_cones_the_proof_does_not_cover_needs_a_known_outcome():
    judged = _claim(P, 1e-9, None, None, cert_proof_cones=["s"])
    assert judged.verdict == "demoted" and "cannot prove" in judged.reason
    assert _claim(P, 1e-9, None, status.EXPECT_INFEASIBLE, cert_proof_cones=["s"]).verdict == "kept"


@pytest.mark.parametrize("expected, proof, verdict", [
    (None, True, "promoted"), (None, False, "kept"), (status.EXPECT_INFEASIBLE, False, "promoted"),
    (status.EXPECT_OPTIMAL, True, "kept"),
])
def test_promotion_follows_the_same_rule(expected, proof, verdict):
    judged = _claim(P, 1e-9, proof, expected, reported=status.PRIMAL_INFEASIBLE_INACCURATE)
    assert judged.verdict == verdict, judged.reason
    limit = _claim(P, 1e-9, proof, expected, reported=status.MAX_ITER_REACHED)
    assert limit.verdict == verdict, limit.reason


# --- the known outcome reaches the rule from runs -------------------------------


def _write_run(path, rows):
    path.mkdir(parents=True)
    (path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (path / "manifest.json").write_text(json.dumps({"run_id": "run", "config": {}}))
    return path


def _row(problem, claim, kkt_record, **metadata):
    return {"run_id": "run", "dataset": "netlib_infeasible", "problem": problem, "solver_id": "s", "solver": "scs",
            "status": claim, "run_time_seconds": 1.0, "kkt": kkt_record,
            "metadata": {"source": f"/data/problem_classes/netlib_data/infeasible/{problem}.mps.gz", **metadata}}


def _record(kind, cert_res, **extra):
    return {"certificate": kind, "cert_res": cert_res, "cert_tol": 1e-3, "valid": True, **extra}


def test_archived_rows_take_the_outcome_their_dataset_states_now(tmp_path):
    """Rows recorded before cplex1's kind was stated say "infeasible"; a
    dual certificate on cplex1 (dual infeasible too) is confirmed, one on
    mondou2 (whose dual is feasible: PDLP's published false certificate)
    needs a proof, which an archived record does not have."""
    rows = [
        _row("cplex1", D, _record(D, 9e-9), expected_status="infeasible"),
        _row("mondou2", D, _record(D, 5.3e-5), expected_status="infeasible"),
        _row("bgdbg1", P, _record(P, 1e-9)),  # no recorded outcome: inferred from the source
    ]
    src = _write_run(tmp_path / "src", rows)
    kkt_verify(src, tmp_path / "out", tol=1e-4)
    out = {json.loads(line)["problem"]: json.loads(line) for line in (tmp_path / "out" / "results.jsonl").open()}
    assert out["cplex1"]["status"] == D
    assert out["mondou2"]["status"] == status.DUAL_INFEASIBLE_INACCURATE
    assert out["bgdbg1"]["status"] == P
    frame = load_results(src, verify=VerifyPolicy(rule="fixed", tol=1e-4)).set_index("problem")
    assert frame.loc["cplex1", "status"] == D
    assert frame.loc["mondou2", "status"] == status.DUAL_INFEASIBLE_INACCURATE
    assert frame.loc["bgdbg1", "status"] == P


def test_a_frame_keeps_the_difference_between_no_proof_and_an_unprovable_cone(tmp_path):
    # In a frame a null cert_proof is NaN either way; cert_proof_cones tells
    # a certificate the check does not cover from a record without proofs.
    rows = [
        {**_row("x", P, _record(P, 1e-9, cert_proof=None, cert_proof_cones=["q"])), "dataset": "other",
         "metadata": {"source": "x.mat"}},
        {**_row("y", P, _record(P, 1e-9)), "dataset": "other", "metadata": {"source": "y.mat"}},
    ]
    policy = VerifyPolicy(rule="fixed", tol=1e-4, missing_is_failure=False)
    frame = load_results(_write_run(tmp_path / "src", rows), verify=policy).set_index("problem")
    assert frame.loc["x", "verify.verdict"] == "demoted"
    assert frame.loc["y", "verify.verdict"] == "missing_residuals"
    assert frame.loc["y", "status"] == P  # tolerated
