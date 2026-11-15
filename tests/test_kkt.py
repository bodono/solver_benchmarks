"""Unit tests for KKT residual computations."""

from __future__ import annotations

import functools

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.analysis.derive import worst_relative_residual


def test_qp_residuals_reject_multiplier_on_absent_upper_bound():
    # min -x subject to x >= 0 is unbounded. At x=0, y=1 satisfies
    # stationarity but incorrectly assigns a multiplier to the absent upper bound.
    res = kkt.qp_residuals(
        sp.csc_matrix((1, 1)),
        np.array([-1.0]),
        sp.eye(1, format="csc"),
        np.array([0.0]),
        np.array([np.inf]),
        np.array([0.0]),
        np.array([1.0]),
    )
    assert res["primal_res_rel"] == 0.0
    assert res["dual_res_rel"] == 0.0
    assert res["duality_gap_rel"] == 0.0
    # (x, y) = (0, 1) is not a KKT point: the multiplier 1 on the absent upper
    # bound, scaled by 1 + |y|, fails the check.
    assert res["dual_bound_res_rel"] == 0.5
    assert worst_relative_residual({"kkt": res}) == 0.5


def test_qp_residuals_reject_multiplier_on_absent_lower_bound():
    # min x subject to x <= 0 is unbounded. At x=0, y=-1 satisfies
    # stationarity but assigns a multiplier to the absent lower bound.
    res = kkt.qp_residuals(
        sp.csc_matrix((1, 1)),
        np.array([1.0]),
        sp.eye(1, format="csc"),
        np.array([-np.inf]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([-1.0]),
    )
    assert res["dual_res_rel"] == 0.0
    assert res["duality_gap_rel"] == 0.0
    assert res["dual_bound_res_rel"] == 0.5


def test_qp_residuals_reject_multiplier_on_free_row():
    # Row 2 is free (no bounds); any multiplier on it is infeasible, whatever its sign.
    for y2 in (2.0, -2.0):
        res = kkt.qp_residuals(
            sp.csc_matrix((1, 1)),
            np.array([-y2]),
            sp.csc_matrix(np.array([[1.0], [1.0]])),
            np.array([0.0, -np.inf]),
            np.array([1.0, np.inf]),
            np.array([0.5]),
            np.array([0.0, y2]),
        )
        assert res["dual_res_rel"] == 0.0
        assert res["dual_bound_res"] == 2.0


def test_qp_residuals_at_optimum():
    # min 0.5 x^T P x + q^T x s.t. l <= A x <= u
    # P = [[4,0],[0,2]], q = [-8, -4], A = I, l = [0,0], u = [10,10].
    # Unconstrained minimum at x = (2, 2); both bounds inactive.
    P = sp.csc_matrix(np.array([[4.0, 0.0], [0.0, 2.0]]))
    q = np.array([-8.0, -4.0])
    A = sp.csc_matrix(np.eye(2))
    l = np.array([0.0, 0.0])
    u = np.array([10.0, 10.0])
    x = np.array([2.0, 2.0])
    y = np.array([0.0, 0.0])
    res = kkt.qp_residuals(P, q, A, l, u, x, y)
    assert res["primal_res"] < 1e-12
    assert res["dual_res"] < 1e-12
    assert res["comp_slack"] < 1e-12
    assert abs(res["duality_gap"]) < 1e-12
    assert res["dual_bound_res"] == 0.0


def test_qp_residuals_with_active_bound():
    # min 0.5 x^2 + q x s.t. x >= 1.
    # P=[2], q=[-0.5]. Unconstrained optimum would be 0.25, bound active at x=1.
    # Stationarity: 2*1 - 0.5 + 1*y = 0  =>  y = -1.5 (lower bound active)
    P = sp.csc_matrix(np.array([[2.0]]))
    q = np.array([-0.5])
    A = sp.csc_matrix(np.array([[1.0]]))
    l = np.array([1.0])
    u = np.array([np.inf])
    x = np.array([1.0])
    y = np.array([-1.5])
    res = kkt.qp_residuals(P, q, A, l, u, x, y)
    assert res["primal_res"] < 1e-12
    assert res["dual_res"] < 1e-12
    assert res["comp_slack"] < 1e-12
    assert abs(res["duality_gap"]) < 1e-12


def test_qp_residuals_detect_infeasible_point():
    P = sp.csc_matrix(np.array([[2.0]]))
    q = np.array([0.0])
    A = sp.csc_matrix(np.array([[1.0]]))
    l = np.array([1.0])
    u = np.array([2.0])
    # x outside bounds, y zero -> large primal residual
    x = np.array([5.0])
    y = np.array([0.0])
    res = kkt.qp_residuals(P, q, A, l, u, x, y)
    assert res["primal_res"] >= 3.0


def test_qp_residuals_sentinel_bounds_gap_immune_to_tiny_multiplier():
    # Regression: dataset readers emit +-1e20 infinity sentinels rather
    # than true infinities (e.g. Maros-Meszaros AUG2DCQP has 20,200 upper
    # bounds at exactly 1e20). A numerically tiny multiplier on a
    # sentinel-bounded row — harmless solver noise — used to add
    # u*y ~ 1e20 * 1e-11 = 1e9 of garbage to the dual objective and comp
    # slack, reporting duality_gap_rel ~ 1.0 at a true optimum.
    # min 0.5 x^2 - x  s.t.  0 <= x <= 0.5 (upper active), sentinel row free.
    P = sp.csc_matrix(np.array([[1.0]]))
    q = np.array([-1.0])
    A = sp.csc_matrix(np.array([[1.0], [1.0]]))
    l = np.array([0.0, -1.0e20])
    u = np.array([0.5, 1.0e20])
    x = np.array([0.5])
    tiny = 1.0e-11
    # Stationarity x + q + y0 + y1 = 0 holds exactly: y0 + y1 = 0.5.
    y = np.array([0.5 - tiny, tiny])
    res = kkt.qp_residuals(P, q, A, l, u, x, y)
    assert res["primal_res"] < 1e-12
    assert res["dual_res"] < 1e-12
    assert res["comp_slack"] < 1e-10
    assert abs(res["duality_gap"]) < 1e-10
    assert res["duality_gap_rel"] < 1e-10


def test_cone_residuals_nonnegative_lp_optimum():
    # min -x s.t. x + s = 1, s >= 0  =>  x* = 1, s* = 0, y* = 1.
    P = sp.csc_matrix((1, 1))
    q = np.array([-1.0])
    A = sp.csc_matrix(np.array([[1.0]]))
    b = np.array([1.0])
    cone = {"l": 1}
    x = np.array([1.0])
    y = np.array([1.0])
    s = np.array([0.0])
    res = kkt.cone_residuals(P, q, A, b, cone, x, y, s)
    assert res["primal_res"] < 1e-12
    assert res["dual_res"] < 1e-12
    assert res["primal_cone_res"] < 1e-12
    assert res["dual_cone_res"] < 1e-12
    assert abs(res["comp_slack"]) < 1e-12
    assert abs(res["duality_gap"]) < 1e-12


def test_cone_residuals_equality_plus_nonnegative():
    # Two rows: first equality, second nonneg.
    # min 0 x s.t. x = 1 (z), x + s = 2, s >= 0.
    # x = 1, s = 1, y_eq free (choose 0), y_nn from stationarity:
    # stationarity: A'y + q = 0 => [1 1] [y_eq; y_nn] = 0 => y_eq = -y_nn.
    # complementarity: s*y_nn = 1*y_nn = 0 => y_nn = 0 => y_eq = 0.
    A = sp.csc_matrix(np.array([[1.0], [1.0]]))
    b = np.array([1.0, 2.0])
    q = np.array([0.0])
    P = sp.csc_matrix((1, 1))
    cone = {"z": 1, "l": 1}
    x = np.array([1.0])
    y = np.array([0.0, 0.0])
    s = np.array([0.0, 1.0])
    res = kkt.cone_residuals(P, q, A, b, cone, x, y, s)
    assert res["primal_res"] < 1e-12
    assert res["dual_res"] < 1e-12
    assert res["primal_cone_res"] < 1e-12
    assert res["dual_cone_res"] < 1e-12


def test_cone_residuals_soc_optimum():
    # min x1 s.t. (x2, x3) = (x1, 3), (x1, x2, x3) in SOC(3) via s = (x1, 3, ?)... simpler:
    # min t s.t. t >= sqrt(x^2 + 1), x = 0. Optimum t=1.
    # Put in form: Ax + s = b, s in SOC(3).
    # Variables: x = (t, x_free). Want s = (t, x_free, 1) in SOC.
    # A = [[1,0],[0,1],[0,0]], s = b - A x, b = (0, 0, 1).
    # At optimum: t=1, x_free=0, s = (0 - 1, 0 - 0, 1 - 0) = (-1, 0, 1). NOT in SOC.
    # Sign convention: s = b - Ax, need s in SOC means 1st >= ||rest||.
    # Let A = [[-1,0],[0,-1],[0,0]], b = (0,0,1). Then s = (t, x_free, 1). For SOC: t >= sqrt(x_free^2 + 1).
    # min t is then min ||(x_free, 1)||2 over x_free -> x_free=0, t=1.
    # q = (1, 0).
    P = sp.csc_matrix((2, 2))
    q = np.array([1.0, 0.0])
    A = sp.csc_matrix(np.array([[-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]))
    b = np.array([0.0, 0.0, 1.0])
    cone = {"q": [3]}
    x = np.array([1.0, 0.0])
    s = np.array([1.0, 0.0, 1.0])
    # SOC is self-dual. y must satisfy A'y + q = 0 and y in SOC, s'y = 0.
    # y = (y0, y1, y2). A'y = (-y0, -y1). Need (-y0, -y1) + (1, 0) = 0 => y0=1, y1=0.
    # s'y = 1*1 + 0*0 + 1*y2 = 1 + y2 = 0 => y2 = -1.
    # Check y in SOC: y0 >= sqrt(y1^2 + y2^2) = 1. 1 >= 1. ✓
    y = np.array([1.0, 0.0, -1.0])
    res = kkt.cone_residuals(P, q, A, b, cone, x, y, s)
    assert res["primal_res"] < 1e-10
    assert res["dual_res"] < 1e-10
    assert res["primal_cone_res"] < 1e-10
    assert res["dual_cone_res"] < 1e-10
    assert abs(res["comp_slack"]) < 1e-10


def test_cone_primal_infeasibility_cert_valid():
    # x + s = 0, x - s = 2, s >= 0 is infeasible (adding: 2x = 2 => x=1; subtracting s = -1).
    # Actually: first row is equality x = 0; second row x + s = 2, s >= 0 means x <= 2.
    # Put x = 0 forced. Second row s = 2. Feasible. Not a good example.
    # Better: x = 0 (z), -x + s = -1 (s >= 0) => x >= 1. With x=0 forced, infeasible.
    A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    b = np.array([0.0, -1.0])
    cone = {"z": 1, "l": 1}
    # Certificate y in K* (y1 free, y2 >= 0), A'y = 0, b'y < 0.
    # A'y = y1 - y2 = 0 => y1 = y2. b'y = 0*y1 + -1*y2 = -y2. Want < 0, so y2 > 0.
    y = np.array([1.0, 1.0])
    cert = kkt.cone_primal_infeasibility_cert(A, b, cone, y)
    assert cert["Aty_inf"] < 1e-12
    assert cert["dual_cone_res"] < 1e-12
    assert cert["bty"] < 0
    assert cert["valid"]


def test_cone_dual_infeasibility_cert_valid():
    # min q'x s.t. Ax + s = b, s >= 0.  Unbounded if exists x with q'x < 0,
    # -Ax in K (i.e., Ax <= 0), Px = 0.
    # q = [-1], A = [[0]], so any x is feasible direction with Ax = 0; q'x = -x < 0 needs x > 0.
    P = sp.csc_matrix((1, 1))
    q = np.array([-1.0])
    A = sp.csc_matrix(np.array([[0.0]]))
    cone = {"l": 1}
    x = np.array([1.0])
    cert = kkt.cone_dual_infeasibility_cert(P, q, A, cone, x)
    assert cert["Px_inf"] < 1e-12
    assert cert["primal_cone_res"] < 1e-12
    assert cert["qtx"] < 0
    assert cert["valid"]


def test_qp_primal_infeasibility_cert():
    # 1 <= x <= 2 and -2 <= -x <= -3 (so x >= 3 via second). Combined infeasible.
    # A = [[1],[-1]], l = [1, -3], u = [2, -2]. Need x with 1<=x<=2 and 2<=x<=3: infeasible.
    A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    l = np.array([1.0, -3.0])
    u = np.array([2.0, -2.0])
    # Certificate: find y with A'y = 0 and u'y+ - l'y- < 0.
    # A' = [[1, -1]]. A'y = y1 - y2 = 0 => y1 = y2 = t.
    # support = u'y+ - l'y-.
    # Try t > 0: y+=(t,t), y-=(0,0). support = 2t + (-2)t = 0. Not strictly negative.
    # Try t < 0: y+=(0,0), y-=(−t,−t). support = 0 - (1·(-t) + (-3)·(-t)) = 0 - (-t + 3t) = -2t = 2|t| > 0. No.
    # Mix: y = (t, s). A'y = t - s = 0 => t = s. covered above.
    # Hmm, try y = (1, -1): A'y = 1 - (-1) = 2. Not in null of A'.
    # This problem actually is infeasible only if strict; the bounds here give x in [1,2] ∩ [2,3] = {2}, which IS feasible.
    # Let me fix: l = [1, -3], u = [2, -2.5] => x in [1,2] ∩ [2.5, 3]: infeasible.
    l = np.array([1.0, -3.0])
    u = np.array([2.0, -2.5])
    # A'y = y1 - y2 = 0 => y1 = y2 = t.
    # support = u'y+ - l'y-.
    # t > 0: support = (2 + (-2.5)) t = -0.5 t < 0. ✓
    y = np.array([1.0, 1.0])
    cert = kkt.qp_primal_infeasibility_cert(A, l, u, y)
    assert cert["Aty_inf"] < 1e-12
    assert cert["support"] < 0
    assert cert["valid"]


def test_psd_projection_clips_negative_eigenvalue():
    # 2x2 symmetric matrix [[1, 2],[2, 1]]. Eigenvalues 3 and -1.
    # SCS vectorization of LOWER triangular (col-major), off-diag scaled by sqrt(2):
    # v = [A11, sqrt(2)*A21, A22] = [1, 2*sqrt(2), 1].
    sqrt2 = np.sqrt(2.0)
    v = np.array([1.0, 2.0 * sqrt2, 1.0])
    proj = kkt._project_psd_triangle(v, 2)
    # Reconstruct the projected matrix.
    a11 = proj[0]
    a21 = proj[1] / sqrt2
    a22 = proj[2]
    mat = np.array([[a11, a21], [a21, a22]])
    eigvals = np.linalg.eigvalsh(mat)
    assert np.all(eigvals >= -1e-12)
    # Frobenius distance: original eigenvalues 3, -1. Closest PSD has eigenvalues 3, 0.
    # So projection distance = |-1 - 0| = 1 in eigenspace -> ||v - proj|| should be sqrt(2)/2 or so.
    assert np.linalg.norm(v - proj) > 0.1


def test_soc_projection_identity_inside():
    # Point inside SOC: (2, 1, 0). 2 >= sqrt(1) = 1. Projection = identity.
    v = np.array([2.0, 1.0, 0.0])
    assert np.allclose(kkt._project_soc(v), v)


def test_soc_projection_zeros_below_polar():
    # Point in polar cone: (-2, 0, 0). Projection = 0.
    v = np.array([-2.0, 0.5, 0.5])
    proj = kkt._project_soc(v)
    assert np.allclose(proj, np.zeros(3))


def test_soc_projection_boundary_case():
    v = np.array([0.0, 3.0, 4.0])  # norm 5; projection should have t = 2.5
    proj = kkt._project_soc(v)
    assert abs(proj[0] - 2.5) < 1e-12
    rest_norm = np.linalg.norm(proj[1:])
    assert abs(rest_norm - 2.5) < 1e-12  # lies on boundary


def test_box_cone_handling_does_not_double_advance_row_cursor():
    """A cone dict with both ``bl`` and ``bu`` keys present must consume
    the box slot exactly once; otherwise downstream slices get
    misaligned and downstream KKT values are computed on wrong slack
    entries. The audit flagged this."""
    # 2 zero entries + box of width 1 + len(bl)=2 -> total 5.
    cone = {"z": 2, "bl": [-1.0, -1.0], "bu": [1.0, 1.0]}
    # Slack/dual vectors of the right size; the projection function
    # must mark "box" as unsupported but consume only one slot.
    s = np.arange(5, dtype=float)
    y = np.arange(5, dtype=float) * -1.0
    s_proj, y_proj, unsupported = kkt._project_cones(cone, s, y)
    assert unsupported == []
    # The two zero rows project to zeros, then the 3-element box slice is
    # projected onto the box cone. Output length must match the input.
    assert s_proj.size == 5
    assert y_proj.size == 5


def test_box_cone_with_only_bl_key():
    """An sdp/box cone with only ``bl`` (or only ``bu``) — a possible
    user mistake — must still consume the right slot width without
    crashing or misaligning."""
    cone = {"bl": [0.0, 0.0, 0.0]}  # box width = 1 + 3 = 4
    s = np.arange(4, dtype=float)
    y = np.arange(4, dtype=float)
    s_proj, y_proj, unsupported = kkt._project_cones(cone, s, y)
    assert unsupported == []
    assert s_proj.size == 4
    assert y_proj.size == 4


def test_box_cone_with_mismatched_bl_bu_sizes_marks_unsupported_safely():
    """Malformed cone where bl and bu disagree in size: do not advance
    the row cursor in a way that would shift later slices."""
    cone = {"bl": [0.0], "bu": [0.0, 0.0]}
    s = np.zeros(8, dtype=float)
    y = np.zeros(8, dtype=float)
    s_proj, y_proj, unsupported = kkt._project_cones(cone, s, y)
    assert "box" in unsupported


# --- infeasibility certificates are held to a residual check, not a sign test ---


def test_cone_primal_cert_outside_dual_cone_is_invalid():
    # b'y < 0, but A'y is far from zero and y has a negative entry on an
    # l-cone row, so y certifies nothing.
    A = sp.csc_matrix(np.array([[1.0], [1.0]]))
    cert = kkt.cone_primal_infeasibility_cert(A, np.array([1.0, 2.0]), {"l": 2}, np.array([-5.0, 1.0]))
    assert cert["bty"] < 0
    assert cert["dual_cone_res"] == 5.0
    assert cert["valid"] is False
    # scaled so that b'y = -1, the worst residual is |A'y| = 5/3
    assert cert["cert_res"] == np.float64(5.0) / 3.0


def test_cone_dual_cert_with_nonzero_px_is_invalid():
    cert = kkt.cone_dual_infeasibility_cert(
        sp.csc_matrix(np.array([[1.0]])), np.array([1.0]), sp.csc_matrix(np.array([[1.0]])), {"l": 1}, np.array([-1.0])
    )
    assert cert["qtx"] < 0 and cert["Px_inf"] == 1.0
    assert cert["valid"] is False


def test_qp_dual_cert_with_nonzero_px_is_invalid():
    cert = kkt.qp_dual_infeasibility_cert(
        sp.csc_matrix(np.array([[1.0]])),
        np.array([1.0]),
        sp.csc_matrix(np.array([[1.0]])),
        np.array([-np.inf]),
        np.array([np.inf]),
        np.array([-1.0]),
    )
    assert cert["qtx"] < 0
    assert cert["valid"] is False


def test_certificates_pass_within_tolerance_and_fail_beyond_it():
    # x = 0 (z) and x >= 1 (l): y = (1, 1) is exact; perturb A'y by 1e-9 and 1e-2.
    A = sp.csc_matrix(np.array([[1.0], [-1.0]]))
    b = np.array([0.0, -1.0])
    for eps, valid in ((1e-9, True), (1e-2, False)):
        cert = kkt.cone_primal_infeasibility_cert(A, b, {"z": 1, "l": 1}, np.array([1.0 + eps, 1.0]))
        assert cert["valid"] is valid
        assert np.isclose(cert["cert_res"], eps)
        assert cert["cert_tol"] == kkt.CERT_TOL
    # The tolerance is a parameter.
    cert = kkt.cone_primal_infeasibility_cert(A, b, {"z": 1, "l": 1}, np.array([1.0 + 1e-6, 1.0]), tol=1e-8)
    assert cert["valid"] is False and cert["cert_tol"] == 1e-8


def test_qp_primal_cert_with_multiplier_on_infinite_bound_is_invalid():
    # 1 <= x and a free row: feasible (x = 1). y = (-1, 1) has A'y = 0 and
    # support -1 only because the free row's multiplier is ignored.
    cert = kkt.qp_primal_infeasibility_cert(
        sp.csc_matrix(np.array([[1.0], [1.0]])),
        np.array([1.0, -np.inf]),
        np.array([np.inf, np.inf]),
        np.array([-1.0, 1.0]),
    )
    assert cert["Aty_inf"] == 0.0
    assert cert["dual_bound_res"] == 1.0
    assert cert["valid"] is False


def test_qp_primal_cert_ignores_roundoff_on_infinite_bounds():
    A = sp.csc_matrix(np.array([[1.0], [-1.0], [1.0]]))
    l = np.array([1.0, -3.0, -np.inf])
    u = np.array([2.0, -2.5, np.inf])
    cert = kkt.qp_primal_infeasibility_cert(A, l, u, np.array([1.0, 1.0, 1e-12]))
    assert cert["support"] < 0
    assert cert["valid"] is True


def test_certificate_cone_residual_is_unknown_for_unsupported_cones():
    # A complex PSD block cannot be checked: report NaN and an unknown
    # verdict instead of a clean zero.
    A = sp.csc_matrix(np.zeros((4, 1)))
    cert = kkt.cone_primal_infeasibility_cert(A, np.ones(4), {"cs": [2]}, -np.ones(4))
    assert np.isnan(cert["dual_cone_res"])
    assert cert["valid"] is None
    assert cert["unsupported_cones"] == ["cs"]
    cert = kkt.cone_dual_infeasibility_cert(sp.csc_matrix((1, 1)), np.array([-1.0]), A, {"cs": [2]}, np.ones(1))
    assert np.isnan(cert["primal_cone_res"])
    assert cert["valid"] is None


def test_exp_cone_certificate_is_checked():
    # y = (-1, -1, -1) is not in the dual exponential cone.
    cert = kkt.cone_primal_infeasibility_cert(sp.csc_matrix(np.zeros((3, 1))), np.ones(3), {"ep": 1}, -np.ones(3))
    assert cert["dual_cone_res"] > 0.1
    assert cert["valid"] is False
    assert "unsupported_cones" not in cert


# --- cone primal residual includes the cone distance of s ---


def test_cone_primal_residual_counts_slack_outside_cone():
    # min x2 s.t. x >= 1 written as -x + s = -1, s >= 0. An adapter that
    # passes s = b - Ax makes Ax + s - b vanish; the point x = (0.5, 1)
    # violates x1 >= 1 and must show up in primal_res_rel.
    A = sp.csc_matrix(-np.eye(2))
    b = np.array([-1.0, -1.0])
    x = np.array([0.5, 1.0])
    res = kkt.cone_residuals(sp.csc_matrix((2, 2)), np.array([0.0, 1.0]), A, b, {"l": 2}, x, np.array([0.0, 1.0]), b - A @ x)
    assert res["primal_eq_res"] == 0.0
    assert res["primal_cone_res"] == 0.5
    assert res["primal_res"] == 0.5
    assert res["primal_res_rel"] == 0.25
    assert res["dual_res_rel"] == 0.0 and res["duality_gap_rel"] == 0.0


def test_cone_primal_residual_counts_equality_violation_through_zero_cone():
    # s = b - Ax on an equality row: the zero-cone distance is |b - Ax|.
    A = sp.csc_matrix(np.array([[1.0]]))
    b = np.array([1.0])
    x = np.array([0.9])
    res = kkt.cone_residuals(sp.csc_matrix((1, 1)), np.array([0.0]), A, b, {"z": 1}, x, np.zeros(1), b - A @ x)
    assert np.isclose(res["primal_res"], 0.1)


# --- exponential, power and box cones ---


def _exp_optimum():
    # min t s.t. (1, 1, t) in K_exp, i.e. s = b - Ax with A = (0, 0, -1)', b = (1, 1, 0).
    # Optimum t = e with dual y = (-e, 0, 1).
    A = sp.csc_matrix(np.array([[0.0], [0.0], [-1.0]]))
    b = np.array([1.0, 1.0, 0.0])
    return A, b, np.array([np.e]), np.array([-np.e, 0.0, 1.0])


def test_exp_cone_residuals_at_optimum_are_small():
    A, b, x, y = _exp_optimum()
    res = kkt.cone_residuals(sp.csc_matrix((1, 1)), np.array([1.0]), A, b, {"ep": 1}, x, y, b - A @ x)
    assert "unsupported_cones" not in res
    for field in ("primal_res_rel", "dual_res_rel", "duality_gap_rel", "primal_cone_res_rel", "dual_cone_res_rel"):
        assert res[field] < 1e-12, (field, res[field])
    assert worst_relative_residual({"kkt": res}) < 1e-12


def test_exp_cone_residuals_detect_slack_outside_cone():
    A, b, _, y = _exp_optimum()
    x = np.array([2.0])  # 1 * exp(1 / 1) = e > 2: s = (1, 1, 2) is outside K_exp
    res = kkt.cone_residuals(sp.csc_matrix((1, 1)), np.array([1.0]), A, b, {"ep": 1}, x, y, b - A @ x)
    assert res["primal_cone_res"] > 0.1


def test_dual_exp_cone_block_swaps_the_roles_of_s_and_y():
    # With an "ed" block, s must lie in K_exp* and y in K_exp.
    s = np.array([-1.0, -1.0, 2.0])  # in K_exp*, not in K_exp
    y = np.array([1.0, 1.0, 3.0])  # in K_exp, not in K_exp*
    s_proj, y_proj, unsupported = kkt._project_cones({"ed": 1}, s, y)
    assert unsupported == []
    assert np.allclose(s_proj, s) and np.allclose(y_proj, y)
    s_proj, y_proj, _ = kkt._project_cones({"ep": 1}, s, y)
    assert not np.allclose(s_proj, s) and not np.allclose(y_proj, y)


# Membership up to ``tol``: shift the point by ``tol`` along a direction
# interior to the cone first. Testing the defining inequality directly is
# ill-conditioned next to the faces (x^a or exp(r/s) with a tiny argument).
def _in_exp(v, tol):
    r, s, t = v[0] - tol, v[1] + tol, v[2] + tol
    with np.errstate(over="ignore"):
        return s > 0 and s * np.exp(r / s) <= t


def _in_exp_dual(v, tol):
    u, w_v, w = v[0] - tol, v[1] + tol, v[2] + tol
    with np.errstate(over="ignore"):
        return u < 0 and -u * np.exp(w_v / u) <= np.e * w


def _in_pow(v, tol, a):
    x, y, z = v[0] + tol, v[1] + tol, v[2]
    return x >= 0 and y >= 0 and x**a * y ** (1 - a) >= abs(z) - tol


def _in_pow_dual(v, tol, a):
    u, w_v, w = v[0] + tol, v[1] + tol, v[2]
    return u >= 0 and w_v >= 0 and (u / a) ** a * (w_v / (1 - a)) ** (1 - a) >= abs(w) - tol


def _check_projection(v, p, in_cone, in_dual, tol=1e-7):
    # Moreau: p = proj_K(v) iff p in K, p - v in K*, and p'(v - p) = 0.
    scale = 1.0 + np.linalg.norm(v)
    assert in_cone(p, tol * scale), (v, p)
    assert in_dual(p - v, tol * scale), (v, p)
    assert abs(p @ (v - p)) <= tol * scale**2, (v, p)


def test_exp_cone_projection_satisfies_moreau_conditions():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(400, 3)) * rng.choice([0.01, 1.0, 100.0], size=(400, 1))
    points = np.vstack([points, [[1.0, 1.0, np.e], [0.0, 0.0, 0.0], [-1.0, 0.0, 2.0], [1.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0], [-2.0, -3.0, 4.0], [3.0, -1.0, -2.0], [1e-9, 1.0, np.e]]])
    projected = kkt._project_exp(points)
    for v, p in zip(points, projected):
        _check_projection(v, p, _in_exp, _in_exp_dual)
    dual = kkt._project_exp_dual(points)
    for v, p in zip(points, dual):
        _check_projection(v, p, _in_exp_dual, _in_exp)


def test_exp_cone_projection_of_a_row_does_not_depend_on_the_others():
    # Each row stops iterating once it has converged, so projecting a large
    # block costs what its rows need, and a row's result is the one it gets
    # on its own.
    rng = np.random.default_rng(2)
    points = rng.normal(size=(300, 3)) * np.exp(3.0 * rng.normal(size=(300, 1)))
    batch = kkt._project_exp(points)
    single = np.vstack([kkt._project_exp(point[None, :]) for point in points])
    np.testing.assert_array_equal(batch, single)


def test_power_cone_projection_satisfies_moreau_conditions():
    rng = np.random.default_rng(1)
    points = rng.normal(size=(300, 3)) * rng.choice([0.01, 1.0, 100.0], size=(300, 1))
    points = np.vstack([points, [[1.0, 1.0, 0.5], [-1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [-1.0, -1.0, 0.1]]])
    for a in (0.3, 0.5, 0.8):
        in_cone = functools.partial(_in_pow, a=a)
        in_dual = functools.partial(_in_pow_dual, a=a)
        for v, p in zip(points, kkt._project_pow(points, a)):
            _check_projection(v, p, in_cone, in_dual)
        for v, p in zip(points, kkt._project_pow_dual(points, a)):
            _check_projection(v, p, in_dual, in_cone)


def test_power_cone_blocks_follow_the_sign_of_the_exponent():
    s = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    s_proj, _, unsupported = kkt._project_cones({"p": [0.5, -0.5]}, s, None)
    assert unsupported == []
    # (1, 1, 1): 1^0.5 1^0.5 = 1 >= |1|, so it is in K_0.5 but not in its dual
    # ((1/0.5)^0.5 (1/0.5)^0.5 = 2 >= 1 also holds, so test a point that differs).
    assert np.allclose(s_proj[:3], s[:3])
    s = np.array([0.0, 0.0, 0.0, 0.2, 0.2, 0.3])  # 0.2^.5 0.2^.5 = 0.2 < 0.3 <= (0.4)^.5 (0.4)^.5
    s_proj, _, _ = kkt._project_cones({"p": [0.5, -0.5]}, s, None)
    assert np.allclose(s_proj[3:], s[3:])
    s_proj, _, _ = kkt._project_cones({"p": [0.5, 0.5]}, s, None)
    assert not np.allclose(s_proj[3:], s[3:])


def _in_box(v, lo, hi, tol):
    t, s = v[0], v[1:]
    lower = np.full_like(s, -np.inf)
    upper = np.full_like(s, np.inf)
    lower[np.isfinite(lo)] = t * lo[np.isfinite(lo)]
    upper[np.isfinite(hi)] = t * hi[np.isfinite(hi)]
    return t >= -tol and np.all(s >= lower - tol) and np.all(s <= upper + tol)


def _in_box_dual(v, lo, hi, tol):
    # tau + min over z in [lo, hi] of sigma'z >= 0
    tau, sigma = v[0], v[1:]
    bound = np.where(sigma >= 0, lo, hi)
    worst = np.zeros_like(sigma)
    active = sigma != 0
    worst[active] = sigma[active] * bound[active]
    return tau + worst.sum() >= -tol


def test_box_cone_projection_satisfies_moreau_conditions():
    rng = np.random.default_rng(2)
    lo = np.array([-1.0, 0.0, -np.inf, 2.0])
    hi = np.array([1.0, np.inf, 3.0, 2.0])
    points = rng.normal(size=(200, 5)) * rng.choice([0.01, 1.0, 10.0], size=(200, 1))
    for v in points:
        p = kkt._project_box(v, lo, hi)
        _check_projection(v, p, lambda w, tol: _in_box(w, lo, hi, tol), lambda w, tol: _in_box_dual(w, lo, hi, tol))
        d = kkt._project_box_dual(v, lo, hi)
        _check_projection(v, d, lambda w, tol: _in_box_dual(w, lo, hi, tol), lambda w, tol: _in_box(w, lo, hi, tol))


def test_box_cone_rows_are_checked():
    cone = {"z": 1, "bl": [-1.0], "bu": [1.0]}
    s = np.array([0.0, 1.0, 0.5])  # t = 1, -1 <= 0.5 <= 1: in the box cone
    s_proj, _, unsupported = kkt._project_cones(cone, s, None)
    assert unsupported == []
    assert np.allclose(s_proj, s)
    s_proj, _, _ = kkt._project_cones(cone, np.array([0.0, 1.0, 2.0]), None)
    assert not np.allclose(s_proj, [0.0, 1.0, 2.0])


# --- PSD projection ---


def _reference_psd_projection(v, n):
    mat = np.zeros((n, n))
    idx = 0
    for col in range(n):
        for row in range(col, n):
            val = v[idx] / (np.sqrt(2.0) if row != col else 1.0)
            mat[row, col] = mat[col, row] = val
            idx += 1
    w, q = np.linalg.eigh(mat)
    proj = (q * np.maximum(w, 0.0)) @ q.T
    out = np.empty_like(v)
    idx = 0
    for col in range(n):
        for row in range(col, n):
            out[idx] = proj[row, col] * (np.sqrt(2.0) if row != col else 1.0)
            idx += 1
    return out


def test_psd_projection_matches_elementwise_reference():
    rng = np.random.default_rng(3)
    for n in (1, 2, 5, 9):
        v = rng.normal(size=n * (n + 1) // 2)
        assert np.allclose(kkt._project_psd_triangle(v, n), _reference_psd_projection(v, n), atol=1e-12)


def test_psd_residuals_with_nonfinite_iterate_are_nan_not_an_error():
    n = 3
    tri = n * (n + 1) // 2
    A = sp.csc_matrix(np.zeros((tri, 1)))
    b = np.zeros(tri)
    res = kkt.cone_residuals(sp.csc_matrix((1, 1)), np.zeros(1), A, b, {"s": [n]}, np.zeros(1), np.full(tri, np.nan), b)
    assert np.isnan(res["dual_cone_res"])
    cert = kkt.cone_primal_infeasibility_cert(A, -np.ones(tri), {"s": [n]}, np.full(tri, np.nan))
    assert np.isnan(cert["dual_cone_res"]) and cert["valid"] is not True


def test_complementarity_has_a_scale_free_variant():
    # The same LP optimum with the objective scaled by 1e6: comp_slack scales
    # with it, comp_slack_rel does not.
    A = sp.csc_matrix(np.array([[1.0]]))
    for scale in (1.0, 1e6):
        y = np.array([scale * (1.0 + 1e-9)])
        res = kkt.cone_residuals(sp.csc_matrix((1, 1)), np.array([-scale]), A, np.array([1.0]), {"l": 1},
                                 np.array([1.0]), y, np.array([1e-9]))
        assert np.isclose(res["comp_slack_rel"], res["comp_slack"] / (1 + abs(res["primal_obj"]) + abs(res["dual_obj"])))
        assert res["comp_slack_rel"] < 1e-9
        qp = kkt.qp_residuals(sp.csc_matrix((1, 1)), np.array([-scale]), A, np.array([-np.inf]), np.array([1.0]),
                              np.array([1.0 - 1e-9]), y)
        assert qp["comp_slack_rel"] < 1e-9
