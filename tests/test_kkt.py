"""Unit tests for KKT residual computations."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt


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
    assert unsupported == ["box"]
    # The two zero rows project to zeros, then the 3-element box slice
    # is left as-is. Output length must match the input.
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
    assert unsupported == ["box"]
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
