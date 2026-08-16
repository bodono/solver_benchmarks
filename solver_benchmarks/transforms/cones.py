"""QP-to-cone conversions shared by conic solver adapters."""

from __future__ import annotations

import logging

import numpy as np
import scipy.sparse as sp

INF_BOUND = 1.0e20
# Bound-hygiene thresholds. The datasets document +-1e20 as the infinity
# sentinel, but several Maros-Meszaros .mat files store it with a few ULPs
# of representation error (e.g. -9.999999999999998e+19); a strict equality
# test misclassified those as genuine finite bounds and materialized
# 1e20-magnitude rows that silently poisoned every downstream solve.
# Policy:
#   |bound| >= _INF_CUT           -> treated as infinite (dropped). Silent
#                                    only within _CANONICAL_RTOL of the
#                                    documented +-1e20 sentinel; warned
#                                    otherwise.
#   _BIG_WARN <= |bound| < _INF_CUT -> kept, but warned: either a corrupted
#                                    sentinel or data scaled badly enough
#                                    to deserve a look.
#   Wrong-signed huge bounds (u <= -_INF_CUT or l >= +_INF_CUT) are kept
#   and warned: they encode an infeasible-scale demand, not infinity, and
#   dropping them would hide a modeling error.
# The canonical band must admit +-inf and float32-stored sentinels
# (float32(1e20) converts to ~1.00000002e+20, relative error 2e-8), so it
# is set well above float32 epsilon but far below any plausible genuine
# value in the cut band.
_INF_CUT = 1.0e19
_BIG_WARN = 1.0e10
_CANONICAL_RTOL = 1.0e-6

_logger = logging.getLogger(__name__)


def _warn_bound_hygiene(l: np.ndarray, u: np.ndarray, inf_l, inf_u) -> None:
    noncanon_u = (inf_u & np.isfinite(u)
                  & (np.abs(u - INF_BOUND) > _CANONICAL_RTOL * INF_BOUND))
    noncanon_l = (inf_l & np.isfinite(l)
                  & (np.abs(l + INF_BOUND) > _CANONICAL_RTOL * INF_BOUND))
    n_noncanon = int(noncanon_u.sum() + noncanon_l.sum())
    if n_noncanon:
        _logger.warning(
            "%d bound(s) with magnitude >= %.0e treated as infinite but not "
            "the canonical +-%.0e sentinel (corrupted sentinel?).",
            n_noncanon, _INF_CUT, INF_BOUND,
        )
    big_u = (~inf_u) & (np.abs(u) >= _BIG_WARN)
    big_l = (~inf_l) & (np.abs(l) >= _BIG_WARN)
    n_big = int(big_u.sum() + big_l.sum())
    if n_big:
        _logger.warning(
            "%d finite bound(s) with magnitude in [%.0e, %.0e) kept as "
            "genuine data; if these are meant to be infinite the problem is "
            "badly scaled.",
            n_big, _BIG_WARN, _INF_CUT,
        )
    wrong_u = u <= -_INF_CUT
    wrong_l = l >= _INF_CUT
    n_wrong = int(wrong_u.sum() + wrong_l.sum())
    if n_wrong:
        _logger.warning(
            "%d bound(s) encode an infeasible-scale demand (u <= -%.0e or "
            "l >= +%.0e); kept as data — check the model.",
            n_wrong, _INF_CUT, _INF_CUT,
        )


def split_qp_bounds(qp: dict):
    a = sp.csc_matrix(qp["A"])
    l = np.asarray(qp["l"], dtype=float)
    u = np.asarray(qp["u"], dtype=float)
    inf_u = u >= _INF_CUT
    inf_l = l <= -_INF_CUT
    _warn_bound_hygiene(l, u, inf_l, inf_u)
    # Equality detection uses a relative tolerance against the larger
    # of |l|, |u|, falling back to an absolute tolerance for tiny
    # bounds. The previous fixed |u-l| < 1e-8 silently treated tiny
    # but distinct bounds (e.g. 0 vs 1e-9) as equality and conversely
    # missed legitimate equalities at scale 1e10 (where 1e-8 of slack
    # is well below floating-point precision of the bounds themselves).
    abs_diff = np.abs(u - l)
    scale = np.maximum.reduce([np.abs(l), np.abs(u), np.ones_like(l)])
    eq = (abs_diff <= 1.0e-12 * scale) & ~inf_u & ~inf_l
    finite_u = (~eq) & ~inf_u
    finite_l = (~eq) & ~inf_l
    return a, l, u, eq, finite_l, finite_u


def qp_to_nonnegative_cone(qp: dict):
    """Convert l <= A x <= u to ZeroCone + NonnegativeCone form.

    The returned form satisfies A_cone x + s = b_cone, where the first z
    entries of s are zero and the remaining entries are nonnegative.
    """
    a, l, u, eq, finite_l, finite_u = split_qp_bounds(qp)
    rows = []
    b_parts = []
    if np.any(eq):
        rows.append(a[eq, :])
        b_parts.append(u[eq])
    if np.any(finite_u):
        rows.append(a[finite_u, :])
        b_parts.append(u[finite_u])
    if np.any(finite_l):
        rows.append(-a[finite_l, :])
        b_parts.append(-l[finite_l])
    if rows:
        cone_a = sp.vstack(rows, format="csc")
        b = np.concatenate(b_parts).astype(float)
    else:
        cone_a = sp.csc_matrix((0, a.shape[1]))
        b = np.array([], dtype=float)
    return cone_a, b, int(np.sum(eq))


def qp_to_scs_box_cone(qp: dict):
    """Convert a QP to SCS data using SCS' box cone when inequalities exist."""
    a, l, u, eq, _, _ = split_qp_bounds(qp)
    m, n = a.shape
    p = sp.csc_matrix(qp["P"])
    # Sanitize sentinel bounds to true infinities for the box cone: the
    # same corrupted / near-sentinel values handled in split_qp_bounds
    # would otherwise reach SCS as finite 1e19..1e20-magnitude box
    # bounds and poison the solve (SCS accepts +-inf natively).
    l = np.where(l <= -_INF_CUT, -np.inf, l)
    u = np.where(u >= _INF_CUT, np.inf, u)
    if np.all(eq):
        a_scs = a.copy()
        b_scs = u.copy()
        cone = {"z": int(np.sum(eq))}
        inv_perm = np.arange(m)
    else:
        order = np.hstack((np.flatnonzero(eq), np.flatnonzero(~eq)))
        inv_perm = np.argsort(order)
        a_scs = sp.vstack((a[eq, :], sp.csc_matrix((1, n)), -a[~eq, :]), format="csc")
        b_scs = np.hstack((u[eq], 1.0, np.zeros(int(np.sum(~eq)))))
        cone = {
            "z": int(np.sum(eq)),
            "bl": l[~eq].tolist(),
            "bu": u[~eq].tolist(),
        }
    data = {"P": p, "A": a_scs, "b": b_scs, "c": np.asarray(qp["q"], dtype=float)}
    return data, cone, inv_perm


def unbox_scs_dual(y, cone, inv_perm):
    if y is None:
        return None
    y = np.asarray(y).copy()
    z = int(cone.get("z", 0))
    if "bl" not in cone and "bu" not in cone:
        return y
    y[z:] *= -1.0
    y = np.delete(y, z)
    return y[inv_perm]
