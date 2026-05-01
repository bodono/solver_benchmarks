"""QP-to-cone conversions shared by conic solver adapters."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

INF_BOUND = 1.0e20


def split_qp_bounds(qp: dict):
    a = sp.csc_matrix(qp["A"])
    l = np.asarray(qp["l"], dtype=float)
    u = np.asarray(qp["u"], dtype=float)
    eq = (np.abs(u - l) < 1.0e-8) & (u < INF_BOUND) & (l > -INF_BOUND)
    finite_u = (~eq) & (u < INF_BOUND)
    finite_l = (~eq) & (l > -INF_BOUND)
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
