"""Helpers for solver APIs that want QP constraints split by type."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.transforms.cones import INF_BOUND, split_qp_bounds


def split_qp_for_range_constraints(qp: dict):
    """Return ``P, q, Aeq, b, G, h_l, h_u`` for ``l <= A x <= u`` QPs."""
    p = sp.csc_matrix(qp["P"])
    q = np.asarray(qp["q"], dtype=float)
    a, l, u, eq, _, _ = split_qp_bounds(qp)
    if np.any(eq):
        aeq = sp.csc_matrix(a[eq, :])
        b = np.asarray(u[eq], dtype=float)
    else:
        aeq = None
        b = None

    ineq = ~eq
    if np.any(ineq):
        g = sp.csc_matrix(a[ineq, :])
        h_l = _solver_bounds(l[ineq])
        h_u = _solver_bounds(u[ineq])
        ineq_indices = np.flatnonzero(ineq)
    else:
        g = None
        h_l = None
        h_u = None
        ineq_indices = np.array([], dtype=int)

    return p, q, aeq, b, g, h_l, h_u, np.flatnonzero(eq), ineq_indices


def combine_qp_duals(
    row_count: int,
    eq_indices,
    eq_dual,
    ineq_indices,
    ineq_dual,
) -> np.ndarray | None:
    if eq_dual is None and ineq_dual is None:
        return None
    y = np.zeros(row_count)
    if eq_dual is not None and len(eq_indices):
        y[np.asarray(eq_indices, dtype=int)] = np.asarray(eq_dual, dtype=float).reshape(-1)
    if ineq_dual is not None and len(ineq_indices):
        y[np.asarray(ineq_indices, dtype=int)] = np.asarray(ineq_dual, dtype=float).reshape(-1)
    return y


def dual_from_lower_upper(z_l, z_u) -> np.ndarray | None:
    if z_l is None and z_u is None:
        return None
    lower = np.asarray(z_l if z_l is not None else 0.0, dtype=float)
    upper = np.asarray(z_u if z_u is not None else 0.0, dtype=float)
    return upper - lower


def _solver_bounds(values) -> np.ndarray:
    bounds = np.asarray(values, dtype=float).copy()
    bounds[bounds <= -INF_BOUND] = -np.inf
    bounds[bounds >= INF_BOUND] = np.inf
    return bounds
