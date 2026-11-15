"""Helpers for handing ``l <= A x <= u`` QPs to solver APIs.

Every QP adapter takes its infinite bounds from :func:`normalize_qp_bounds`,
so all solvers see the same problem whatever spelling a loader used for an
absent bound, and leaves out the rows that constrain nothing
(:func:`free_row_mask`), mapping their duals back as zero.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.transforms.cones import _INF_CUT, split_qp_bounds

# Bounds at or beyond this magnitude are infinite. The datasets document
# +-1e20 as the sentinel, some Maros-Meszaros files store it with a few ULPs
# of error (-9.999999999999998e19), and the transform layer
# (``split_qp_bounds``) and the MM / QPS readers already cut at 1e19, so the
# adapters share that cutoff rather than each testing for 1e20.
INF_CUT = _INF_CUT


def normalize_qp_bounds(l, u) -> tuple[np.ndarray, np.ndarray]:
    """Return float copies of ``l`` and ``u`` with sentinels replaced by +-inf.

    Only the correct side is converted, as in ``split_qp_bounds``: a lower
    bound <= -1e19 becomes -inf and an upper bound >= 1e19 becomes +inf. A
    wrong-signed huge bound (``u <= -1e19`` or ``l >= 1e19``) encodes an
    infeasible demand, not infinity, and is kept as data.
    """
    lower = np.array(l, dtype=float, copy=True).reshape(-1)
    upper = np.array(u, dtype=float, copy=True).reshape(-1)
    lower[lower <= -INF_CUT] = -np.inf
    upper[upper >= INF_CUT] = np.inf
    return lower, upper


def free_row_mask(l, u) -> np.ndarray:
    """Rows with no finite bound on either side (after normalization)."""
    lower, upper = normalize_qp_bounds(l, u)
    return np.isneginf(lower) & np.isposinf(upper)


def scatter_rows(row_count: int, kept, values) -> np.ndarray | None:
    """Place per-kept-row ``values`` at the original row indices, zero elsewhere.

    ``kept`` is a boolean mask or an index array over the original rows.
    Returns None when ``values`` is None.
    """
    if values is None:
        return None
    out = np.zeros(int(row_count))
    out[np.asarray(kept)] = np.asarray(values, dtype=float).reshape(-1)
    return out


def split_qp_for_range_constraints(qp: dict):
    """Return ``P, q, Aeq, b, G, h_l, h_u, eq_indices, ineq_indices``.

    Rows whose bounds are both infinite are left out of ``G`` (and so of
    ``ineq_indices``): they constrain nothing, PIQP warns about each one and
    ProxQP pays for them in every iteration. :func:`combine_qp_duals` gives
    them a zero multiplier.
    """
    p = sp.csc_matrix(qp["P"])
    q = np.asarray(qp["q"], dtype=float)
    a, l, u, eq, finite_l, finite_u = split_qp_bounds(qp)
    if np.any(eq):
        aeq = sp.csc_matrix(a[eq, :])
        b = np.asarray(u[eq], dtype=float)
    else:
        aeq = None
        b = None

    ineq = ~eq & (finite_l | finite_u)
    if np.any(ineq):
        g = sp.csc_matrix(a[ineq, :])
        # The masks from split_qp_bounds already apply the shared cutoff.
        h_l = np.where(finite_l[ineq], l[ineq], -np.inf)
        h_u = np.where(finite_u[ineq], u[ineq], np.inf)
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
