"""Row layouts shared by the cone-form adapters (ECOS, CVXOPT, SDPA)."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.transforms.cones import split_qp_bounds


def qp_dual_in_native_order(qp, y_eq, z_ineq, n_eq):
    """Reassemble cone-form duals into the QP's native row order.

    The QP went to the solver through ``qp_to_nonnegative_cone``: equality
    rows in ``A x = b`` with multiplier ``y_eq``, then ``G x + s = h``,
    ``s >= 0`` with ``G = [A_finite_u; -A_finite_l]`` and multiplier
    ``z_ineq >= 0``. The QP convention ``y_qp = lambda_u - lambda_l`` gives::

        y_qp[eq]       = y_eq
        y_qp[finite_u] = +z_u
        y_qp[finite_l] = -z_l

    Returns None when both duals are missing or the layouts disagree.
    """
    n_rows = sp.csc_matrix(qp["A"]).shape[0]
    if y_eq is None and z_ineq is None:
        return None

    _a, _l, _u, eq, finite_l, finite_u = split_qp_bounds(qp)
    eq_rows = np.flatnonzero(eq)
    fu_rows = np.flatnonzero(finite_u)
    fl_rows = np.flatnonzero(finite_l)
    y_eq_arr = np.asarray(y_eq, dtype=float).ravel() if y_eq is not None else np.zeros(n_eq)
    z_arr = np.asarray(z_ineq, dtype=float).ravel() if z_ineq is not None else np.zeros(0)

    if fu_rows.size + fl_rows.size != z_arr.size:
        return None

    y = np.zeros(n_rows)
    if eq_rows.size:
        y[eq_rows] += y_eq_arr[: eq_rows.size]
    cursor = 0
    if fu_rows.size:
        y[fu_rows] += z_arr[cursor : cursor + fu_rows.size]
        cursor += fu_rows.size
    if fl_rows.size:
        y[fl_rows] += -z_arr[cursor : cursor + fl_rows.size]
    return y


def psd_triangle_to_full(dim: int) -> sp.csc_matrix:
    """Map a length-``dim*(dim+1)/2`` PSD triangle vec (canonical col-major
    lower order with √2 off-diagonal scaling) to its ``dim*dim`` full-matrix
    vec in column-major order: the layout SDPA takes and CVXOPT's BLAS
    unpacked 'L' storage.
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    col = 0
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for j in range(dim):
        for i in range(j, dim):
            if i == j:
                rows.append(i + j * dim)
                cols.append(col)
                data.append(1.0)
            else:
                rows.extend([i + j * dim, j + i * dim])
                cols.extend([col, col])
                data.extend([inv_sqrt2, inv_sqrt2])
            col += 1
    return sp.csc_matrix((data, (rows, cols)), shape=(dim * dim, dim * (dim + 1) // 2))
