"""MaxCut SDP relaxation construction.

Given a weighted graph with weight matrix ``W`` (symmetric,
non-negative, zero diagonal), the standard Goemans-Williamson SDP
relaxation of MaxCut is::

    maximize    ¼ * trace(L X)
    subject to  diag(X) = 1
                X ⪰ 0

where ``L = diag(W·1) − W`` is the graph Laplacian. Expressing this
in the canonical CONE form (``min q' x s.t. A x + s = b, s in K``)
of the codebase: ``x = y`` is the SDP dual (one variable per
diagonal-fix constraint), ``q = -1`` (we *minimize* the dual; primal
MaxCut SDP value at optimum equals ``-q'x + constant``), ``A`` columns
hold ``vec(A_k) = vec(e_k e_k')`` (the k-th diagonal selector), and
``b = vec(¼ L)`` in the canonical PSD layout.

The constant ``¼ * sum W`` that the MaxCut value picks up under the
``½(1 − X[i,j])`` substitution is recorded on
``problem.metadata["maxcut_constant"]`` rather than baked into the
objective.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def maxcut_sdp_cone_problem(weights: np.ndarray) -> tuple[dict, dict]:
    """Build the MaxCut SDP relaxation as a CONE problem.

    Parameters
    ----------
    weights:
        ``(n, n)`` symmetric weight matrix. Asymmetric inputs are
        symmetrized via ``½ (W + W^T)``. The diagonal is zeroed before
        the Laplacian is computed (self-loops carry no signal in
        MaxCut).

    Returns
    -------
    (problem_dict, metadata)
        ``problem_dict`` is the canonical CONE-form dict (``P`` set to
        None, ``q`` / ``A`` / ``b`` / ``cone`` populated). ``metadata``
        carries ``num_nodes``, ``total_weight`` (``sum W / 2``), and
        ``maxcut_constant`` (``¼ sum W``) so callers can recover the
        absolute MaxCut SDP value from ``-q'x + maxcut_constant``.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(
            f"MaxCut weights must be a square matrix; got shape {weights.shape}"
        )
    n = int(weights.shape[0])
    sym = 0.5 * (weights + weights.T)
    np.fill_diagonal(sym, 0.0)
    laplacian = np.diag(sym.sum(axis=1)) - sym
    c_matrix = 0.25 * laplacian

    triangle_size = n * (n + 1) // 2
    sqrt2 = float(np.sqrt(2.0))

    b_dense = np.zeros(triangle_size, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for j in range(n):
        for i in range(j, n):
            idx = _psd_triangle_index(n, i, j)
            value = c_matrix[i, j]
            if i == j:
                b_dense[idx] = value
            else:
                # Off-diagonals carry the canonical √2 scaling so the
                # canonical inner product matches the SDPA convention.
                b_dense[idx] = sqrt2 * value

    # Each diagonal-fix constraint A_k = e_k e_k^T contributes a
    # single non-zero in the canonical vec at the (k, k) diagonal
    # position (no √2 because diagonal entries are unscaled).
    for k in range(n):
        rows.append(_psd_triangle_index(n, k, k))
        cols.append(k)
        data.append(1.0)

    a_matrix = sp.csc_matrix(
        (data, (rows, cols)),
        shape=(triangle_size, n),
    )
    cone = {"s": [n]}
    problem = {
        "P": None,
        "q": -np.ones(n),  # min -1'y  ⇔  max 1'y  ⇔  primal ¼ trace(L X)
        "r": 0.0,
        "A": a_matrix,
        "b": b_dense,
        "n": n,
        "m": int(triangle_size),
        "cone": cone,
        "obj_type": "min",
    }
    metadata = {
        "num_nodes": n,
        "total_weight": float(0.5 * sym.sum()),
        # The Goemans-Williamson identity:
        #   max sum_{i<j} W[i,j] (1 − X[i,j]) / 2 + constant
        # gives ``¼ sum W`` as the constant offset. Callers can
        # recover the absolute MaxCut SDP value from
        # ``primal = -q'x + maxcut_constant``.
        "maxcut_constant": float(0.25 * sym.sum()),
    }
    return problem, metadata


def _psd_triangle_index(order: int, i: int, j: int) -> int:
    """Index of ``X[i, j]`` in a column-major lower-triangle vec."""
    if i < j:
        i, j = j, i
    return j * order - j * (j - 1) // 2 + (i - j)
