"""MaxCut SDP relaxation construction.

Given a weighted graph with weight matrix ``W`` (symmetric,
non-negative, zero diagonal), the standard Goemans-Williamson SDP
relaxation of MaxCut is::

    maximize    ¼ * trace(L X)
    subject to  diag(X) = 1
                X ⪰ 0

where ``L = diag(W·1) − W`` is the graph Laplacian. The Lagrangian
gives the dual::

    minimize    1' y
    subject to  Diag(y) − ¼ L  ⪰  0

In the codebase's CONE form (``min q' x s.t. A x + s = b, s in K``)
this is encoded with::

    q = +ones(n)
    A_k = -e_k e_k^T   (negative diagonal selector for each k)
    b   = -vec(¼ L)    (canonical √2-scaled PSD layout)

so that ``s = b - A x = -¼L + Diag(y) = Diag(y) - ¼L``, which lies
in the PSD cone exactly when ``Diag(y) ⪰ ¼L``. By LP/SDP duality
the optimum of the dual equals the optimum of the primal, so::

    q' x  =  1' y  =  max ¼ tr(L X)  =  MaxCut SDP relaxation value

Pre-fix this module flipped all three signs (``q = -1``, ``b = +¼L``,
positive A selectors), encoding the dual of the **opposite**
problem (``min ¼ tr(L X)``). The reported objective then had no
useful relationship to the MaxCut bound.

Note on the Goemans-Williamson identity:
    cut value = (1/2) sum_{i<j} W[i,j] − (1/4) tr(W X)
              = ¼ tr(L X)             [since diag(X) = 1 and L = D − W]

so ``¼ tr(L X)`` is the SDP value directly — there is no separate
constant offset to recover.
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
        ``problem_dict`` is the canonical CONE-form dict; the solver's
        ``q' x`` at optimum equals ``max ¼ tr(L X)`` directly (no
        constant offset). ``metadata`` carries ``num_nodes`` and
        ``total_weight`` (``sum_{i<j} W[i,j]``, useful for
        normalizing the bound against a trivial ``cut ≤ total``
        baseline).
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

    # b = -vec(¼ L) so that s = b - A x = Diag(y) - ¼L ∈ PSD; the
    # √2 scaling is the canonical PSD-vec convention.
    b_dense = np.zeros(triangle_size, dtype=float)
    for j in range(n):
        for i in range(j, n):
            idx = _psd_triangle_index(n, i, j)
            value = -c_matrix[i, j]
            if i == j:
                b_dense[idx] = value
            else:
                b_dense[idx] = sqrt2 * value

    # A's k-th column is -e_k e_k^T in canonical PSD vec, so
    # A x = -Diag(y), and s = b - A x = -¼L + Diag(y).
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for k in range(n):
        rows.append(_psd_triangle_index(n, k, k))
        cols.append(k)
        data.append(-1.0)

    a_matrix = sp.csc_matrix(
        (data, (rows, cols)),
        shape=(triangle_size, n),
    )
    cone = {"s": [n]}
    problem = {
        "P": None,
        "q": np.ones(n),  # min 1'y; at optimum, q'x = SDP relaxation value.
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
        # ``total_weight`` is the sum of edge weights (each undirected
        # edge counted once); a trivial upper bound for any cut is
        # ``total_weight`` (cut everything).
        "total_weight": float(0.5 * sym.sum()),
    }
    return problem, metadata


def _psd_triangle_index(order: int, i: int, j: int) -> int:
    """Index of ``X[i, j]`` in a column-major lower-triangle vec."""
    if i < j:
        i, j = j, i
    return j * order - j * (j - 1) // 2 + (i - j)
