"""PSD triangle vec ordering helpers.

The benchmark suite uses *column-major lower-triangular* (also called
``"triu_col_major"`` for symmetric matrices) as the canonical PSD vec
ordering — this matches SCS and the KKT residual code in
``solver_benchmarks.analysis.kkt``. For ``n=3`` the canonical entries
are ``M[0,0], M[1,0], M[2,0], M[1,1], M[2,1], M[2,2]``.

Solvers that expect *row-major lower-triangular* ordering (Clarabel,
SDPA via ``sdpa-python``) need a per-block permutation. The helpers
below produce that permutation and apply it to flat vectors and to row
slices of cone matrices.
"""

from __future__ import annotations

import numpy as np


def col_major_to_row_major_perm(n: int) -> np.ndarray:
    """Return ``p`` such that ``v_row[p[k]] == v_col[k]`` for all ``k``,
    i.e. ``np.empty_like(v); v_row[p] = v_col`` produces row-major from
    col-major and ``v_col = v_row[p]`` produces col-major from row-major.
    """
    p = np.empty(n * (n + 1) // 2, dtype=np.intp)
    k = 0
    for j in range(n):
        for i in range(j, n):
            p[k] = i * (i + 1) // 2 + j
            k += 1
    return p


def col_major_to_row_major_vec(v: np.ndarray, n: int) -> np.ndarray:
    """Convert a length-``n*(n+1)/2`` col-major-lower vec to row-major-lower."""
    p = col_major_to_row_major_perm(n)
    out = np.empty_like(v)
    out[p] = v
    return out


def row_major_to_col_major_vec(v: np.ndarray, n: int) -> np.ndarray:
    """Convert a length-``n*(n+1)/2`` row-major-lower vec to col-major-lower."""
    p = col_major_to_row_major_perm(n)
    return v[p]


def cone_row_perm_canonical_to_row_major(cone: dict, m: int) -> np.ndarray:
    """Return row index array ``idx`` such that, given canonical (col-major
    lower) cone-form ``A`` and ``b``, ``A[idx, :]`` and ``b[idx]`` give the
    matrix and rhs in *row-major lower* PSD ordering.

    Non-PSD rows map to themselves.
    """
    idx = np.arange(m, dtype=np.intp)
    row = 0
    for name, value in cone.items():
        if name in ("f", "z", "l"):
            row += int(value)
            continue
        if name == "q":
            for dim in _as_list(value):
                row += int(dim)
            continue
        if name == "s":
            for dim in _as_list(value):
                size = int(dim)
                tri = size * (size + 1) // 2
                p = col_major_to_row_major_perm(size)
                # We want: A_row_major[k] = A_canonical[idx_block[k]]
                # where idx_block satisfies "k-th row in row-major form is the
                # canonical row holding the same entry". v_row[p[c]] = v_col[c]
                # means row-major position p[c] holds the same value as
                # col-major position c, so idx_block[p[c]] = c, i.e. argsort(p).
                idx[row : row + tri] = row + np.argsort(p)
                row += tri
            continue
    return idx


def cone_vec_canonical_to_row_major(v: np.ndarray, cone: dict) -> np.ndarray:
    """Permute s-cone blocks of ``v`` from col-major lower to row-major lower."""
    return _apply_cone_vec_perm(v, cone, direction="canonical_to_row_major")


def cone_vec_row_major_to_canonical(v: np.ndarray, cone: dict) -> np.ndarray:
    """Permute s-cone blocks of ``v`` from row-major lower to col-major lower."""
    return _apply_cone_vec_perm(v, cone, direction="row_major_to_canonical")


def _apply_cone_vec_perm(v: np.ndarray, cone: dict, *, direction: str) -> np.ndarray:
    out = np.asarray(v, dtype=float).copy()
    row = 0
    for name, value in cone.items():
        if name in ("f", "z", "l"):
            row += int(value)
            continue
        if name == "q":
            for dim in _as_list(value):
                row += int(dim)
            continue
        if name == "s":
            for dim in _as_list(value):
                size = int(dim)
                tri = size * (size + 1) // 2
                block = v[row : row + tri]
                if direction == "canonical_to_row_major":
                    out[row : row + tri] = col_major_to_row_major_vec(block, size)
                else:
                    out[row : row + tri] = row_major_to_col_major_vec(block, size)
                row += tri
            continue
    return out


def _as_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]
