"""Tests for PSD triangle vec ordering helpers."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.transforms.psd import (
    col_major_to_row_major_perm,
    col_major_to_row_major_vec,
    cone_row_perm_canonical_to_row_major,
    cone_vec_canonical_to_row_major,
    cone_vec_row_major_to_canonical,
    row_major_to_col_major_vec,
)


def _symmetric_matrix(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, n))
    return 0.5 * (a + a.T)


def _col_major_lower_vec(matrix: np.ndarray, scale: bool = False) -> np.ndarray:
    n = matrix.shape[0]
    out = np.empty(n * (n + 1) // 2)
    k = 0
    for j in range(n):
        for i in range(j, n):
            value = matrix[i, j]
            if scale and i != j:
                value *= np.sqrt(2.0)
            out[k] = value
            k += 1
    return out


def _row_major_lower_vec(matrix: np.ndarray, scale: bool = False) -> np.ndarray:
    n = matrix.shape[0]
    out = np.empty(n * (n + 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i, j]
            if scale and i != j:
                value *= np.sqrt(2.0)
            out[k] = value
            k += 1
    return out


def test_col_major_to_row_major_perm_n3():
    # For n=3: col-major lower order is [(0,0),(1,0),(2,0),(1,1),(2,1),(2,2)]
    # row-major lower order is        [(0,0),(1,0),(1,1),(2,0),(2,1),(2,2)].
    # So entry k=2 of col-major (which is (2,0)) lives at row-major index 3.
    p = col_major_to_row_major_perm(3)
    assert list(p) == [0, 1, 3, 2, 4, 5]


def test_col_major_to_row_major_perm_n4_is_inverse_of_argsort():
    p = col_major_to_row_major_perm(4)
    pinv = np.argsort(p)
    # Applying p then argsort(p) round-trips to the identity.
    assert list(p[pinv]) == list(range(p.size))
    assert list(pinv[p]) == list(range(p.size))


def test_vec_round_trip_through_row_major():
    for n in (1, 2, 3, 4, 7):
        m = _symmetric_matrix(n)
        v_col = _col_major_lower_vec(m)
        v_row = col_major_to_row_major_vec(v_col, n)
        assert np.allclose(v_row, _row_major_lower_vec(m))
        v_back = row_major_to_col_major_vec(v_row, n)
        assert np.allclose(v_back, v_col)


def test_cone_row_perm_handles_mixed_layout():
    # Cone has 2 zero rows, then a 3x3 PSD block (6 rows), then 2 nonneg rows.
    cone = {"z": 2, "s": [3], "l": 2}
    idx = cone_row_perm_canonical_to_row_major(cone, 10)
    # Identity for non-PSD rows.
    assert list(idx[:2]) == [0, 1]
    assert list(idx[8:]) == [8, 9]
    # PSD rows are permuted within [2, 8). For n=3, p=[0,1,3,2,4,5];
    # canonical-to-row-major idx is argsort(p) shifted by start.
    p = col_major_to_row_major_perm(3)
    expected = 2 + np.argsort(p)
    assert list(idx[2:8]) == list(expected)


def test_cone_vec_round_trip_mixed_layout():
    cone = {"l": 2, "s": [4], "z": 1}
    rng = np.random.default_rng(7)
    m = _symmetric_matrix(4)
    psd_block = _col_major_lower_vec(m, scale=True)
    canonical = np.concatenate([rng.normal(size=2), psd_block, rng.normal(size=1)])
    row_major = cone_vec_canonical_to_row_major(canonical, cone)
    # Non-PSD entries unchanged.
    assert np.allclose(row_major[:2], canonical[:2])
    assert np.allclose(row_major[-1:], canonical[-1:])
    # PSD entries match the row-major-lower vec of the same matrix.
    assert np.allclose(row_major[2:12], _row_major_lower_vec(m, scale=True))
    back = cone_vec_row_major_to_canonical(row_major, cone)
    assert np.allclose(back, canonical)


def test_cone_vec_handles_multiple_psd_blocks_and_q_block():
    cone = {"q": [3], "s": [2, 3]}
    rng = np.random.default_rng(11)
    m1 = _symmetric_matrix(2)
    m2 = _symmetric_matrix(3)
    canonical = np.concatenate(
        [
            rng.normal(size=3),
            _col_major_lower_vec(m1, scale=True),
            _col_major_lower_vec(m2, scale=True),
        ]
    )
    row_major = cone_vec_canonical_to_row_major(canonical, cone)
    assert np.allclose(row_major[:3], canonical[:3])
    assert np.allclose(row_major[3:6], _row_major_lower_vec(m1, scale=True))
    assert np.allclose(row_major[6:12], _row_major_lower_vec(m2, scale=True))


def test_cone_row_perm_treats_f_and_z_as_skip():
    cone = {"f": 3, "s": [2]}
    idx = cone_row_perm_canonical_to_row_major(cone, 6)
    assert list(idx[:3]) == [0, 1, 2]
    p = col_major_to_row_major_perm(2)
    assert list(idx[3:]) == list(3 + np.argsort(p))
