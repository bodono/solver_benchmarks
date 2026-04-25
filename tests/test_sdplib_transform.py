"""Tests for the SDPLIB JLD2-to-cone transform's PSD vec ordering."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.transforms.sdplib import _col_major_lower_index, _psd_vec


def test_col_major_lower_index_n3():
    # n=3 col-major lower order: [(0,0),(1,0),(2,0),(1,1),(2,1),(2,2)]
    expected = {(0, 0): 0, (1, 0): 1, (2, 0): 2, (1, 1): 3, (2, 1): 4, (2, 2): 5}
    for (i, j), idx in expected.items():
        assert _col_major_lower_index(i, j, 3) == idx
        # Symmetric input (i < j) should map to the same index.
        assert _col_major_lower_index(j, i, 3) == idx


def test_psd_vec_round_trips_to_known_layout():
    # Build a 3x3 symmetric matrix with distinct nonzero entries so we can
    # verify each entry lands at the expected col-major lower position.
    matrix = np.array(
        [
            [10.0, 2.0, 3.0],
            [2.0, 40.0, 5.0],
            [3.0, 5.0, 60.0],
        ]
    )
    vec = _psd_vec(sp.csc_matrix(matrix)).toarray().reshape(-1)
    # Expected col-major lower vec with √2 off-diagonal scaling.
    s = np.sqrt(2.0)
    expected = np.array([10.0, 2.0 * s, 3.0 * s, 40.0, 5.0 * s, 60.0])
    assert np.allclose(vec, expected)


def test_psd_vec_diagonal_only():
    matrix = sp.diags([1.0, 2.0, 3.0, 4.0]).tocsc()
    vec = _psd_vec(matrix).toarray().reshape(-1)
    # For diagonal input the vec has zeros at every off-diagonal position.
    expected = np.zeros(10)
    expected[0] = 1.0  # (0,0)
    expected[4] = 2.0  # (1,1) — col 1 has 3 entries below diag, so diag is at 4
    expected[7] = 3.0  # (2,2) — col 2 has 2 entries below diag, so diag is at 7
    expected[9] = 4.0  # (3,3)
    assert np.allclose(vec, expected)
