"""Round-trip tests for the QP-split helpers used by piqp/proxqp.

These helpers were entirely uncovered before; a sign error in
combine_qp_duals would silently feed the wrong dual into KKT.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.solvers.qp_split import (
    combine_qp_duals,
    dual_from_lower_upper,
    split_qp_for_range_constraints,
)


def _qp_with_eq_and_ineq() -> dict:
    return {
        "P": sp.csc_matrix(np.eye(2)),
        "q": np.array([0.0, 0.0]),
        "A": sp.csc_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([1.0, 0.0, -np.inf]),
        "u": np.array([1.0, np.inf, 2.0]),
        "n": 2,
        "m": 3,
        "obj_type": "min",
    }


def test_split_qp_separates_eq_and_ineq_rows():
    qp = _qp_with_eq_and_ineq()
    p, q, aeq, b, g, h_l, h_u, eq_idx, ineq_idx = split_qp_for_range_constraints(qp)
    assert p.shape == (2, 2)
    assert q.tolist() == [0.0, 0.0]
    assert aeq is not None
    assert aeq.toarray().tolist() == [[1.0, 1.0]]
    assert b.tolist() == [1.0]
    assert g is not None
    # Two ineq rows preserved.
    assert g.toarray().tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert h_l.tolist() == [0.0, -np.inf]
    assert h_u.tolist() == [np.inf, 2.0]
    assert eq_idx.tolist() == [0]
    assert ineq_idx.tolist() == [1, 2]


def test_combine_qp_duals_round_trip_preserves_row_alignment():
    """eq + ineq slots must be deposited at the original row indices."""
    qp = _qp_with_eq_and_ineq()
    _, _, _, _, _, _, _, eq_idx, ineq_idx = split_qp_for_range_constraints(qp)
    eq_dual = np.array([3.0])
    ineq_dual = np.array([1.0, -2.0])
    y = combine_qp_duals(qp["A"].shape[0], eq_idx, eq_dual, ineq_idx, ineq_dual)
    assert y is not None
    # Row 0 (eq) -> 3.0, row 1 (first ineq) -> 1.0, row 2 (second ineq) -> -2.0.
    assert y.tolist() == [3.0, 1.0, -2.0]


def test_combine_qp_duals_returns_none_when_both_missing():
    assert combine_qp_duals(3, np.array([0]), None, np.array([1, 2]), None) is None


def test_dual_from_lower_upper_handles_one_sided_inputs():
    # OSQP-style returns lower / upper duals separately; the canonical
    # QP dual is upper - lower.
    z_l = np.array([0.0, 1.0])
    z_u = np.array([2.0, 0.0])
    assert dual_from_lower_upper(z_l, z_u).tolist() == [2.0, -1.0]
    # Either side absent should not crash; absent side acts as zero.
    assert dual_from_lower_upper(None, np.array([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert dual_from_lower_upper(np.array([1.0, 2.0]), None).tolist() == [-1.0, -2.0]
    assert dual_from_lower_upper(None, None) is None
