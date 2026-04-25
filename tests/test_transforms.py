"""Tests for QP-to-cone transforms."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.transforms.cones import (
    INF_BOUND,
    qp_to_nonnegative_cone,
    qp_to_scs_box_cone,
    split_qp_bounds,
    unbox_scs_dual,
)


def _sample_mixed_qp():
    # 3 rows: one equality (row 0), one upper-bounded inequality (row 1),
    # one two-sided inequality (row 2).
    return {
        "P": sp.csc_matrix([[2.0, 0.0], [0.0, 2.0]]),
        "q": np.array([1.0, -1.0]),
        "A": sp.csc_matrix(
            [
                [1.0, 1.0],   # equality
                [2.0, 0.0],   # -inf <= 2 x1 <= 5
                [0.0, 3.0],   # -1 <= 3 x2 <= 4
            ]
        ),
        "l": np.array([3.0, -INF_BOUND, -1.0]),
        "u": np.array([3.0, 5.0, 4.0]),
        "n": 2,
        "m": 3,
    }


def test_split_qp_bounds_classifies_rows():
    qp = _sample_mixed_qp()
    _, _, _, eq, finite_l, finite_u = split_qp_bounds(qp)
    assert list(eq) == [True, False, False]
    assert list(finite_u) == [False, True, True]
    assert list(finite_l) == [False, False, True]


def test_qp_to_nonnegative_cone_row_layout():
    qp = _sample_mixed_qp()
    a_cone, b, z = qp_to_nonnegative_cone(qp)

    # First z rows are the equalities, then upper bounds (u - Ax >= 0),
    # then lower bounds encoded as (-A x >= -l).
    assert z == 1
    assert a_cone.shape == (4, 2)
    assert b.shape == (4,)

    expected = np.array(
        [
            [1.0, 1.0],    # equality (row 0)
            [2.0, 0.0],    # finite_u row 1
            [0.0, 3.0],    # finite_u row 2
            [0.0, -3.0],   # finite_l row 2 -> negated
        ]
    )
    assert np.allclose(a_cone.toarray(), expected)
    assert np.allclose(b, np.array([3.0, 5.0, 4.0, 1.0]))


def test_qp_to_nonnegative_cone_all_equality():
    qp = {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([0.0]),
        "A": sp.csc_matrix([[1.0]]),
        "l": np.array([2.0]),
        "u": np.array([2.0]),
        "n": 1,
        "m": 1,
    }
    a_cone, b, z = qp_to_nonnegative_cone(qp)
    assert z == 1
    assert a_cone.shape == (1, 1)
    assert np.allclose(b, [2.0])


def test_qp_to_nonnegative_cone_empty():
    # All bounds infinite -> no cone rows.
    qp = {
        "P": sp.csc_matrix((1, 1)),
        "q": np.array([0.0]),
        "A": sp.csc_matrix([[1.0]]),
        "l": np.array([-INF_BOUND]),
        "u": np.array([INF_BOUND]),
        "n": 1,
        "m": 1,
    }
    a_cone, b, z = qp_to_nonnegative_cone(qp)
    assert z == 0
    assert a_cone.shape == (0, 1)
    assert b.shape == (0,)


def test_qp_to_scs_box_cone_all_equality():
    qp = {
        "P": sp.csc_matrix((2, 2)),
        "q": np.array([1.0, 2.0]),
        "A": sp.csc_matrix([[1.0, 0.0], [0.0, 1.0]]),
        "l": np.array([3.0, 4.0]),
        "u": np.array([3.0, 4.0]),
        "n": 2,
        "m": 2,
    }
    data, cone, inv_perm = qp_to_scs_box_cone(qp)
    assert cone == {"z": 2}
    assert data["A"].shape == (2, 2)
    assert np.allclose(data["b"], [3.0, 4.0])
    assert np.array_equal(inv_perm, np.arange(2))


def test_qp_to_scs_box_cone_mixed_structure():
    qp = _sample_mixed_qp()
    data, cone, inv_perm = qp_to_scs_box_cone(qp)
    # Equalities first, then a dummy row for the box cone 't' component,
    # then negated inequality rows.
    assert cone["z"] == 1
    assert cone["bl"] == [-INF_BOUND, -1.0]
    assert cone["bu"] == [5.0, 4.0]
    # Rows: 1 eq + 1 box-t + 2 inequalities.
    assert data["A"].shape == (4, 2)
    assert np.allclose(data["b"], [3.0, 1.0, 0.0, 0.0])
    # inv_perm permutes SCS row ordering back to original row ordering.
    assert inv_perm.shape == (3,)


def test_unbox_scs_dual_all_equality_is_identity():
    cone = {"z": 2}
    inv_perm = np.array([0, 1])
    y_scs = np.array([0.5, -0.25])
    y_qp = unbox_scs_dual(y_scs, cone, inv_perm)
    assert np.allclose(y_qp, y_scs)


def test_unbox_scs_dual_round_trips_permutation_and_sign():
    # Two equality rows + two inequality rows; SCS layout is [eq, t, ineq].
    # We only need a self-consistent cone+perm here.
    cone = {"z": 2, "bl": [-INF_BOUND, 0.0], "bu": [1.0, 5.0]}
    # Original order (0..3): eq, ineq, eq, ineq.
    # Reordered to [eq, eq, ineq, ineq] then inv_perm permutes back.
    inv_perm = np.array([0, 2, 1, 3])
    # SCS dual: [y_eq, y_eq, t_dummy, -y_ineq, -y_ineq] (5 entries total).
    y_scs = np.array([0.1, -0.2, 99.0, -0.5, 0.4])
    y_qp = unbox_scs_dual(y_scs, cone, inv_perm)
    # The '99.0' box-t entry is removed; remaining ineq entries are sign-flipped,
    # then inv_perm remaps to original row order.
    # After dropping index 2 (the 't'): [0.1, -0.2, -0.5, 0.4]
    # After flipping the inequality tail: [0.1, -0.2, 0.5, -0.4]
    # After inv_perm=[0,2,1,3]: pick indices [0,2,1,3] -> [0.1, 0.5, -0.2, -0.4]
    assert np.allclose(y_qp, [0.1, 0.5, -0.2, -0.4])


def test_unbox_scs_dual_none_passthrough():
    assert unbox_scs_dual(None, {"z": 0}, np.array([])) is None
