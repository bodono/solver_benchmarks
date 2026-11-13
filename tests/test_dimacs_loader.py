"""DIMACS files are posed as the problem the file states.

Two things are pinned here. The rows of a cone problem follow one fixed
layout (zero, nonnegative, second-order, PSD), whatever order a dictionary
lists its keys in, and a PSD block is the lower triangle of the symmetric
part with the off-diagonal entries times ``sqrt(2)``. Both went wrong in the
DIMACS loader, so the bundled problems are also solved and compared with the
optima published for the 7th DIMACS challenge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core.problem import (
    CONE,
    QP,
    ProblemData,
    cone_items_in_row_order,
    cone_row_count,
)
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.datasets.dimacs import _read_dimacs
from solver_benchmarks.solvers.registry import get_solver
from solver_benchmarks.transforms.psd import (
    cone_row_perm_canonical_to_row_major,
    cone_vec_canonical_to_row_major,
)

SQRT2 = np.sqrt(2.0)

# max b'y (= min c'x) as published for the 7th DIMACS challenge. The harness
# minimises -b'y, so a solver's objective is minus this number.
PUBLISHED_OPTIMUM = {
    "nb_L1": -13.01227,
    "qssp30": -6.496675,
    "sched_50_50_scaled": 7.852038,
    "truss5": 132.6356779,
    "copo14": 0.0,
}


def _write_sedumi(path: Path, at: np.ndarray, b: np.ndarray, c: np.ndarray, k: dict) -> Path:
    """Write a SeDuMi-format file. ``at`` has one row per cone entry."""
    scipy.io.savemat(
        path,
        {
            "A": sp.csc_matrix(at.T),
            "b": np.asarray(b, dtype=float).reshape(-1, 1),
            "c": np.asarray(c, dtype=float).reshape(-1, 1),
            "K": {name: np.asarray(value, dtype=float).reshape(1, -1) for name, value in k.items()},
        },
    )
    return path


def test_rows_stay_in_sedumi_order_whatever_order_k_lists_its_fields(tmp_path: Path):
    # SeDuMi stacks free, linear, second-order, PSD. The fields of K are
    # stored here as q, s, l, f, the order must not leak into the problem.
    rng = np.random.default_rng(0)
    rows = 2 + 3 + (3 + 4) + 4
    at = rng.standard_normal((rows, 5))
    at[-3] = at[-2]  # the 2 x 2 PSD block is symmetric as stored
    c = rng.standard_normal(rows)
    c[-3] = c[-2]
    b = rng.standard_normal(5)
    path = _write_sedumi(
        tmp_path / "mixed.mat", at, b, c, {"q": [3, 4], "s": [2], "l": [3], "f": [2]}
    )

    problem = _read_dimacs(path)

    assert list(problem["cone"]) == ["z", "l", "q", "s"]
    assert problem["cone"] == {"z": 2, "l": 3, "q": [3, 4], "s": [2]}
    linear = 2 + 3 + 3 + 4
    np.testing.assert_array_equal(problem["A"].toarray()[:linear], at[:linear])
    np.testing.assert_array_equal(problem["b"][:linear], c[:linear])
    np.testing.assert_array_equal(problem["q"], -b)
    assert problem["A"].shape == (linear + 3, 5)
    ProblemData("dimacs", "mixed", CONE, problem)  # passes the layout check


def test_psd_block_is_the_scaled_lower_triangle(tmp_path: Path):
    # One 3 x 3 block. Row j * 3 + i of the file is entry (i, j).
    rng = np.random.default_rng(1)
    matrices = []
    for _ in range(3):  # C, A_1, A_2
        m = rng.standard_normal((3, 3))
        matrices.append(m + m.T)
    c_mat, a_mats = matrices[0], matrices[1:]
    at = np.column_stack([a.reshape(-1, order="F") for a in a_mats])
    path = _write_sedumi(
        tmp_path / "psd.mat", at, np.array([1.0, 2.0]), c_mat.reshape(-1, order="F"), {"s": [3]}
    )

    problem = _read_dimacs(path)

    def triangle(m: np.ndarray) -> np.ndarray:
        # (0,0) (1,0) (2,0) (1,1) (2,1) (2,2), off-diagonal times sqrt 2.
        return np.array(
            [m[0, 0], SQRT2 * m[1, 0], SQRT2 * m[2, 0], m[1, 1], SQRT2 * m[2, 1], m[2, 2]]
        )

    assert problem["cone"] == {"s": [3]}
    np.testing.assert_allclose(problem["b"], triangle(c_mat), rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        problem["A"].toarray(),
        np.column_stack([triangle(a) for a in a_mats]),
        rtol=0,
        atol=1e-15,
    )
    # The triangle keeps the inner product: <C, A_1> = b' a_1.
    assert problem["b"] @ problem["A"].toarray()[:, 0] == pytest.approx(np.sum(c_mat * a_mats[0]))


def test_psd_block_stored_as_one_triangle_is_read_as_its_symmetric_part(tmp_path: Path):
    # SeDuMi asks that the symmetric part of c - A'y be PSD. Data that is not
    # symmetric as stored (in fap09 and fap25 each constraint matrix is a
    # single entry) is read that way; forcing the two triangles to agree
    # instead, as the loader used to, is a different problem in general.
    upper_only = np.array([[1.0, 4.0], [0.0, 3.0]])
    c_mat = np.array([[2.0, 0.0], [6.0, 5.0]])
    path = _write_sedumi(
        tmp_path / "onetriangle.mat",
        upper_only.reshape(-1, 1, order="F"),
        np.array([1.0]),
        c_mat.reshape(-1, order="F"),
        {"s": [2]},
    )

    problem = _read_dimacs(path)

    assert problem["cone"] == {"s": [2]}
    np.testing.assert_allclose(problem["A"].toarray().ravel(), [1.0, SQRT2 * 2.0, 3.0])
    np.testing.assert_allclose(problem["b"], [2.0, SQRT2 * 3.0, 5.0])


def test_cone_sizes_must_account_for_every_row(tmp_path: Path):
    at = np.ones((6, 2))
    path = _write_sedumi(tmp_path / "short.mat", at, np.ones(2), np.ones(6), {"l": [2], "s": [2]})
    path_ok = _write_sedumi(tmp_path / "ok.mat", at, np.ones(2), np.ones(6), {"l": [2], "q": [4]})

    with pytest.raises(ValueError, match="K accounts for 6 rows"):
        _read_dimacs(_write_sedumi(tmp_path / "long.mat", np.ones((7, 2)), np.ones(2), np.ones(7), {"l": [2], "s": [2]}))
    assert _read_dimacs(path)["cone"] == {"l": 2, "s": [2]}
    assert _read_dimacs(path_ok)["cone"] == {"l": 2, "q": [4]}


def test_rotated_cones_are_refused(tmp_path: Path):
    path = _write_sedumi(tmp_path / "r.mat", np.ones((3, 1)), np.ones(1), np.ones(3), {"r": [3]})
    with pytest.raises(ValueError, match="rotated"):
        _read_dimacs(path)


@pytest.mark.parametrize("name", sorted(PUBLISHED_OPTIMUM))
def test_bundled_problem_reaches_its_published_optimum(name: str):
    # nb_L1, qssp30 and sched_50_50_scaled have linear and second-order rows,
    # the pair that used to reach Clarabel in the wrong order (it reported
    # dual infeasibility on two of them and an optimum of 0 on qssp30).
    # truss5 and copo14 have PSD blocks, which used to be scaled wrongly
    # (truss5 came out as 187.575, sqrt 2 times its optimum).
    pytest.importorskip("clarabel")
    from solver_benchmarks.core import status

    problem = get_dataset("dimacs")().load_problem(name)
    result = get_solver("clarabel")({"verbose": False}).solve(problem, Path("."))

    assert result.status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}
    assert -result.objective_value == pytest.approx(PUBLISHED_OPTIMUM[name], rel=2e-6, abs=2e-6)
    assert result.kkt["primal_cone_res"] < 1e-8
    assert result.kkt["dual_cone_res"] < 1e-8


# Small on disk and in memory (the two order-3375 torus files are small on
# disk and take about a gigabyte to convert). Every cone combination in the
# bundle is here: l+q, l+s, l+q+s, s alone, several blocks, order-1 blocks.
SMALL_PROBLEMS = (
    "copo14", "filter48_socp", "filtinf1", "hinf12", "hinf13", "minphase", "nb", "nb_L1",
    "nql30", "qssp30", "qssp30old", "sched_50_50_orig", "sched_50_50_scaled", "truss5",
)


def test_bundled_problems_have_a_consistent_layout():
    # The problems are converted and checked, not solved.
    dataset = get_dataset("dimacs")()
    available = {spec.name for spec in dataset.list_problems()}
    assert set(SMALL_PROBLEMS) <= available
    for name in SMALL_PROBLEMS:
        problem = dataset.load_problem(name).cone
        cone = problem["cone"]
        assert list(cone) == [key for key in ("z", "l", "q", "s") if key in cone], name
        assert cone_row_count(cone) == problem["A"].shape[0] == problem["b"].size, name
        assert all(cone.values()), name  # no empty cone is listed
    kinds = {tuple(dataset.load_problem(name).cone["cone"]) for name in SMALL_PROBLEMS}
    assert kinds == {("l", "q"), ("l", "s"), ("l", "q", "s"), ("s",)}


# ---------------------------------------------------------------------------
# The row layout is fixed; no consumer may follow the dictionary's key order.
# ---------------------------------------------------------------------------


def _mixed_cone_point():
    """A point with a different residual in every cone, so that a consumer
    that reads the rows in the wrong order gets a different answer."""
    cone = {"z": 1, "l": 2, "q": [3], "s": [2]}
    s = np.array([0.0, 1.0, -2.0, 1.0, 3.0, 0.0, 1.0, 0.0, -1.0])
    y = np.array([5.0, -1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 3.0])
    return cone, s, y


@pytest.mark.parametrize(
    "order",
    [("q", "l", "s", "z"), ("s", "q", "l", "z"), ("l", "q", "s", "z"), ("z", "s", "l", "q")],
)
def test_kkt_cone_residuals_do_not_depend_on_key_order(order):
    cone, s, y = _mixed_cone_point()
    m = s.size
    a = sp.eye(m, 2, format="csc")
    x = np.array([1.0, -1.0])
    b = a @ x + s
    q = -(a.T @ y)
    shuffled = {key: cone[key] for key in order}

    p = sp.csc_matrix((2, 2))

    expected = kkt.cone_residuals(p, q, a, b, cone, x, y, s)
    got = kkt.cone_residuals(p, q, a, b, shuffled, x, y, s)

    assert expected["primal_cone_res"] > 0.5 and expected["dual_cone_res"] > 0.5
    for key in ("primal_cone_res", "dual_cone_res", "comp_slack", "primal_res", "dual_res"):
        assert got[key] == expected[key], key


def test_kkt_reads_the_zero_cone_first_when_an_adapter_appends_it():
    # The SCS adapter renames the free cone "f" to "z", which puts the key
    # last in its dictionary while the rows stay first.
    s = np.array([0.0, 0.0, 2.0, 1.0])
    y = np.array([-3.0, -4.0, 0.0, 0.0])
    a = sp.eye(4, 1, format="csc")
    x = np.array([0.0])
    result = kkt.cone_residuals(
        sp.csc_matrix((1, 1)), -(a.T @ y), a, s, {"l": 2, "z": 2}, x, y, s
    )
    assert result["primal_cone_res"] == 0.0
    assert result["dual_cone_res"] == 0.0


def test_clarabel_and_sdpa_cone_parsers_do_not_depend_on_key_order():
    from solver_benchmarks.solvers.sdpa_adapter import _cone_blocks

    canonical = {"z": 1, "l": 2, "q": [3], "s": [2]}
    shuffled = {"q": [3], "s": [2], "l": 2, "f": 1}
    assert _cone_blocks(shuffled) == _cone_blocks(canonical)
    assert _cone_blocks(canonical) == [("f", 0, 1, 1), ("l", 1, 2, 2), ("q", 3, 3, 3), ("s", 6, 3, 2)]

    clarabel = pytest.importorskip("clarabel")
    from solver_benchmarks.solvers.clarabel_adapter import _parse_cones

    cones, keep_rows, cone_dict = _parse_cones(shuffled, 9, clarabel)
    assert [type(cone).__name__ for cone in cones] == [
        "ZeroConeT", "NonnegativeConeT", "SecondOrderConeT", "PSDTriangleConeT",
    ]
    assert keep_rows == list(range(9))
    assert cone_dict == canonical


def test_psd_row_permutation_does_not_depend_on_key_order():
    canonical = {"l": 2, "q": [3], "s": [3]}
    shuffled = {"s": [3], "q": [3], "l": 2}
    m = 2 + 3 + 6
    v = np.arange(float(m))

    np.testing.assert_array_equal(
        cone_row_perm_canonical_to_row_major(shuffled, m),
        cone_row_perm_canonical_to_row_major(canonical, m),
    )
    np.testing.assert_array_equal(
        cone_vec_canonical_to_row_major(v, shuffled), cone_vec_canonical_to_row_major(v, canonical)
    )
    # Only the PSD rows move.
    assert list(cone_row_perm_canonical_to_row_major(canonical, m)[:5]) == [0, 1, 2, 3, 4]
    # Box rows come before the PSD rows and are not sized by these helpers.
    box = {"bl": [0.0], "bu": [1.0], "s": [2]}
    with pytest.raises(ValueError, match="box"):
        cone_row_perm_canonical_to_row_major(box, 5)
    with pytest.raises(ValueError, match="box"):
        cone_vec_canonical_to_row_major(np.zeros(5), box)


def test_cone_items_in_row_order():
    # SCS's order: z, l, box, q, s, cs, ep, ed, p; unknown keys last.
    cone = {
        "mystery": 1, "s": [2], "ed": 1, "q": [3], "bu": [1.0], "bl": [0.0], "cs": [2],
        "l": 2, "ep": 1, "f": 1, "p": [0.5],
    }
    assert [key for key, _ in cone_items_in_row_order(cone)] == [
        "f", "l", "bu", "bl", "q", "s", "cs", "ep", "ed", "p", "mystery",
    ]
    del cone["mystery"]
    assert cone_row_count(cone) == 1 + 2 + 2 + 3 + 3 + 4 + 3 + 3 + 3
    assert cone_row_count({"l": 2, "mystery": 4}) is None


def test_cone_row_count_of_a_box():
    # One row for the scaling variable plus one per bound.
    assert cone_row_count({"bl": [0.0, 0.0], "bu": [1.0, 1.0]}) == 3
    assert cone_row_count({"z": 1, "bl": [0.0, 0.0, 0.0], "bu": [1.0, 1.0, 1.0], "q": [3]}) == 8
    # An empty box takes no rows, as in SCS.
    assert cone_row_count({"l": 2, "bl": [], "bu": []}) == 2
    # One-sided or uneven: the longer side sets the width.
    assert cone_row_count({"bl": [0.0, 0.0]}) == 3
    assert cone_row_count({"bl": [0.0], "bu": [1.0, 2.0]}) == 3


def test_kkt_takes_no_rows_for_an_empty_box():
    s = np.array([1.0, 2.0, 1.0, 0.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 0.0, -1.0])
    a = sp.eye(5, 1, format="csc")
    x = np.array([0.0])
    cone = {"l": 2, "bl": [], "bu": [], "q": [3]}
    with_box = kkt.cone_residuals(sp.csc_matrix((1, 1)), -(a.T @ y), a, s, cone, x, y, s)
    without = kkt.cone_residuals(
        sp.csc_matrix((1, 1)), -(a.T @ y), a, s, {"l": 2, "q": [3]}, x, y, s
    )
    assert with_box["primal_cone_res"] == without["primal_cone_res"] == 0.0
    assert with_box["dual_cone_res"] == without["dual_cone_res"] == 0.0


def _cone_problem(cone: dict, rows: int) -> dict:
    return {
        "P": None, "A": sp.csc_matrix((rows, 2)), "b": np.zeros(rows), "q": np.zeros(2),
        "r": 0.0, "n": 2, "m": rows, "cone": cone, "obj_type": "min",
    }


def test_problem_data_refuses_cone_keys_out_of_row_order():
    with pytest.raises(ValueError, match="not in row order"):
        ProblemData("d", "p", CONE, _cone_problem({"q": [3], "l": 2}, 5))
    with pytest.raises(ValueError, match="not in row order"):
        ProblemData("d", "p", CONE, _cone_problem({"l": 2, "z": 1}, 3))
    ProblemData("d", "p", CONE, _cone_problem({"z": 1, "l": 2, "q": [3]}, 6))


def test_problem_data_refuses_a_cone_that_does_not_account_for_the_rows():
    with pytest.raises(ValueError, match="accounts for 5 rows but A has 6"):
        ProblemData("d", "p", CONE, _cone_problem({"l": 2, "q": [3]}, 6))
    short_b = _cone_problem({"l": 5}, 5)
    short_b["b"] = np.zeros(3)
    with pytest.raises(ValueError, match="b has 3 entries but A has 5 rows"):
        ProblemData("d", "p", CONE, short_b)
    # A key the harness cannot size is not checked, and neither is a QP.
    ProblemData("d", "p", CONE, _cone_problem({"l": 2, "mystery": 3}, 6))
    ProblemData("d", "p", QP, {"cone": {"q": [3], "l": 2}})


def test_results_from_before_the_corrected_conversion_are_not_resumed():
    from solver_benchmarks.core.config import _signature_dataset_options

    # Signatures recorded before the change hashed the options as given.
    assert _signature_dataset_options("dimacs", {}) != {}
    assert _signature_dataset_options("dimacs", {"subset": "all"}) != {"subset": "all"}
    assert _signature_dataset_options("dimacs", {"subset": "all"})["subset"] == "all"
