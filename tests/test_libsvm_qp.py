"""Tests for the libsvm_qp dataset adapter.

Synthesizes tiny LIBSVM-format files so the tests run without network
access. Covers parsing, both QP shapes (SVM dual + Markowitz), the
linear and RBF kernels, the subset filter, the integrity-check on
download, and the registry registration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Synthetic LIBSVM data fixtures.
# ---------------------------------------------------------------------------


# Tiny binary-classification dataset:
# - 4 samples, 3 features
# - class +1 vs -1
# - sparse representation (some entries zero)
TINY_LIBSVM = """\
+1 1:1.0 2:0.5
+1 1:0.5 3:1.0
-1 1:-0.5 2:1.0
-1 2:-0.5 3:-1.0
"""


# Variant using {0, 1} labels (common in LIBSVM datasets) — adapter
# must coerce to ±1 for the SVM dual.
LIBSVM_ZERO_ONE = """\
1 1:1.0
0 1:-1.0
1 1:0.5
0 1:-0.5
"""


def _write_libsvm(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _local_dataset(folder: Path, **options):
    """Return a LibsvmQPDataset reading from ``folder``.

    Subclasses per-call so test fixtures don't leak the patched
    ``folder`` across tests.
    """
    from solver_benchmarks.datasets.libsvm_qp import LibsvmQPDataset

    class _LocalLIBSVM(LibsvmQPDataset):
        @property
        def folder(self) -> Path:  # type: ignore[override]
            return folder

        @property
        def data_dir(self) -> Path:  # type: ignore[override]
            return folder

    return _LocalLIBSVM(**options)


# ---------------------------------------------------------------------------
# Registry + module setup.
# ---------------------------------------------------------------------------


def test_libsvm_qp_is_registered():
    from solver_benchmarks.datasets import registry

    cls = registry.get_dataset("libsvm_qp")
    assert cls.dataset_id == "libsvm_qp"


# ---------------------------------------------------------------------------
# LIBSVM parser.
# ---------------------------------------------------------------------------


def test_read_libsvm_file_parses_sparse_data(tmp_path: Path):
    from solver_benchmarks.datasets.libsvm_qp import read_libsvm_file

    path = _write_libsvm(tmp_path / "tiny.libsvm", TINY_LIBSVM)
    x, y = read_libsvm_file(path)
    assert x.shape == (4, 3)
    assert y.tolist() == [1.0, 1.0, -1.0, -1.0]
    # Sparse: row 0 has feature 0 = 1.0 and feature 1 = 0.5.
    assert x[0, 0] == 1.0
    assert x[0, 1] == 0.5
    assert x[0, 2] == 0.0  # not specified → zero
    # Row 3 only has features 1 and 2 (LIBSVM 1-indexed → feature 1 maps to col 1).
    assert x[3, 0] == 0.0
    assert x[3, 1] == -0.5
    assert x[3, 2] == -1.0


def test_read_libsvm_file_skips_blank_lines_and_comments(tmp_path: Path):
    from solver_benchmarks.datasets.libsvm_qp import read_libsvm_file

    body = "\n# comment line\n+1 1:1.0\n\n-1 1:-1.0\n"
    path = _write_libsvm(tmp_path / "with_blanks.libsvm", body)
    x, y = read_libsvm_file(path)
    assert x.shape == (2, 1)
    assert y.tolist() == [1.0, -1.0]


# ---------------------------------------------------------------------------
# Download integrity check.
# ---------------------------------------------------------------------------


def test_parse_libsvm_first_line_rejects_html_response():
    """If the LIBSVM server returns an HTML error page, the first
    line won't have a numeric label — we must surface that as a
    ValueError before committing to the cache."""
    from solver_benchmarks.datasets.libsvm_qp import _parse_libsvm_first_line

    body = b"<html>\n<body>Not Found</body>\n</html>\n"
    with pytest.raises(ValueError, match="not a number"):
        _parse_libsvm_first_line(body)


def test_parse_libsvm_first_line_rejects_empty_body():
    from solver_benchmarks.datasets.libsvm_qp import _parse_libsvm_first_line

    with pytest.raises(ValueError, match="empty"):
        _parse_libsvm_first_line(b"\n# only a comment\n")


def test_parse_libsvm_first_line_accepts_valid_data():
    from solver_benchmarks.datasets.libsvm_qp import _parse_libsvm_first_line

    # Should not raise.
    _parse_libsvm_first_line(b"+1 1:0.5\n-1 2:1.0\n")


# ---------------------------------------------------------------------------
# SVM dual QP construction.
# ---------------------------------------------------------------------------


def test_svm_dual_qp_shape_and_constraint_layout():
    from solver_benchmarks.datasets.libsvm_qp import svm_dual_qp

    x = np.array([[1.0, 0.5], [0.5, 1.0], [-0.5, 1.0], [0.0, -0.5]])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    qp = svm_dual_qp(x, y, kernel="linear", c_upper=1.0)
    n = 4
    # Variables: α (one per sample). Constraints: y'α = 0 (1 row) plus
    # 0 ≤ α ≤ C box (n rows).
    assert qp["P"].shape == (n, n)
    assert qp["q"].shape == (n,)
    assert qp["q"].tolist() == [-1.0, -1.0, -1.0, -1.0]  # min -1' α
    assert qp["A"].shape == (n + 1, n)
    # First row is the equality y'α = 0.
    assert qp["l"][0] == 0.0 and qp["u"][0] == 0.0
    # Box rows: 0 ≤ α_i ≤ 1.
    assert (qp["l"][1:] == 0.0).all()
    assert (qp["u"][1:] == 1.0).all()


def test_svm_dual_qp_p_is_psd_and_symmetric():
    """``Q[i,j] = y_i y_j K(x_i, x_j)`` is PSD (the kernel is PSD,
    so the outer y product preserves semidefiniteness). Verify by
    checking eigenvalues."""
    from solver_benchmarks.datasets.libsvm_qp import svm_dual_qp

    rng = np.random.default_rng(0)
    n, d = 6, 3
    x = rng.standard_normal((n, d))
    y = rng.choice([-1.0, 1.0], size=n)
    qp = svm_dual_qp(x, y, kernel="linear")
    p = qp["P"].toarray()
    assert np.allclose(p, p.T)
    eigvals = np.linalg.eigvalsh(p)
    assert eigvals.min() > -1e-10  # PSD up to numerical noise.


def test_svm_dual_qp_linear_kernel_low_rank_when_d_lt_n():
    """For the linear kernel ``Q = D Y X X' Y D`` where Y = diag(y),
    so ``rank(Q) ≤ rank(X X') ≤ d``. With n=6, d=2 we expect rank ≤ 2.
    This is the regime where ECOS' QP→SOCP reformulation hits the
    rank-deficient PSD path."""
    from solver_benchmarks.datasets.libsvm_qp import svm_dual_qp

    rng = np.random.default_rng(42)
    n, d = 6, 2
    x = rng.standard_normal((n, d))
    y = rng.choice([-1.0, 1.0], size=n)
    qp = svm_dual_qp(x, y, kernel="linear")
    p = qp["P"].toarray()
    rank = int(np.sum(np.linalg.eigvalsh(p) > 1e-9))
    assert rank <= d


def test_svm_dual_qp_rbf_kernel_full_rank():
    """RBF kernel typically gives a full-rank Gram matrix."""
    from solver_benchmarks.datasets.libsvm_qp import svm_dual_qp

    rng = np.random.default_rng(0)
    n, d = 5, 3
    x = rng.standard_normal((n, d))
    y = rng.choice([-1.0, 1.0], size=n)
    qp = svm_dual_qp(x, y, kernel="rbf", gamma=0.5)
    p = qp["P"].toarray()
    eigvals = np.linalg.eigvalsh(p)
    assert eigvals.min() > 0.0  # full-rank PSD.


def test_svm_dual_qp_coerces_zero_one_labels():
    """LIBSVM datasets sometimes use {0, 1} or {1, 2} labels instead
    of {-1, +1}; the adapter must coerce to ±1 before forming Q."""
    from solver_benchmarks.datasets.libsvm_qp import svm_dual_qp

    x = np.eye(2)
    y = np.array([0.0, 1.0])  # Should become [-1, +1].
    qp = svm_dual_qp(x, y, kernel="linear")
    # First row of A is the y'α = 0 equality. With coerced y = [-1, +1]:
    a_eq_row = qp["A"].toarray()[0]
    assert a_eq_row.tolist() == [-1.0, 1.0]


def test_svm_dual_qp_rejects_unknown_kernel():
    from solver_benchmarks.datasets.libsvm_qp import svm_dual_qp

    x = np.eye(2)
    y = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match="Unknown kernel"):
        svm_dual_qp(x, y, kernel="laplacian")


# ---------------------------------------------------------------------------
# Markowitz QP construction.
# ---------------------------------------------------------------------------


def test_markowitz_qp_shape_and_constraint_layout():
    from solver_benchmarks.datasets.libsvm_qp import markowitz_qp

    rng = np.random.default_rng(0)
    n_samples, n_assets = 20, 4
    returns = rng.standard_normal((n_samples, n_assets))
    qp = markowitz_qp(returns, risk_aversion=1.0)
    assert qp["P"].shape == (n_assets, n_assets)
    assert qp["q"].shape == (n_assets,)
    # First row: 1' w = 1 equality. Then n_assets nonneg-bound rows.
    assert qp["A"].shape == (n_assets + 1, n_assets)
    assert qp["l"][0] == 1.0 and qp["u"][0] == 1.0
    assert (qp["l"][1:] == 0.0).all()
    assert (qp["u"][1:] == np.inf).all()


def test_markowitz_qp_p_is_sample_covariance():
    """``P`` is the unbiased sample covariance of the rows."""
    from solver_benchmarks.datasets.libsvm_qp import markowitz_qp

    returns = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    qp = markowitz_qp(returns, risk_aversion=1.0)
    expected = np.cov(returns, rowvar=False)
    assert np.allclose(qp["P"].toarray(), expected)
    # q = -mu (with risk_aversion=1).
    assert np.allclose(qp["q"], -returns.mean(axis=0))


def test_markowitz_qp_requires_at_least_two_samples():
    from solver_benchmarks.datasets.libsvm_qp import markowitz_qp

    with pytest.raises(ValueError, match="at least 2 samples"):
        markowitz_qp(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# list_problems / load_problem end-to-end.
# ---------------------------------------------------------------------------


@pytest.fixture
def libsvm_data_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "libsvm_data"
    _write_libsvm(folder / "heart.libsvm", TINY_LIBSVM)
    _write_libsvm(folder / "diabetes.libsvm", TINY_LIBSVM)
    return folder


def test_list_problems_returns_svm_dual_specs_by_default(libsvm_data_folder: Path):
    dataset = _local_dataset(libsvm_data_folder)
    specs = dataset.list_problems()
    names = sorted(spec.name for spec in specs)
    # Default kind is svm_dual; default subset includes heart + diabetes
    # (both present in the fixture).
    assert names == ["svm_dual_diabetes", "svm_dual_heart"]
    for spec in specs:
        assert spec.kind == "qp"
        assert spec.metadata["kind"] == "svm_dual"


def test_list_problems_respects_kind_markowitz(libsvm_data_folder: Path):
    dataset = _local_dataset(libsvm_data_folder, kind="markowitz")
    specs = dataset.list_problems()
    names = sorted(spec.name for spec in specs)
    assert names == ["markowitz_diabetes", "markowitz_heart"]


def test_list_problems_rejects_unknown_kind(libsvm_data_folder: Path):
    dataset = _local_dataset(libsvm_data_folder, kind="exotic")
    with pytest.raises(ValueError, match="Unknown libsvm_qp kind"):
        dataset.list_problems()


def test_list_problems_filters_by_subset(libsvm_data_folder: Path):
    dataset = _local_dataset(libsvm_data_folder, subset="heart")
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["svm_dual_heart"]


def test_load_problem_returns_well_formed_svm_dual(libsvm_data_folder: Path):
    dataset = _local_dataset(libsvm_data_folder)
    pd = dataset.load_problem("svm_dual_heart")
    qp = pd.qp
    n = qp["P"].shape[0]
    # 4 LIBSVM rows in our tiny fixture ⇒ n = 4 dual variables.
    assert n == 4
    # P is symmetric PSD.
    p_dense = qp["P"].toarray()
    assert np.allclose(p_dense, p_dense.T)
    # Constraints layout: 1 equality + n bounds.
    assert qp["A"].shape == (n + 1, n)
    assert pd.metadata["num_samples"] == 4
    assert pd.metadata["num_features"] == 3


def test_load_problem_returns_well_formed_markowitz(libsvm_data_folder: Path):
    dataset = _local_dataset(libsvm_data_folder, kind="markowitz")
    pd = dataset.load_problem("markowitz_heart")
    qp = pd.qp
    # Markowitz vars = num features (assets), not num samples.
    assert qp["P"].shape == (3, 3)
    assert qp["A"].shape == (4, 3)  # 1 equality + 3 nonneg bounds.


def test_load_problem_respects_max_samples_subsampling(libsvm_data_folder: Path):
    """``max_samples`` deterministically subsamples the rows so the
    QP stays small even on larger LIBSVM datasets."""
    # Build a 20-row LIBSVM file; with max_samples=5, the dual QP
    # should have n = 5.
    body_lines = []
    for i in range(20):
        label = "+1" if i % 2 == 0 else "-1"
        body_lines.append(f"{label} 1:{i}.0")
    _write_libsvm(libsvm_data_folder / "heart.libsvm", "\n".join(body_lines) + "\n")
    dataset = _local_dataset(libsvm_data_folder, subset="heart", max_samples=5)
    pd = dataset.load_problem("svm_dual_heart")
    assert pd.qp["P"].shape == (5, 5)
    assert pd.metadata["num_samples"] == 5


# ---------------------------------------------------------------------------
# End-to-end: solve a small SVM-dual QP through one of the QP solvers.
# ---------------------------------------------------------------------------


def test_svm_dual_qp_is_solvable_by_clarabel(tmp_path: Path, libsvm_data_folder: Path):
    """Round-trip: build the QP from synthetic LIBSVM data, solve it
    with Clarabel, and check the result is OPTIMAL with feasible α
    (sums of y_i α_i ≈ 0 and 0 ≤ α ≤ C)."""
    pytest.importorskip("clarabel")
    from solver_benchmarks.core import status
    from solver_benchmarks.solvers.clarabel_adapter import ClarabelSolverAdapter

    dataset = _local_dataset(libsvm_data_folder, subset="heart")
    pd = dataset.load_problem("svm_dual_heart")
    adapter = ClarabelSolverAdapter({"verbose": False})
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = adapter.solve(pd, artifacts)
    assert result.status == status.OPTIMAL
    # Reasonable KKT residuals.
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


def test_markowitz_qp_is_solvable_by_clarabel(tmp_path: Path, libsvm_data_folder: Path):
    pytest.importorskip("clarabel")
    from solver_benchmarks.core import status
    from solver_benchmarks.solvers.clarabel_adapter import ClarabelSolverAdapter

    dataset = _local_dataset(libsvm_data_folder, kind="markowitz", subset="heart")
    pd = dataset.load_problem("markowitz_heart")
    adapter = ClarabelSolverAdapter({"verbose": False})
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = adapter.solve(pd, artifacts)
    assert result.status == status.OPTIMAL
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4


# ---------------------------------------------------------------------------
# _subsample utility.
# ---------------------------------------------------------------------------


def test_subsample_passthrough_when_under_limit():
    from solver_benchmarks.datasets.libsvm_qp import _subsample

    x = sp.csr_matrix(np.eye(3))
    y = np.array([1.0, -1.0, 1.0])
    x_sub, y_sub = _subsample(x, y, max_samples=5)
    assert x_sub.shape == x.shape
    assert (y_sub == y).all()


def test_subsample_takes_evenly_spaced_rows():
    from solver_benchmarks.datasets.libsvm_qp import _subsample

    x = sp.csr_matrix(np.arange(20).reshape(20, 1).astype(float))
    y = np.arange(20, dtype=float)
    x_sub, y_sub = _subsample(x, y, max_samples=5)
    assert x_sub.shape == (5, 1)
    # Step = 20 // 5 = 4, so indices = [0, 4, 8, 12, 16].
    assert y_sub.tolist() == [0.0, 4.0, 8.0, 12.0, 16.0]


# ---------------------------------------------------------------------------
# _normalize_subset.
# ---------------------------------------------------------------------------


def test_normalize_subset_default_returns_curated_set():
    from solver_benchmarks.datasets.libsvm_qp import (
        LIBSVM_DEFAULT_SUBSET,
        _normalize_subset,
    )

    assert _normalize_subset(None) == set(LIBSVM_DEFAULT_SUBSET)
    assert _normalize_subset("default") == set(LIBSVM_DEFAULT_SUBSET)


def test_normalize_subset_accepts_comma_string():
    from solver_benchmarks.datasets.libsvm_qp import _normalize_subset

    assert _normalize_subset("heart, diabetes ") == {"heart", "diabetes"}


def test_normalize_subset_accepts_list():
    from solver_benchmarks.datasets.libsvm_qp import _normalize_subset

    assert _normalize_subset(["heart", "splice"]) == {"heart", "splice"}
