"""QP problems derived from LIBSVM datasets.

Two QP shapes are produced from the same LIBSVM ``(X, y)`` data,
which is a common ML benchmark workload pattern absent from the
classical QP test sets (Maros-Meszaros, QPLIB):

1. **SVM dual** (the standard kernel SVM dual)::

       min  ½ α'Q α − 1' α
       s.t. y' α = 0
            0 ≤ α_i ≤ C

   where ``Q[i,j] = y[i] y[j] K(x[i], x[j])`` and K is a kernel
   (linear: ``K = x'z``; RBF: ``K = exp(-γ ‖x − z‖²)``). ``Q`` is
   PSD; for the linear kernel its rank is bounded by the feature
   dimension, so ECOS' eigendecomposition fallback in QP→SOCP gets
   exercised on rank-deficient ``P``.

2. **Markowitz portfolio**::

       min  ½ w'Σ w − μ' w
       s.t. 1' w = 1
            w_i ≥ 0

   where ``Σ`` is the sample covariance of the rows of ``X`` (treated
   as returns) and ``μ`` is the mean. Smaller (``n × n_assets``) than
   the SVM dual and a different conditioning regime (Σ has rank
   ≤ min(n, d), often < n_assets).

Data files are downloaded from the LIBSVM datasets archive on first
use and cached under ``problem_classes/libsvm_data``. The default
subset is the smallest binary-classification datasets so a fresh
prepare completes in a few seconds and the solves run in well under
a second per instance.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec

from .base import Dataset, atomic_write_bytes, validate_gzip_payload

LIBSVM_BASE_URL = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
)

# Default curated subset: small binary classification datasets that
# produce dense small QPs solvable in well under a second by IPM
# solvers. Each entry maps the dataset id (used as the problem name)
# to the LIBSVM filename hosted at ``LIBSVM_BASE_URL``.
LIBSVM_DEFAULT_SUBSET: dict[str, str] = {
    "heart": "heart",
    "breast-cancer": "breast-cancer",
    "australian": "australian",
    "diabetes": "diabetes",
    "ionosphere": "ionosphere_scale",
    "german-numer": "german.numer",
}


class LibsvmQPDataset(Dataset):
    """Generate QP problems from LIBSVM classification / regression data.

    Options:
        kind: ``svm_dual`` (default) or ``markowitz``.
        kernel: ``linear`` (default) or ``rbf``. SVM-dual only.
        gamma: RBF gamma. Default ``1 / num_features``. SVM-dual only.
        C: SVM regularization upper bound. Default 1.0.
        max_samples: subsample rows to at most this many. Default 200
            (keeps the dense kernel matrix small enough for IPM).
        subset: comma-separated list of LIBSVM dataset names, or a
            list. Default ``LIBSVM_DEFAULT_SUBSET``.
        risk_aversion: Markowitz λ (objective is ``½ w'Σ w − λ μ' w``).
            Default 1.0. Markowitz-only.
    """

    dataset_id = "libsvm_qp"
    description = (
        "QP problems derived from LIBSVM data (SVM dual or Markowitz). "
        "Realistic ML / finance workload shapes complementing the "
        "classical Maros-Meszaros / QPLIB sets."
    )
    data_source = (
        "external download from "
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    )
    data_patterns = ("*.libsvm", "*.libsvm.gz")
    prepare_command = "python scripts/prepare_libsvm_qp.py"
    automatic_download = True

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "libsvm_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        subset = _normalize_subset(self.options.get("subset"))
        kind = str(self.options.get("kind", "svm_dual"))
        if kind not in ("svm_dual", "markowitz"):
            raise ValueError(
                f"Unknown libsvm_qp kind {kind!r}. Use 'svm_dual' or 'markowitz'."
            )
        specs: list[ProblemSpec] = []
        for libsvm_name in sorted(subset):
            path = self.folder / f"{libsvm_name}.libsvm"
            if not path.is_file():
                gz_path = path.with_suffix(".libsvm.gz")
                if gz_path.is_file():
                    path = gz_path
                else:
                    continue  # not yet downloaded
            problem_name = f"{kind}_{libsvm_name}"
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=problem_name,
                    kind=QP,
                    path=path,
                    metadata={
                        "source": str(path),
                        "kind": kind,
                        "libsvm_dataset": libsvm_name,
                    },
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        kind = spec.metadata["kind"]
        x, y = read_libsvm_file(spec.path)
        x, y = _subsample(
            x, y, max_samples=int(self.options.get("max_samples", 200))
        )
        if kind == "svm_dual":
            qp = svm_dual_qp(
                x,
                y,
                kernel=str(self.options.get("kernel", "linear")),
                gamma=self.options.get("gamma"),
                c_upper=float(self.options.get("C", 1.0)),
            )
        elif kind == "markowitz":
            qp = markowitz_qp(
                x,
                risk_aversion=float(self.options.get("risk_aversion", 1.0)),
            )
        else:
            raise KeyError(name)
        return ProblemData(
            self.dataset_id,
            name,
            QP,
            qp,
            metadata={
                **dict(spec.metadata),
                "num_samples": int(x.shape[0]),
                "num_features": int(x.shape[1]),
            },
        )

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        if problem_names:
            names = list(problem_names)
        elif all_problems:
            # No "all" remote enumeration; we restrict to the default
            # curated subset since LIBSVM hosts hundreds of datasets,
            # most of which are too large to fit a small-QP benchmark.
            names = list(LIBSVM_DEFAULT_SUBSET)
        else:
            names = list(LIBSVM_DEFAULT_SUBSET)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_libsvm_dataset(name, self.folder)


# ---------------------------------------------------------------------------
# Download + parsing.
# ---------------------------------------------------------------------------


def download_libsvm_dataset(name: str, folder: Path) -> Path:
    """Download a LIBSVM binary-classification dataset into ``folder``.

    The remote files at ``LIBSVM_BASE_URL`` are plain text (no gzip);
    we still validate the response by parsing the first non-empty
    line so a truncated download surfaces immediately.
    """
    remote_filename = LIBSVM_DEFAULT_SUBSET.get(name, name)
    target = folder / f"{name}.libsvm"
    if target.exists():
        return target
    url = f"{LIBSVM_BASE_URL}/{remote_filename}"
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read()
    if remote_filename.endswith(".gz") or url.endswith(".gz"):
        validate_gzip_payload(body)
        # Decompress on disk so list_problems doesn't need to handle gz.
        import gzip as _gzip

        body = _gzip.decompress(body)
    # Validate the file is parsable LIBSVM format before committing.
    _parse_libsvm_first_line(body)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, body)
    return target


def read_libsvm_file(path: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    """Parse a LIBSVM-format file into ``(X, y)``.

    LIBSVM format: each non-empty line is::

        <label> <idx>:<value> <idx>:<value> ...

    Indices are 1-based. We use a manual parser rather than depending
    on sklearn so the dataset has no extra Python dependencies.
    """
    if path.suffix == ".gz" or path.name.endswith(".libsvm.gz"):
        import gzip as _gzip

        with _gzip.open(path, "rb") as handle:
            raw = handle.read()
    else:
        raw = path.read_bytes()
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    labels: list[float] = []
    max_col = 0
    # ``row_idx`` increments only on accepted (non-blank, non-comment)
    # rows so the COO indices stay aligned with the labels list,
    # which is what determines the final matrix dimension.
    row_idx = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        labels.append(float(parts[0]))
        for token in parts[1:]:
            if ":" not in token:
                continue
            col_str, value_str = token.split(":", 1)
            col = int(col_str) - 1  # LIBSVM is 1-indexed.
            if col < 0:
                continue
            value = float(value_str)
            rows.append(row_idx)
            cols.append(col)
            data.append(value)
            if col > max_col:
                max_col = col
        row_idx += 1
    n_rows = len(labels)
    n_cols = max_col + 1
    x = sp.csr_matrix(
        (np.asarray(data, dtype=float), (rows, cols)),
        shape=(n_rows, n_cols),
    )
    y = np.asarray(labels, dtype=float)
    return x, y


def _parse_libsvm_first_line(body: bytes) -> None:
    """Validate that the first non-empty line is well-formed.

    Used as a download-integrity check: a truncated download with a
    valid HTTP 200 will sometimes produce HTML or partial content
    that does not parse as LIBSVM; surface it immediately rather
    than baking the corruption into the cache.
    """
    text = body.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 1:
            raise ValueError("LIBSVM file's first non-empty line is empty.")
        try:
            float(parts[0])
        except ValueError as exc:
            raise ValueError(
                f"LIBSVM file's first label {parts[0]!r} is not a number; "
                "the download may be HTML or partial."
            ) from exc
        for token in parts[1:]:
            if ":" in token:
                col_str, value_str = token.split(":", 1)
                int(col_str)
                float(value_str)
        return
    raise ValueError("LIBSVM file is empty (no non-comment lines).")


# ---------------------------------------------------------------------------
# QP problem construction.
# ---------------------------------------------------------------------------


def svm_dual_qp(
    x: sp.spmatrix | np.ndarray,
    y: np.ndarray,
    *,
    kernel: str = "linear",
    gamma: float | None = None,
    c_upper: float = 1.0,
) -> dict:
    """Build the SVM-dual QP::

        min  ½ α'Q α − 1' α
        s.t. y' α = 0
             0 ≤ α ≤ C

    with ``Q[i,j] = y[i] y[j] K(x[i], x[j])``. Labels ``y`` are
    coerced to ``±1`` (any nonzero positive becomes ``+1``, anything
    else becomes ``-1``) — many LIBSVM binary files use ``{0, 1}`` or
    ``{1, 2}`` instead of ``{-1, +1}``.
    """
    y = np.where(np.asarray(y, dtype=float) > 0.0, 1.0, -1.0)
    n = int(y.size)
    if sp.issparse(x):
        x_dense = np.asarray(x.toarray(), dtype=np.float64)  # type: ignore[union-attr]
    else:
        x_dense = np.asarray(x, dtype=np.float64)
    if kernel == "linear":
        gram = x_dense @ x_dense.T
    elif kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / max(1, x_dense.shape[1])
        # Pairwise squared distances via the standard expansion.
        sq_norms = np.sum(x_dense * x_dense, axis=1)
        sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (x_dense @ x_dense.T)
        sq_dist = np.maximum(sq_dist, 0.0)
        gram = np.exp(-float(gamma) * sq_dist)
    else:
        raise ValueError(f"Unknown kernel {kernel!r}; use 'linear' or 'rbf'.")
    # Symmetrize to absorb numerical asymmetry in (X X').
    gram = 0.5 * (gram + gram.T)
    p = (y[:, None] * y[None, :]) * gram
    p = 0.5 * (p + p.T)  # explicit re-symmetrize after labels.

    # Constraints: y' α = 0 (one equality), 0 ≤ α ≤ C (n bounds).
    # Stack as one (n+1) × n constraint matrix with l, u.
    a_eq = sp.csr_matrix(y.reshape(1, -1))
    a_box = sp.eye(n, format="csr")
    a_full = sp.vstack([a_eq, a_box], format="csc")
    l_full = np.concatenate([[0.0], np.zeros(n)])
    u_full = np.concatenate([[0.0], np.full(n, float(c_upper))])
    return {
        "P": sp.csc_matrix(p),
        "q": -np.ones(n),
        "r": 0.0,
        "A": a_full,
        "l": l_full,
        "u": u_full,
        "n": n,
        "m": int(a_full.shape[0]),
        "obj_type": "min",
    }


def markowitz_qp(
    x: sp.spmatrix | np.ndarray,
    *,
    risk_aversion: float = 1.0,
) -> dict:
    """Build a long-only Markowitz QP with weights summing to 1::

        min  ½ w'Σ w − λ μ' w
        s.t. 1' w = 1
             w ≥ 0

    Treats the rows of ``X`` as samples (returns) and the columns as
    assets, computing ``Σ`` and ``μ`` from the sample statistics.
    """
    if sp.issparse(x):
        x_dense = np.asarray(x.toarray(), dtype=np.float64)  # type: ignore[union-attr]
    else:
        x_dense = np.asarray(x, dtype=np.float64)
    n_samples, n_assets = x_dense.shape
    if n_samples < 2:
        raise ValueError("Markowitz QP requires at least 2 samples.")
    mu = x_dense.mean(axis=0)
    centered = x_dense - mu
    cov = (centered.T @ centered) / (n_samples - 1)
    cov = 0.5 * (cov + cov.T)

    a_eq = sp.csr_matrix(np.ones((1, n_assets)))
    a_box = sp.eye(n_assets, format="csr")
    a_full = sp.vstack([a_eq, a_box], format="csc")
    l_full = np.concatenate([[1.0], np.zeros(n_assets)])
    u_full = np.concatenate([[1.0], np.full(n_assets, np.inf)])
    return {
        "P": sp.csc_matrix(cov),
        "q": -float(risk_aversion) * mu,
        "r": 0.0,
        "A": a_full,
        "l": l_full,
        "u": u_full,
        "n": int(n_assets),
        "m": int(a_full.shape[0]),
        "obj_type": "min",
    }


def _subsample(
    x: sp.spmatrix, y: np.ndarray, *, max_samples: int
) -> tuple[sp.spmatrix, np.ndarray]:
    """Deterministically subsample to at most ``max_samples`` rows.

    Selection is deterministic (every k-th row) rather than random so
    repeat runs of the same dataset produce the same QP and the
    benchmark's resume / cache-hash logic stays consistent.
    """
    n = x.shape[0]
    if n <= max_samples:
        return x, y
    step = n // max_samples
    indices = np.arange(0, n, step, dtype=int)[:max_samples]
    return x[indices, :], y[indices]


# ---------------------------------------------------------------------------
# Subset normalization.
# ---------------------------------------------------------------------------


def _normalize_subset(value: Any) -> set[str]:
    if value is None or value == "default":
        return set(LIBSVM_DEFAULT_SUBSET)
    if value == "all":
        return set(LIBSVM_DEFAULT_SUBSET)  # only the curated subset is supported.
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}
