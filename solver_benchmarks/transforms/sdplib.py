"""Direct SDPLIB JLD2-to-SCS-cone parsing without CVXPY."""

from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def list_sdplib_tar(tar_path: Path) -> dict[str, int]:
    """Return a mapping from problem name to the tar member size in bytes.

    The size is exposed so callers (and the runner-level size filter) can
    reason about per-member sizes without having to crack open the archive
    again.
    """
    if not tar_path.exists():
        return {}
    with tarfile.open(tar_path) as archive:
        return {
            Path(member.name).stem: int(member.size)
            for member in archive
            if member.name.endswith(".jld2")
        }


def extract_from_tar(tar_path: Path, name: str, cache_dir: Path) -> Path:
    target = cache_dir / f"{name}.jld2"
    if target.exists():
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    member_name = f"{name}.jld2"
    with tarfile.open(tar_path) as archive:
        member = archive.getmember(member_name)
        source = archive.extractfile(member)
        if source is None:
            raise FileNotFoundError(member_name)
        target.write_bytes(source.read())
    return target


def read_sdplib_jld2(path: Path) -> dict:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install h5py to load SDPLIB .jld2 problems") from exc

    with h5py.File(path, "r") as handle:
        m = int(np.array(handle["m"]).item())
        n = int(np.array(handle["n"]).item())
        c = np.array(handle["c"], dtype=float).reshape(-1)
        fs = [_parse_sparse_matrix(handle, i) for i in range(m + 1)]

    rows = n * (n + 1) // 2
    columns = []
    for i in range(m):
        columns.append(-_psd_vec(fs[i + 1]))
    a = sp.hstack(columns, format="csc") if columns else sp.csc_matrix((rows, 0))
    b = -_psd_vec(fs[0]).toarray().reshape(-1)
    return {
        "P": None,
        "A": a,
        "b": b,
        "q": c,
        "r": 0.0,
        "n": int(a.shape[1]),
        "m": int(a.shape[0]),
        "cone": {"s": [n]},
        "obj_type": "min",
    }


def _parse_sparse_matrix(handle, idx: int) -> sp.csc_matrix:
    ref = handle[handle["F"][idx]]
    value = np.array(ref).item()
    m = int(value[0])
    n = int(value[1])
    colptr = np.array(handle[value[2]], dtype=np.int64).reshape(-1) - 1
    rowidx = np.array(handle[value[3]], dtype=np.int64).reshape(-1) - 1
    data = np.array(handle[value[4]], dtype=float).reshape(-1)
    matrix = sp.csc_matrix((data, rowidx, colptr), shape=(m, n))
    return 0.5 * (matrix + matrix.T)


def _psd_vec(matrix: sp.spmatrix) -> sp.csc_matrix:
    """Vectorize a symmetric matrix in col-major lower-triangular order
    (the canonical PSD vec convention used by SCS and the KKT module).

    Vectorized over the COO triple so block sizes in the hundreds-of-
    thousands of nnz are handled without a Python-level inner loop.
    """
    matrix = sp.tril(matrix, format="coo")
    n = matrix.shape[0]
    if matrix.nnz == 0:
        return sp.csc_matrix((n * (n + 1) // 2, 1))
    rows = matrix.row.astype(np.int64, copy=False)
    cols = matrix.col.astype(np.int64, copy=False)
    values = matrix.data.astype(float, copy=False)
    flat = _col_major_lower_index_array(rows, cols, n)
    scale = np.where(rows != cols, np.sqrt(2.0), 1.0)
    data = scale * values
    return sp.csc_matrix(
        (data, (flat, np.zeros(flat.size, dtype=np.int64))),
        shape=(n * (n + 1) // 2, 1),
    )


def _col_major_lower_index(i: int, j: int, n: int) -> int:
    if i < j:
        i, j = j, i
    return j * n - j * (j - 1) // 2 + (i - j)


def _col_major_lower_index_array(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Vectorized version of _col_major_lower_index over COO arrays."""
    # Ensure i >= j elementwise, swapping where necessary.
    upper = rows < cols
    i = np.where(upper, cols, rows)
    j = np.where(upper, rows, cols)
    return j * n - j * (j - 1) // 2 + (i - j)
