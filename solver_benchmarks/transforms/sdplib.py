"""Direct SDPLIB JLD2-to-SCS-cone parsing without CVXPY."""

from __future__ import annotations

import tarfile
from collections.abc import Sequence
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


def read_sdplib_block_sizes(path: Path) -> tuple[int, ...] | None:
    """The block structure a converted SDPLIB file records, or None if it has none.

    The converted files keep SDPLIB's block structure next to the matrices,
    as ``blockvec`` in SDPA notation: a positive entry is a semidefinite
    block of that order, a negative entry a diagonal block of that many
    nonnegative variables, in order along the diagonal of the stored
    matrices. (For the 92 bundled problems it equals the third line of the
    original ``.dat-s`` file.)
    """
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install h5py to load SDPLIB .jld2 problems") from exc

    with h5py.File(path, "r") as handle:
        if "blockvec" not in handle:
            return None
        values = np.array(handle["blockvec"], dtype=float).reshape(-1)
    sizes = tuple(int(round(value)) for value in values)
    if not sizes or any(size == 0 or size != value for size, value in zip(sizes, values, strict=True)):
        raise ValueError(f"{path.name}: blockvec {values.tolist()!r} is not a list of block sizes")
    return sizes


def split_into_declared_blocks(problem: dict, block_sizes: Sequence[int]) -> dict:
    """Give a problem read from the converted archive the blocks SDPLIB declares.

    The converted archive stores each constraint as one ``n x n`` matrix, so
    ``read_sdplib_jld2`` returns a single positive semidefinite cone of order
    ``n``. SDPLIB states 39 of its 92 problems as block diagonal: several
    semidefinite blocks, and sometimes a diagonal block of plain nonnegative
    variables. Stored as one block, every entry between two blocks becomes a
    row of ``A`` that is entirely zero with a zero right-hand side, and a
    solver has to factor one matrix of order ``n`` where the problem as
    published has many small ones.

    ``block_sizes`` is the block structure in SDPA notation, in order along
    the diagonal of the merged matrix: a positive entry is a semidefinite
    block of that order, a negative entry a diagonal block of that many
    nonnegative variables. The result has one ``l`` row per variable of each
    diagonal block followed by one ``s`` block per semidefinite block, both
    in the declared order. This is the layout ``sdpa_to_cone_problem``
    produces from the original file. The rows that are dropped are the
    entries outside every block; a nonzero there means ``block_sizes`` does
    not describe this problem, and raises.
    """
    cone = {key: value for key, value in dict(problem["cone"]).items() if value}
    orders = list(cone.get("s", []))
    if set(cone) != {"s"} or len(orders) != 1:
        raise ValueError(
            "split_into_declared_blocks expects a single semidefinite block, "
            f"got cone {problem['cone']!r}"
        )
    n = int(orders[0])
    sizes = [int(size) for size in block_sizes]
    if any(size == 0 for size in sizes) or sum(abs(size) for size in sizes) != n:
        raise ValueError(
            f"Block sizes {tuple(sizes)} do not add up to the order {n} of the stored matrix"
        )
    if sizes == [n]:
        return problem

    linear_rows: list[np.ndarray] = []
    psd_rows: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        if size < 0:
            diagonal = offset + np.arange(-size, dtype=np.int64)
            linear_rows.append(_col_major_lower_index_array(diagonal, diagonal, n))
        else:
            # Column-major lower triangle of the block: column j, rows i >= j.
            cols, rows = np.triu_indices(size)
            psd_rows.append(
                _col_major_lower_index_array(
                    offset + rows.astype(np.int64), offset + cols.astype(np.int64), n
                )
            )
        offset += abs(size)
    kept = np.concatenate(linear_rows + psd_rows)

    a = sp.csr_matrix(problem["A"], copy=True)
    a.eliminate_zeros()
    b = np.asarray(problem["b"], dtype=float).reshape(-1)
    outside = np.ones(a.shape[0], dtype=bool)
    outside[kept] = False
    stray = np.flatnonzero(outside & ((np.diff(a.indptr) > 0) | (b != 0.0)))
    if stray.size:
        i, j = _col_major_lower_entry(int(stray[0]), n)
        raise ValueError(
            f"Entry ({i}, {j}) of the stored matrices is nonzero but lies outside the "
            f"declared blocks {tuple(sizes)} ({stray.size} such entries)"
        )

    split = dict(problem)
    split["A"] = a[kept, :].tocsc()
    split["b"] = b[kept]
    split["m"] = int(kept.size)
    num_linear = sum(-size for size in sizes if size < 0)
    split["cone"] = {}
    if num_linear:
        split["cone"]["l"] = int(num_linear)
    psd_orders = [size for size in sizes if size > 0]
    if psd_orders:
        split["cone"]["s"] = psd_orders
    return split


def _col_major_lower_entry(index: int, n: int) -> tuple[int, int]:
    """Inverse of ``_col_major_lower_index``: the entry (i, j), i >= j, at ``index``."""
    j = 0
    while index >= n - j:
        index -= n - j
        j += 1
    return j + index, j


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
