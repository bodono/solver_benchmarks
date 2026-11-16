"""Direct SDPLIB JLD2-to-SCS-cone parsing without CVXPY."""

from __future__ import annotations

import functools
import json
import shutil
import tarfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.atomic import (
    atomic_write_bytes,
    atomic_write_stream,
    remove_stale_temporaries,
)

# A temporary file of an extraction older than this was left by a worker that
# was killed during the copy. A live copy writes to its file continuously (the
# largest member, 395 MB, takes seconds), so an hour without a write cannot be
# a live extraction, even on a slow or heavily loaded machine.
STALE_TEMPORARY_AGE_SECONDS = 3600.0


def list_sdplib_tar(tar_path: Path, index_path: Path | None = None) -> dict[str, int]:
    """Return a mapping from problem name to the tar member size in bytes.

    The size is exposed so callers (and the runner-level size filter) can
    reason about per-member sizes without having to crack open the archive
    again.

    Listing the members of the gzip-compressed archive decompresses all of
    it (911 MB for the bundled one, about a second). With ``index_path`` the
    listing is kept in that JSON file, with the archive's size and
    modification time, and read from there while they still match. The index
    is written atomically, with the mode ``open()`` would give it, and a
    failure to write it is ignored.
    """
    if not tar_path.exists():
        return {}
    stat = tar_path.stat()
    stamp = {"archive_size": int(stat.st_size), "archive_mtime_ns": int(stat.st_mtime_ns)}
    if index_path is not None:
        cached = _read_member_index(index_path, stamp)
        if cached is not None:
            return cached
    with tarfile.open(tar_path) as archive:
        members = {
            Path(member.name).stem: int(member.size)
            for member in archive
            if member.name.endswith(".jld2")
        }
    if index_path is not None:
        payload = json.dumps({**stamp, "members": members}, sort_keys=True).encode("utf-8")
        try:
            atomic_write_bytes(index_path, payload)
        except OSError:
            pass
    return members


def _read_member_index(index_path: Path, stamp: dict[str, int]) -> dict[str, int] | None:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if any(index.get(key) != value for key, value in stamp.items()):
            return None
        return {str(name): int(size) for name, size in index["members"].items()}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def extract_from_tar(
    tar_path: Path, name: str, cache_dir: Path, expected_size: int | None = None
) -> Path:
    """Extract member ``<name>.jld2`` to ``cache_dir`` and return its path.

    The member is copied to a temporary file in ``cache_dir`` and renamed
    onto ``<name>.jld2`` once complete, so a reader never sees a partial
    file: a second worker loading the same problem meanwhile extracts its
    own copy, and a worker killed during the copy leaves nothing at the
    target; its temporary file (``.<name>.jld2.<random>.tmp``) is removed
    by a later call once it is an hour old (``STALE_TEMPORARY_AGE_SECONDS``).
    A file already at the target is used as it is, unless
    ``expected_size`` (the member's size) is given and differs, as for a
    file truncated by an interrupted copy before extraction was atomic.
    A new file gets the mode ``open()`` would give it (0o666 less the
    umask), so on a shared data directory other users can read what one
    user extracted.

    The archive is read once, up to the member: the compressed archive
    cannot seek back, so looking the member up by name first (which lists
    the whole archive) and then reading it would decompress it twice.
    """
    sizes = None if expected_size is None else {name: int(expected_size)}
    return extract_many_from_tar(tar_path, [name], cache_dir, expected_sizes=sizes)[name]


def extract_many_from_tar(
    tar_path: Path,
    names: Sequence[str],
    cache_dir: Path,
    expected_sizes: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Extract members ``<name>.jld2`` for each of ``names`` in one pass over the archive.

    Returns the path of each. Each file is written as ``extract_from_tar``
    writes one, and one already at its target (with the expected size, when
    ``expected_sizes`` gives it) is kept. Reading stops after the last
    member wanted. A name with no member raises FileNotFoundError.
    """
    paths = {name: cache_dir / f"{name}.jld2" for name in names}
    # Also when nothing is extracted: the temporary files of a killed copy
    # stay behind after another worker has extracted the member.
    for target in paths.values():
        remove_stale_temporaries(target, STALE_TEMPORARY_AGE_SECONDS)
    wanted = {
        f"{name}.jld2": name
        for name, target in paths.items()
        if not (
            target.exists()
            and (
                expected_sizes is None
                or name not in expected_sizes
                or target.stat().st_size == int(expected_sizes[name])
            )
        )
    }
    if not wanted:
        return paths
    with tarfile.open(tar_path) as archive:
        for member in archive:
            name = wanted.pop(member.name, None)
            if name is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise FileNotFoundError(member.name)
            # Copied in pieces: the largest member is 395 MB.
            with source:
                atomic_write_stream(paths[name], functools.partial(shutil.copyfileobj, source))
            if not wanted:
                break
    if wanted:
        raise FileNotFoundError(f"{tar_path.name} has no member {', '.join(sorted(wanted))}")
    return paths


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
        try:
            values = np.array(handle["blockvec"], dtype=float).reshape(-1)
        except (TypeError, ValueError, OSError) as exc:
            raise ValueError(f"{path.name}: blockvec is not a list of block sizes ({exc})") from exc
    if (
        values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values == 0)
        or np.any(values != np.round(values))
    ):
        raise ValueError(f"{path.name}: blockvec {values.tolist()!r} is not a list of block sizes")
    return tuple(int(value) for value in values)


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
