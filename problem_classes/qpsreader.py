"""Read MPS LP files into the canonical (A, c, l, u) tuple via HiGHS."""

from __future__ import annotations

import bz2
import contextlib
import gzip
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse


def readMpsLp(filename):
    """Read an MPS LP file and return ``(mat, c, l, u)`` in canonical bound form.

    Supports plain ``.mps``, gzip-compressed ``.mps.gz``, and bzip2-compressed
    ``.mps.bz2`` files. HiGHS' ``readModel`` only accepts plain MPS files on
    disk, so compressed inputs are first decompressed to a temporary file.
    """

    path = Path(filename)
    suffix = "".join(path.suffixes[-2:]).lower() if path.suffixes else ""

    if suffix.endswith(".gz") or path.suffix.lower() == ".gz":
        with _decompressed_temp_file(path, gzip.open) as tmp:
            return _read_uncompressed_mps(tmp)
    if suffix.endswith(".bz2") or path.suffix.lower() == ".bz2":
        with _decompressed_temp_file(path, bz2.open) as tmp:
            return _read_uncompressed_mps(tmp)
    return _read_uncompressed_mps(str(path))


@contextlib.contextmanager
def _decompressed_temp_file(path: Path, opener):
    fd, tmp_path = tempfile.mkstemp(suffix=".mps")
    try:
        with os.fdopen(fd, "wb") as out_handle, opener(path, "rb") as in_handle:
            out_handle.write(in_handle.read())
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _read_uncompressed_mps(filename: str):
    import highspy

    h = highspy.Highs()
    # Silence HiGHS console output; otherwise readModel pollutes stdout.
    with contextlib.suppress(Exception):
        h.silent()
    read_status = h.readModel(str(filename))
    warning_status = getattr(highspy.HighsStatus, "kWarning", None)
    if read_status not in (highspy.HighsStatus.kOk, warning_status):
        raise RuntimeError(
            f"HiGHS failed to read MPS file {filename!r} (status={read_status!r})"
        )
    lp = h.getLp()
    if lp.num_col_ == 0:
        raise RuntimeError(f"HiGHS read empty model from {filename!r}")
    raw = lp.a_matrix_
    a = sparse.csc_matrix(
        (np.asarray(raw.value_, dtype=float), raw.index_, raw.start_),
        shape=(raw.num_row_, raw.num_col_),
    )
    c = np.asarray(lp.col_cost_, dtype=float)
    n = c.size
    x_low = np.asarray(lp.col_lower_, dtype=float)
    x_upper = np.asarray(lp.col_upper_, dtype=float)
    c_low = np.asarray(lp.row_lower_, dtype=float)
    c_upper = np.asarray(lp.row_upper_, dtype=float)

    finite_var_bounds = (x_low > -1.0e20) | (x_upper < 1.0e20)
    l = np.hstack((c_low, x_low[finite_var_bounds]))
    u = np.hstack((c_upper, x_upper[finite_var_bounds]))
    mat = sparse.vstack(
        (a, sparse.eye(n, format="csr")[finite_var_bounds, :]),
        format="csc",
    )
    return mat, c, l, u
