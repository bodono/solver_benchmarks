"""Read MPS (and QPS) files into the canonical (A, c, l, u) form via HiGHS."""

from __future__ import annotations

import bz2
import contextlib
import gzip
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import sparse

# Bound magnitudes at or beyond this are infinity sentinels (the MPS
# datasets document +-1e20, sometimes stored with representation error;
# HiGHS hands the literal values back). Normalized to true np.inf on the
# correct side only — a wrong-signed huge bound (u <= -1e19 or
# l >= +1e19) encodes an infeasible-scale demand, not infinity, and is
# kept as data (same policy as transforms/cones.py).
_INF_SENTINEL = 1.0e19


def _sanitize_lower(v: np.ndarray) -> np.ndarray:
    return np.where(v <= -_INF_SENTINEL, -np.inf, v)


def _sanitize_upper(v: np.ndarray) -> np.ndarray:
    return np.where(v >= _INF_SENTINEL, np.inf, v)


class QuadraticDataIgnoredWarning(UserWarning):
    """A reader that returns only the linear part dropped a quadratic objective.

    ``readMpsLp`` returns ``(A, c, l, u)``; ``readMpsProblem`` returns the
    quadratic objective as well.
    """


class MpsProblem(NamedTuple):
    """An LP read from an MPS file, as the minimisation the harness solves:
    minimise ``c'x + offset`` subject to ``l <= mat x <= u``."""

    mat: sparse.csc_matrix
    c: np.ndarray
    l: np.ndarray
    u: np.ndarray
    # Constant term of the objective of the minimisation: the file's constant,
    # with its sign changed if the file states a maximisation.
    offset: float
    # Whether the file states a maximisation. ``c``, ``offset`` and
    # ``hessian`` are then minus the file's, and the optimum of the file is
    # minus the optimum of the minimisation.
    maximise: bool
    # The symmetric ``P`` of a quadratic objective ``0.5 x'Px`` (QUADOBJ,
    # QMATRIX or QSECTION, as HiGHS reads them), or None for an LP.
    hessian: sparse.csc_matrix | None = None


def readMpsProblem(filename) -> MpsProblem:
    """Read an MPS file as the minimisation the harness solves.

    Supports plain ``.mps``, gzip-compressed ``.mps.gz``, and bzip2-compressed
    ``.mps.bz2`` files. HiGHS' ``readModel`` only accepts plain MPS files on
    disk, so compressed inputs are first decompressed, in pieces, to a
    temporary file. A quadratic objective section is returned as
    ``hessian``.
    """

    path = Path(filename)
    suffix = "".join(path.suffixes[-2:]).lower() if path.suffixes else ""

    if suffix.endswith(".gz") or path.suffix.lower() == ".gz":
        with _decompressed_temp_file(path, gzip.open) as tmp:
            return _read_uncompressed_mps_problem(tmp, source_name=path.name)
    if suffix.endswith(".bz2") or path.suffix.lower() == ".bz2":
        with _decompressed_temp_file(path, bz2.open) as tmp:
            return _read_uncompressed_mps_problem(tmp, source_name=path.name)
    return _read_uncompressed_mps_problem(str(path))


def readMpsLp(filename):
    """Read an MPS LP file and return ``(mat, c, l, u)`` in canonical bound form.

    The objective's constant term is left out; ``readMpsProblem`` returns it.
    A quadratic objective section (QPS) is left out too, with a
    ``QuadraticDataIgnoredWarning``: the four values are then only the
    linear part of the problem the file states.
    """
    return _linear_part(readMpsProblem(filename), filename)


@contextlib.contextmanager
def _decompressed_temp_file(path: Path, opener):
    fd, tmp_path = tempfile.mkstemp(suffix=".mps")
    try:
        with os.fdopen(fd, "wb") as out_handle, opener(path, "rb") as in_handle:
            # In pieces: decompressed MIPLIB and Mittelmann files reach
            # gigabytes.
            shutil.copyfileobj(in_handle, out_handle)
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _read_uncompressed_mps(filename: str):
    return _linear_part(_read_uncompressed_mps_problem(filename), filename)


def _linear_part(problem: MpsProblem, filename) -> tuple:
    if problem.hessian is not None and problem.hessian.nnz:
        warnings.warn(
            f"MPS file {Path(filename).name} has a quadratic objective section "
            "(QUADOBJ/QMATRIX/QSECTION), which readMpsLp leaves out; use "
            "readMpsProblem to keep it.",
            QuadraticDataIgnoredWarning,
            stacklevel=3,
        )
    return tuple(problem[:4])


def _read_uncompressed_mps_problem(filename: str, source_name: str | None = None) -> MpsProblem:
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
    # Everything downstream minimises. A file that HiGHS reads as a
    # maximisation is posed as the minimisation of minus its objective,
    # constant term included, as the CBLIB and QPLIB loaders do. The sense
    # is read without a fallback: a HiGHS build that does not report it must
    # fail here, not hand back a maximisation as a minimisation.
    offset = float(lp.offset_)
    maximise = bool(lp.sense_ == highspy.ObjSense.kMaximize)
    if maximise:
        c = -c
        offset = -offset
    n = c.size
    x_low = _sanitize_lower(np.asarray(lp.col_lower_, dtype=float))
    x_upper = _sanitize_upper(np.asarray(lp.col_upper_, dtype=float))
    if read_status == warning_status:
        x_low = _free_lower_bounds_below_negative_upper(
            filename, lp, x_low, x_upper, source_name or Path(filename).name
        )
    c_low = _sanitize_lower(np.asarray(lp.row_lower_, dtype=float))
    c_upper = _sanitize_upper(np.asarray(lp.row_upper_, dtype=float))

    finite_var_bounds = np.isfinite(x_low) | np.isfinite(x_upper)
    l = np.hstack((c_low, x_low[finite_var_bounds]))
    u = np.hstack((c_upper, x_upper[finite_var_bounds]))
    mat = sparse.vstack(
        (a, sparse.eye(n, format="csr")[finite_var_bounds, :]),
        format="csc",
    )
    hessian = _symmetric_hessian(h.getModel().hessian_, n)
    if hessian is not None and maximise:
        hessian = -hessian
    return MpsProblem(mat, c, l, u, offset, maximise, hessian)


def _symmetric_hessian(hessian, n: int) -> sparse.csc_matrix | None:
    """The full symmetric matrix of HiGHS' Hessian, or None if there is none.

    HiGHS stores the Hessian of ``0.5 x'Qx`` by columns, as its lower
    triangle (``kTriangular``, what it builds from QUADOBJ, QMATRIX and
    QSECTION) or whole (``kSquare``).
    """
    dim = int(getattr(hessian, "dim_", 0))
    if dim == 0 or len(hessian.value_) == 0:
        return None
    stored = sparse.csc_matrix(
        (np.asarray(hessian.value_, dtype=float), np.asarray(hessian.index_), np.asarray(hessian.start_)),
        shape=(dim, dim),
    )
    if dim != n:
        raise RuntimeError(f"HiGHS returned a Hessian of order {dim} for {n} columns")
    import highspy

    if hessian.format_ == highspy.HessianFormat.kSquare:
        return stored
    lower = sparse.tril(stored, format="csc")
    return sparse.csc_matrix(lower + sparse.tril(lower, -1).T)


# Bound types that state a lower bound (MI and FR state -infinity).
_LOWER_BOUND_TYPES = frozenset({"LO", "FX", "FR", "MI", "BV", "LI"})
# Section keywords, which start in the first column (a free-format data
# line may too, so the keyword decides).
_MPS_SECTIONS = frozenset(
    {
        "NAME", "OBJSENSE", "OBJSENSEMPS", "OBJNAME", "ROWS", "USERCUTS", "LAZYCONS",
        "COLUMNS", "RHS", "RANGES", "BOUNDS", "SOS", "QUADOBJ", "QMATRIX", "QSECTION",
        "QCMATRIX", "CSECTION", "INDICATORS", "GENCONS", "PWLOBJ", "ENDATA",
    }
)


def _free_lower_bounds_below_negative_upper(
    filename: str, lp, x_low: np.ndarray, x_upper: np.ndarray, source_name: str
) -> np.ndarray:
    """Apply the usual MPS convention to a negative upper bound alone.

    A column with an ``UP`` bound below zero and no lower bound of its own
    is free below in CPLEX, Gurobi and lp_solve. HiGHS keeps the default
    lower bound 0 and only warns about the inconsistent bounds, which
    would pose an infeasible problem. Columns that show this (lower bound
    0, upper bound negative, after a HiGHS warning) are checked against the
    BOUNDS section of the file, streamed: those with no lower bound stated
    get -infinity. A lower bound the file does state is kept, and the
    problem stays infeasible, as written.
    """
    suspects = np.flatnonzero((x_low == 0.0) & (x_upper < 0.0))
    if suspects.size == 0:
        return x_low
    names = list(lp.col_names_)
    if len(names) != x_low.size:
        return x_low
    wanted = {names[i]: i for i in suspects}
    stated: set[str] = set()
    in_bounds = False
    with open(filename, encoding="ascii", errors="replace") as handle:
        for line in handle:
            fields = line.split()
            if not fields or line.startswith("*"):
                continue
            if not line[0].isspace() and fields[0].upper() in _MPS_SECTIONS:
                in_bounds = fields[0].upper() == "BOUNDS"
                continue
            if in_bounds and fields[0].upper() in _LOWER_BOUND_TYPES:
                # Free format may leave out the bound set's name, so the
                # column is the second or the third field.
                stated.update(name for name in fields[1:3] if name in wanted)
    freed = [i for name, i in wanted.items() if name not in stated]
    if not freed:
        return x_low
    warnings.warn(
        f"{source_name}: {len(freed)} column(s) with a negative upper bound and "
        "no lower bound are taken as free below (as CPLEX and Gurobi read them); HiGHS "
        "kept the default lower bound 0",
        UserWarning,
        stacklevel=4,
    )
    x_low = x_low.copy()
    x_low[freed] = -np.inf
    return x_low
