"""Tests for the MPS reader."""

from __future__ import annotations

import bz2
import gzip
from pathlib import Path

import numpy as np
import pytest

from problem_classes import qpsreader


_TINY_MPS = """\
NAME          TINY
ROWS
 N  COST
 L  C1
 G  C2
 E  C3
COLUMNS
    X1   COST        1.0   C1   1.0
    X1   C2          1.0
    X2   COST        1.0   C1   1.0
    X2   C3          1.0
RHS
    RHS   C1   10.0   C2   2.0
    RHS   C3   5.0
BOUNDS
 LO BND   X1   0.0
 UP BND   X1   8.0
 LO BND   X2   0.0
ENDATA
"""


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_readMpsLp_plain_mps(tmp_path: Path):
    path = _write(tmp_path / "tiny.mps", _TINY_MPS)
    a, c, l, u = qpsreader.readMpsLp(str(path))
    # 2 columns, 3 row constraints + 2 finite-bound variables = 5 rows.
    assert c.shape == (2,)
    assert a.shape[0] == l.shape[0] == u.shape[0]
    assert a.shape[1] == 2
    # Row constraints reproduce the CONST rows: C1 (<=10), C2 (>=2), C3 (==5).
    # The order may vary but all bounds should be present in (l, u).
    bounds = sorted(zip(l[: a.shape[0] - 2].tolist(), u[: a.shape[0] - 2].tolist()))
    assert bounds == sorted([(-np.inf, 10.0), (2.0, np.inf), (5.0, 5.0)])


def test_readMpsLp_gzip(tmp_path: Path):
    plain = _write(tmp_path / "tiny.mps", _TINY_MPS)
    plain_result = qpsreader.readMpsLp(str(plain))

    gz_path = tmp_path / "tiny.mps.gz"
    gz_path.write_bytes(gzip.compress(_TINY_MPS.encode()))
    gz_result = qpsreader.readMpsLp(str(gz_path))

    # Both readers should produce identical canonical tuples.
    assert np.allclose(plain_result[1], gz_result[1])
    assert plain_result[0].shape == gz_result[0].shape
    assert np.allclose(plain_result[0].toarray(), gz_result[0].toarray())
    assert np.allclose(plain_result[2], gz_result[2])
    assert np.allclose(plain_result[3], gz_result[3])


def test_readMpsLp_bz2(tmp_path: Path):
    bz2_path = tmp_path / "tiny.mps.bz2"
    bz2_path.write_bytes(bz2.compress(_TINY_MPS.encode()))
    a, c, l, u = qpsreader.readMpsLp(str(bz2_path))
    assert c.shape == (2,)
    assert a.shape[1] == 2


def test_readMpsLp_missing_file_raises(tmp_path: Path):
    with pytest.raises(RuntimeError):
        qpsreader.readMpsLp(str(tmp_path / "does_not_exist.mps"))


def test_readMpsLp_missing_gz_raises(tmp_path: Path):
    # Without the fix, qpsreader silently stripped ``.gz`` and read whatever
    # ``.mps`` happened to live alongside; verify that a missing ``.gz`` errors
    # rather than returning empty matrices.
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        qpsreader.readMpsLp(str(tmp_path / "does_not_exist.mps.gz"))
