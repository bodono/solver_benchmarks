"""Tests for the MPS reader."""

from __future__ import annotations

import bz2
import gzip
import sys
import types
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

_SENTINEL_BOUND_MPS = """\
NAME          SENT
ROWS
 N  COST
 G  C1
COLUMNS
    X1   COST        1.0   C1   1.0
    X2   COST        1.0   C1   1.0
RHS
    RHS   C1   1.0
BOUNDS
 UP BND   X1   1.0e+20
 LO BND   X2   -1.0e+20
 UP BND   X2   1.0e+20
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


def test_readMpsLp_accepts_highs_warning_with_populated_model(monkeypatch, tmp_path: Path):
    class FakeStatus:
        kOk = "ok"
        kWarning = "warning"
        kError = "error"

    class FakeMatrix:
        num_row_ = 1
        num_col_ = 1
        value_ = [1.0]
        index_ = [0]
        start_ = [0, 1]

    class FakeLp:
        num_col_ = 1
        a_matrix_ = FakeMatrix()
        col_cost_ = [1.0]
        col_lower_ = [0.0]
        col_upper_ = [1.0e30]
        row_lower_ = [-1.0e30]
        row_upper_ = [1.0]
        sense_ = "minimize"

    class FakeObjSense:
        kMinimize = "minimize"
        kMaximize = "maximize"

    class FakeHighs:
        def silent(self):
            pass

        def readModel(self, filename):
            return FakeStatus.kWarning

        def getLp(self):
            return FakeLp()

    fake_highspy = types.SimpleNamespace(
        Highs=FakeHighs, HighsStatus=FakeStatus, ObjSense=FakeObjSense
    )
    monkeypatch.setitem(sys.modules, "highspy", fake_highspy)

    path = tmp_path / "warning.mps"
    path.write_text(_TINY_MPS)
    a, c, l, u = qpsreader._read_uncompressed_mps(str(path))

    assert a.shape == (2, 1)
    assert c.tolist() == [1.0]
    # HiGHS-style 1e30 infinities are normalized to true np.inf.
    assert l.tolist() == [-np.inf, 0.0]
    assert u.tolist() == [1.0, np.inf]


def test_readMpsLp_missing_file_raises(tmp_path: Path):
    with pytest.raises(RuntimeError):
        qpsreader.readMpsLp(str(tmp_path / "does_not_exist.mps"))


def test_readMpsLp_missing_gz_raises(tmp_path: Path):
    # Without the fix, qpsreader silently stripped ``.gz`` and read whatever
    # ``.mps`` happened to live alongside; verify that a missing ``.gz`` errors
    # rather than returning empty matrices.
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        qpsreader.readMpsLp(str(tmp_path / "does_not_exist.mps.gz"))


def test_readMpsLp_warns_on_quadratic_section(tmp_path: Path):
    """A QPS file with a quadratic section should warn (HiGHS getLp() drops it).

    The warning fires before HiGHS attempts to parse, so even if HiGHS
    rejects the QPS file the user gets a clear signal about the dropped
    quadratic data.
    """
    qps_body = _TINY_MPS.replace(
        "ENDATA\n",
        "QUADOBJ\n    X1   X1   2.0\nENDATA\n",
    )
    path = _write(tmp_path / "tiny.qps", qps_body)
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        try:
            qpsreader.readMpsLp(str(path))
        except RuntimeError:
            pass  # HiGHS may also fail to parse; the warning is what we test.
    quadratic_warnings = [
        w for w in record if issubclass(w.category, qpsreader.QuadraticDataIgnoredWarning)
    ]
    assert quadratic_warnings, "Expected a QuadraticDataIgnoredWarning"


def test_readMpsLp_does_not_warn_on_plain_lp(tmp_path: Path):
    path = _write(tmp_path / "tiny.mps", _TINY_MPS)
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        qpsreader.readMpsLp(str(path))
    quadratic_warnings = [
        w for w in record if issubclass(w.category, qpsreader.QuadraticDataIgnoredWarning)
    ]
    assert quadratic_warnings == []


def test_readMpsLp_normalizes_sentinel_bounds_to_inf(tmp_path: Path):
    # +-1e20 bound values are infinity sentinels, not data: they must come
    # back as true np.inf, and a variable whose bounds are both sentinels
    # must not get a bound row at all (previously X1's u leaked as 1e20).
    path = _write(tmp_path / "sent.mps", _SENTINEL_BOUND_MPS)
    a, c, l, u = qpsreader.readMpsLp(str(path))
    # 1 constraint row + 1 bound row for X1 (finite default lower of 0);
    # X2 is free after sentinel normalization so its row is dropped.
    assert a.shape == (2, 2)
    assert l.tolist() == [1.0, 0.0]
    assert u.tolist() == [np.inf, np.inf]


_MAX_BODY = """\
ROWS
 N  COST
 L  C1
 L  C2
COLUMNS
    X    COST   3.0   C1   1.0
    X    C2     1.0
    Y    COST   2.0   C1   1.0
    Y    C2     3.0
RHS
    RHS   C1   4.0   C2   6.0
BOUNDS
 UP BND   X   3.0
ENDATA
"""


@pytest.mark.parametrize(
    ("objsense", "expected_cost"),
    [
        # maximise 3x + 2y is posed as minimise -3x - 2y.
        ("OBJSENSE\n    MAX\n", [-3.0, -2.0]),
        ("OBJSENSE\n    MAXIMIZE\n", [-3.0, -2.0]),
        ("OBJSENSE MAX\n", [-3.0, -2.0]),
        ("OBJSENSE\n    MIN\n", [3.0, 2.0]),
        ("", [3.0, 2.0]),
    ],
)
def test_readMpsLp_poses_a_maximisation_as_a_minimisation(
    tmp_path: Path, objsense: str, expected_cost: list[float]
):
    pytest.importorskip("highspy")
    path = _write(tmp_path / "sense.mps", "NAME          SENSE\n" + objsense + _MAX_BODY)
    a, c, l, u = qpsreader.readMpsLp(str(path))
    assert c.tolist() == expected_cost
    # The constraints do not depend on the sense.
    assert a.toarray().tolist() == [[1.0, 1.0], [1.0, 3.0], [1.0, 0.0], [0.0, 1.0]]
    assert l.tolist() == [-np.inf, -np.inf, 0.0, 0.0]
    assert u.tolist() == [4.0, 6.0, 3.0, np.inf]


@pytest.mark.parametrize(("suffix", "opener"), [(".mps.gz", gzip.open), (".mps.bz2", bz2.open)])
def test_readMpsLp_sense_survives_compression(tmp_path: Path, suffix: str, opener):
    pytest.importorskip("highspy")
    body = "NAME          SENSE\nOBJSENSE\n    MAX\n" + _MAX_BODY
    path = tmp_path / f"sense{suffix}"
    with opener(path, "wb") as handle:
        handle.write(body.encode())
    _, c, _, _ = qpsreader.readMpsLp(str(path))
    assert c.tolist() == [-3.0, -2.0]


def test_readMpsLp_fails_if_highs_does_not_report_the_sense(monkeypatch, tmp_path: Path):
    # A build without the sense must not hand a maximisation back as a
    # minimisation: there is no fallback.
    highspy = pytest.importorskip("highspy")
    monkeypatch.delattr(highspy, "ObjSense")
    path = _write(tmp_path / "sense.mps", "NAME          SENSE\nOBJSENSE\n    MAX\n" + _MAX_BODY)
    with pytest.raises(AttributeError, match="ObjSense"):
        qpsreader.readMpsLp(str(path))


def test_mps_dataset_solves_a_maximisation_to_its_optimum(tmp_path: Path):
    # maximise 3x + 2y, x + y <= 4, x + 3y <= 6, 0 <= x <= 3, y >= 0: the
    # optimum is 11 at (3, 1). Read as a minimisation of 3x + 2y, which is
    # what the loader used to do, the answer is 0 at the origin.
    pytest.importorskip("highspy")
    from solver_benchmarks.core import status
    from solver_benchmarks.datasets import get_dataset
    from solver_benchmarks.solvers.registry import get_solver

    data_root = tmp_path / "problem_classes"
    folder = data_root / "netlib_data" / "feasible"
    folder.mkdir(parents=True)
    _write(folder / "maxlp.mps", "NAME          MAXLP\nOBJSENSE\n    MAX\n" + _MAX_BODY)

    dataset = get_dataset("netlib")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("maxlp")
    assert problem.qp["q"].tolist() == [-3.0, -2.0]
    assert problem.qp["obj_type"] == "min"

    result = get_solver("highs")({"verbose": False}).solve(problem, tmp_path)
    assert result.status == status.OPTIMAL
    assert result.objective_value == pytest.approx(-11.0, abs=1e-7)


def test_mps_dataset_reports_an_unbounded_maximisation_as_unbounded(tmp_path: Path):
    # maximise x + y with x - y <= 1 and x, y >= 0 has no upper bound. Read as
    # a minimisation it has the optimum 0 at the origin.
    pytest.importorskip("highspy")
    from solver_benchmarks.core import status
    from solver_benchmarks.datasets import get_dataset
    from solver_benchmarks.solvers.registry import get_solver

    data_root = tmp_path / "problem_classes"
    folder = data_root / "netlib_data" / "feasible"
    folder.mkdir(parents=True)
    _write(folder / "unbounded.mps", "\n".join([
        "NAME          UNB", "OBJSENSE", "    MAX", "ROWS", " N  COST", " L  C1", "COLUMNS",
        "    X    COST   1.0   C1   1.0", "    Y    COST   1.0   C1   -1.0",
        "RHS", "    RHS   C1   1.0", "ENDATA", "",
    ]))

    problem = get_dataset("netlib")(repo_root=tmp_path, data_root=data_root).load_problem("unbounded")
    result = get_solver("highs")({"verbose": False}).solve(problem, tmp_path)
    assert result.status in {status.DUAL_INFEASIBLE, status.PRIMAL_OR_DUAL_INFEASIBLE}
