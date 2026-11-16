"""CBF reader: assembly cost, header-only inspection, listing and signs.

The reader used to build one sparse block per CBF cone domain and stack
them, which costs memory and time in proportion to (number of domains) x
(number of variables): an SOCP with tens of thousands of small cones ran
out of memory. Listing parsed every file in the folder in full, and a
worker loading one problem listed (so parsed) every file again.
"""

from __future__ import annotations

import gzip
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from solver_benchmarks.datasets import cblib as cblib_module
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.datasets.cblib import inspect_cbf, read_cbf_cone_problem


def _many_q_cones_cbf(path: Path, ncones: int, dim: int = 3) -> Path:
    n = ncones * dim
    lines = ["VER", "3", "OBJSENSE", "MIN", "VAR", f"{n} {ncones}"]
    lines += [f"Q {dim}"] * ncones
    lines += ["CON", f"{n} {ncones}"]
    lines += [f"Q {dim}"] * ncones
    lines += ["OBJACOORD", str(ncones)]
    lines += [f"{k * dim} 1.0" for k in range(ncones)]
    lines += ["ACOORD", str(n)]
    lines += [f"{i} {i} 1.0" for i in range(n)]
    lines += ["BCOORD", str(ncones)]
    lines += [f"{k * dim} 1.0" for k in range(ncones)]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_cbf_memory_grows_with_nonzeros_not_cones_times_variables(tmp_path: Path):
    # 4000 second-order cones on 6000 variables, 12000 nonzeros. Assembled
    # block by block this peaked near 200 MB (and quadrupled with every
    # doubling of the cone count); assembled by one row permutation it
    # needs a few MB.
    path = _many_q_cones_cbf(tmp_path / "many.cbf", 2000)
    tracemalloc.start()
    try:
        problem, _ = read_cbf_cone_problem(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert problem["A"].shape == (12000, 6000)
    assert problem["cone"] == {"q": [3] * 4000}
    assert peak < 25e6, f"peak {peak / 1e6:.0f} MB"


# A file with every supported cone, as variable and as constraint domains,
# interleaved, with nonzero b, an objective constant and a maximisation.
_MIXED_CBF = """VER
3
OBJSENSE
MAX
VAR
13 7
L+ 1
Q 3
F 1
EXP 3
L- 1
L= 1
EXP* 3
CON
13 7
Q 2
L- 2
EXP* 3
L+ 1
F 1
L= 1
EXP 3
OBJACOORD
3
0 1.5
4 -2.0
11 0.5
OBJBCOORD
7.0
ACOORD
16
0 0 1.0
0 3 2.0
1 1 -1.0
2 2 3.0
2 5 1.0
3 4 1.0
4 6 -4.0
5 7 1.0
6 8 2.0
7 9 1.0
8 10 1.0
9 11 1.0
10 0 1.0
10 12 -1.0
11 12 1.0
12 3 0.5
BCOORD
6
0 1.0
2 -3.0
5 2.0
8 4.0
10 -1.0
12 2.5
"""


def _reference_cone_rows(body: str):
    """Row-by-row reading of the CBF conventions, independent of the reader.

    Variable domain: the row is ``-x_i`` with zero right-hand side, so the
    slack ``b - A x`` is ``x_i``. Constraint domain: the row is ``-(A x)_k``
    with right-hand side ``b_k``, so the slack is ``(A x + b)_k``. ``L-``
    rows are negated. EXP and EXP* triples are reversed. Rows are grouped
    z, l, q, ep, ed, variable domains first within each group.
    """
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    nvar, nvd = map(int, lines[lines.index("VAR") + 1].split())
    var_domains = [lines[lines.index("VAR") + 2 + k].split() for k in range(nvd)]
    ncon, ncd = map(int, lines[lines.index("CON") + 1].split())
    con_domains = [lines[lines.index("CON") + 2 + k].split() for k in range(ncd)]
    a = np.zeros((ncon, nvar))
    start = lines.index("ACOORD")
    for k in range(int(lines[start + 1])):
        i, j, v = lines[start + 2 + k].split()
        a[int(i), int(j)] = float(v)
    b = np.zeros(ncon)
    start = lines.index("BCOORD")
    for k in range(int(lines[start + 1])):
        i, v = lines[start + 2 + k].split()
        b[int(i)] = float(v)
    groups: dict[str, list] = {"L=": [], "L": [], "Q": [], "EXP": [], "EXP*": []}
    soc_dims: list[int] = []
    for domains, rows_of, rhs_of in (
        (var_domains, lambda r: -np.eye(nvar)[r], lambda r: 0.0),
        (con_domains, lambda r: -a[r], lambda r: b[r]),
    ):
        offset = 0
        for name, dim in domains:
            dim = int(dim)
            rows = [(rows_of(r), rhs_of(r)) for r in range(offset, offset + dim)]
            offset += dim
            if name == "F":
                continue
            if name == "L-":
                rows = [(-row, -rhs) for row, rhs in rows]
            if name in ("EXP", "EXP*"):
                rows = rows[::-1]
            key = "L" if name in ("L+", "L-") else name
            if name == "Q":
                soc_dims.append(dim)
            groups[key].extend(rows)
    ordered = [row for key in ("L=", "L", "Q", "EXP", "EXP*") for row in groups[key]]
    a_ref = np.array([row for row, _ in ordered])
    b_ref = np.array([rhs for _, rhs in ordered])
    return a_ref, b_ref, soc_dims, {key: len(rows) for key, rows in groups.items()}


def test_read_cbf_matches_the_row_by_row_reading_of_every_cone(tmp_path: Path):
    path = tmp_path / "mixed.cbf"
    path.write_text(_MIXED_CBF)
    problem, metadata = read_cbf_cone_problem(path)
    a_ref, b_ref, soc_dims, counts = _reference_cone_rows(_MIXED_CBF)

    assert sp.isspmatrix_csc(problem["A"])
    np.testing.assert_array_equal(problem["A"].toarray(), a_ref)
    np.testing.assert_array_equal(problem["b"], b_ref)
    assert problem["cone"] == {
        "z": counts["L="],
        "l": counts["L"],
        "q": soc_dims,
        "ep": counts["EXP"] // 3,
        "ed": counts["EXP*"] // 3,
    }
    # Maximisation: the minimisation of minus the objective, constant included.
    q = np.zeros(13)
    q[[0, 4, 11]] = [-1.5, 2.0, -0.5]
    np.testing.assert_array_equal(problem["q"], q)
    assert problem["r"] == -7.0
    assert problem["obj_type"] == "min"
    assert metadata["original_obj_type"] == "max"


_MAX_LP_CBF = """VER
3
OBJSENSE
MAX
VAR
2 1
L+ 2
CON
3 2
L- 2
L+ 1
OBJACOORD
2
0 1.0
1 2.0
OBJBCOORD
3.0
ACOORD
4
0 0 1.0
0 1 1.0
1 1 1.0
2 0 -1.0
BCOORD
3
0 -4.0
1 -3.0
2 1.0
"""


def test_cbf_signs_solve_to_the_published_optimum(tmp_path: Path):
    """max x0 + 2 x1 + 3 s.t. x0 + x1 <= 4, x1 <= 3, x0 <= 1, x >= 0.

    The optimum is x = (1, 3), objective 10. Posed as a minimisation the
    recorded objective is -10, and the metadata says the file maximised.
    L- rows with a nonzero BCOORD, L+ with a nonzero BCOORD and OBJBCOORD
    all enter it.
    """
    pytest.importorskip("clarabel")
    from solver_benchmarks.solvers import get_solver

    folder = tmp_path / "problem_classes" / "cblib_data"
    folder.mkdir(parents=True)
    (folder / "maxlp.cbf").write_text(_MAX_LP_CBF)
    dataset = get_dataset("cblib")(repo_root=tmp_path, data_root=tmp_path / "problem_classes")
    [spec] = dataset.list_problems()
    assert spec.metadata["original_obj_type"] == "max"
    problem = dataset.load_problem("maxlp")
    assert problem.metadata["original_obj_type"] == "max"

    result = get_solver("clarabel")({"verbose": False}).solve(problem, tmp_path / "artifacts")
    assert result.status == "optimal"
    # The worker adds the constant term r to the solver's objective.
    assert result.objective_value + problem.cone["r"] == pytest.approx(-10.0, abs=1e-6)


_TINY_CBF = "VER\n3\nVAR\n1 1\nL+ 1\nOBJACOORD\n1\n0 1.0\n"


def _dataset(tmp_path: Path, **options):
    return get_dataset("cblib")(
        repo_root=tmp_path, data_root=tmp_path / "problem_classes", **options
    )


def _cblib_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "problem_classes" / "cblib_data"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def test_load_problem_reads_only_its_own_file(tmp_path: Path, monkeypatch):
    folder = _cblib_folder(tmp_path)
    for k in range(5):
        (folder / f"other{k}.cbf").write_text(_TINY_CBF)
    (folder / "tiny.cbf").write_text(_TINY_CBF)
    opened: list[str] = []
    real = cblib_module._iter_cbf_lines

    def spy(path):
        opened.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(cblib_module, "_iter_cbf_lines", spy)
    problem = _dataset(tmp_path).load_problem("tiny")

    assert problem.cone["q"].tolist() == [1.0]
    assert set(opened) == {"tiny.cbf"}


def test_load_problem_keeps_the_listing_filters(tmp_path: Path):
    folder = _cblib_folder(tmp_path)
    (folder / "tiny.cbf").write_text(_TINY_CBF)
    (folder / "other.cbf").write_text(_TINY_CBF)
    with pytest.raises(KeyError):
        _dataset(tmp_path, subset="other").load_problem("tiny")
    with pytest.raises(KeyError):
        _dataset(tmp_path, subset_kind="socp").load_problem("tiny")
    with pytest.raises(KeyError):
        _dataset(tmp_path, max_size_mb=1e-6).load_problem("tiny")
    with pytest.raises(KeyError):
        _dataset(tmp_path).load_problem("missing")
    assert _dataset(tmp_path, subset_kind="lp").load_problem("tiny").name == "tiny"


def test_inspect_reads_the_header_only(tmp_path: Path):
    # Everything after the first coordinate section is unreadable: the
    # header still inspects, the full read fails.
    body = "VER\n3\nOBJSENSE\nMAX\nVAR\n2 1\nL+ 2\nCON\n1 1\nQ 1\nACOORD\nnot a number\n"
    path = tmp_path / "header.cbf.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(body)
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) - 6])  # truncated tail as well

    metadata = inspect_cbf(path)

    assert metadata["num_variables"] == 2
    assert metadata["constraint_cones"] == {"q": [1]}
    assert metadata["original_obj_type"] == "max"
    with pytest.raises((ValueError, EOFError, OSError)):
        read_cbf_cone_problem(path)


def test_listing_skips_an_unreadable_file_with_a_warning(tmp_path: Path):
    folder = _cblib_folder(tmp_path)
    (folder / "tiny.cbf").write_text(_TINY_CBF)
    (folder / "broken.cbf.gz").write_bytes(gzip.compress(_TINY_CBF.encode())[:12])
    (folder / "garbled.cbf").write_text("VER\n3\nVAR\nx y\n")
    # An intact gzip header over a damaged deflate stream raises zlib.error,
    # which is neither an OSError nor a ValueError.
    header = gzip.compress(_TINY_CBF.encode())[:10]
    (folder / "damaged.cbf.gz").write_bytes(header + b"\x07" + b"\x00" * 20)

    with pytest.warns(UserWarning) as caught:
        specs = _dataset(tmp_path).list_problems()

    assert [spec.name for spec in specs] == ["tiny"]
    messages = " ".join(str(warning.message) for warning in caught)
    assert "broken.cbf.gz" in messages and "damaged.cbf.gz" in messages


def test_plain_and_gzipped_copies_are_one_problem(tmp_path: Path):
    folder = _cblib_folder(tmp_path)
    (folder / "tiny.cbf").write_text(_TINY_CBF)
    (folder / "tiny.cbf.gz").write_bytes(gzip.compress(_TINY_CBF.encode()))
    (folder / "zipped.cbf.gz").write_bytes(gzip.compress(_TINY_CBF.encode()))

    specs = _dataset(tmp_path).list_problems()

    assert [(spec.name, spec.path.name) for spec in specs] == [
        ("tiny", "tiny.cbf"),
        ("zipped", "zipped.cbf.gz"),
    ]


def test_a_plain_and_gzipped_pair_is_filtered_by_the_smaller_copy(tmp_path: Path):
    folder = _cblib_folder(tmp_path)
    padded = _TINY_CBF + "# padding\n" * 5000
    (folder / "tiny.cbf").write_text(padded)
    (folder / "tiny.cbf.gz").write_bytes(gzip.compress(padded.encode()))
    gz_size = (folder / "tiny.cbf.gz").stat().st_size
    dataset = _dataset(tmp_path, max_size_mb=0.01)

    (spec,) = dataset.visible_problems()
    assert spec.path == folder / "tiny.cbf"
    assert spec.metadata["size_bytes"] == gz_size
    assert dataset.load_problem("tiny").metadata["size_bytes"] == gz_size


def test_loading_an_unreadable_file_reports_why(tmp_path: Path):
    folder = _cblib_folder(tmp_path)
    (folder / "broken.cbf.gz").write_bytes(gzip.compress(_TINY_CBF.encode())[:12])

    with pytest.raises(EOFError):
        _dataset(tmp_path).load_problem("broken")


def test_requested_names_leave_out_present_files_the_listing_filters(tmp_path: Path):
    folder = _cblib_folder(tmp_path)
    (folder / "lp.cbf").write_text(_TINY_CBF)  # linear only
    (folder / "psd.cbf").write_text("VER\n3\nPSDVAR\n1\n2\n")  # unsupported
    dataset = _dataset(tmp_path, subset="lp,psd,absent", subset_kind="socp")

    assert dataset.list_problems() == []
    # On disk but filtered out: not missing. Not on disk: requested.
    assert dataset.requested_problem_names() == {"absent"}
    assert _dataset(tmp_path, subset="lp,psd,absent").requested_problem_names() == {"lp", "absent"}
