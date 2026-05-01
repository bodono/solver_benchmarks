"""Tests for the dc_opf dataset, MATPOWER parser, and DC OPF LP.

Synthesizes tiny MATPOWER ``.m`` files (a 3-bus economic-dispatch
toy and a 2-bus minimal example) so the tests run without network
access. Covers the parser, the LP construction (power balance,
reference bus, generator bounds, line flow limits, linear cost),
the dataset's list_problems / load_problem flow, the subset filter,
and end-to-end solving.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# MATPOWER fixtures.
# ---------------------------------------------------------------------------


# A 3-bus DC OPF: 2 generators (at buses 1 and 2), 1 load (at bus 3),
# 3 lines forming a triangle. Cheap generator at bus 1 ($10/MWh),
# expensive at bus 2 ($30/MWh). Load = 100 MW at bus 3. Generator
# limits and line ratings are loose so the optimum is to source
# everything from generator 1 (60 MW) and the rest from generator 2.
# Actually with 100 MW load and balanced topology the optimum
# depends on line limits — fixture is sized so optimum is interior.
THREE_BUS = """\
function mpc = case3
mpc.version = '2';
mpc.baseMVA = 100;

%% bus data
%   bus_id  type  Pd     Qd  Gs  Bs  area  Vm  Va  baseKV  zone  Vmax  Vmin
mpc.bus = [
    1  3   0.0   0   0   0   1   1   0   135   1   1.05   0.95;
    2  2   0.0   0   0   0   1   1   0   135   1   1.05   0.95;
    3  1   100.0 0   0   0   1   1   0   135   1   1.05   0.95;
];

%% generator data
%   bus  Pg    Qg  Qmax  Qmin  Vg  mBase  status  Pmax  Pmin  ...
mpc.gen = [
    1  0   0   100   -100   1   100   1   200   0;
    2  0   0   100   -100   1   100   1   200   0;
];

%% branch data
%   fbus  tbus  r     x     b   rateA  rateB  rateC  ratio  angle  status  angmin  angmax
mpc.branch = [
    1  2   0.0   0.1   0   200   200   200   0   0   1   -360   360;
    2  3   0.0   0.1   0   200   200   200   0   0   1   -360   360;
    1  3   0.0   0.1   0   200   200   200   0   0   1   -360   360;
];

%% generator cost data
%   model  startup  shutdown  n  c1  c0
mpc.gencost = [
    2  0   0   2  10   0;
    2  0   0   2  30   0;
];
"""


# A degenerate 2-bus case with one constraint to test edge cases:
# 1 generator, 1 load, 1 line. All flow goes through the line.
TWO_BUS = """\
function mpc = case2tiny
mpc.version = '2';
mpc.baseMVA = 100;
mpc.bus = [
    1  3   0.0   0   0   0   1   1   0   135   1   1.05   0.95;
    2  1   50.0  0   0   0   1   1   0   135   1   1.05   0.95;
];
mpc.gen = [
    1  0   0   100   -100   1   100   1   100   0;
];
mpc.branch = [
    1  2   0.0   0.1   0   100   100   100   0   0   1   -360   360;
];
mpc.gencost = [
    2  0   0   2  20   5;
];
"""


def _write_m(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.m"
    path.write_text(body)
    return path


def _local_dataset(folder: Path, **options):
    from solver_benchmarks.datasets.dc_opf import DCOPFDataset

    class _Local(DCOPFDataset):
        @property
        def folder(self) -> Path:  # type: ignore[override]
            return folder

        @property
        def data_dir(self) -> Path:  # type: ignore[override]
            return folder

    return _Local(**options)


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


def test_dc_opf_is_registered():
    from solver_benchmarks.datasets import registry

    cls = registry.get_dataset("dc_opf")
    assert cls.dataset_id == "dc_opf"


# ---------------------------------------------------------------------------
# MATPOWER parser.
# ---------------------------------------------------------------------------


def test_parse_matpower_extracts_scalar_basemva():
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case

    case = parse_matpower_case(THREE_BUS)
    assert case["baseMVA"] == 100.0


def test_parse_matpower_extracts_bus_matrix_with_correct_shape():
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case

    case = parse_matpower_case(THREE_BUS)
    bus = case["bus"]
    assert bus.shape == (3, 13)
    # First column: bus IDs.
    assert bus[:, 0].astype(int).tolist() == [1, 2, 3]
    # Bus 1 is the reference (type = 3).
    assert bus[0, 1] == 3.0


def test_parse_matpower_extracts_gen_branch_gencost():
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case

    case = parse_matpower_case(THREE_BUS)
    assert case["gen"].shape == (2, 10)
    assert case["branch"].shape == (3, 13)
    assert case["gencost"].shape == (2, 6)


def test_parse_matpower_strips_inline_percent_comments(tmp_path: Path):
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case

    body = """\
function mpc = inline
mpc.baseMVA = 100;
mpc.bus = [
    1 3 0 0 0 0 1 1 0 135 1 1.05 0.95;  % reference bus
    2 1 50 0 0 0 1 1 0 135 1 1.05 0.95;
];
mpc.gen = [
    1 0 0 100 -100 1 100 1 100 0;
];
mpc.branch = [
    1 2 0 0.1 0 100 100 100 0 0 1 -360 360;
];
mpc.gencost = [
    2 0 0 2 20 0;
];
"""
    case = parse_matpower_case(body)
    assert case["bus"].shape == (2, 13)


def test_parse_matpower_rejects_missing_basemva():
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case

    body = "function mpc = bad\nmpc.bus = [1 3 0 0 0 0 1 1 0 135 1 1.05 0.95;];\n"
    with pytest.raises(ValueError, match="baseMVA"):
        parse_matpower_case(body)


def test_parse_matpower_rejects_missing_matrix():
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case

    body = "function mpc = bad\nmpc.baseMVA = 100;\n"
    with pytest.raises(ValueError, match="mpc.bus"):
        parse_matpower_case(body)


def test_parse_matpower_rejects_inconsistent_row_widths():
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case

    body = """\
function mpc = bad
mpc.baseMVA = 100;
mpc.bus = [
    1 3 0 0 0 0 1 1 0 135 1 1.05 0.95;
    2 1 50 0 0;
];
mpc.gen = [
    1 0 0 100 -100 1 100 1 100 0;
];
mpc.branch = [
    1 2 0 0.1 0 100 100 100 0 0 1 -360 360;
];
mpc.gencost = [
    2 0 0 2 20 0;
];
"""
    with pytest.raises(ValueError, match="inconsistent width"):
        parse_matpower_case(body)


# ---------------------------------------------------------------------------
# DC OPF LP construction.
# ---------------------------------------------------------------------------


def test_dc_opf_lp_shape_and_variable_layout():
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case
    from solver_benchmarks.transforms.dc_opf import dc_opf_lp

    case = parse_matpower_case(THREE_BUS)
    problem, metadata = dc_opf_lp(case)
    n_gen = 2
    n_bus = 3
    n_vars = n_gen + n_bus
    assert problem["P"].shape == (n_vars, n_vars)
    assert problem["P"].nnz == 0  # LP, P is zero.
    assert problem["q"].shape == (n_vars,)
    # Linear costs in $/MW: row col 4 of gencost is c1 = [10, 30].
    # Convert to per-unit by multiplying by baseMVA = 100 → [1000, 3000].
    assert problem["q"][:n_gen].tolist() == [1000.0, 3000.0]
    # Theta variables have zero cost.
    assert (problem["q"][n_gen:] == 0.0).all()
    assert metadata["num_buses"] == 3
    assert metadata["num_generators"] == 2
    assert metadata["num_branches_active"] == 3


def test_dc_opf_lp_includes_reference_bus_constraint():
    """θ_ref must be fixed to 0; the LP must include an equality row
    that picks out the reference-bus angle."""
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case
    from solver_benchmarks.transforms.dc_opf import dc_opf_lp

    case = parse_matpower_case(THREE_BUS)
    problem, metadata = dc_opf_lp(case)
    # The reference-bus row is the row right after the n_bus power-
    # balance equality rows. Power-balance has n_bus = 3 rows, so
    # the reference row sits at index 3.
    a = problem["A"].toarray()
    n_gen = 2
    ref_row = a[3]
    expected_col = n_gen + metadata["reference_bus_index"]
    assert ref_row[expected_col] == 1.0
    # The bound is exactly zero.
    assert problem["l"][3] == 0.0 and problem["u"][3] == 0.0


def test_dc_opf_lp_generator_bounds_use_per_unit():
    """Generator bounds in the LP are per-unit (MW / baseMVA)."""
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case
    from solver_benchmarks.transforms.dc_opf import dc_opf_lp

    case = parse_matpower_case(THREE_BUS)
    problem, _ = dc_opf_lp(case)
    # Bounds rows are at the bottom (last n_gen rows). Generator 1
    # has Pmax = 200 MW = 2.0 p.u., Pmin = 0.
    # Last 2 rows are the gen bounds.
    n_total = problem["A"].shape[0]
    pg_lower = problem["l"][n_total - 2 : n_total]
    pg_upper = problem["u"][n_total - 2 : n_total]
    assert pg_lower.tolist() == [0.0, 0.0]
    assert pg_upper.tolist() == [2.0, 2.0]


def test_dc_opf_lp_skips_unlimited_lines():
    """A branch with rateA = 0 (MATPOWER convention for unlimited)
    must not produce a flow-limit row."""
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case
    from solver_benchmarks.transforms.dc_opf import dc_opf_lp

    body = THREE_BUS.replace(
        "    1  2   0.0   0.1   0   200",
        "    1  2   0.0   0.1   0   0  ",
    )
    case = parse_matpower_case(body)
    _, metadata = dc_opf_lp(case)
    # 3 lines total, 1 unlimited → 2 flow-limit rows.
    assert metadata["num_lines_with_flow_limit"] == 2


def test_dc_opf_lp_solves_to_known_optimum_with_clarabel(tmp_path: Path):
    """Three-bus economic-dispatch problem: gen 1 ($10/MWh) wants to
    serve the entire 100 MW load. Line 1→3 has limit 200 MW so it
    can; gen 2 idles. Optimum cost = 10 * 100 = $1000."""
    pytest.importorskip("clarabel")
    from solver_benchmarks.core import status
    from solver_benchmarks.core.problem import QP, ProblemData
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case
    from solver_benchmarks.solvers.clarabel_adapter import ClarabelSolverAdapter
    from solver_benchmarks.transforms.dc_opf import dc_opf_lp

    case = parse_matpower_case(THREE_BUS)
    problem, _ = dc_opf_lp(case)
    pd = ProblemData("test", "case3", QP, problem)
    adapter = ClarabelSolverAdapter({"verbose": False})
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = adapter.solve(pd, artifacts)
    assert result.status == status.OPTIMAL
    # The LP minimum equals the dispatch cost in $/h.
    # 100 MW * $10/MWh = $1000/h.
    # In LP coords: q = [1000, 3000, 0, 0, 0], optimum at Pg = [1, 0]
    # in p.u. so q'x = 1000.
    assert result.objective_value == pytest.approx(1000.0, abs=1.0)
    assert result.kkt is not None
    assert result.kkt["primal_res_rel"] < 1e-4
    assert result.kkt["dual_res_rel"] < 1e-4


def test_dc_opf_lp_two_bus_minimal_case_solves(tmp_path: Path):
    pytest.importorskip("clarabel")
    from solver_benchmarks.core import status
    from solver_benchmarks.core.problem import QP, ProblemData
    from solver_benchmarks.datasets.dc_opf import parse_matpower_case
    from solver_benchmarks.solvers.clarabel_adapter import ClarabelSolverAdapter
    from solver_benchmarks.transforms.dc_opf import dc_opf_lp

    case = parse_matpower_case(TWO_BUS)
    problem, metadata = dc_opf_lp(case)
    assert metadata["num_buses"] == 2
    assert metadata["num_generators"] == 1
    pd = ProblemData("test", "case2tiny", QP, problem)
    result = ClarabelSolverAdapter({"verbose": False}).solve(pd, tmp_path)
    assert result.status == status.OPTIMAL


def test_dc_opf_lp_rejects_case_with_no_buses():
    from solver_benchmarks.transforms.dc_opf import dc_opf_lp

    case = {
        "baseMVA": 100.0,
        "bus": np.zeros((0, 13)),
        "gen": np.zeros((1, 10)),
        "branch": np.zeros((0, 13)),
        "gencost": np.zeros((0, 6)),
    }
    with pytest.raises(ValueError, match="no buses"):
        dc_opf_lp(case)


def test_dc_opf_lp_rejects_case_with_no_generators():
    from solver_benchmarks.transforms.dc_opf import dc_opf_lp

    case = {
        "baseMVA": 100.0,
        "bus": np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 135, 1, 1.05, 0.95]]),
        "gen": np.zeros((0, 10)),
        "branch": np.zeros((0, 13)),
        "gencost": np.zeros((0, 6)),
    }
    with pytest.raises(ValueError, match="no generators"):
        dc_opf_lp(case)


# ---------------------------------------------------------------------------
# Dataset list_problems / load_problem.
# ---------------------------------------------------------------------------


@pytest.fixture
def dc_opf_data_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "dc_opf_data"
    _write_m(folder, "case3", THREE_BUS)
    _write_m(folder, "case2tiny", TWO_BUS)
    return folder


def test_list_problems_returns_specs_for_all_local_files(dc_opf_data_folder: Path):
    dataset = _local_dataset(dc_opf_data_folder)
    names = sorted(spec.name for spec in dataset.list_problems())
    assert names == ["case2tiny", "case3"]


def test_list_problems_filters_by_subset(dc_opf_data_folder: Path):
    dataset = _local_dataset(dc_opf_data_folder, subset="case3")
    names = [spec.name for spec in dataset.list_problems()]
    assert names == ["case3"]


def test_load_problem_returns_canonical_qp_form_lp(dc_opf_data_folder: Path):
    dataset = _local_dataset(dc_opf_data_folder)
    pd = dataset.load_problem("case3")
    qp = pd.qp
    assert qp["P"].nnz == 0  # LP.
    # 2 generators + 3 buses = 5 vars.
    assert qp["q"].size == 5
    assert pd.metadata["num_buses"] == 3
    assert pd.metadata["num_generators"] == 2
    assert pd.metadata["format"] == "matpower-dc-opf"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def test_normalize_subset_none_means_no_filter():
    from solver_benchmarks.datasets.dc_opf import _normalize_subset

    assert _normalize_subset(None) is None
    assert _normalize_subset("all") is None


def test_normalize_subset_accepts_comma_string_and_list():
    from solver_benchmarks.datasets.dc_opf import _normalize_subset

    assert _normalize_subset("case5, case9") == {"case5", "case9"}
    assert _normalize_subset(["case14"]) == {"case14"}
