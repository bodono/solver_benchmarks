import gzip
from pathlib import Path

import numpy as np
import pytest

from solver_benchmarks.datasets import cblib as cblib_module
from solver_benchmarks.datasets import get_dataset, list_datasets
from solver_benchmarks.datasets import mpc_qpbenchmark as mpc_module
from solver_benchmarks.datasets import qplib as qplib_module
from solver_benchmarks.solvers import get_solver, list_solvers


_TINY_MPS = """\
NAME          TINY
ROWS
 N  COST
 L  C1
COLUMNS
    X1   COST        1.0   C1   1.0
    X2   COST        2.0   C1   1.0
RHS
    RHS  C1   3.0
BOUNDS
 LO BND  X1   0.0
 LO BND  X2   0.0
ENDATA
"""


def test_required_datasets_are_registered():
    required = {
        "cblib",
        "cutest_qp",
        "kennington",
        "maros_meszaros",
        "mpc_qpbenchmark",
        "netlib",
        "miplib",
        "miplib_lp_relaxation",
        "qplib",
        "mittelmann",
        "sdplib",
        "dimacs",
        "synthetic_cone",
    }

    assert required.issubset(set(list_datasets()))


def test_required_solvers_are_registered():
    required = {
        "qtqp",
        "scs",
        "clarabel",
        "osqp",
        "mosek",
        "gurobi",
        "pdlp",
        "highs",
        "proxqp",
        "piqp",
        "sdpa",
        "cplex",
    }

    assert required.issubset(set(list_solvers()))
    assert "cone" in get_solver("clarabel").supported_problem_kinds


def test_commercial_adapters_use_solver_specific_modules():
    from solver_benchmarks.solvers.gurobi_adapter import GurobiSolverAdapter
    from solver_benchmarks.solvers.mosek_adapter import MosekSolverAdapter

    assert get_solver("gurobi") is GurobiSolverAdapter
    assert get_solver("mosek") is MosekSolverAdapter
    assert not list(Path("solver_benchmarks/solvers").glob("legacy_*.py"))


def test_synthetic_dataset_loads_qp():
    dataset = get_dataset("synthetic_qp")()
    problems = dataset.list_problems()
    problem = dataset.load_problem(problems[0].name)

    assert {problem.name for problem in problems} == {"one_variable_eq", "one_variable_lp"}
    assert problem.kind == "qp"
    assert problem.qp["P"].shape == (1, 1)
    assert problem.qp["A"].shape == (1, 1)


def test_synthetic_cone_dataset_loads_cone():
    dataset = get_dataset("synthetic_cone")()
    problems = dataset.list_problems()
    problem = dataset.load_problem(problems[0].name)

    assert {problem.name for problem in problems} == {"one_variable_cone_lp"}
    assert problem.kind == "cone"
    assert problem.cone["A"].shape == (1, 1)
    assert problem.cone["cone"] == {"l": 1}


def test_dataset_data_status_reports_local_problem_counts():
    synthetic = get_dataset("synthetic_qp")()
    dimacs = get_dataset("dimacs")()

    assert synthetic.data_status().available
    assert synthetic.data_status().problem_count == 2
    assert dimacs.data_status().available
    assert dimacs.data_status().problem_count > 0


def test_cblib_cbf_parser_loads_supported_continuous_problem(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "cblib_data"
    folder.mkdir(parents=True)
    (folder / "tiny.cbf").write_text(
        "\n".join(
            [
                "VER",
                "1",
                "OBJSENSE",
                "MIN",
                "VAR",
                "2 2",
                "L+ 1",
                "F 1",
                "CON",
                "2 2",
                "L= 1",
                "L+ 1",
                "OBJACOORD",
                "1",
                "1 2.0",
                "ACOORD",
                "2",
                "0 0 1.0",
                "1 1 -1.0",
                "BCOORD",
                "2",
                "0 -1.0",
                "1 3.0",
            ]
        )
        + "\n"
    )

    dataset = get_dataset("cblib")(repo_root=tmp_path, data_root=data_root)
    problems = dataset.list_problems()
    problem = dataset.load_problem("tiny")

    assert [spec.name for spec in problems] == ["tiny"]
    assert problem.kind == "cone"
    assert problem.cone["A"].shape == (3, 2)
    assert problem.cone["cone"] == {"z": 1, "l": 2}
    assert problem.cone["q"].tolist() == [0.0, 2.0]


def test_cblib_prepare_uses_default_small_subset(monkeypatch, tmp_path: Path):
    calls = []

    def fake_download(name, folder):
        calls.append((name, folder))
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{name}.cbf.gz"

    monkeypatch.setattr(cblib_module, "download_cblib_problem", fake_download)
    dataset = get_dataset("cblib")(repo_root=tmp_path, data_root=tmp_path / "problem_classes")

    dataset.prepare_data()

    assert [name for name, _ in calls] == list(cblib_module.CBLIB_DEFAULT_SUBSET)


def test_mpc_qpbenchmark_npz_loader_converts_qp_schema(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "mpc_qpbenchmark_data"
    folder.mkdir(parents=True)
    np.savez(
        folder / "toy.npz",
        P=np.eye(2),
        q=np.array([1.0, 2.0]),
        G=np.array([[1.0, 0.0]]),
        h=np.array([3.0]),
        A=np.array([[0.0, 1.0]]),
        b=np.array([4.0]),
        lb=np.array([-1.0, -2.0]),
        ub=np.array([5.0, 6.0]),
    )

    dataset = get_dataset("mpc_qpbenchmark")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("toy")

    assert problem.kind == "qp"
    assert problem.qp["P"].shape == (2, 2)
    assert problem.qp["A"].shape == (4, 2)
    assert problem.qp["l"].tolist() == pytest.approx([-1.0e20, 4.0, -1.0, -2.0])
    assert problem.qp["u"].tolist() == pytest.approx([3.0, 4.0, 5.0, 6.0])


def test_mpc_prepare_uses_default_small_subset(monkeypatch, tmp_path: Path):
    calls = []

    def fake_download(name, folder):
        calls.append((name, folder))
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{name}.npz"

    monkeypatch.setattr(mpc_module, "download_mpc_qpbenchmark_problem", fake_download)
    dataset = get_dataset("mpc_qpbenchmark")(
        repo_root=tmp_path,
        data_root=tmp_path / "problem_classes",
    )

    dataset.prepare_data()

    assert [name for name, _ in calls] == list(mpc_module.MPC_QPBENCHMARK_DEFAULT_SUBSET)


def test_netlib_dataset_loads_plain_mps_problem(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "netlib_data" / "feasible"
    folder.mkdir(parents=True)
    (folder / "tiny.mps").write_text(_TINY_MPS)

    dataset = get_dataset("netlib")(repo_root=tmp_path, data_root=data_root)
    spec = next(spec for spec in dataset.list_problems() if spec.name == "tiny")
    problem = dataset.load_problem(spec.name)

    assert problem.kind == "qp"
    assert problem.qp["A"].shape[1] == 2
    assert problem.qp["q"].tolist() == [1.0, 2.0]


def test_netlib_dataset_loads_gzipped_mps_problem(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "netlib_data" / "feasible"
    folder.mkdir(parents=True)
    (folder / "tiny.mps.gz").write_bytes(gzip.compress(_TINY_MPS.encode()))

    dataset = get_dataset("netlib")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("tiny")

    assert problem.kind == "qp"
    assert problem.qp["A"].shape[1] == 2
    assert problem.qp["q"].tolist() == [1.0, 2.0]


def test_kennington_dataset_round_trips_through_qpsreader(tmp_path: Path):
    # Regression test for the qpsreader ``.gz`` handling bug — previously
    # ``.mps.gz`` files were silently treated as missing and loaded as 0x0.
    data_root = tmp_path / "problem_classes"
    folder = data_root / "kennington"
    folder.mkdir(parents=True)
    (folder / "tiny.mps.gz").write_bytes(gzip.compress(_TINY_MPS.encode()))

    dataset = get_dataset("kennington")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("tiny")
    assert problem.qp["A"].shape[1] == 2


def test_qplib_index_and_subset_filtering(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "qplib_data"
    folder.mkdir(parents=True)
    (folder / "list_convex_qps.txt").write_text("CCB (1)\n-------\n8790\n\nDCL (1)\n--------\n8495\n")
    (folder / "QPLIB_8790.qplib").write_text("")
    (folder / "QPLIB_8495.qplib").write_text("")

    all_dataset = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root)
    ccb_dataset = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root, subset="ccb")

    assert qplib_module.qplib_index(folder) == {"8790": "ccb", "8495": "dcl"}
    assert {spec.name for spec in all_dataset.list_problems()} == {"8790", "8495"}
    assert [spec.name for spec in ccb_dataset.list_problems()] == ["8790"]
