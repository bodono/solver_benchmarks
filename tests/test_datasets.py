import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io
from click.testing import CliRunner

from solver_benchmarks.cli import main
from solver_benchmarks.datasets import cblib as cblib_module
from solver_benchmarks.datasets import dimacs as dimacs_module
from solver_benchmarks.datasets import get_dataset, list_datasets
from solver_benchmarks.datasets import mpc_qpbenchmark as mpc_module
from solver_benchmarks.datasets import mps as mps_module
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
        "liu_pataki",
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


def test_liu_pataki_dataset_converts_sedumi_psd_blocks(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "liu_pataki_data"
    folder.mkdir(parents=True)
    scipy.io.savemat(
        folder / "weak_clean_1_2_7.mat",
        {
            "A": np.array([[1.0, 2.0, 2.0, 3.0]]),
            "b": np.array([[5.0]]),
            "c": np.array([[4.0], [6.0], [6.0], [8.0]]),
            "K": {"s": np.array([[2]])},
        },
    )

    dataset = get_dataset("liu_pataki")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("weak_clean_1_2_7")

    assert problem.kind == "cone"
    assert problem.cone["cone"] == {"z": 1, "s": [2]}
    assert problem.cone["q"].tolist() == pytest.approx([4.0, 6.0 * np.sqrt(2.0), 8.0])
    assert problem.cone["b"].tolist() == pytest.approx([5.0, 0.0, 0.0, 0.0])
    assert problem.cone["A"].toarray() == pytest.approx(
        np.array(
            [
                [1.0, 2.0 * np.sqrt(2.0), 3.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        )
    )
    assert problem.metadata["classification"] == "weak"
    assert problem.metadata["conditioning"] == "clean"
    assert problem.metadata["constraint_count"] == 1
    assert problem.metadata["block_dim"] == 2
    assert problem.metadata["sample_index"] == 7


def test_liu_pataki_dataset_filters_bundled_groups():
    dataset = get_dataset("liu_pataki")(
        classification="weak",
        conditioning="messy",
        constraint_count=20,
        block_dim=10,
    )

    problems = dataset.list_problems()

    assert len(problems) == 100
    assert problems[0].name == "weak_messy_20_10_1"
    assert problems[-1].name == "weak_messy_20_10_100"
    assert {spec.metadata["classification"] for spec in problems} == {"weak"}
    assert {spec.metadata["conditioning"] for spec in problems} == {"messy"}
    assert {spec.metadata["constraint_count"] for spec in problems} == {20}


def test_dataset_data_status_reports_local_problem_counts():
    synthetic = get_dataset("synthetic_qp")()
    dimacs = get_dataset("dimacs")()
    liu_pataki = get_dataset("liu_pataki")()

    assert synthetic.data_status().available
    assert synthetic.data_status().problem_count == 2
    assert dimacs.data_status().available
    assert dimacs.data_status().problem_count > 0
    assert liu_pataki.data_status().available
    assert liu_pataki.data_status().problem_count == 800



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


def test_validate_gzip_payload_rejects_truncated_archive():
    """A truncated gzip stream (header valid, CRC/EOF missing) must be
    rejected. The previous one-byte ``GzipFile.read(1)`` probe passed
    on these payloads, atomically writing them to the cache and baking
    the corruption in for every subsequent run.
    """
    import gzip as _gzip

    import pytest

    from solver_benchmarks.datasets.base import validate_gzip_payload

    full = _gzip.compress(b"hello world" * 100)
    # Drop the trailing CRC32+ISIZE (8 bytes) so the EOF marker is missing.
    truncated = full[:-4]
    with pytest.raises((EOFError, OSError)):
        validate_gzip_payload(truncated)


def test_validate_gzip_payload_accepts_well_formed_archive():
    import gzip as _gzip

    from solver_benchmarks.datasets.base import validate_gzip_payload

    # No exception expected.
    validate_gzip_payload(_gzip.compress(b"hello"))


def test_cblib_download_does_not_atomically_commit_truncated_gzip(
    monkeypatch, tmp_path: Path
):
    """Pin the contract that a truncated gzip from the network is
    rejected before atomic_write_bytes lands it on disk."""
    import gzip as _gzip
    import urllib.request

    import pytest

    folder = tmp_path / "problem_classes" / "cblib_data"
    full = _gzip.compress(b"\n".join([b"VER", b"1", b"OBJSENSE", b"MIN"]))
    truncated = full[:-4]

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: _FakeResponse(truncated),
    )
    with pytest.raises((EOFError, OSError)):
        cblib_module.download_cblib_problem("never_committed", folder)
    # No file was written at the cache target.
    assert not (folder / "never_committed.cbf.gz").exists()


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


def test_kennington_prepare_uses_bundled_cache_and_solves(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    # NETLIB serves Kennington in EMPS (compressed-MPS) format which
    # qpsreader cannot parse, so the dataset is intentionally bundled-only.
    # Pin that prepare never touches the network and that a real bundled
    # instance round-trips through prepare → load → solve.
    from solver_benchmarks.core import status

    def fail_if_network_is_used(*args, **kwargs):
        raise AssertionError("Kennington bundled prepare unexpectedly used the network")

    monkeypatch.setattr(mps_module.urllib.request, "urlopen", fail_if_network_is_used)
    data_root = tmp_path / "problem_classes"
    dataset = get_dataset("kennington")(repo_root=repo_root, data_root=data_root)

    dataset.prepare_data(["ken-07"])

    target = data_root / "kennington" / "ken-07.mps.gz"
    assert target.exists()
    assert (
        target.read_bytes()
        == (repo_root / "problem_classes/kennington/ken-07.mps.gz").read_bytes()
    )

    problem = dataset.load_problem("ken-07")
    solver_cls = get_solver("clarabel")
    assert problem.kind in solver_cls.supported_problem_kinds
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = solver_cls({"verbose": False, "max_iter": 200}).solve(problem, artifacts)
    assert result.status in {
        status.OPTIMAL,
        status.OPTIMAL_INACCURATE,
        *status.ANY_INFEASIBLE,
    }


def test_dimacs_prepare_uses_bundled_cache_before_network(
    monkeypatch, tmp_path: Path, repo_root: Path
):
    def fail_if_network_is_used(*args, **kwargs):
        raise AssertionError("DIMACS bundled prepare unexpectedly used the network")

    monkeypatch.setattr(dimacs_module.urllib.request, "urlopen", fail_if_network_is_used)
    data_root = tmp_path / "problem_classes"
    dataset = get_dataset("dimacs")(
        repo_root=repo_root,
        data_root=data_root,
    )

    dataset.prepare_data(["nb"])

    target = data_root / "dimacs_data" / "nb.mat.gz"
    assert target.exists()
    assert (
        target.read_bytes()
        == (repo_root / "problem_classes/dimacs_data/nb.mat.gz").read_bytes()
    )


def test_sdplib_dataset_publishes_tar_member_sizes(tmp_path: Path):
    """Tar members share ProblemSpec.path with the whole archive, so
    SDPLIB must surface per-member sizes via metadata so the runner-level
    size filter can compare against the member, not the archive."""
    from solver_benchmarks.datasets.base import filter_problem_specs_by_size

    data_root = tmp_path / "problem_classes"
    folder = data_root / "sdplib_data"
    folder.mkdir(parents=True)

    with tarfile.open(folder / "sdplib.tar", "w") as archive:
        for name, size in [
            ("tar_small.jld2", 10),
            ("tar_large.jld2", 1_500_001),
        ]:
            data = b"x" * size
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    dataset = get_dataset("sdplib")(repo_root=tmp_path, data_root=data_root)
    specs = dataset.list_problems()

    assert [spec.name for spec in specs] == ["tar_large", "tar_small"]
    sizes = {spec.name: spec.metadata["size_bytes"] for spec in specs}
    assert sizes == {"tar_small": 10, "tar_large": 1_500_001}

    # The runner-level filter must drop the oversized member even though
    # the whole sdplib.tar archive (which is the spec.path) is over 1 MB.
    filtered = filter_problem_specs_by_size(specs, 1.0)
    assert [spec.name for spec in filtered] == ["tar_small"]


def test_generic_size_filter_applies_to_dataset_visibility_and_cli(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "sdplib_data"
    folder.mkdir(parents=True)

    with tarfile.open(folder / "sdplib.tar", "w") as archive:
        for name, size in [
            ("tar_small.jld2", 10),
            ("tar_large.jld2", 1_500_001),
        ]:
            data = b"x" * size
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    dataset = get_dataset("sdplib")(
        repo_root=tmp_path,
        data_root=data_root,
        max_size_mb=1.0,
    )
    status = dataset.data_status()

    assert [spec.name for spec in dataset.visible_problems()] == ["tar_small"]
    assert status.available
    assert status.problem_count == 1
    assert "1 of 2 problems visible after filters" in status.message
    with pytest.raises(KeyError):
        dataset.problem_by_name("tar_large")

    result = CliRunner().invoke(
        main,
        [
            "list",
            "problems",
            "sdplib",
            "--repo-root",
            str(tmp_path),
            "--option",
            f"data_root={data_root}",
            "--option",
            "max_size_mb=1.0",
        ],
    )

    assert result.exit_code == 0
    assert "tar_small" in result.output
    assert "tar_large" not in result.output


def test_mps_size_filter_uses_smallest_duplicate_encoding(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "miplib_data"
    folder.mkdir(parents=True)
    uncompressed = folder / "dup.mps"
    compressed = folder / "dup.mps.gz"
    large_only = folder / "large_only.mps"
    uncompressed.write_bytes(b"x" * 1_500_001)
    compressed.write_bytes(gzip.compress(b"x" * 10))
    large_only.write_bytes(b"x" * 1_500_001)

    dataset = get_dataset("miplib")(
        repo_root=tmp_path,
        data_root=data_root,
        max_size_mb=1.0,
    )
    visible = {spec.name: spec for spec in dataset.visible_problems()}

    assert set(visible) == {"dup"}
    assert visible["dup"].path == uncompressed
    assert visible["dup"].metadata["size_bytes"] == compressed.stat().st_size
    assert dataset.problem_by_name("dup").path == uncompressed
    with pytest.raises(KeyError):
        dataset.problem_by_name("large_only")


def test_miplib_prepare_default_downloads_tiny_subset(monkeypatch, tmp_path: Path):
    calls = []

    def fake_download(name, folder):
        calls.append((name, folder))

    monkeypatch.setattr(mps_module, "_download_miplib_problem", fake_download)
    data_root = tmp_path / "problem_classes"
    dataset = get_dataset("miplib")(repo_root=tmp_path, data_root=data_root)

    dataset.prepare_data()

    assert [name for name, _ in calls] == list(mps_module.MIPLIB_DEFAULT_SUBSET)
    assert all(folder == data_root / "miplib_data" for _, folder in calls)


def test_miplib_prepare_max_size_uses_remote_benchmark_sizes(
    monkeypatch,
    tmp_path: Path,
):
    calls = []
    sizes = {
        "tiny": 999_999,
        "large": 1_000_001,
    }

    monkeypatch.setattr(
        mps_module,
        "_miplib_remote_problem_names",
        lambda: ["tiny", "large"],
    )
    monkeypatch.setattr(
        mps_module,
        "_miplib_remote_size_bytes",
        lambda name: sizes[name],
    )
    monkeypatch.setattr(
        mps_module,
        "_download_miplib_problem",
        lambda name, folder: calls.append((name, folder)),
    )
    data_root = tmp_path / "problem_classes"
    dataset = get_dataset("miplib")(
        repo_root=tmp_path,
        data_root=data_root,
        max_size_mb=1.0,
    )

    dataset.prepare_data()

    assert [name for name, _ in calls] == ["tiny"]


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


def _write_minimal_qplib(path: Path, *, direction: str = "minimize") -> None:
    """Write a tiny QPLIB-format file for a 1-D max problem ``-(x-1)^2``."""
    body = "\n".join(
        [
            "TINY",
            "QCB",
            direction,
            "1 variables",
            "0 constraints",
            "1 quadratic objective",
            "1 1 -2.0",  # P = -2 in lower-tri (x^2 coefficient = -1, doubled by 1/2 conv)
            "0.0",  # q default
            "1",  # 1 non-default q
            "1 2.0",  # q[1] = 2.0
            "-1.0",  # r = -1
            "1.0e30",  # infty marker
            "0.0",  # lx default
            "0",  # 0 non-default lx
            "0.0",  # ux default
            "0",
            "0.0",  # x0 default
            "0",
            "0.0",  # y0 default
            "0",
            "0.0",  # w0 default
            "0",
            "",
        ]
    )
    path.write_text(body)


def test_qplib_loader_negates_max_data_and_reports_min_obj_type(tmp_path: Path):
    """A max-form QPLIB file must be loaded as min-form so the worker
    does not double-negate the reported objective value."""
    data_root = tmp_path / "problem_classes"
    folder = data_root / "qplib_data"
    folder.mkdir(parents=True)
    (folder / "list_convex_qps.txt").write_text("CCB (1)\n-------\n9999\n")
    _write_minimal_qplib(folder / "QPLIB_9999.qplib", direction="maximize")

    dataset = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("9999")

    qp = problem.qp
    # obj_type is min-form so the worker leaves the reported objective alone.
    assert qp["obj_type"] == "min"
    # Original direction is preserved as metadata for reporting.
    assert problem.metadata.get("original_obj_type") == "max"
    # Quadratic + linear data was negated; constant offset r flipped sign.
    assert qp["r"] == 1.0  # original r = -1, negated for min form
    assert qp["q"][0] == -2.0  # original q[0] = 2, negated


def test_qplib_loader_handles_min_direction_without_negation(tmp_path: Path):
    """Min-form QPLIB files must NOT be negated."""
    data_root = tmp_path / "problem_classes"
    folder = data_root / "qplib_data"
    folder.mkdir(parents=True)
    (folder / "list_convex_qps.txt").write_text("CCB (1)\n-------\n8888\n")
    _write_minimal_qplib(folder / "QPLIB_8888.qplib", direction="minimize")

    dataset = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("8888")
    qp = problem.qp
    assert qp["obj_type"] == "min"
    assert problem.metadata.get("original_obj_type") == "min"
    assert qp["r"] == -1.0  # unchanged
    assert qp["q"][0] == 2.0  # unchanged


def test_qplib_loader_with_constraints_assembles_a_and_bounds(tmp_path: Path):
    """The single-pass parser must produce the same A/l/u shapes as
    the old multi-read implementation when constraints are present.

    Lays out a 1-variable, 1-constraint min problem with a single A
    nonzero, default lower=0 and upper=10."""
    data_root = tmp_path / "problem_classes"
    folder = data_root / "qplib_data"
    folder.mkdir(parents=True)
    (folder / "list_convex_qps.txt").write_text("CCB (1)\n-------\n7777\n")
    body = "\n".join(
        [
            "TINY_WITH_CON",
            "QCB",
            "minimize",
            "1 variables",
            "1 constraints",  # m = 1
            "0 quadratic objective",  # nnz_Ptriu = 0
            "0.0",  # q default
            "0",
            "0.0",  # r
            "1 linear constraint coefficient",  # nnz_A = 1
            "1 1 1.0",  # A[0,0] = 1
            "1.0e30",  # infty marker
            "0.0",  # l default
            "0",
            "10.0",  # u default
            "0",
            "0.0",  # lx default
            "0",
            "5.0",  # ux default
            "0",
            "0.0",  # x0 default
            "0",
            "0.0",  # y0 default
            "0",
            "0.0",  # w0 default
            "0",
            "",
        ]
    )
    (folder / "QPLIB_7777.qplib").write_text(body)

    dataset = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("7777")
    qp = problem.qp
    # Combined: 1 user constraint + 1 variable bound row -> m = 2.
    assert qp["m"] == 2
    assert qp["A"].shape == (2, 1)
    # First row is the user constraint with coefficient 1.
    assert qp["A"].toarray()[0, 0] == 1.0
    # Bounds: [user_l=0, var_l=0] / [user_u=10, var_u=5].
    assert list(qp["l"]) == [0.0, 0.0]
    assert list(qp["u"]) == [10.0, 5.0]


def test_qplib_loader_tolerates_blank_lines(tmp_path: Path):
    """The new cursor strips blank lines so a stray trailing blank or
    extra blank between sections doesn't shift later parses."""
    data_root = tmp_path / "problem_classes"
    folder = data_root / "qplib_data"
    folder.mkdir(parents=True)
    (folder / "list_convex_qps.txt").write_text("CCB (1)\n-------\n6666\n")
    body = "\n".join(
        [
            "TINY_BLANK",
            "QCB",
            "",  # spurious blank line before direction
            "minimize",
            "1 variables",
            "0 constraints",
            "0 quadratic objective",
            "0.0",
            "0",
            "0.0",
            "1.0e30",
            "0.0",
            "0",
            "0.0",
            "0",
            "",  # spurious blank line in middle
            "0.0",
            "0",
            "0.0",
            "0",
            "0.0",
            "0",
            "",
        ]
    )
    (folder / "QPLIB_6666.qplib").write_text(body)
    dataset = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root)
    problem = dataset.load_problem("6666")
    assert problem.qp["n"] == 1
