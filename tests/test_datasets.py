import gzip
import io
import re
import tarfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io
from click.testing import CliRunner

from solver_benchmarks.cli import main
from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.data_prepare import data_prepare_selection
from solver_benchmarks.datasets import cblib as cblib_module
from solver_benchmarks.datasets import dimacs as dimacs_module
from solver_benchmarks.datasets import get_dataset, list_datasets
from solver_benchmarks.datasets import mpc_qpbenchmark as mpc_module
from solver_benchmarks.datasets import mps as mps_module
from solver_benchmarks.datasets import qplib as qplib_module
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.datasets.dc_opf import MATPOWER_REVISION
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
        "cpo3",
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


@pytest.mark.parametrize("name,row", [
    ("LIPMWALK4", 0), ("LIPMWALK10", 0), ("LIPMWALK12", 1),
    ("LIPMWALK18", 1), ("LIPMWALK20", 0), ("LIPMWALK28", 1),
])
def test_mpc_loader_clips_known_roundoff_bounds(name, row, repo_root):
    dataset = get_dataset("mpc_qpbenchmark")(repo_root=repo_root)
    if not (dataset.folder / f"{name}.npz").exists():
        pytest.skip("MPC benchmark data not downloaded")
    problem = dataset.load_problem(name)
    assert problem.qp["A"].getrow(row).nnz == 0
    assert problem.qp["u"][row] >= 0.0


def test_maros_meszaros_loader_normalizes_sentinel_bounds(tmp_path: Path):
    # The archived .mat files store +-1e20 infinity sentinels, in a few
    # files with ULPs of representation error; the loader must hand back
    # true infinities so no finite 1e20-scale bound reaches solvers or
    # the KKT checker.
    import scipy.sparse

    from problem_classes.maros_meszaros import MarosMeszaros

    scipy.io.savemat(
        tmp_path / "TOY.mat",
        {
            "P": scipy.sparse.eye(2, format="csc"),
            "q": np.zeros(2),
            "r": 0.0,
            "A": scipy.sparse.eye(2, format="csc"),
            "l": np.array([-1.0e20, 0.0]),
            "u": np.array([9.999999999999998e19, 2.0]),
            "n": 2,
            "m": 2,
        },
    )

    instance = MarosMeszaros(str(tmp_path / "TOY"), "TOY")

    assert instance.l.tolist() == [-np.inf, 0.0]
    assert instance.u.tolist() == [np.inf, 2.0]


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


def _assert_same_problem(loaded, reference) -> None:
    assert (loaded.dataset_id, loaded.name, loaded.kind) == (
        reference.dataset_id,
        reference.name,
        reference.kind,
    )
    assert loaded.metadata == reference.metadata
    assert loaded.qp.keys() == reference.qp.keys()
    for key, value in reference.qp.items():
        other = loaded.qp[key]
        if hasattr(value, "toarray"):
            assert other.shape == value.shape and (other != value).nnz == 0, key
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(other, value, err_msg=key)
        else:
            assert other == value, key


def _load_counting_listings(monkeypatch, dataset, name):
    """``dataset.load_problem(name)`` and the folders it listed meanwhile."""
    listed = []
    real = mps_module.group_problem_files

    def counting(folder, suffixes):
        listed.append(folder)
        return real(folder, suffixes)

    monkeypatch.setattr(mps_module, "group_problem_files", counting)
    loaded = dataset.load_problem(name)
    monkeypatch.setattr(mps_module, "group_problem_files", real)
    return loaded, listed


def _load_from_listing(dataset, name):
    """The problem as ``load_problem`` gives it when it looks the name up in the listing."""
    dataset._problem_from_its_files = lambda _name: None
    try:
        return dataset.load_problem(name)
    finally:
        del dataset._problem_from_its_files


@pytest.mark.parametrize(
    ("dataset_name", "options", "problem"),
    [
        ("netlib", {}, "afiro"),
        ("netlib", {"subset": "infeasible"}, "itest2"),
        ("kennington", {}, "cre-c"),
    ],
)
def test_mps_load_problem_finds_its_file_without_listing_the_folder(
    monkeypatch, repo_root: Path, dataset_name: str, options: dict, problem: str
):
    dataset = get_dataset(dataset_name)(
        repo_root=repo_root, data_root=repo_root / "problem_classes", **options
    )
    reference = _load_from_listing(dataset, problem)

    loaded, listed = _load_counting_listings(monkeypatch, dataset, problem)

    assert listed == []
    _assert_same_problem(loaded, reference)
    assert loaded.metadata["source"].endswith(f"{problem}.mps.gz")


def test_mps_load_problem_of_a_plain_and_gzipped_pair_matches_the_listing(monkeypatch, tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "netlib_data" / "feasible"
    folder.mkdir(parents=True)
    (folder / "tiny.mps").write_text(_TINY_MPS)
    (folder / "tiny.mps.gz").write_bytes(gzip.compress(_TINY_MPS.encode()))
    (folder / "zipped.mps.gz").write_bytes(gzip.compress(_TINY_MPS.encode()))
    dataset = get_dataset("netlib")(repo_root=tmp_path, data_root=data_root)

    for name in ("tiny", "zipped"):
        reference = _load_from_listing(dataset, name)
        loaded, listed = _load_counting_listings(monkeypatch, dataset, name)
        assert listed == []
        _assert_same_problem(loaded, reference)
    assert loaded.metadata["size_bytes"] == (folder / "zipped.mps.gz").stat().st_size
    tiny = dataset.load_problem("tiny")
    assert tiny.metadata["source"] == str(folder / "tiny.mps")
    assert tiny.metadata["size_bytes"] == (folder / "tiny.mps.gz").stat().st_size

    # A problem the size filter leaves out, or one that is not there, is
    # still not found.
    small = get_dataset("netlib")(repo_root=tmp_path, data_root=data_root, max_size_mb=1e-6)
    with pytest.raises(KeyError):
        small.load_problem("tiny")
    with pytest.raises(KeyError):
        dataset.load_problem("absent")


def test_mps_load_problem_reads_only_what_the_listing_lists(repo_root: Path):
    dataset = get_dataset("netlib")(repo_root=repo_root, data_root=repo_root / "problem_classes")
    # A name with path components would reach a file of another folder (and
    # pose it with this folder's expected outcome).
    assert (repo_root / "problem_classes" / "netlib_data" / "infeasible" / "itest2.mps.gz").exists()
    with pytest.raises(KeyError):
        dataset.load_problem("../infeasible/itest2")

    class WithoutAfiro(type(dataset)):
        def list_problems(self):
            return [spec for spec in super().list_problems() if spec.name != "afiro"]

    hiding = WithoutAfiro(repo_root=repo_root, data_root=repo_root / "problem_classes")
    with pytest.raises(KeyError):
        hiding.load_problem("afiro")
    assert hiding.load_problem("adlittle").name == "adlittle"


def test_mps_listing_computes_the_shared_metadata_once(monkeypatch, repo_root: Path):
    dataset = get_dataset("netlib")(repo_root=repo_root, data_root=repo_root / "problem_classes")
    calls = []
    real = type(dataset).expected_metadata

    def counting(self):
        calls.append(1)
        return real(self)

    monkeypatch.setattr(type(dataset), "expected_metadata", counting)
    specs = dataset.list_problems()

    assert len(specs) > 90
    assert len(calls) == 1
    assert {spec.metadata["expected_status"] for spec in specs} == {"optimal"}


def test_bundled_netlib_greenbea_infeasible_variant_is_infeasible(
    tmp_path: Path, repo_root: Path
):
    """Guard against replacing the infeasible original with the repaired LP."""
    from solver_benchmarks.core import status

    data_dir = repo_root / "problem_classes" / "netlib_data"
    with gzip.open(data_dir / "feasible" / "greenbea.mps.gz", "rb") as handle:
        feasible_payload = handle.read().replace(b"\r\n", b"\n")
    with gzip.open(data_dir / "infeasible" / "greenbea.mps.gz", "rb") as handle:
        infeasible_payload = handle.read().replace(b"\r\n", b"\n")
    assert infeasible_payload != feasible_payload

    dataset = get_dataset("netlib")(repo_root=repo_root, subset="infeasible")
    problem = dataset.load_problem("greenbea")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    result = get_solver("highs")({"verbose": False}).solve(problem, artifacts)

    assert result.status == status.PRIMAL_INFEASIBLE


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


@pytest.mark.parametrize(
    ("dataset_id", "problem_name", "kind"),
    [
        ("dimacs", "nb", "cone"),
        ("liu_pataki", "infeas_clean_10_10_1", "cone"),
        ("maros_meszaros", "CVXQP1_S", "qp"),
        ("netlib", "afiro", "qp"),
        ("sdplib", "truss1", "cone"),
    ],
)
def test_bundled_dataset_round_trips_through_prepare_load_solve(
    monkeypatch,
    tmp_path: Path,
    repo_root: Path,
    dataset_id: str,
    problem_name: str,
    kind: str,
):
    # Smoke test that each bundled dataset can prepare → load → solve a
    # real bundled instance end-to-end, with the network forbidden. This
    # is the same shape of check that originally caught the Kennington
    # EMPS-format bug as an xfail in the network suite.
    from solver_benchmarks.core import status

    def fail_if_network(*args, **kwargs):
        raise AssertionError(f"{dataset_id} bundled prepare unexpectedly used the network")

    monkeypatch.setattr(dimacs_module.urllib.request, "urlopen", fail_if_network)
    monkeypatch.setattr(mps_module.urllib.request, "urlopen", fail_if_network)

    # Datasets with a real prepare step write into a fresh data_root. The
    # bundled-no-prepare datasets read directly from the source repo.
    if dataset_id in {"dimacs", "sdplib"}:
        data_root = tmp_path / "problem_classes"
        if dataset_id == "sdplib":
            sdplib_folder = data_root / "sdplib_data"
            sdplib_folder.mkdir(parents=True)
            (sdplib_folder / "sdplib.tar").symlink_to(
                repo_root / "problem_classes/sdplib_data/sdplib.tar"
            )
        dataset = get_dataset(dataset_id)(repo_root=repo_root, data_root=data_root)
    else:
        dataset = get_dataset(dataset_id)(repo_root=repo_root)

    dataset.prepare_data([problem_name])
    problem = dataset.load_problem(problem_name)
    assert problem.kind == kind

    solver_cls = get_solver("clarabel")
    assert problem.kind in solver_cls.supported_problem_kinds
    artifacts = tmp_path / "artifacts" / dataset_id
    artifacts.mkdir(parents=True)
    result = solver_cls({"verbose": False, "max_iter": 500}).solve(problem, artifacts)

    # Smoke contract: the solver consumed the problem and returned a
    # real status, not a crash. Convergence quality (optimal vs. max-iter)
    # is left to other tests since some bundled datasets (e.g. liu_pataki)
    # ship deliberately hard infeasibility-detection instances.
    assert result.status not in {status.SOLVER_ERROR, status.WORKER_ERROR}


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


def test_sdplib_feasibility_subsets(tmp_path: Path):
    """The four infeasible SDPLIB instances form their own subset, like the
    NETLIB infeasible LPs: the default lists only feasible problems."""
    import pytest

    data_root = tmp_path / "problem_classes"
    folder = data_root / "sdplib_data"
    folder.mkdir(parents=True)
    with tarfile.open(folder / "sdplib.tar", "w") as archive:
        for name in ("arch0.jld2", "infd1.jld2", "infp2.jld2", "theta1.jld2"):
            info = tarfile.TarInfo(name)
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))

    def names(**options):
        return [spec.name for spec in get_dataset("sdplib")(repo_root=tmp_path, data_root=data_root, **options).list_problems()]

    assert names() == ["arch0", "theta1"]
    assert names(subset="feasible") == ["arch0", "theta1"]
    assert names(subset="infeasible") == ["infd1", "infp2"]
    assert names(subset="all") == ["arch0", "infd1", "infp2", "theta1"]
    with pytest.raises(ValueError, match="Unknown SDPLIB subset"):
        names(subset="bogus")


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


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        ("MINIMIZE", "min"),
        ("Minimize   # objective sense", "min"),
        ("MAXIMIZE", "max"),
        ("maximize # not to minimize", "max"),
    ],
)
def test_qplib_loader_reads_the_sense_word_in_any_case(
    tmp_path: Path, direction: str, expected: str
):
    """The sense is the first word of its line. A test for the substring
    'minimize' read MINIMIZE as a maximisation and negated the objective."""
    data_root = tmp_path / "problem_classes"
    folder = data_root / "qplib_data"
    folder.mkdir(parents=True)
    (folder / "list_convex_qps.txt").write_text("CCB (1)\n-------\n7777\n")
    _write_minimal_qplib(folder / "QPLIB_7777.qplib", direction=direction)

    problem = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root).load_problem("7777")

    assert problem.metadata.get("original_obj_type") == expected
    assert problem.qp["q"][0] == (2.0 if expected == "min" else -2.0)


def test_qplib_loader_rejects_an_unknown_sense(tmp_path: Path):
    data_root = tmp_path / "problem_classes"
    folder = data_root / "qplib_data"
    folder.mkdir(parents=True)
    (folder / "list_convex_qps.txt").write_text("CCB (1)\n-------\n6666\n")
    _write_minimal_qplib(folder / "QPLIB_6666.qplib", direction="1 variables")

    dataset = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root)
    with pytest.raises(ValueError, match="objective sense"):
        dataset.load_problem("6666")


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


def _installed_without_checkout(tmp_path: Path, monkeypatch) -> Path:
    """Pretend the package was installed from a wheel: its parent is a
    site-packages holding only the code of problem_classes, and the current
    directory (returned) is not a checkout either."""
    from solver_benchmarks.datasets import base as base_module

    site = tmp_path / "site-packages"
    (site / "solver_benchmarks" / "datasets").mkdir(parents=True)
    (site / "problem_classes").mkdir()
    (site / "problem_classes" / "qpsreader.py").write_text("")
    monkeypatch.setattr(base_module, "__file__", str(site / "solver_benchmarks" / "datasets" / "base.py"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return elsewhere


def _site_files(tmp_path: Path) -> list[str]:
    return sorted(str(path.relative_to(tmp_path)) for path in (tmp_path / "site-packages").rglob("*"))


def test_default_repo_root_never_takes_site_packages(tmp_path, monkeypatch):
    from solver_benchmarks.datasets.base import default_repo_root

    cwd = _installed_without_checkout(tmp_path, monkeypatch)

    assert default_repo_root() == cwd.resolve()


def test_bundled_data_without_a_root_is_a_clear_error(tmp_path, monkeypatch):
    _installed_without_checkout(tmp_path, monkeypatch)
    before = _site_files(tmp_path)

    netlib = get_dataset("netlib")()
    with pytest.raises(ValueError, match="no bundled data") as excinfo:
        netlib.data_status()
    assert "pip install -e" in str(excinfo.value) and "--data-root" in str(excinfo.value)

    config = tmp_path / "netlib.yaml"
    config.write_text(
        "run: {dataset: netlib, include: [afiro], timeout_seconds: 60}\n"
        "solvers: [{id: highs, solver: highs, settings: {}}]\n"
    )
    result = CliRunner().invoke(main, ["run", str(config), "--run-dir", str(tmp_path / "run")])
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)  # a clean Error:, no traceback
    assert "no bundled data" in result.output
    assert _site_files(tmp_path) == before


def test_data_free_datasets_and_existing_runs_work_without_a_checkout(tmp_path, monkeypatch):
    _installed_without_checkout(tmp_path, monkeypatch)
    before = _site_files(tmp_path)
    runner = CliRunner()

    listed = runner.invoke(main, ["list", "problems", "synthetic_qp"])
    assert listed.exit_code == 0, listed.output
    assert "one_variable_lp" in listed.output

    config = tmp_path / "synthetic.yaml"
    config.write_text(
        "run: {dataset: synthetic_qp, include: [one_variable_lp], timeout_seconds: 60}\n"
        "solvers: [{id: highs, solver: highs, settings: {}}]\n"
    )
    run_dir = tmp_path / "synthetic_run"
    result = runner.invoke(main, ["run", str(config), "--run-dir", str(run_dir), "--no-stream-output"])
    assert result.exit_code == 0, result.output
    summary = runner.invoke(main, ["summary", str(run_dir)])
    assert summary.exit_code == 0, summary.output
    assert "optimal" in summary.output
    assert _site_files(tmp_path) == before


def test_data_root_works_without_a_checkout_or_a_repo_root(tmp_path, monkeypatch, repo_root):
    _installed_without_checkout(tmp_path, monkeypatch)
    before = _site_files(tmp_path)
    data_root = repo_root / "problem_classes"

    netlib = get_dataset("netlib")(data_root=str(data_root))
    assert netlib.data_status().available

    config = tmp_path / "netlib.yaml"
    config.write_text(
        "run: {dataset: netlib, include: [afiro], timeout_seconds: 60}\n"
        "solvers: [{id: highs, solver: highs, settings: {}}]\n"
    )
    run_dir = tmp_path / "netlib_run"
    runner = CliRunner()
    result = runner.invoke(main, ["run", str(config), "--run-dir", str(run_dir), "--data-root", str(data_root),
                                  "--no-stream-output"])
    assert result.exit_code == 0, result.output
    rows = (run_dir / "results.jsonl").read_text()
    assert '"status": "optimal"' in rows
    summary = runner.invoke(main, ["summary", str(run_dir)])
    assert summary.exit_code == 0, summary.output
    assert _site_files(tmp_path) == before


def test_default_repo_root_takes_the_checkout_or_a_working_directory_with_data(tmp_path, monkeypatch):
    from solver_benchmarks.datasets import base as base_module

    checkout = tmp_path / "checkout"
    (checkout / "solver_benchmarks" / "datasets").mkdir(parents=True)
    (checkout / "problem_classes").mkdir()
    (checkout / "pyproject.toml").write_text("")
    monkeypatch.setattr(base_module, "__file__", str(checkout / "solver_benchmarks" / "datasets" / "base.py"))
    monkeypatch.chdir(tmp_path)
    assert base_module.default_repo_root() == checkout.resolve()

    work = tmp_path / "work"
    (work / "problem_classes").mkdir(parents=True)
    monkeypatch.chdir(work)
    assert base_module.default_repo_root() == work.resolve()


def test_explicit_roots_work_without_a_checkout(tmp_path, monkeypatch, repo_root):
    _installed_without_checkout(tmp_path, monkeypatch)

    netlib = get_dataset("netlib")(repo_root=repo_root)
    assert netlib.data_status().available
    runner = CliRunner()
    result = runner.invoke(main, ["data", "status", "netlib", "--repo-root", str(repo_root)])
    assert result.exit_code == 0, result.output
    assert "\tavailable\t" in result.output


# ---------------------------------------------------------------------------
# Dataset options: names are validated, subsets accept a single scalar name.
# ---------------------------------------------------------------------------

def test_every_registered_dataset_declares_the_options_it_reads():
    """The names each dataset module reads from ``options`` are the names its
    classes declare (with the generic ones the base class reads)."""
    import inspect
    import sys

    from solver_benchmarks.datasets.base import GENERIC_OPTIONS

    modules: dict[str, set[str]] = {}
    for name, cls in dataset_registry.DATASETS.items():
        assert cls.option_names is not None, name
        for klass in cls.__mro__:
            if isinstance(klass.__dict__.get("option_names"), frozenset):
                modules.setdefault(klass.__module__, set()).update(klass.option_names)
    pattern = re.compile(r"""options(?:\.get\(|\[)["']([A-Za-z_]+)["']""")
    for module, declared in modules.items():
        read = set(pattern.findall(inspect.getsource(sys.modules[module])))
        assert read - GENERIC_OPTIONS == declared, module


@pytest.mark.parametrize(
    ("dataset", "options", "message"),
    [
        ("libsvm_qp", {"kernal": "rbf"}, "'kernal' (did you mean 'kernel'?)"),
        ("libsvm_qp", {"c": 10.0}, "'c' (did you mean 'C'?)"),
        ("sdplib", {"block": "merged"}, "'block' (did you mean 'blocks'?)"),
        ("maros_meszaros", {"max_size": 0.1}, "'max_size' (did you mean 'max_size_mb'?)"),
        ("maros_meszaros", {"subset": ["AUG2D"]}, "'subset'"),
        ("kennington", {"subset": "ken-07"}, "'subset'"),
        ("dimacs", {"subset": "default"}, "'subset'"),
        ("liu_pataki", {"subset": "infeas_clean_10_10_1"}, "'subset'"),
        ("mittelmann", {"subset": ["qap15"]}, "'subset'"),
        ("miplib", {"subset": "markshare_4_0"}, "'subset'"),
    ],
)
def test_an_option_the_dataset_does_not_read_is_rejected(dataset, options, message):
    with pytest.raises(ValueError, match="Unknown dataset option") as excinfo:
        get_dataset(dataset)(**options)
    assert message in str(excinfo.value)
    assert dataset in str(excinfo.value)


@pytest.mark.parametrize("subset", ["all", "feasable", ["afiro", "adlittle"]])
def test_netlib_rejects_a_subset_that_is_not_a_feasibility_class(subset):
    with pytest.raises(ValueError, match="Unknown NETLIB subset"):
        get_dataset("netlib")(subset=subset)


def test_netlib_accepts_its_two_subsets(repo_root: Path):
    for subset in ("feasible", "infeasible"):
        assert get_dataset("netlib")(repo_root=repo_root, subset=subset).list_problems()


def _readme_dataset_options() -> list[tuple[str, str]]:
    """``(dataset, option)`` for each row of the README's dataset options table."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    table = readme.split("Useful dataset options:", 1)[1].split("### ", 1)[0]
    pairs = []
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[1].startswith("`"):
            continue
        datasets = (
            list_datasets()
            if cells[0] == "All file-backed datasets"
            else re.findall(r"`([a-z_]+)`", cells[0])
        )
        for option in sorted(set(re.findall(r"`([A-Za-z_]+)=", cells[1]))):
            pairs.extend((dataset, option) for dataset in datasets)
    return pairs


def test_every_option_the_readme_documents_is_accepted():
    pairs = _readme_dataset_options()
    assert ("libsvm_qp", "C") in pairs and ("cblib", "include_unsupported") in pairs
    for dataset, option in pairs:
        value = "feasible" if (dataset, option) == ("netlib", "subset") else "x"
        get_dataset(dataset).validate_options({option: value})


def _name_subset_file(tmp_path: Path, dataset: str, name: str) -> None:
    root = tmp_path / "problem_classes"
    files = {
        "qplib": root / "qplib_data" / f"QPLIB_{name}.qplib",
        "cutest_qp": root / "cutest_qp_data" / f"{name}.npz",
        "mpc_qpbenchmark": root / "mpc_qpbenchmark_data" / f"{name}.npz",
        "cblib": root / "cblib_data" / f"{name}.cbf",
        "mittelmann_sdp": root / "mittelmann_sdp_data" / f"{name}.dat-s",
        "tsplib_sdp": root / "tsplib_data" / f"{name}.tsp",
        "dc_opf": root / "dc_opf_data" / MATPOWER_REVISION / f"{name}.mat",
        "libsvm_qp": root / "libsvm_data" / f"{name}.libsvm",
    }
    path = files[dataset]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("VER\n3\nVAR\n1 1\nL+ 1\n" if dataset == "cblib" else "")


_NAME_SUBSET_DATASETS = (
    "qplib",
    "cutest_qp",
    "mpc_qpbenchmark",
    "cblib",
    "mittelmann_sdp",
    "tsplib_sdp",
    "dc_opf",
    "libsvm_qp",
)


@pytest.mark.parametrize("dataset", _NAME_SUBSET_DATASETS)
def test_a_scalar_subset_is_the_single_name_it_spells(tmp_path: Path, dataset):
    """YAML reads ``subset: 8790`` as an int, and so does ``--option subset=8790``."""
    for name in ("8790", "8515"):
        _name_subset_file(tmp_path, dataset, name)
    instance = get_dataset(dataset)(
        repo_root=tmp_path, data_root=tmp_path / "problem_classes", subset=8790
    )
    expected = "svm_dual_8790" if dataset == "libsvm_qp" else "8790"
    assert [spec.name for spec in instance.list_problems()] == [expected]
    assert instance.requested_problem_names() == {expected}


def test_a_numeric_qplib_subset_works_from_a_config_and_the_cli(tmp_path: Path, monkeypatch):
    _name_subset_file(tmp_path, "qplib", "8790")
    _name_subset_file(tmp_path, "qplib", "8515")
    config = parse_run_config(
        {
            "run": {"dataset": "qplib", "dataset_options": {"subset": 8790}},
            "solvers": [{"id": "s", "solver": "clarabel"}],
        }
    )
    options = config.datasets[0].dataset_options
    dataset = get_dataset("qplib")(repo_root=tmp_path, **options)
    assert [spec.name for spec in dataset.list_problems()] == ["8790"]

    result = CliRunner().invoke(
        main, ["list", "problems", "qplib", "--repo-root", str(tmp_path), "--option", "subset=8790"]
    )
    assert result.exit_code == 0, result.output
    assert [line.split("\t")[0] for line in result.output.splitlines()] == ["8790"]

    downloads = []
    monkeypatch.setattr(qplib_module, "download_qplib_problem", lambda name, folder: downloads.append(name))
    get_dataset("qplib")(repo_root=tmp_path, subset=8515).prepare_data()
    assert downloads == ["8515"]


def test_requested_problem_names(tmp_path: Path):
    def requested(dataset, **options):
        return get_dataset(dataset)(repo_root=tmp_path, **options).requested_problem_names()

    # Nothing on disk: the names come from the option alone.
    assert requested("qplib", subset=["ccb", "8495"]) == {"8790", "8792", "8991", "8495"}
    assert requested("qplib", subset="all") is None
    assert requested("qplib") is None
    assert requested("cutest_qp", subset="default") == {"HS35", "HS35MOD", "QAFIRO", "QPCBLEND"}
    assert requested("cblib", subset="nb.cbf.gz, qssp30") == {"nb", "qssp30"}
    assert requested("libsvm_qp", subset="a1a", kind="markowitz") == {"markowitz_a1a"}
    assert requested("dc_opf", subset="feasible") is None
    assert requested("dc_opf", subset="case9") == {"case9"}
    assert requested("sdplib", subset="infeasible") is None
    assert requested("maros_meszaros") is None


def test_a_single_name_subset_is_prepared_by_name_where_the_subset_lists_problems():
    def selection(dataset, subset):
        config = parse_run_config(
            {
                "run": {"dataset": dataset, "dataset_options": {"subset": subset}},
                "solvers": [{"id": "s", "solver": "clarabel"}],
            }
        )
        return data_prepare_selection(config, config.datasets[0])

    for dataset in _NAME_SUBSET_DATASETS:
        lists = get_dataset(dataset).subset_lists_problems
        assert lists == (dataset != "dc_opf"), dataset
    assert selection("mpc_qpbenchmark", "LIPMWALK0") == (["LIPMWALK0"], False)
    assert selection("cutest_qp", "HS35") == (["HS35"], False)
    assert selection("tsplib_sdp", "burma14") == (["burma14"], False)
    assert selection("qplib", "ccb") == (["ccb"], False)
    # ``feasible`` is a class of cases, not a case name.
    assert selection("dc_opf", "feasible") == (None, False)


def test_qplib_category_subsets_work_under_another_data_root(tmp_path: Path, monkeypatch):
    """The category list is bundled with the checkout, not with the data."""
    data_root = tmp_path / "elsewhere"
    folder = data_root / "qplib_data"
    folder.mkdir(parents=True)
    for problem_id in ("8790", "8495"):
        (folder / f"QPLIB_{problem_id}.qplib").write_text("")
    dataset = get_dataset("qplib")(repo_root=tmp_path, data_root=data_root, subset="ccb")

    assert [spec.name for spec in dataset.list_problems()] == ["8790"]
    assert dataset.list_problems()[0].metadata["qplib_category"] == "ccb"

    downloads = []
    monkeypatch.setattr(qplib_module, "download_qplib_problem", lambda name, folder: downloads.append(name))
    dataset.prepare_data()
    assert downloads == ["8790", "8792", "8991"]


@pytest.mark.parametrize(
    ("dataset", "folder", "plain", "gz"),
    [
        ("dimacs", "dimacs_data", "p.mat", "p.mat.gz"),
        ("mittelmann_sdp", "mittelmann_sdp_data", "p.dat-s", "p.dat-s.gz"),
        ("tsplib_sdp", "tsplib_data", "p.tsp", "p.tsp.gz"),
        ("mittelmann", "mittelmann", "p.mps", "p.mps.gz"),
    ],
)
def test_a_plain_and_gzipped_pair_is_filtered_by_the_smaller_copy(
    tmp_path: Path, dataset, folder, plain, gz
):
    path = tmp_path / "problem_classes" / folder
    path.mkdir(parents=True)
    (path / plain).write_bytes(b"x" * 50_000)
    (path / gz).write_bytes(b"x" * 500)
    instance = get_dataset(dataset)(
        repo_root=tmp_path, data_root=tmp_path / "problem_classes", max_size_mb=0.001
    )

    (spec,) = instance.visible_problems()
    assert spec.name == "p"
    assert spec.path == path / plain  # read from the plain copy
    assert spec.metadata["size_bytes"] == 500


def test_miplib_prepare_sizes_local_files_on_disk_and_heads_only_missing_ones(
    monkeypatch, tmp_path: Path
):
    folder = tmp_path / "problem_classes" / "miplib_data"
    folder.mkdir(parents=True)
    (folder / "local_small.mps.gz").write_bytes(b"x" * 10)
    (folder / "local_large.mps.gz").write_bytes(b"x" * 2_000_000)
    heads = []

    def remote_size(name):
        heads.append(name)
        return {"remote_small": 100, "remote_large": 5_000_000}[name]

    downloads = []
    monkeypatch.setattr(
        mps_module,
        "_miplib_remote_problem_names",
        lambda: ["local_large", "local_small", "remote_large", "remote_small"],
    )
    monkeypatch.setattr(mps_module, "_miplib_remote_size_bytes", remote_size)
    monkeypatch.setattr(mps_module, "_download_miplib_problem", lambda name, f: downloads.append(name))
    dataset = get_dataset("miplib")(
        repo_root=tmp_path, data_root=tmp_path / "problem_classes", max_size_mb=1.0
    )

    dataset.prepare_data()

    assert sorted(heads) == ["remote_large", "remote_small"]
    assert downloads == ["local_small", "remote_small"]


def test_miplib_prepare_without_network_uses_the_local_instances(monkeypatch, tmp_path: Path):
    folder = tmp_path / "problem_classes" / "miplib_data"
    folder.mkdir(parents=True)
    (folder / "a.mps.gz").write_bytes(b"x" * 10)
    (folder / "b.mps.gz").write_bytes(b"x" * 10)

    def offline():
        raise RuntimeError("Could not fetch MIPLIB benchmark index: no network")

    def no_request(*args, **kwargs):
        raise AssertionError("no request expected")

    downloads = []
    monkeypatch.setattr(mps_module, "_miplib_remote_problem_names", offline)
    monkeypatch.setattr(mps_module, "_miplib_remote_size_bytes", no_request)
    monkeypatch.setattr(mps_module, "_download_miplib_problem", lambda name, f: downloads.append(name))
    dataset = get_dataset("miplib")(
        repo_root=tmp_path, data_root=tmp_path / "problem_classes", max_size_mb=5.0
    )

    with pytest.warns(UserWarning, match="no network; using the 2 MIPLIB instances"):
        dataset.prepare_data()
    assert downloads == ["a", "b"]

    # With nothing on disk, the failure stands.
    empty = get_dataset("miplib")(repo_root=tmp_path, data_root=tmp_path / "none", max_size_mb=5.0)
    with pytest.raises(RuntimeError, match="no network"):
        empty.prepare_data()


def test_miplib_prepare_without_network_counts_plain_instances(monkeypatch, tmp_path: Path):
    folder = tmp_path / "problem_classes" / "miplib_data"
    folder.mkdir(parents=True)
    (folder / "plain.mps").write_bytes(b"x" * 10)

    def offline(*args, **kwargs):
        raise RuntimeError("Could not fetch MIPLIB benchmark index: no network")

    def no_request(*args, **kwargs):
        raise AssertionError("no request expected")

    monkeypatch.setattr(mps_module, "_miplib_remote_problem_names", offline)
    monkeypatch.setattr(mps_module, "_miplib_remote_size_bytes", no_request)
    monkeypatch.setattr(mps_module.urllib.request, "urlopen", no_request)
    dataset = get_dataset("miplib")(
        repo_root=tmp_path, data_root=tmp_path / "problem_classes", max_size_mb=5.0
    )

    with pytest.warns(UserWarning, match="using the 1 MIPLIB instances"):
        dataset.prepare_data()
    assert sorted(path.name for path in folder.iterdir()) == ["plain.mps"]
