import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.cli import main
from solver_benchmarks.core.config import parse_environment_run_config, parse_run_config
from solver_benchmarks.core.env_runner import run_environment_matrix
from solver_benchmarks.core.problem import CONE, QP, ProblemSpec
from solver_benchmarks.core.result import ProblemResult
from solver_benchmarks.core.runner import _filter_by_size, run_benchmark
from solver_benchmarks.core.storage import ResultStore
from solver_benchmarks.datasets import registry as dataset_registry
from solver_benchmarks.datasets.base import Dataset
from solver_benchmarks.solvers import registry as solver_registry
from solver_benchmarks.solvers.base import SolverAdapter


def test_runner_writes_results_logs_and_resumes(tmp_path: Path, repo_root: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
                "parallelism": 1,
                "resume": True,
            },
            "solvers": [
                {
                    "id": "scs_smoke",
                    "solver": "scs",
                    "settings": {
                        "verbose": False,
                        "eps_abs": 1e-6,
                        "eps_rel": 1e-6,
                        "max_iters": 1000,
                    },
                }
            ],
        }
    )

    store = run_benchmark(config, repo_root=repo_root)
    results_path = store.results_jsonl_path
    rows = results_path.read_text().strip().splitlines()

    assert len(rows) == 1
    record = json.loads(rows[0])
    assert record["status"] == "optimal"
    artifact_dir = Path(record["artifact_dir"])
    assert (artifact_dir / "stdout.log").exists()
    assert (artifact_dir / "stderr.log").exists()
    assert (artifact_dir / "result.json").exists()
    assert store.results_parquet_path.exists()
    assert record["metadata"]["runtime"]["python_version"]
    assert "scs" in record["metadata"]["runtime"]["solver_package_versions"]

    run_benchmark(config, run_dir=store.run_dir, repo_root=repo_root)
    assert len(results_path.read_text().strip().splitlines()) == 1

    df = load_results(store.run_dir)
    assert len(df) == 1
    assert df.loc[0, "solver_id"] == "scs_smoke"


def test_run_cli_uses_config_stem_name_and_copies_source_config(tmp_path: Path, repo_root: Path):
    config_path = tmp_path / "named_smoke_run.yaml"
    output_dir = tmp_path / "runs"
    config_text = f"""
run:
  dataset: synthetic_qp
  output_dir: {output_dir}
  include:
    - one_variable_eq
solvers:
  - id: scs_smoke
    solver: scs
    settings:
      verbose: false
      max_iters: 1000
"""
    config_path.write_text(config_text)

    result = CliRunner().invoke(
        main,
        ["run", str(config_path), "--repo-root", str(repo_root)],
    )

    assert result.exit_code == 0, result.output
    run_dir = Path(result.output.strip().splitlines()[-1])
    assert run_dir.parent == output_dir
    assert re.fullmatch(
        r"named_smoke_run_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_UTC",
        run_dir.name,
    )
    assert (run_dir / "run_config.yaml").read_text() == config_text
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["config"]["name"] == "named_smoke_run"
    assert manifest["config"]["config_hash"] not in run_dir.name
    assert "synthetic_qp" not in run_dir.name


def test_run_cli_can_disable_solver_streaming_without_changing_settings(
    monkeypatch, tmp_path: Path
):
    config_path = tmp_path / "verbose_override.yaml"
    config_path.write_text(
        """
run:
  dataset: synthetic_qp
  output_dir: runs
solvers:
  - id: scs_smoke
    solver: scs
    settings:
      verbose: true
  - id: qtqp_smoke
    solver: qtqp
    settings:
      max_iter: 10
"""
    )
    seen = {}

    def fake_run_benchmark(config, **kwargs):
        seen["settings"] = [solver.settings for solver in config.solvers]
        seen["stream_output"] = kwargs["stream_output"]
        seen["stream_solver_output"] = kwargs["stream_solver_output"]
        return SimpleNamespace(run_dir=tmp_path / "runs" / "fake_run")

    monkeypatch.setattr("solver_benchmarks.cli.run_benchmark", fake_run_benchmark)

    result = CliRunner().invoke(
        main,
        ["run", str(config_path), "--no-stream-output"],
    )

    assert result.exit_code == 0, result.output
    assert seen["settings"] == [
        {"verbose": True},
        {"max_iter": 10},
    ]
    assert seen["stream_output"] is True
    assert seen["stream_solver_output"] is False


def test_env_run_cli_can_disable_solver_streaming(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "env_stream_override.yaml"
    config_path.write_text(
        """
run:
  dataset: synthetic_qp
  output_dir: runs
environments:
  - id: current
    python: python
    solvers:
      - id: scs_current
        solver: scs
        settings:
          verbose: true
"""
    )
    seen = {}

    def fake_run_environment_matrix(config, **kwargs):
        seen["settings"] = [
            solver.settings
            for environment in config.environments
            for solver in environment.solvers
        ]
        seen["stream_output"] = kwargs["stream_output"]
        seen["stream_solver_output"] = kwargs["stream_solver_output"]
        return tmp_path / "runs" / "fake_env_run"

    monkeypatch.setattr(
        "solver_benchmarks.cli.run_environment_matrix",
        fake_run_environment_matrix,
    )

    result = CliRunner().invoke(
        main,
        ["env", "run", str(config_path), "--no-stream-output"],
    )

    assert result.exit_code == 0, result.output
    assert seen["settings"] == [{"verbose": True}]
    assert seen["stream_output"] is True
    assert seen["stream_solver_output"] is False


def test_run_benchmark_resolves_relative_output_dir_under_repo_root(
    monkeypatch, tmp_path: Path
):
    class EmptyDataset(Dataset):
        dataset_id = "empty_for_output_dir"

        def list_problems(self):
            return []

        def load_problem(self, name):
            raise KeyError(name)

    monkeypatch.setitem(
        dataset_registry.DATASETS, "empty_for_output_dir", EmptyDataset
    )
    config = parse_run_config(
        {
            "run": {"dataset": "empty_for_output_dir", "output_dir": "results"},
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=tmp_path)

    assert store.run_dir.parent == tmp_path / "results"


def test_result_store_normalizes_nonfinite_values_for_parquet(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "solver", "solver": "scs", "settings": {}}],
        }
    )
    store = ResultStore.create(config, run_dir=tmp_path / "run")

    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="p1",
            problem_kind=QP,
            solver_id="solver",
            solver="scs",
            status="optimal",
            objective_value=1.0,
            iterations=1,
            run_time_seconds=0.1,
        )
    )
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="synthetic_qp",
            problem="p2",
            problem_kind=QP,
            solver_id="solver",
            solver="scs",
            status="max_iter_reached",
            objective_value=math.nan,
            iterations=None,
            run_time_seconds=0.2,
        )
    )

    records = [json.loads(line) for line in store.results_jsonl_path.read_text().splitlines()]
    df = load_results(store.run_dir)

    assert records[1]["objective_value"] is None
    assert store.results_parquet_path.exists()
    assert len(df) == 2
    assert df.loc[df["problem"] == "p2", "objective_value"].isna().all()


def test_parquet_rewrite_handles_legacy_string_nan(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "output_dir": str(tmp_path / "runs")},
            "solvers": [{"id": "solver", "solver": "scs", "settings": {}}],
        }
    )
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    store.results_jsonl_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": store.run_id,
                        "dataset": "synthetic_qp",
                        "problem": "p1",
                        "problem_kind": QP,
                        "solver_id": "solver",
                        "solver": "scs",
                        "status": "optimal",
                        "objective_value": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "run_id": store.run_id,
                        "dataset": "synthetic_qp",
                        "problem": "p2",
                        "problem_kind": QP,
                        "solver_id": "solver",
                        "solver": "scs",
                        "status": "max_iter_reached",
                        "objective_value": "nan",
                    }
                ),
            ]
        )
        + "\n"
    )

    store.rewrite_parquet()
    df = load_results(store.run_dir)

    assert len(df) == 2
    assert df.loc[df["problem"] == "p2", "objective_value"].isna().all()


def test_unsupported_combinations_skip_by_default(monkeypatch, tmp_path: Path, repo_root: Path):
    class FakeConeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="fake_cone", name="cone_problem", kind=CONE)]

    class FakeQPSolver(SolverAdapter):
        solver_name = "fake_qp"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):
            raise AssertionError("Unsupported solver should not be invoked")

    monkeypatch.setitem(dataset_registry.DATASETS, "fake_cone", FakeConeDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "fake_qp", FakeQPSolver)

    config = parse_run_config(
        {
            "run": {
                "dataset": "fake_cone",
                "output_dir": str(tmp_path / "runs"),
                "include": ["cone_problem"],
                "parallelism": 1,
            },
            "solvers": [{"id": "fake_qp_skip", "solver": "fake_qp", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=repo_root)
    df = load_results(store.run_dir)

    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped_unsupported"
    assert df.loc[0, "problem"] == "cone_problem"
    assert store.events_path.exists()


def test_pdlp_skips_cleanly_when_unavailable_or_non_lp(tmp_path: Path, repo_root: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
                "parallelism": 1,
            },
            "solvers": [{"id": "pdlp_smoke", "solver": "pdlp", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=repo_root)
    df = load_results(store.run_dir)

    assert len(df) == 1
    assert df.loc[0, "status"] == "skipped_unsupported"


def test_pdlp_defaults_do_not_enable_glop_or_polishing():
    pytest.importorskip("ortools.pdlp.solvers_pb2")
    from solver_benchmarks.solvers.pdlp_adapter import _pdlp_parameters_from_settings

    parameters = _pdlp_parameters_from_settings({})

    assert not parameters.presolve_options.use_glop
    assert not parameters.use_feasibility_polishing
    assert not parameters.apply_feasibility_polishing_after_limits_reached
    assert not parameters.apply_feasibility_polishing_if_solver_is_interrupted
    assert not parameters.use_diagonal_qp_trust_region_solver


def test_pdlp_glop_presolve_requires_explicit_opt_in():
    pytest.importorskip("ortools.pdlp.solvers_pb2")
    from solver_benchmarks.solvers.pdlp_adapter import _pdlp_parameters_from_settings

    parameters = _pdlp_parameters_from_settings({"use_glop": True})

    assert parameters.presolve_options.use_glop


def test_filter_by_size_drops_oversized_paths_only(tmp_path: Path):
    small = tmp_path / "small.bin"
    large = tmp_path / "large.bin"
    small.write_bytes(b"x" * 10)
    large.write_bytes(b"x" * 1_500_001)

    specs = [
        ProblemSpec(dataset_id="d", name="small", kind=QP, path=small),
        ProblemSpec(dataset_id="d", name="large", kind=QP, path=large),
        ProblemSpec(dataset_id="d", name="synth", kind=QP, path=None),
        ProblemSpec(dataset_id="d", name="missing", kind=QP, path=tmp_path / "nope.bin"),
    ]

    assert [spec.name for spec in _filter_by_size(specs, None)] == [
        "small",
        "large",
        "synth",
        "missing",
    ]
    assert [spec.name for spec in _filter_by_size(specs, 1.0)] == [
        "small",
        "synth",
        "missing",
    ]


def test_filter_by_size_prefers_metadata_size_over_path_stat(tmp_path: Path):
    """Datasets that pack many problems into one shared file (e.g. SDPLIB
    tar members) must be able to advertise per-member sizes via
    metadata["size_bytes"]; the filter must use those instead of the
    shared path's size on disk."""
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"x" * 5_000_000)  # 5 MB on disk

    specs = [
        ProblemSpec(
            dataset_id="d",
            name="member_small",
            kind=QP,
            path=archive,
            metadata={"size_bytes": 10},
        ),
        ProblemSpec(
            dataset_id="d",
            name="member_large",
            kind=QP,
            path=archive,
            metadata={"size_bytes": 1_500_001},
        ),
    ]

    assert [spec.name for spec in _filter_by_size(specs, 1.0)] == ["member_small"]


def test_pdlp_linear_cone_accepts_free_zero_cone_key():
    pytest.importorskip("ortools.linear_solver.linear_solver_pb2")

    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.solvers.pdlp_adapter import _build_lp_model_from_linear_cone

    cone_problem = {
        "A": sp.csr_matrix([[1.0], [-1.0]]),
        "b": np.array([1.0, -1.0]),
        "q": np.array([1.0]),
        "cone": {"f": 1, "l": 1},
    }

    model = _build_lp_model_from_linear_cone(cone_problem)

    assert len(model.constraint) == 2
    assert model.constraint[0].lower_bound == pytest.approx(1.0)
    assert model.constraint[0].upper_bound == pytest.approx(1.0)
    assert not model.constraint[1].HasField("lower_bound")
    assert model.constraint[1].upper_bound == pytest.approx(-1.0)


def test_auto_prepare_data_invokes_dataset_prepare(monkeypatch, tmp_path: Path, repo_root: Path):
    called = {}

    class FakeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def prepare_data(self, problem_names=None, all_problems=False):
            called["problem_names"] = problem_names
            called["all_problems"] = all_problems

        def list_problems(self):
            return []

        def data_status(self):
            return type(
                "Status",
                (),
                {"message": "fake dataset has no problems"},
            )()

    monkeypatch.setitem(dataset_registry.DATASETS, "fake_empty", FakeDataset)
    config = parse_run_config(
        {
            "run": {
                "dataset": "fake_empty",
                "output_dir": str(tmp_path / "runs"),
                "include": ["needed"],
                "auto_prepare_data": True,
            },
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    run_benchmark(config, repo_root=repo_root)

    assert called == {"problem_names": ["needed"], "all_problems": False}


def test_auto_prepare_data_honors_subset_all(monkeypatch, tmp_path: Path, repo_root: Path):
    called = {}

    class FakeDataset:
        def __init__(self, repo_root=None, **options):
            pass

        def prepare_data(self, problem_names=None, all_problems=False):
            called["problem_names"] = problem_names
            called["all_problems"] = all_problems

        def list_problems(self):
            return []

        def data_status(self):
            return type(
                "Status",
                (),
                {"available": True, "message": "fake dataset has no problems"},
            )()

    monkeypatch.setitem(dataset_registry.DATASETS, "fake_all", FakeDataset)
    config = parse_run_config(
        {
            "run": {
                "dataset": "fake_all",
                "dataset_options": {"subset": "all"},
                "output_dir": str(tmp_path / "runs"),
                "auto_prepare_data": True,
            },
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    run_benchmark(config, repo_root=repo_root)

    assert called == {"problem_names": None, "all_problems": True}


def test_run_cli_missing_downloadable_data_reports_exact_commands(
    monkeypatch,
    tmp_path: Path,
):
    class MissingDownloadDataset(Dataset):
        dataset_id = "remote_fake"
        description = "Missing remote fixture."
        automatic_download = True

        @property
        def data_dir(self):
            return self.problem_classes_dir / "remote_fake_data"

        def list_problems(self):
            return []

        def load_problem(self, name):
            raise AssertionError("No problem should be loaded")

    monkeypatch.setitem(dataset_registry.DATASETS, "remote_fake", MissingDownloadDataset)
    config_path = tmp_path / "remote_fake.json"
    config_path.write_text(
        json.dumps(
            {
                "run": {
                    "dataset": "remote_fake",
                    "output_dir": str(tmp_path / "runs"),
                    "include": ["needed"],
                },
                "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
            }
        )
    )

    result = CliRunner().invoke(
        main,
        ["run", str(config_path), "--repo-root", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert (
        f"bench data prepare remote_fake --repo-root {tmp_path} --problem needed"
        in result.output
    )
    assert (
        f"bench run {config_path} --prepare-data --repo-root {tmp_path}"
        in result.output
    )


def test_data_status_reports_exact_prepare_command(monkeypatch, tmp_path: Path):
    class MissingDownloadDataset(Dataset):
        dataset_id = "remote_status_fake"
        description = "Missing remote fixture."
        automatic_download = True

        @property
        def data_dir(self):
            return self.problem_classes_dir / "remote_status_fake_data"

        def list_problems(self):
            return []

        def load_problem(self, name):
            raise AssertionError("No problem should be loaded")

    monkeypatch.setitem(
        dataset_registry.DATASETS, "remote_status_fake", MissingDownloadDataset
    )

    result = CliRunner().invoke(
        main,
        ["data", "status", "remote_status_fake", "--repo-root", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "missing" in result.output
    assert f"bench data prepare remote_status_fake --repo-root {tmp_path}" in result.output


def test_runner_records_environment_metadata(tmp_path: Path, repo_root: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
                "parallelism": 1,
            },
            "solvers": [
                {
                    "id": "scs_env",
                    "solver": "scs",
                    "settings": {"verbose": False, "max_iters": 1000},
                }
            ],
        }
    )

    store = run_benchmark(
        config,
        repo_root=repo_root,
        environment_id="scs_3_2",
        environment_metadata={"scs": "3.2.0"},
    )
    record = json.loads(store.results_jsonl_path.read_text().splitlines()[0])

    assert record["metadata"]["environment_id"] == "scs_3_2"
    assert record["metadata"]["environment_metadata"] == {"scs": "3.2.0"}


def test_runner_iterates_over_multiple_datasets(monkeypatch, tmp_path: Path, repo_root: Path):
    class _FakeDatasetA:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="ds_a", name="prob_one", kind=QP)]

    class _FakeDatasetB:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [
                ProblemSpec(dataset_id="ds_b", name="prob_one", kind=QP),
                ProblemSpec(dataset_id="ds_b", name="prob_two", kind=QP),
            ]

    class _StubSolver(SolverAdapter):
        solver_name = "stub"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):
            raise AssertionError("Stub solver should be replaced before solve")

    monkeypatch.setitem(dataset_registry.DATASETS, "ds_a", _FakeDatasetA)
    monkeypatch.setitem(dataset_registry.DATASETS, "ds_b", _FakeDatasetB)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", _StubSolver)

    def _fake_run_subprocess(
        cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output
    ):
        from types import SimpleNamespace

        payload_path = Path(cmd[-1])
        payload = json.loads(payload_path.read_text())
        artifact_dir = Path(payload["artifacts_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
            artifact_dir=str(artifact_dir),
        ).to_record()
        (artifact_dir / "worker_result.json").write_text(json.dumps(record))
        stdout_path.write_text("")
        stderr_path.write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", _fake_run_subprocess
    )

    config = parse_run_config(
        {
            "run": {
                "datasets": ["ds_a", "ds_b"],
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
            },
            "solvers": [{"id": "stub_solver", "solver": "stub", "settings": {}}],
        }
    )
    store = run_benchmark(config, repo_root=repo_root)
    df = load_results(store.run_dir)

    # Three problems total across the two datasets.
    assert len(df) == 3
    assert set(df["dataset"]) == {"ds_a", "ds_b"}
    # Same problem name in different datasets is preserved as two rows.
    a_rows = df[df["dataset"] == "ds_a"]["problem"].tolist()
    b_rows = sorted(df[df["dataset"] == "ds_b"]["problem"].tolist())
    assert a_rows == ["prob_one"]
    assert b_rows == ["prob_one", "prob_two"]


def test_runner_resume_keys_are_dataset_aware(monkeypatch, tmp_path: Path):
    """Resume must not conflate identically-named problems across datasets."""

    class _DatasetA:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="ds_a", name="shared", kind=QP)]

    class _DatasetB:
        def __init__(self, repo_root=None, **options):
            pass

        def list_problems(self):
            return [ProblemSpec(dataset_id="ds_b", name="shared", kind=QP)]

    class _StubSolver(SolverAdapter):
        solver_name = "stub"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    monkeypatch.setitem(dataset_registry.DATASETS, "ds_a", _DatasetA)
    monkeypatch.setitem(dataset_registry.DATASETS, "ds_b", _DatasetB)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", _StubSolver)

    config = parse_run_config(
        {
            "run": {
                "datasets": ["ds_a", "ds_b"],
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
                "resume": True,
            },
            "solvers": [{"id": "stub_solver", "solver": "stub", "settings": {}}],
        }
    )

    # Pre-populate one dataset's row so resume only skips that dataset's task.
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    store.write_result(
        ProblemResult(
            run_id=store.run_id,
            dataset="ds_a",
            problem="shared",
            problem_kind=QP,
            solver_id="stub_solver",
            solver="stub",
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.0,
        )
    )

    completed = store.completed_keys()
    assert ("ds_a", "shared", "stub_solver") in completed
    assert ("ds_b", "shared", "stub_solver") not in completed


def test_runner_distinguishes_repeats_of_one_adapter_via_explicit_ids(monkeypatch, tmp_path: Path, repo_root: Path):
    """Same registry name listed twice with distinct ids must produce two
    independent dataset slots: separate artifact directories, separate
    resume keys, and distinct ``dataset`` values on each row."""

    seen_options: list[dict] = []

    class _ConfigurableDataset:
        def __init__(self, repo_root=None, **options):
            seen_options.append(dict(options))
            self._variant = options.get("variant", "default")

        def list_problems(self):
            return [ProblemSpec(dataset_id=self._variant, name="shared", kind=QP)]

    class _StubSolver(SolverAdapter):
        solver_name = "stub"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):  # pragma: no cover
            raise AssertionError

    monkeypatch.setitem(dataset_registry.DATASETS, "configurable", _ConfigurableDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", _StubSolver)

    def _fake_run_subprocess(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        from types import SimpleNamespace

        payload_path = Path(cmd[-1])
        payload = json.loads(payload_path.read_text())
        artifact_dir = Path(payload["artifacts_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        record = ProblemResult(
            run_id=payload["run_id"],
            dataset=payload["dataset"],
            problem=payload["problem"],
            problem_kind=payload["problem_kind"],
            solver_id=payload["solver"]["id"],
            solver=payload["solver"]["solver"],
            status="optimal",
            objective_value=0.0,
            iterations=1,
            run_time_seconds=0.01,
            artifact_dir=str(artifact_dir),
        ).to_record()
        (artifact_dir / "worker_result.json").write_text(json.dumps(record))
        stdout_path.write_text("")
        stderr_path.write_text("")
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    monkeypatch.setattr(
        "solver_benchmarks.core.runner._run_subprocess", _fake_run_subprocess
    )

    config = parse_run_config(
        {
            "run": {
                "datasets": [
                    {
                        "id": "feasible",
                        "name": "configurable",
                        "dataset_options": {"variant": "feasible"},
                    },
                    {
                        "id": "infeasible",
                        "name": "configurable",
                        "dataset_options": {"variant": "infeasible"},
                    },
                ],
                "output_dir": str(tmp_path / "runs"),
                "parallelism": 1,
            },
            "solvers": [{"id": "stub_solver", "solver": "stub", "settings": {}}],
        }
    )

    store = run_benchmark(config, repo_root=repo_root)
    df = load_results(store.run_dir)

    # Both variants ran the adapter with their own options.
    assert {opts.get("variant") for opts in seen_options} == {"feasible", "infeasible"}
    # Two rows, one per dataset id, even though both share the same problem
    # name and the same registry adapter.
    assert sorted(df["dataset"].tolist()) == ["feasible", "infeasible"]
    assert set(df["problem"]) == {"shared"}
    # Artifact directories are slugged on the dataset id so they don't collide.
    assert (store.run_dir / "problems" / "feasible" / "shared" / "stub_solver").exists()
    assert (store.run_dir / "problems" / "infeasible" / "shared" / "stub_solver").exists()
    # Resume tracks (id, problem, solver_id), so both entries are independent.
    completed = store.completed_keys()
    assert ("feasible", "shared", "stub_solver") in completed
    assert ("infeasible", "shared", "stub_solver") in completed


def test_worker_loads_dataset_via_registry_name_not_entry_id(monkeypatch, tmp_path: Path, repo_root: Path):
    """When an entry uses an explicit ``id`` distinct from its registry
    ``name``, the worker must look the adapter up by ``dataset_name``.
    Looking up by the entry id (which is not in the registry) would make
    every solve fail with ``worker_error``."""

    import numpy as np
    import scipy.sparse as sp

    from solver_benchmarks.core import status as status_module
    from solver_benchmarks.core.problem import ProblemData
    from solver_benchmarks.core.result import SolverResult
    from solver_benchmarks.solvers.base import SolverAdapter
    from solver_benchmarks.worker import run_payload

    seen_dataset_init = []

    class _ConfigurableDataset:
        def __init__(self, repo_root=None, **options):
            seen_dataset_init.append(options)

        def list_problems(self):
            return [ProblemSpec(dataset_id="configurable", name="tiny", kind=QP)]

        def load_problem(self, name):
            qp = {
                "P": sp.csc_matrix(np.array([[1.0]])),
                "q": np.array([0.0]),
                "r": 0.0,
                "A": sp.csc_matrix(np.array([[1.0]])),
                "l": np.array([0.0]),
                "u": np.array([0.0]),
                "n": 1,
                "m": 1,
                "obj_type": "min",
            }
            return ProblemData("configurable", name, QP, qp)

    class _StubSolver(SolverAdapter):
        solver_name = "stub"
        supported_problem_kinds = {QP}

        def solve(self, problem, artifacts_dir):
            return SolverResult(
                status=status_module.OPTIMAL,
                objective_value=0.0,
                iterations=1,
                run_time_seconds=0.001,
                info={},
            )

    monkeypatch.setitem(dataset_registry.DATASETS, "configurable", _ConfigurableDataset)
    monkeypatch.setitem(solver_registry.SOLVERS, "stub", _StubSolver)

    artifacts_dir = tmp_path / "artifacts"
    payload = {
        "run_id": "run",
        "dataset": "feasible",          # entry id — NOT a registry key
        "dataset_name": "configurable", # registry key for the adapter
        "dataset_options": {"variant": "feasible"},
        "problem": "tiny",
        "problem_kind": QP,
        "solver": {"id": "stub_solver", "solver": "stub", "settings": {}},
        "artifacts_dir": str(artifacts_dir),
        "repo_root": str(repo_root),
    }

    result = run_payload(payload)
    assert result.status == "optimal", result.error
    # Result identity preserves the entry id, while the lookup used the name.
    assert result.dataset == "feasible"
    assert seen_dataset_init == [{"variant": "feasible"}]


def test_environment_matrix_runs_current_python_and_preserves_manifest(tmp_path: Path, repo_root: Path):
    config = parse_environment_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
                "parallelism": 1,
            },
            "environments": [
                {
                    "id": "current_a",
                    "python": sys.executable,
                    "metadata": {"label": "current-a"},
                    "solvers": [
                        {
                            "id": "scs_current_a",
                            "solver": "scs",
                            "settings": {"verbose": False, "max_iters": 1000},
                        }
                    ],
                },
                {
                    "id": "current_b",
                    "python": sys.executable,
                    "metadata": {"label": "current-b"},
                    "solvers": [
                        {
                            "id": "scs_current_b",
                            "solver": "scs",
                            "settings": {"verbose": False, "max_iters": 1000},
                        }
                    ],
                },
            ],
        }
    )

    run_dir = run_environment_matrix(
        config,
        run_dir=tmp_path / "matrix",
        repo_root=repo_root,
        stream_output=False,
    )
    df = load_results(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert set(df["solver_id"]) == {"scs_current_a", "scs_current_b"}
    assert set(df["metadata.environment_id"]) == {"current_a", "current_b"}
    assert set(df["status"]) == {"optimal"}
    assert {solver["id"] for solver in manifest["config"]["solvers"]} == {
        "scs_current_a",
        "scs_current_b",
    }
