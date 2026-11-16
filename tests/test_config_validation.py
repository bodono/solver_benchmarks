"""Config files fail loudly and early on mistakes, and parse numbers as written."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from solver_benchmarks.cli import main
from solver_benchmarks.core.config import (
    load_environment_run_config,
    load_run_config,
    parse_environment_run_config,
    parse_run_config,
)
from solver_benchmarks.core.storage import ResultStore, slugify

REPO_ROOT = Path(__file__).resolve().parent.parent


def _minimal(**run):
    return {"run": {"dataset": "synthetic_qp", **run}, "solvers": [{"id": "s", "solver": "scs"}]}


@pytest.mark.parametrize(
    "path",
    sorted(
        path
        for path in (REPO_ROOT / "configs").rglob("*")
        if path.suffix in {".yaml", ".yml", ".json"}
    ),
    ids=lambda path: str(path.relative_to(REPO_ROOT / "configs")),
)
def test_every_bundled_config_parses(path: Path):
    raw = yaml.safe_load(path.read_text())
    if "environments" in raw:
        load_environment_run_config(path)
    else:
        load_run_config(path)


def test_yaml_exponent_floats_parse_as_floats(tmp_path: Path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "run: {dataset: synthetic_qp, timeout_seconds: 1e2}\n"
        "solvers:\n"
        "  - {id: tight, solver: scs, settings: {eps_abs: 1e-9, eps_rel: 1.0e-9, max_iters: 1e6}}\n"
        "  - id: sweep\n"
        "    solver: scs\n"
        "    sweep: {rho_x: [1.0e-6, 1.0e6, 1E+3]}\n"
        "    settings: {name: 1e-6x, version: 1.0.0}\n"
    )
    config = load_run_config(path)
    tight = config.solvers[0].settings
    assert tight["eps_abs"] == 1e-9 and isinstance(tight["eps_abs"], float)
    assert tight["eps_rel"] == 1e-9
    assert tight["max_iters"] == 1e6
    assert config.timeout_seconds == 100.0
    assert [solver.settings["rho_x"] for solver in config.solvers[1:]] == [1e-6, 1e6, 1e3]
    # Strings that are not numbers stay strings.
    assert config.solvers[1].settings["name"] == "1e-6x"
    assert config.solvers[1].settings["version"] == "1.0.0"


@pytest.mark.parametrize(
    "raw, unknown, suggestion",
    [
        (_minimal(timeout=60), "timeout", "timeout_seconds"),
        (_minimal(paralellism=8), "paralellism", "parallelism"),
        ({**_minimal(), "solvres": []}, "solvres", "solvers"),
        (
            {"run": {"dataset": "synthetic_qp"},
             "solvers": [{"id": "a", "solver": "scs", "setting": {"eps_abs": 1e-9}}]},
            "setting",
            "settings",
        ),
        (
            {"run": {"datasets": [{"name": "synthetic_qp", "dataset_option": {}}]},
             "solvers": [{"id": "a", "solver": "scs"}]},
            "dataset_option",
            "dataset_options",
        ),
    ],
)
def test_unknown_keys_are_rejected_with_a_suggestion(raw, unknown, suggestion):
    with pytest.raises(ValueError, match=rf"{unknown}.*{suggestion}"):
        parse_run_config(raw)


def test_unknown_environment_keys_are_rejected():
    raw = {
        "run": {"dataset": "synthetic_qp"},
        "environments": [{"id": "e", "pyhton": "python", "solvers": [{"id": "a", "solver": "scs"}]}],
    }
    with pytest.raises(ValueError, match="pyhton.*python"):
        parse_environment_run_config(raw)


def test_manifest_shaped_configs_still_parse():
    """Environment matrices hand each child run a manifest-shaped config."""
    config = parse_run_config(_minimal())
    assert parse_run_config(json.loads(json.dumps(config.to_manifest(), default=str))) == config


@pytest.mark.parametrize(
    "raw",
    [
        {"run": None, "dataset": "synthetic_qp", "solvers": [{"id": "s", "solver": "scs"}]},
        _minimal(dataset_options=None, include=None, exclude=None),
        {"run": {"datasets": [{"name": "synthetic_qp", "dataset_options": None,
                               "include": None}]},
         "solvers": [{"id": "s", "solver": "scs", "settings": None}]},
    ],
)
def test_empty_sections_count_as_empty(raw):
    config = parse_run_config(raw)
    assert config.datasets[0].dataset_options == {}
    assert config.solvers[0].settings == {}


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"run": None, "solvers": [{"id": "s", "solver": "scs"}]}, "dataset"),
        ({"run": {"dataset": "synthetic_qp"}, "solvers": {"s": {"solver": "scs"}}}, "solvers"),
        ({"run": {"dataset": "synthetic_qp"}, "solvers": ["scs"]}, "solver entry"),
        ({"run": "synthetic_qp", "solvers": [{"id": "s", "solver": "scs"}]}, "run"),
        (_minimal(dataset_options=["subset"]), "dataset_options"),
    ],
)
def test_malformed_sections_raise_value_errors(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_run_config(raw)


def test_unknown_solver_and_dataset_names_fail_at_parse_time():
    with pytest.raises(ValueError, match="clarabell.*clarabel"):
        parse_run_config({"run": {"dataset": "synthetic_qp"},
                          "solvers": [{"id": "s", "solver": "clarabell"}]})
    with pytest.raises(ValueError, match="netlb.*netlib"):
        parse_run_config({"run": {"dataset": "netlb"}, "solvers": [{"id": "s", "solver": "scs"}]})
    with pytest.raises(ValueError, match="nosuch"):
        parse_environment_run_config(
            {"run": {"dataset": "synthetic_qp"},
             "environments": [{"id": "e", "solvers": [{"id": "s", "solver": "nosuch"}]}]}
        )


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (
            {"dataset": "libsvm_qp", "dataset_options": {"kernal": "rbf"}},
            "'kernal' (did you mean 'kernel'?)",
        ),
        (
            {"dataset": "maros_meszaros", "dataset_options": {"subset": ["AUG2D", "CVXQP1_S"]}},
            "Unknown dataset option for 'maros_meszaros': 'subset'",
        ),
        (
            {"datasets": [{"name": "dimacs", "dataset_options": {"max_size": 0.1}}]},
            "'max_size' (did you mean 'max_size_mb'?)",
        ),
        (
            {"datasets": ["kennington"], "dataset_options": {"subset": "ken-07"}},
            "run.dataset_options: 'subset' is not an option of any listed dataset (kennington)",
        ),
        ({"dataset": "netlib", "dataset_options": {"subset": "all"}}, "Unknown NETLIB subset 'all'"),
    ],
)
def test_unknown_dataset_options_fail_at_parse_time(run, message):
    with pytest.raises(ValueError) as excinfo:
        parse_run_config({"run": run, "solvers": [{"id": "s", "solver": "scs"}]})
    assert message in str(excinfo.value)


def test_cli_reports_dataset_option_errors_without_a_traceback(tmp_path: Path):
    path = tmp_path / "typo.yaml"
    path.write_text(
        f"run: {{dataset: libsvm_qp, output_dir: {tmp_path / 'runs'}, "
        "dataset_options: {kernal: rbf, C: 10}}\n"
        "solvers: [{id: s, solver: clarabel}]\n"
    )
    result = CliRunner().invoke(main, ["run", str(path)])
    assert result.exit_code != 0
    assert "kernal" in result.output and "kernel" in result.output
    assert not isinstance(result.exception, (ValueError, KeyError))
    assert not (tmp_path / "runs").exists()


def test_cli_reports_config_errors_without_a_traceback(tmp_path: Path):
    path = tmp_path / "typo.yaml"
    path.write_text(
        f"run: {{dataset: synthetic_qp, output_dir: {tmp_path / 'runs'}, timeout: 5}}\n"
        "solvers: [{id: s, solver: clarabell}]\n"
    )
    result = CliRunner().invoke(main, ["run", str(path)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "timeout" in result.output and not isinstance(result.exception, (ValueError, KeyError))
    assert not (tmp_path / "runs").exists()

    env = tmp_path / "env.yaml"
    env.write_text("run: {dataset: synthetic_qp}\nenvironments: [{id: e, solvers: [{id: s, solver: nosuch}]}]\n")
    result = CliRunner().invoke(main, ["env", "run", str(env)])
    assert result.exit_code != 0 and "nosuch" in result.output
    assert not isinstance(result.exception, (ValueError, KeyError))


def test_sweep_ids_with_exponents_get_their_own_artifact_directories(tmp_path: Path):
    config = parse_run_config(
        {"run": {"dataset": "synthetic_qp"},
         "solvers": [{"id": "scs", "solver": "scs", "sweep": {"scale": [1e-6, 1e6]}}]}
    )
    ids = [solver.id for solver in config.solvers]
    assert ids == ["scs__scale=1e-06", "scs__scale=1e+06"]
    store = ResultStore.create(config, run_dir=tmp_path / "run")
    dirs = {store.problem_solver_dir("synthetic_qp", "p", solver_id) for solver_id in ids}
    assert len(dirs) == 2
    assert slugify("osqp+polish") != slugify("osqp-polish")


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"run": {"dataset": "synthetic_qp"},
          "solvers": [{"id": "Fast", "solver": "scs"}, {"id": "fast", "solver": "osqp"}]},
         "Fast.*fast"),
        ({"run": {"datasets": [{"name": "synthetic_qp", "id": "a b"},
                               {"name": "synthetic_qp", "id": "a-b"}]},
          "solvers": [{"id": "s", "solver": "scs"}]},
         "a b.*a-b"),
    ],
)
def test_ids_that_share_an_artifact_directory_are_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_run_config(raw)


def test_ids_that_share_an_artifact_directory_are_rejected_across_environments():
    raw = {
        "run": {"dataset": "synthetic_qp"},
        "environments": [
            {"id": "a", "solvers": [{"id": "OSQP", "solver": "osqp"}]},
            {"id": "b", "solvers": [{"id": "osqp", "solver": "osqp"}]},
        ],
    }
    with pytest.raises(ValueError, match="OSQP.*osqp"):
        parse_environment_run_config(raw)


def test_relative_data_root_resolves_against_the_repo_root(monkeypatch, tmp_path: Path):
    """Like run.output_dir, and as before when run from the repository root:
    not against the config file's directory or the caller's."""
    from solver_benchmarks.core.result import ProblemResult
    from solver_benchmarks.core.runner import run_benchmark

    repo = tmp_path / "repo"
    data = repo / "mydata" / "netlib_data" / "feasible"
    data.mkdir(parents=True)
    afiro = REPO_ROOT / "problem_classes" / "netlib_data" / "feasible" / "afiro.mps.gz"
    (data / "afiro.mps.gz").write_bytes(afiro.read_bytes())
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "rel.yaml"
    path.write_text(
        "run:\n  output_dir: " + str(tmp_path / "runs") + "\n"
        "  datasets:\n    - name: netlib\n      include: [afiro]\n"
        "      dataset_options: {data_root: mydata}\n"
        "solvers: [{id: s, solver: scs}]\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_run_config(path)
    assert config.datasets[0].dataset_options["data_root"] == "mydata"

    payloads = []

    def fake(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        payloads.append(payload)
        record = ProblemResult(
            run_id=payload["run_id"], dataset=payload["dataset"], problem=payload["problem"],
            problem_kind=payload["problem_kind"], solver_id="s", solver="scs",
            status="optimal", objective_value=0.0, iterations=1, run_time_seconds=0.01,
            artifact_dir=payload["artifacts_dir"],
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, timed_out=False)

    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", fake)
    store = run_benchmark(config, repo_root=repo)
    resolved = str((repo / "mydata").resolve())
    assert [payload["problem"] for payload in payloads] == ["afiro"]
    assert payloads[0]["dataset_options"]["data_root"] == resolved
    manifest = json.loads(store.manifest_path.read_text())
    assert manifest["config"]["datasets"][0]["dataset_options"]["data_root"] == resolved


def test_run_finds_bundled_data_from_outside_the_repository(
    monkeypatch, tmp_path: Path
):
    """bench run without --repo-root defaults like the other commands: the
    current directory if it holds problem_classes, else the package's repo."""
    from solver_benchmarks.core.result import ProblemResult
    from solver_benchmarks.core.runner import run_benchmark

    seen = {}

    def fake(cmd, *, cwd, timeout, stdout_path, stderr_path, stream_output):
        payload = json.loads(Path(cmd[-1]).read_text())
        seen["cwd"] = cwd
        record = ProblemResult(
            run_id=payload["run_id"], dataset=payload["dataset"], problem=payload["problem"],
            problem_kind=payload["problem_kind"], solver_id="s", solver="scs",
            status="optimal", objective_value=0.0, iterations=1, run_time_seconds=0.01,
            artifact_dir=payload["artifacts_dir"],
        ).to_record()
        (Path(payload["artifacts_dir"]) / "worker_result.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0, timed_out=False)

    monkeypatch.setattr("solver_benchmarks.core.runner._run_subprocess", fake)
    monkeypatch.chdir(tmp_path)
    config = parse_run_config(
        {"run": {"dataset": "netlib", "include": ["afiro"], "output_dir": str(tmp_path / "runs")},
         "solvers": [{"id": "s", "solver": "scs"}]}
    )
    store = run_benchmark(config)
    assert seen["cwd"] == REPO_ROOT
    assert len(store.results_jsonl_path.read_text().splitlines()) == 1


def test_environment_matrix_records_a_missing_python_and_continues(tmp_path: Path):
    from solver_benchmarks.core.env_runner import run_environment_matrix

    config = parse_environment_run_config(
        {
            "run": {"dataset": "synthetic_qp", "include": ["one_variable_eq"]},
            "environments": [
                {"id": "broken", "python": str(tmp_path / "no-venv" / "bin" / "python"),
                 "solvers": [{"id": "scs_broken", "solver": "scs"}]},
                {"id": "bad_install", "python": sys.executable,
                 "install": ["{python} -c 'x = {unknown_field}'"],
                 "solvers": [{"id": "scs_bad_install", "solver": "scs"}]},
                {"id": "missing_install", "python": sys.executable,
                 "install": [str(tmp_path / "no-such-installer")],
                 "solvers": [{"id": "scs_missing_install", "solver": "scs"}]},
                {"id": "good", "python": sys.executable,
                 "solvers": [{"id": "scs_good", "solver": "scs", "settings": {"verbose": False}}]},
            ],
        }
    )
    with pytest.raises(
        RuntimeError,
        match=r"broken \(run\), bad_install \(install\), missing_install \(install\)",
    ):
        run_environment_matrix(
            config, run_dir=tmp_path / "matrix", repo_root=REPO_ROOT, stream_output=False
        )
    rows = [json.loads(line) for line in (tmp_path / "matrix" / "results.jsonl").read_text().splitlines()]
    assert {row["solver_id"] for row in rows} >= {"scs_good"}
    events = [json.loads(line) for line in (tmp_path / "matrix" / "events.jsonl").read_text().splitlines()]
    failed = {event["environment_id"]: event for event in events if "environment_id" in event}
    assert failed["broken"]["message"] == "environment_run_failed"
    assert "no-venv" in failed["broken"]["error"]
    assert failed["missing_install"]["message"] == "environment_install_failed"
    assert "unknown_field" in failed["bad_install"]["error"]
