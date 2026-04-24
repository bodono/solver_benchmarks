from pathlib import Path

from solver_benchmarks.core.config import load_run_config, parse_run_config
from solver_benchmarks.solvers.base import settings_with_defaults


def test_parse_run_config_supports_multiple_solver_variants(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq"],
                "parallelism": 2,
            },
            "solvers": [
                {"id": "scs_a", "solver": "scs", "settings": {"eps_abs": 1e-4}},
                {"id": "scs_b", "solver": "scs", "settings": {"eps_abs": 1e-6}},
            ],
        }
    )

    assert config.dataset == "synthetic_qp"
    assert config.parallelism == 2
    assert not config.auto_prepare_data
    assert [solver.id for solver in config.solvers] == ["scs_a", "scs_b"]
    assert config.solvers[0].settings != config.solvers[1].settings


def test_parse_run_config_supports_auto_prepare_data(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "auto_prepare_data": True,
            },
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    assert config.auto_prepare_data


def test_config_hash_changes_with_solver_settings(tmp_path: Path):
    first = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "output_dir": str(tmp_path)},
            "solvers": [{"id": "scs", "solver": "scs", "settings": {"eps_abs": 1e-4}}],
        }
    )
    second = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp", "output_dir": str(tmp_path)},
            "solvers": [{"id": "scs", "solver": "scs", "settings": {"eps_abs": 1e-6}}],
        }
    )

    assert first.config_hash != second.config_hash


def test_solver_settings_are_verbose_by_default():
    assert settings_with_defaults({})["verbose"] is True
    assert settings_with_defaults({"eps_abs": 1.0e-6})["verbose"] is True
    assert settings_with_defaults({"verbose": False})["verbose"] is False


def test_example_configs_do_not_opt_out_of_verbose_by_default():
    for config_path in Path("configs").glob("*.*"):
        config = load_run_config(config_path)
        assert all(
            solver.settings.get("verbose", True) is not False
            for solver in config.solvers
        )
