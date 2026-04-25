from pathlib import Path

import pytest

from solver_benchmarks.core.config import (
    load_run_config,
    parse_environment_run_config,
    parse_run_config,
)
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


def test_parse_environment_run_config():
    config = parse_environment_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": "runs",
                "include": ["one_variable_lp"],
            },
            "environments": [
                {
                    "id": "osqp_1_0",
                    "python": ".venv-osqp-1.0/bin/python",
                    "install": ["{python} -m pip install osqp==1.0.0"],
                    "metadata": {"osqp": "1.0.0"},
                    "solvers": [
                        {
                            "id": "osqp_1_0_default",
                            "solver": "osqp",
                            "settings": {"eps_abs": 1.0e-6},
                        }
                    ],
                },
                {
                    "id": "osqp_1_1",
                    "python": ".venv-osqp-1.1/bin/python",
                    "solvers": [
                        {
                            "id": "osqp_1_1_default",
                            "solver": "osqp",
                            "settings": {"eps_abs": 1.0e-6},
                        }
                    ],
                },
            ],
        }
    )

    assert config.run.dataset == "synthetic_qp"
    assert [env.id for env in config.environments] == ["osqp_1_0", "osqp_1_1"]
    assert config.environments[0].install == ["{python} -m pip install osqp==1.0.0"]
    assert config.environments[0].solvers[0].id == "osqp_1_0_default"


def test_parse_run_config_expands_solver_sweep_cross_product():
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp"},
            "solvers": [
                {
                    "id": "scs",
                    "solver": "scs",
                    "settings": {"verbose": False, "max_iters": 1000},
                    "sweep": {
                        "eps_abs": [1.0e-4, 1.0e-6],
                        "linear_solver": ["direct", "indirect"],
                    },
                    "id_template": "scs_abs{eps_abs:g}_{linear_solver}",
                }
            ],
        }
    )

    assert [solver.id for solver in config.solvers] == [
        "scs_abs0.0001_direct",
        "scs_abs0.0001_indirect",
        "scs_abs1e-06_direct",
        "scs_abs1e-06_indirect",
    ]
    assert config.solvers[0].settings == {
        "verbose": False,
        "max_iters": 1000,
        "eps_abs": 1.0e-4,
        "linear_solver": "direct",
    }
    assert config.solvers[-1].settings["eps_abs"] == 1.0e-6
    assert config.solvers[-1].settings["linear_solver"] == "indirect"


def test_parse_run_config_generates_default_sweep_ids():
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp"},
            "solvers": [
                {
                    "id": "scs",
                    "solver": "scs",
                    "sweep": {
                        "alpha": [1.5],
                        "normalize": [True, False],
                    },
                }
            ],
        }
    )

    assert [solver.id for solver in config.solvers] == [
        "scs__alpha=1.5__normalize=true",
        "scs__alpha=1.5__normalize=false",
    ]


def test_parse_environment_run_config_expands_solver_sweeps():
    config = parse_environment_run_config(
        {
            "run": {"dataset": "synthetic_qp"},
            "environments": [
                {
                    "id": "current",
                    "solvers": [
                        {
                            "id": "scs_env",
                            "solver": "scs",
                            "sweep": {"eps_abs": [1.0e-4, 1.0e-6]},
                            "id_template": "scs_env_{eps_abs:g}",
                        }
                    ],
                }
            ],
        }
    )

    assert [solver.id for solver in config.environments[0].solvers] == [
        "scs_env_0.0001",
        "scs_env_1e-06",
    ]


def test_parse_run_config_rejects_duplicate_sweep_ids():
    with pytest.raises(ValueError, match="Duplicate solver id"):
        parse_run_config(
            {
                "run": {"dataset": "synthetic_qp"},
                "solvers": [
                    {
                        "id": "scs",
                        "solver": "scs",
                        "sweep": {"eps_abs": [1.0e-4, 1.0e-6]},
                        "id_template": "scs_duplicate",
                    }
                ],
            }
        )


def test_parse_environment_run_config_rejects_duplicate_solver_ids_across_envs():
    with pytest.raises(ValueError, match="Duplicate solver id across environments"):
        parse_environment_run_config(
            {
                "run": {"dataset": "synthetic_qp"},
                "environments": [
                    {
                        "id": "env_a",
                        "solvers": [{"id": "scs_same", "solver": "scs"}],
                    },
                    {
                        "id": "env_b",
                        "solvers": [{"id": "scs_same", "solver": "scs"}],
                    },
                ],
            }
        )
