from pathlib import Path

import pytest

from solver_benchmarks.core.config import (
    DatasetConfig,
    load_run_config,
    manifest_dataset_entries,
    parse_environment_run_config,
    parse_run_config,
    resolve_output_dir,
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


def test_load_run_config_uses_config_stem_as_default_run_name(tmp_path: Path):
    config_path = tmp_path / "scs_anderson_sweep.yaml"
    config_path.write_text(
        """
run:
  dataset: synthetic_qp
solvers:
  - id: scs
    solver: scs
    settings: {}
"""
    )

    config = load_run_config(config_path)

    assert config.name == "scs_anderson_sweep"


def test_relative_output_dir_is_repo_root_relative(tmp_path: Path):
    config_path = tmp_path / "configs" / "relative_output.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        """
run:
  dataset: synthetic_qp
  output_dir: results
solvers:
  - id: scs
    solver: scs
    settings: {}
"""
    )

    config = load_run_config(config_path)
    resolved = resolve_output_dir(config, tmp_path)

    assert config.output_dir == Path("results")
    assert resolved.output_dir == (tmp_path / "results").resolve()


def test_default_output_dir_is_results():
    config = parse_run_config(
        {
            "run": {"dataset": "synthetic_qp"},
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    assert config.output_dir == Path("results")


def test_explicit_run_name_overrides_config_stem(tmp_path: Path):
    config_path = tmp_path / "file_name.yaml"
    config_path.write_text(
        """
run:
  name: publication_sweep
  dataset: synthetic_qp
solvers:
  - id: scs
    solver: scs
    settings: {}
"""
    )

    config = load_run_config(config_path)

    assert config.name == "publication_sweep"


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


def test_config_hash_ignores_run_name(tmp_path: Path):
    first = parse_run_config(
        {
            "run": {"name": "first", "dataset": "synthetic_qp", "output_dir": str(tmp_path)},
            "solvers": [{"id": "scs", "solver": "scs", "settings": {"eps_abs": 1e-4}}],
        }
    )
    second = parse_run_config(
        {
            "run": {"name": "second", "dataset": "synthetic_qp", "output_dir": str(tmp_path)},
            "solvers": [{"id": "scs", "solver": "scs", "settings": {"eps_abs": 1e-4}}],
        }
    )

    assert first.config_hash == second.config_hash


def test_solver_settings_are_verbose_by_default():
    assert settings_with_defaults({})["verbose"] is True
    assert settings_with_defaults({"eps_abs": 1.0e-6})["verbose"] is True
    assert settings_with_defaults({"verbose": False})["verbose"] is False


def test_example_configs_do_not_opt_out_of_verbose_by_default():
    config_paths = [
        *Path("configs").glob("*.json"),
        *Path("configs").glob("*.yaml"),
        *Path("configs").glob("*.yml"),
    ]
    for config_path in config_paths:
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


def test_parse_run_config_accepts_datasets_list_of_strings(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "datasets": ["synthetic_qp", "synthetic_lp"],
                "output_dir": str(tmp_path / "runs"),
            },
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    assert [dataset.name for dataset in config.datasets] == [
        "synthetic_qp",
        "synthetic_lp",
    ]
    assert all(dataset.dataset_options == {} for dataset in config.datasets)


def test_parse_run_config_accepts_datasets_list_of_mappings(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "datasets": [
                    {
                        "name": "netlib_lp",
                        "dataset_options": {"variant": "feasible"},
                        "include": ["afiro"],
                    },
                    {
                        "name": "synthetic_qp",
                        "exclude": ["one_variable_lp"],
                    },
                ],
                "output_dir": str(tmp_path / "runs"),
            },
            "solvers": [{"id": "scs", "solver": "scs", "settings": {}}],
        }
    )

    netlib, synthetic = config.datasets
    assert netlib.name == "netlib_lp"
    assert netlib.dataset_options == {"variant": "feasible"}
    assert netlib.include == ["afiro"]
    assert synthetic.exclude == ["one_variable_lp"]
    # Run-level include is empty so synthetic falls through to no include filter.
    include, exclude = config.effective_filters(synthetic)
    assert include == []
    assert exclude == ["one_variable_lp"]


def test_parse_run_config_rejects_dataset_and_datasets_together(tmp_path: Path):
    with pytest.raises(ValueError, match="not both"):
        parse_run_config(
            {
                "run": {
                    "dataset": "synthetic_qp",
                    "datasets": ["synthetic_lp"],
                    "output_dir": str(tmp_path / "runs"),
                },
                "solvers": [{"id": "scs", "solver": "scs"}],
            }
        )


def test_parse_run_config_rejects_duplicate_dataset_names(tmp_path: Path):
    with pytest.raises(ValueError, match="Duplicate dataset"):
        parse_run_config(
            {
                "run": {
                    "datasets": ["synthetic_qp", "synthetic_qp"],
                    "output_dir": str(tmp_path / "runs"),
                },
                "solvers": [{"id": "scs", "solver": "scs"}],
            }
        )


def test_run_config_dataset_property_errors_for_multiple(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "datasets": ["synthetic_qp", "synthetic_lp"],
                "output_dir": str(tmp_path / "runs"),
            },
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    with pytest.raises(ValueError, match="datasets"):
        _ = config.dataset


def test_run_config_dataset_level_include_overrides_run_level(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "datasets": [
                    {"name": "netlib_lp", "include": ["afiro"]},
                    {"name": "synthetic_qp"},
                ],
                "include": ["one_variable_eq"],
                "output_dir": str(tmp_path / "runs"),
            },
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    netlib, synthetic = config.datasets
    netlib_include, _ = config.effective_filters(netlib)
    synthetic_include, _ = config.effective_filters(synthetic)
    assert netlib_include == ["afiro"]
    assert synthetic_include == ["one_variable_eq"]


def test_legacy_dataset_options_apply_as_default_for_datasets_list(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "datasets": [
                    {"name": "netlib_lp"},
                    {
                        "name": "maros_meszaros",
                        "dataset_options": {"override": "yes"},
                    },
                ],
                "dataset_options": {"shared": "value"},
                "output_dir": str(tmp_path / "runs"),
            },
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    netlib, maros = config.datasets
    assert netlib.dataset_options == {"shared": "value"}
    assert maros.dataset_options == {"shared": "value", "override": "yes"}


def test_manifest_dataset_entries_handles_legacy_and_new_shapes():
    legacy = manifest_dataset_entries(
        {
            "dataset": "synthetic_qp",
            "dataset_options": {"foo": "bar"},
            "include": ["one_variable_eq"],
            "exclude": [],
        }
    )
    assert legacy == [
        {
            "id": "synthetic_qp",
            "name": "synthetic_qp",
            "dataset_options": {"foo": "bar"},
            "include": ["one_variable_eq"],
            "exclude": [],
        }
    ]

    multi = manifest_dataset_entries(
        {
            "datasets": [
                {
                    "name": "netlib_lp",
                    "dataset_options": {},
                    "include": ["afiro"],
                    "exclude": [],
                },
                {
                    "name": "synthetic_qp",
                    "dataset_options": {},
                    "include": [],
                    "exclude": ["bad_problem"],
                },
            ],
            "include": ["fallback"],
            "exclude": ["global_bad"],
        }
    )
    netlib, synthetic = multi
    # Dataset-level include is set, so run-level fallback is not applied.
    assert netlib["include"] == ["afiro"]
    # Empty dataset-level include falls back to run-level include.
    assert synthetic["include"] == ["fallback"]
    # Exclude is unioned across run-level and dataset-level.
    assert "global_bad" in netlib["exclude"]
    assert "global_bad" in synthetic["exclude"]
    assert "bad_problem" in synthetic["exclude"]


def test_parse_run_config_allows_same_dataset_twice_with_distinct_ids(tmp_path: Path):
    config = parse_run_config(
        {
            "run": {
                "datasets": [
                    {
                        "id": "netlib_feasible",
                        "name": "netlib_lp",
                        "dataset_options": {"subset": "feasible"},
                    },
                    {
                        "id": "netlib_infeasible",
                        "name": "netlib_lp",
                        "dataset_options": {"subset": "infeasible"},
                    },
                ],
                "output_dir": str(tmp_path / "runs"),
            },
            "solvers": [{"id": "scs", "solver": "scs"}],
        }
    )
    feasible, infeasible = config.datasets
    assert feasible.id == "netlib_feasible"
    assert feasible.name == "netlib_lp"
    assert feasible.dataset_options == {"subset": "feasible"}
    assert infeasible.id == "netlib_infeasible"
    assert infeasible.name == "netlib_lp"
    assert infeasible.dataset_options == {"subset": "infeasible"}


def test_parse_run_config_rejects_duplicate_dataset_ids(tmp_path: Path):
    with pytest.raises(ValueError, match="Duplicate dataset id"):
        parse_run_config(
            {
                "run": {
                    "datasets": [
                        {"id": "shared", "name": "netlib_lp"},
                        {"id": "shared", "name": "synthetic_qp"},
                    ],
                    "output_dir": str(tmp_path / "runs"),
                },
                "solvers": [{"id": "scs", "solver": "scs"}],
            }
        )


def test_dataset_config_id_defaults_to_name():
    dataset = DatasetConfig(name="synthetic_qp")
    assert dataset.id == "synthetic_qp"
    explicit = DatasetConfig(name="synthetic_qp", id="qp_run")
    assert explicit.id == "qp_run"
    # Name remains the registry key even when id is given.
    assert explicit.name == "synthetic_qp"


def test_manifest_dataset_entries_surfaces_explicit_dataset_id():
    entries = manifest_dataset_entries(
        {
            "datasets": [
                {
                    "id": "netlib_feasible",
                    "name": "netlib_lp",
                    "dataset_options": {"subset": "feasible"},
                    "include": [],
                    "exclude": [],
                },
                {
                    "name": "synthetic_qp",
                    "dataset_options": {},
                    "include": [],
                    "exclude": [],
                },
            ],
        }
    )
    assert entries[0]["id"] == "netlib_feasible"
    assert entries[0]["name"] == "netlib_lp"
    # Default id == name when no explicit id is given.
    assert entries[1]["id"] == "synthetic_qp"


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
