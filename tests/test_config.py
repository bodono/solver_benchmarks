from pathlib import Path

from solver_benchmarks.core.config import parse_run_config


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
