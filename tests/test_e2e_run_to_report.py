"""End-to-end test: run_benchmark → load_results → write_run_report.

Pins the contract that the entire pipeline runs without errors and
produces the expected on-disk artifacts. Previously the runner and
the report layer each had their own tests but no test chained them,
so a contract drift between them would only show up in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.analysis.markdown_report import write_run_report
from solver_benchmarks.core.config import parse_run_config
from solver_benchmarks.core.runner import run_benchmark
from solver_benchmarks.solvers import get_solver


def test_run_then_report_produces_artifacts(tmp_path: Path, repo_root: Path):
    if not get_solver("scs").is_available():
        pytest.skip("scs not installed")

    config = parse_run_config(
        {
            "run": {
                "dataset": "synthetic_qp",
                "output_dir": str(tmp_path / "runs"),
                "include": ["one_variable_eq", "one_variable_lp"],
                "parallelism": 1,
            },
            "solvers": [
                {
                    "id": "scs_e2e",
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

    df = load_results(store.run_dir)
    assert len(df) == 2
    assert set(df["status"]) == {"optimal"}

    report_dir = tmp_path / "report"
    outputs = write_run_report(store.run_dir, output_dir=report_dir, repo_root=repo_root)
    output_paths = {Path(path).name for path in outputs}

    # Core report files always present.
    assert "index.md" in output_paths
    assert "README.md" in output_paths
    assert "solver_metrics.csv" in output_paths
    assert "completion.csv" in output_paths
    assert "kkt_summary.csv" in output_paths

    # Performance plot got rendered.
    assert any(name.startswith("performance_profile_") for name in output_paths)
    assert any(name.startswith("cactus_") for name in output_paths)

    # Markdown is non-empty and references the dataset and solver.
    markdown = (report_dir / "index.md").read_text()
    assert "Benchmark Report" in markdown
    assert "scs_e2e" in markdown
    assert "synthetic_qp" in markdown

    # README.md is content-equivalent (either symlink target or written
    # copy). The PR 7 change made it a symlink-or-copy of index.md.
    readme = (report_dir / "README.md").read_text()
    assert readme == markdown
