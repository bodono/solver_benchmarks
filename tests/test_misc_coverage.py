"""Misc coverage tests pinning behaviors flagged in the audit:

- ``bench list datasets`` / ``bench list solvers`` end-to-end (only
  hit by a CI smoke step previously).
- ``make_run_id`` / ``_datasets_slug`` truncation for many-dataset runs.
- ``kkt_summary`` / ``kkt_certificate_summary`` shape and aggregation
  paths (these report functions were never imported by the test suite
  before).
"""

from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner

from solver_benchmarks.analysis.tables import (
    kkt_certificate_summary,
    kkt_summary,
)
from solver_benchmarks.cli import main
from solver_benchmarks.core.config import (
    DatasetConfig,
    RunConfig,
    SolverConfig,
)
from solver_benchmarks.core.storage import _datasets_slug, make_run_id


def test_cli_list_datasets_emits_each_registered_dataset():
    result = CliRunner().invoke(main, ["list", "datasets"])
    assert result.exit_code == 0, result.output
    # At least the always-registered datasets show up; the CI smoke
    # step asserts the registry, so here we just confirm something
    # was actually printed.
    assert "synthetic_qp" in result.output


def test_cli_list_solvers_marks_optional_extras():
    result = CliRunner().invoke(main, ["list", "solvers"])
    assert result.exit_code == 0, result.output
    # Each line is "<name>\t<availability>\t<kinds>"; check that
    # availability column exists.
    for line in result.output.strip().splitlines():
        parts = line.split("\t")
        assert len(parts) == 3
        assert parts[1] in {"available", "missing optional extra"}


def test_make_run_id_truncates_many_dataset_slug():
    """A multi-dataset run name longer than 64 chars should collapse to
    "multi-N" rather than producing a multi-line filesystem path."""
    long_names = [f"dataset_with_a_long_name_{idx:02d}" for idx in range(10)]
    config = RunConfig(
        datasets=[DatasetConfig(name=n) for n in long_names],
        solvers=[SolverConfig(id="scs", solver="scs")],
    )
    slug = _datasets_slug(config)
    assert slug == "multi-10"
    run_id = make_run_id(config)
    assert "multi-10" in run_id


def test_make_run_id_keeps_single_dataset_slug_intact():
    config = RunConfig(
        datasets=[DatasetConfig(name="netlib_lp")],
        solvers=[SolverConfig(id="scs", solver="scs")],
    )
    assert _datasets_slug(config) == "netlib_lp"


def test_kkt_summary_returns_per_solver_residual_aggregates():
    frame = pd.DataFrame(
        [
            {
                "solver_id": "a",
                "status": "optimal",
                "kkt.primal_res_rel": 1e-8,
                "kkt.dual_res_rel": 2e-8,
                "kkt.comp_slack": 1e-9,
                "kkt.duality_gap_rel": 3e-9,
            },
            {
                "solver_id": "a",
                "status": "optimal",
                "kkt.primal_res_rel": 1e-6,
                "kkt.dual_res_rel": 2e-6,
                "kkt.comp_slack": 1e-7,
                "kkt.duality_gap_rel": 3e-7,
            },
            {
                "solver_id": "b",
                "status": "solver_error",
                "kkt.primal_res_rel": None,
                "kkt.dual_res_rel": None,
            },
        ]
    )
    summary = kkt_summary(frame)
    by_solver = summary.set_index("solver_id")
    # Solver a has two successful rows with KKT data.
    assert by_solver.loc["a", "success_count"] == 2
    assert by_solver.loc["a", "kkt_count"] == 2
    assert by_solver.loc["a", "primal_res_rel_max"] == pytest.approx(1e-6)
    assert by_solver.loc["a", "primal_res_rel_median"] == pytest.approx(
        (1e-8 + 1e-6) / 2
    )


def test_kkt_certificate_summary_handles_mix_of_valid_and_invalid():
    frame = pd.DataFrame(
        [
            {
                "solver_id": "a",
                "status": "primal_infeasible",
                "kkt.valid": True,
                "kkt.Aty_rel": 1e-9,
                "kkt.Px_rel": None,
            },
            {
                "solver_id": "a",
                "status": "primal_infeasible",
                "kkt.valid": False,
                "kkt.Aty_rel": 1e-3,
                "kkt.Px_rel": None,
            },
            {
                "solver_id": "b",
                "status": "primal_infeasible",
                "kkt.valid": None,  # no certificate
                "kkt.Aty_rel": None,
                "kkt.Px_rel": None,
            },
        ]
    )
    summary = kkt_certificate_summary(frame)
    by_solver = summary.set_index("solver_id")
    # Solver a: 2 infeasibility claims, both have a cert (notna), one valid.
    assert by_solver.loc["a", "infeasible_count"] == 2
    assert by_solver.loc["a", "cert_count"] == 2
    assert by_solver.loc["a", "cert_valid"] == 1
    assert by_solver.loc["a", "cert_invalid"] == 1
    # Solver b: 1 claim, no cert produced.
    assert by_solver.loc["b", "infeasible_count"] == 1
    assert by_solver.loc["b", "cert_count"] == 0
    assert by_solver.loc["b", "cert_valid"] == 0


def test_kkt_summary_empty_frame_returns_empty_table():
    summary = kkt_summary(pd.DataFrame())
    assert summary.empty
    assert "primal_res_rel_max" in summary.columns


def test_kkt_certificate_summary_empty_frame_returns_empty_table():
    summary = kkt_certificate_summary(pd.DataFrame())
    assert summary.empty
    assert "cert_valid" in summary.columns
