"""Problems with a known outcome are scored against it.

On a problem known to be infeasible or unbounded only the matching
certificate counts as solved, and a claimed optimum is a failure; on a
problem known or assumed to be solvable, a certificate is a failure.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from solver_benchmarks.analysis.outcomes import expected_outcomes, success_mask
from solver_benchmarks.analysis.profiles import performance_profile
from solver_benchmarks.analysis.tables import solver_metrics
from solver_benchmarks.core import status
from solver_benchmarks.datasets.dc_opf import DCOPFDataset
from solver_benchmarks.datasets.liu_pataki import LiuPatakiDataset
from solver_benchmarks.datasets.mps import NetlibDataset
from solver_benchmarks.datasets.sdplib import SDPLIBDataset


def _frame(rows):
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("expected", "row_status", "solved"),
    [
        # Known infeasible: either certificate, nothing else.
        (status.EXPECT_INFEASIBLE, status.PRIMAL_INFEASIBLE, True),
        (status.EXPECT_INFEASIBLE, status.DUAL_INFEASIBLE, True),
        (status.EXPECT_INFEASIBLE, status.OPTIMAL, False),
        (status.EXPECT_INFEASIBLE, status.OPTIMAL_INACCURATE, False),
        (status.EXPECT_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE, False),
        (status.EXPECT_INFEASIBLE, status.PRIMAL_OR_DUAL_INFEASIBLE, False),
        (status.EXPECT_INFEASIBLE, status.TIME_LIMIT, False),
        # Known unbounded: only a dual-infeasibility certificate.
        (status.EXPECT_UNBOUNDED, status.DUAL_INFEASIBLE, True),
        (status.EXPECT_UNBOUNDED, status.PRIMAL_INFEASIBLE, False),
        (status.EXPECT_UNBOUNDED, status.OPTIMAL, False),
        # Known solvable, and unknown (assumed solvable): optimal only.
        (status.EXPECT_OPTIMAL, status.OPTIMAL, True),
        (status.EXPECT_OPTIMAL, status.PRIMAL_INFEASIBLE, False),
        (status.EXPECT_OPTIMAL, status.DUAL_INFEASIBLE, False),
        (None, status.OPTIMAL, True),
        (None, status.PRIMAL_INFEASIBLE, False),
    ],
)
def test_success_follows_expected_outcome(expected, row_status, solved):
    frame = _frame(
        [
            {
                "dataset": "d",
                "problem": "p",
                "status": row_status,
                "metadata.expected_status": expected,
            }
        ]
    )
    assert bool(success_mask(frame).iloc[0]) is solved


def test_custom_success_statuses_apply_only_to_solvable_problems():
    frame = _frame(
        [
            {"dataset": "d", "problem": "a", "status": status.OPTIMAL_INACCURATE},
            {
                "dataset": "d",
                "problem": "b",
                "status": status.OPTIMAL_INACCURATE,
                "metadata.expected_status": status.EXPECT_INFEASIBLE,
            },
        ]
    )
    loose = {status.OPTIMAL, status.OPTIMAL_INACCURATE}
    assert success_mask(frame, loose).tolist() == [True, False]


def test_certificate_found_invalid_is_a_failure():
    frame = _frame(
        [
            {
                "dataset": "d",
                "problem": p,
                "status": status.PRIMAL_INFEASIBLE,
                "metadata.expected_status": status.EXPECT_INFEASIBLE,
                "kkt.valid": valid,
            }
            for p, valid in (("ok", True), ("bad", False), ("unchecked", None))
        ]
    )
    assert success_mask(frame).tolist() == [True, False, True]


def test_archived_rows_infer_outcome_from_dataset_and_source():
    frame = _frame(
        [
            {"dataset": "liu_pataki", "problem": "weak_clean_10_10_1", "metadata.source": "x.mat"},
            {
                "dataset": "netlib",
                "problem": "cplex1",
                "metadata.source": "/data/problem_classes/netlib_data/infeasible/cplex1.mps",
            },
            {
                "dataset": "netlib",
                "problem": "afiro",
                "metadata.source": "/data/problem_classes/netlib_data/feasible/afiro.mps",
            },
            {"dataset": "maros_meszaros", "problem": "qafiro", "metadata.source": "qafiro.mat"},
            {"dataset": "sdplib", "problem": "infp1", "metadata.source": "sdplib.tar"},
            {"dataset": "sdplib", "problem": "arch0", "metadata.source": "sdplib.tar"},
            # Aliased config entry ids: the source path still identifies the data.
            {
                "dataset": "netlib_infeasible",
                "problem": "bgdbg1",
                "metadata.source": "/v/problem_classes/netlib_data/infeasible/bgdbg1.mps.gz",
            },
            {
                "dataset": "sdplib0",
                "problem": "infd1",
                "metadata.source": "/v/problem_classes/sdplib_data/sdplib.tar",
            },
            {
                "dataset": "sdp_lp",
                "problem": "weak_clean_10_10_1",
                "metadata.source": "/v/problem_classes/liu_pataki_data/weak_clean_10_10_1.mat",
            },
        ]
    )
    assert expected_outcomes(frame).tolist() == [
        status.EXPECT_INFEASIBLE,
        status.EXPECT_INFEASIBLE,
        None,
        None,
        status.EXPECT_INFEASIBLE,
        None,
        status.EXPECT_INFEASIBLE,
        status.EXPECT_INFEASIBLE,
        status.EXPECT_INFEASIBLE,
    ]


def test_recorded_outcome_wins_over_inference():
    frame = _frame(
        [
            {
                "dataset": "netlib",
                "problem": "x",
                "metadata.source": "/netlib_data/infeasible/x.mps",
                "metadata.expected_status": status.EXPECT_OPTIMAL,
            }
        ]
    )
    assert expected_outcomes(frame).tolist() == [status.EXPECT_OPTIMAL]


def _mixed_results():
    """Solver A answers both problems correctly; solver B claims the infeasible one is optimal."""
    rows = []
    for solver, infeasible_status in (("A", status.PRIMAL_INFEASIBLE), ("B", status.OPTIMAL)):
        rows.append(
            {
                "dataset": "d",
                "problem": "feasible",
                "solver_id": solver,
                "status": status.OPTIMAL,
                "run_time_seconds": 1.0,
                "iterations": 10,
            }
        )
        rows.append(
            {
                "dataset": "d",
                "problem": "infeasible",
                "solver_id": solver,
                "status": infeasible_status,
                "run_time_seconds": 1.0,
                "iterations": 10,
                "metadata.expected_status": status.EXPECT_INFEASIBLE,
            }
        )
    return pd.DataFrame(rows)


def test_solver_metrics_count_certificates_and_reject_false_optima():
    metrics = solver_metrics(_mixed_results()).set_index("solver_id")
    success = [c for c in metrics.columns if "success" in c and "count" in c]
    assert success, metrics.columns
    assert metrics.loc["A", success[0]] == 2
    assert metrics.loc["B", success[0]] == 1


def test_performance_profile_treats_false_optimum_as_failure():
    profile = performance_profile(_mixed_results(), metric="run_time_seconds")
    # One column per solver: the fraction of problems within ratio tau.
    assert profile["A"].max() == pytest.approx(1.0)
    assert profile["B"].max() == pytest.approx(0.5)


def test_liu_pataki_records_expected_infeasible(tmp_path):
    folder = tmp_path / "liu_pataki_data"
    folder.mkdir()
    for name in ("infeas_clean_10_10_1", "weak_messy_20_10_3"):
        (folder / f"{name}.mat").write_bytes(b"")
    specs = LiuPatakiDataset(data_root=tmp_path).list_problems()
    assert {s.metadata["expected_status"] for s in specs} == {status.EXPECT_INFEASIBLE}
    assert len(specs) == 2


@pytest.mark.parametrize(
    ("subset", "expected"),
    [("infeasible", status.EXPECT_INFEASIBLE), ("feasible", status.EXPECT_OPTIMAL)],
)
def test_netlib_subsets_record_expected_outcome(tmp_path, subset, expected):
    folder = tmp_path / "netlib_data" / subset
    folder.mkdir(parents=True)
    (folder / "toy.mps").write_text("NAME toy\nENDATA\n")
    specs = NetlibDataset(data_root=tmp_path, subset=subset).list_problems()
    assert [s.metadata["expected_status"] for s in specs] == [expected]


def test_dc_opf_records_labels_as_expected_outcome(tmp_path):
    ds = DCOPFDataset(data_root=tmp_path)
    ds.data_dir.mkdir(parents=True)
    (ds.data_dir / "feasibility.json").write_text(
        json.dumps({"case5": "feasible", "case_bad": "infeasible"})
    )
    for name in ("case5", "case_bad", "case_unlabelled"):
        (ds.data_dir / f"{name}.mat").write_bytes(b"")
    got = {s.name: s.metadata.get("expected_status") for s in ds.list_problems()}
    assert got == {
        "case5": status.EXPECT_OPTIMAL,
        "case_bad": status.EXPECT_INFEASIBLE,
        "case_unlabelled": None,
    }


def test_sdplib_records_expected_outcome(tmp_path):
    ds = SDPLIBDataset(data_root=tmp_path, subset="all")
    ds.folder.mkdir(parents=True)
    for name in ("arch0", "infp1", "infd2"):
        (ds.folder / f"{name}.dat-s").write_text("")
    got = {s.name: s.metadata["expected_status"] for s in ds.list_problems()}
    assert got == {
        "arch0": status.EXPECT_OPTIMAL,
        "infp1": status.EXPECT_INFEASIBLE,
        "infd2": status.EXPECT_INFEASIBLE,
    }
