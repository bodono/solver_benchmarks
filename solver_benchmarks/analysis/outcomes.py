"""Score result rows against each problem's known outcome.

A problem known to be infeasible or unbounded is solved only by an
infeasibility certificate of the right kind (``status.CERTIFICATE_SUCCESS``);
any claim of optimality on it is a failure. A problem known or assumed to
have an optimal solution is solved only by a status in the caller's success
set, so a certificate on it is a failure. Every success count, profile and
plot goes through :func:`success_mask`, so this is the single definition.

The expected outcome comes from ``metadata.expected_status`` (written by the
datasets that know it). Runs recorded before that field existed fall back to
what the dataset id and problem source already say (see
:func:`_inferred_outcome`).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from solver_benchmarks.core import status
from solver_benchmarks.datasets.sdplib import SDPLIB_INFEASIBLE

EXPECTED_COLUMN = "metadata.expected_status"


def _inferred_outcome(dataset: object, problem: object, source: object) -> str | None:
    """Outcome of archived rows that predate ``metadata.expected_status``.

    ``dataset`` is the config entry id, which a run may alias (e.g.
    ``netlib_infeasible``), so the data folder in the recorded source path is
    checked first and the dataset id is only a fallback.
    """
    path = source.replace("\\", "/") if isinstance(source, str) else ""
    if "/netlib_data/infeasible/" in path:
        return status.EXPECT_INFEASIBLE
    if "/liu_pataki_data/" in path or dataset == "liu_pataki":
        # Every instance is infeasible or weakly infeasible by construction.
        return status.EXPECT_INFEASIBLE
    if ("/sdplib_data/" in path or dataset == "sdplib") and problem in SDPLIB_INFEASIBLE:
        return status.EXPECT_INFEASIBLE
    return None


def expected_outcomes(results: pd.DataFrame) -> pd.Series:
    """Expected outcome of every row (None where unknown, i.e. assumed optimal)."""
    if results.empty:
        return pd.Series([], index=results.index, dtype=object)
    recorded = _column(results, EXPECTED_COLUMN)
    dataset = _column(results, "dataset")
    problem = _column(results, "problem")
    source = _column(results, "metadata.source")
    values = []
    for rec, ds, prob, src in zip(recorded, dataset, problem, source):
        if isinstance(rec, str) and rec in status.EXPECTED_OUTCOMES:
            values.append(rec)
        else:
            values.append(_inferred_outcome(ds, prob, src))
    return pd.Series(values, index=results.index, dtype=object)


def success_mask(results: pd.DataFrame, success_statuses: Iterable[str] | None = None) -> pd.Series:
    """Boolean Series: did each row reach its problem's correct outcome?

    ``success_statuses`` (default ``status.SOLUTION_PRESENT``) applies to
    problems expected or assumed to have an optimal solution. Rows of
    problems known to be infeasible or unbounded succeed only with the
    matching certificate status, and only if the harness did not find that
    certificate invalid (``kkt.valid`` False).
    """
    if results.empty:
        return pd.Series([], index=results.index, dtype=bool)
    ok_statuses = set(status.SOLUTION_PRESENT if success_statuses is None else success_statuses)
    statuses = results["status"]
    expected = expected_outcomes(results)
    certificate_expected = expected.isin(list(status.CERTIFICATE_SUCCESS))
    optimal_ok = statuses.isin(ok_statuses) & ~certificate_expected
    cert_ok = pd.Series(False, index=results.index)
    for outcome, accepted in status.CERTIFICATE_SUCCESS.items():
        cert_ok |= (expected == outcome) & statuses.isin(accepted)
    if "kkt.valid" in results.columns:
        cert_ok &= ~results["kkt.valid"].map(lambda v: v is False or v is np.False_)
    return (optimal_ok | cert_ok).astype(bool)


def _column(results: pd.DataFrame, name: str) -> pd.Series:
    if name in results.columns:
        return results[name]
    return pd.Series([None] * len(results), index=results.index, dtype=object)
