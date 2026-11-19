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

A report scores the same rows many times (every table, plot and slice), so
:func:`with_expected_outcomes` can attach each row's expected outcome once as
the helper column ``OUTCOME_COLUMN``; :func:`expected_outcomes` then reads it
instead of deriving it again. The column depends only on the row's dataset,
problem, source and recorded expected status, never on its status, so slices
of the frame and status rewrites keep it valid; rows added without it (the
``not_attempted`` padding of planned solves) are derived on demand.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from solver_benchmarks.core import status
from solver_benchmarks.datasets.dimacs import DIMACS_INFEASIBLE
from solver_benchmarks.datasets.sdplib import SDPLIB_INFEASIBLE

EXPECTED_COLUMN = "metadata.expected_status"
# Set on every row load_results verified (see analysis/verify.py).
VERDICT_COLUMN = "verify.verdict"
# Helper column holding each row's expected outcome (see with_expected_outcomes).
# "" stands for an unknown outcome, so a missing value means "not derived yet".
OUTCOME_COLUMN = "__expected_outcome"
_UNKNOWN = ""


def _inferred_outcome(dataset: object, problem: object, source: object) -> str | None:
    """Outcome of archived rows that predate ``metadata.expected_status``.

    This is the per-row definition; :func:`_compute_expected_outcomes`
    applies it to a whole frame at once.

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
    if ("/dimacs_data/" in path or dataset == "dimacs") and problem in DIMACS_INFEASIBLE:
        return status.EXPECT_INFEASIBLE
    return None


def expected_outcomes(results: pd.DataFrame) -> pd.Series:
    """Expected outcome of every row (None where unknown, i.e. assumed optimal)."""
    if results.empty:
        return pd.Series([], index=results.index, dtype=object)
    if OUTCOME_COLUMN not in results.columns:
        return _compute_expected_outcomes(results)
    attached = results[OUTCOME_COLUMN].to_numpy(dtype=object, na_value=None).copy()
    pending = pd.isna(attached)
    if pending.any():
        attached[pending] = _compute_expected_outcomes(results.loc[pending]).to_numpy(dtype=object)
    attached[attached == _UNKNOWN] = None
    return pd.Series(attached, index=results.index, dtype=object)


def with_expected_outcomes(results: pd.DataFrame) -> pd.DataFrame:
    """``results`` with each row's expected outcome attached as ``OUTCOME_COLUMN``.

    Returns ``results`` itself when the column is already there. The column
    is a scoring helper, not data: the tables built from such a frame select
    their own columns, and rows written out must not include it. Editing a
    row's dataset, problem, ``metadata.source`` or ``metadata.expected_status``
    after attaching leaves its attached outcome stale; attach again on a
    frame without the column instead.
    """
    if OUTCOME_COLUMN in results.columns:
        return results
    outcomes = expected_outcomes(results).to_numpy(dtype=object).copy()
    outcomes[pd.isna(outcomes)] = _UNKNOWN
    return results.assign(**{OUTCOME_COLUMN: outcomes})


def _compute_expected_outcomes(results: pd.DataFrame) -> pd.Series:
    """Derive the expected outcomes of ``results``, without ``OUTCOME_COLUMN``.

    The vectorized form of a recorded ``metadata.expected_status`` where it
    is valid and :func:`_inferred_outcome` everywhere else.
    """
    recorded = _column(results, EXPECTED_COLUMN).astype(object)
    known = recorded.isin(list(status.EXPECTED_OUTCOMES)).to_numpy(dtype=bool)
    out = pd.Series(np.where(known, recorded.to_numpy(), None), index=results.index, dtype=object)
    if known.all():
        return out
    rows = results.loc[~known]
    dataset = _column(rows, "dataset")
    problem = _column(rows, "problem")
    path = _string_values(_column(rows, "metadata.source")).str.replace("\\", "/", regex=False)

    def in_path(text: str) -> np.ndarray:
        return path.str.contains(text, regex=False).fillna(False).to_numpy(dtype=bool)

    def dataset_is(name: str) -> np.ndarray:
        return dataset.isin([name]).to_numpy(dtype=bool)

    def problem_in(names) -> np.ndarray:
        return problem.isin(list(names)).to_numpy(dtype=bool)

    infeasible = (
        in_path("/netlib_data/infeasible/")
        | in_path("/liu_pataki_data/")
        | dataset_is("liu_pataki")
        | ((in_path("/sdplib_data/") | dataset_is("sdplib")) & problem_in(SDPLIB_INFEASIBLE))
        | ((in_path("/dimacs_data/") | dataset_is("dimacs")) & problem_in(DIMACS_INFEASIBLE))
    )
    inferred = np.full(len(rows), None, dtype=object)
    inferred[infeasible] = status.EXPECT_INFEASIBLE
    out.iloc[np.flatnonzero(~known)] = inferred
    return out


def _string_values(values: pd.Series) -> pd.Series:
    """``values`` as a string Series, with every non-string value missing."""
    if values.dtype == object:
        is_string = values.map(lambda value: isinstance(value, str)).to_numpy(dtype=bool)
        values = values.where(is_string, None)
    return values.astype("string")


def success_mask(results: pd.DataFrame, success_statuses: Iterable[str] | None = None) -> pd.Series:
    """Boolean Series: did each row reach its problem's correct outcome?

    ``success_statuses`` (default ``status.SOLUTION_PRESENT``) applies to
    problems expected or assumed to have an optimal solution. Rows of
    problems known to be infeasible or unbounded succeed only with the
    matching certificate status, and only if the harness did not find that
    certificate invalid. A row the verification judged (it has a
    ``verify.verdict``, as every row ``load_results`` verifies does) carries
    that judgement in its status, made at the row's own certificate
    tolerance; any other row is held to the check recorded at solve time
    (``kkt.valid`` False is a failure).
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
        invalid = results["kkt.valid"].map(lambda v: v is False or v is np.False_)
        if VERDICT_COLUMN in results.columns:
            invalid &= results[VERDICT_COLUMN].isna()
        cert_ok &= ~invalid.astype(bool)
    return (optimal_ok | cert_ok).astype(bool)


def _column(results: pd.DataFrame, name: str) -> pd.Series:
    if name in results.columns:
        return results[name]
    return pd.Series([None] * len(results), index=results.index, dtype=object)
