"""Independent KKT verification of result statuses.

Every solve records KKT residuals that the harness computed itself from the
returned point or certificate and the problem data (``analysis/kkt.py``).
Verification holds each row's claim to them: a claimed solution must have
relative residuals at or below a tolerance, a claimed infeasibility
certificate a scaled certificate residual at or below one, and (with
promotion) an inaccurate, iteration- or time-limited stop whose point passes
becomes ``optimal``. :func:`classify` is the one definition of that rule; the
loader (``load_results``, verified by default) and ``bench kkt-verify`` both
call it.

The tolerance of a row is, by default, ``VERIFY_SLACK`` times the tolerance
its configuration asked the solver for (``SolverAdapter.requested_tolerance``),
for its solutions and its infeasibility certificates alike (the solvers' own
infeasibility tolerances are in incomparable units; see
``RequestedTolerance``). A fixed tolerance for every row is the other rule.
The requested tolerance comes from the row (the worker records it since this
module exists), else the run's manifest, else nowhere: a row that needs one
and has none raises :class:`RequestedToleranceUnknown`, which names the
remedies, instead of being judged at a guess.

Verification depends only on a row's reported status (the status before any
earlier verification, ``metadata.kkt_verify.original_status``), its KKT
record, its time-limit markers and the policy, so re-verifying a verified run
under the same policy changes nothing.

pandas is imported only by the frame functions: ``derive`` imports this
module, and the MOSEK adapter imports ``derive`` in its workers.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from solver_benchmarks.analysis.kkt import certificate_check
from solver_benchmarks.core import status

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

# The factor between the tolerance a solver was asked for and the one its
# result is held to. Solvers measure their residuals in their own norms and
# scalings, most after equilibrating the problem, so a solution that honestly
# meets eps by the solver's measure can sit a few times above eps by the
# harness's; ten absorbs that and still rejects a claim off by more than a
# decade. It is the factor of the published 3.3.2 figures (bench kkt-verify
# --tol 10*eps) and of the 3.3.3 suite's verify.py.
VERIFY_SLACK = 10.0

KKT_FIELDS = ("primal_res_rel", "dual_res_rel", "duality_gap_rel")
# QP-form records also carry the multipliers on absent bounds (one-sided and
# free rows), which must be zero; the duality gap cannot see them. Records
# written before this residual existed do not have it and are checked on the
# other fields alone, so re-verifying an archived run does not demote them.
QP_FIELDS = ("dual_bound_res_rel",)
# Cone-form records also carry the distance of s and y to their cones; a point
# that satisfies Ax + s = b with s outside K is not feasible, so these are part
# of the check whenever the record has them (QP-form records have none). The
# distance of s is scaled like the primal equality residual and the distance
# of y by 1 + |y|, so the check depends neither on the units of the problem
# nor on the size of A and q, and projection roundoff in a large cone stays
# small.
CONE_FIELDS = ("primal_cone_res_rel", "dual_cone_res_rel")
# Records written before the relative distances existed only carry the
# absolute ones; they are used in that case.
_CONE_FALLBACK = {"primal_cone_res_rel": "primal_cone_res", "dual_cone_res_rel": "dual_cone_res"}
# Statuses that carry a returned point which may still pass the check.
PROMOTABLE = frozenset({status.OPTIMAL_INACCURATE, status.MAX_ITER_REACHED, status.TIME_LIMIT})
# Certificate statuses that count as a success, and what they become when
# their certificate fails the check (see kkt.certificate_check).
CERTIFICATE_DEMOTION = {
    status.PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE_INACCURATE,
    status.DUAL_INFEASIBLE: status.DUAL_INFEASIBLE_INACCURATE,
}
# With promote, the converse: inaccurate certificate statuses whose
# certificate is fully checked and passes become the accurate status.
CERTIFICATE_PROMOTION = {inaccurate: accurate for accurate, inaccurate in CERTIFICATE_DEMOTION.items()}
# Limit statuses whose record may hold a certificate instead of a point;
# such a row is promoted to the certificate's status under the same rule.
LIMIT_STATUSES = frozenset({status.MAX_ITER_REACHED, status.TIME_LIMIT})
# The statuses that claim a success: a solution or a certificate.
CLAIMS = frozenset(status.SOLUTION_PRESENT) | frozenset(CERTIFICATE_DEMOTION)
# Where the worker records the returned point's objective in the convention
# of objective_value (constant term and sense applied); a row promoted to
# optimal without an objective takes it from there.
POINT_OBJECTIVE = "primal_obj_reported"

# Values of the verify.verdict column (and of metadata.kkt_verify.verdict):
# - kept: the reported status stands after the check (a claim that passed, or
#   an inaccurate or limit stop that did not pass, or could not be checked).
# - demoted: a claim whose residuals (or certificate) fail the check.
# - promoted: an inaccurate or limit stop whose point (or certificate) passes.
# - over_time_limit: a stop returned after the run's time limit; never promoted.
# - missing_residuals: a claimed solution without complete residuals, or a
#   claimed certificate with none recorded (demoted unless missing residuals
#   are tolerated).
# - unverified_cones: a claimed solution that passes every residual the
#   harness can compute; its cone distances are unknown (complex PSD).
# - unverified_certificate: a claimed certificate that passes every residual
#   the harness can compute; its distance to a cone it cannot project is
#   unknown (a claim with no certificate recorded is missing_residuals).
# - not_checked: a status that claims nothing and cannot be promoted.
VERDICTS = (
    "kept",
    "demoted",
    "promoted",
    "over_time_limit",
    "missing_residuals",
    "unverified_cones",
    "unverified_certificate",
    "not_checked",
)
RULES = ("auto", "requested", "fixed")
# The verify.* columns load_results adds, in order.
VERIFY_COLUMNS = (
    "verify.verdict",
    "verify.rule",
    "verify.tol",
    "verify.requested_tol",
    "verify.requested_tol_source",
    "verify.worst",
    "verify.worst_field",
    "verify.reason",
    "verify.objective_from",
)
# The tolerance tag make_campaign appends to solver ids (clarabel_1e-4).
SOLVER_ID_TOL_TAG = re.compile(r"_(\d(?:\.\d+)?e-\d+)$")

REMEDIES = (
    "pass --requested-tol SOLVER_ID=TOL for each solver named (Python: "
    "VerifyPolicy(requested_tol={...})), or --tol-from-solver-id to read it from a tolerance tag "
    "such as _1e-4 at the end of the solver id, or --verify-tol T to hold every row to one "
    "tolerance, or --unverified for the solvers' own statuses (verify=False)"
)


class VerifyInputError(ValueError):
    """A run, policy or option verification cannot use."""


class RequestedToleranceUnknown(VerifyInputError):
    """Rows that need a requested tolerance whose solvers have none.

    ``unknown`` maps each such solver id to why its tolerance is unknown.
    The message lists them and the remedies (``REMEDIES``).
    """

    def __init__(self, unknown: Mapping[str, str]) -> None:
        self.unknown = dict(unknown)
        lines = "; ".join(f"{solver_id}: {reason}" for solver_id, reason in sorted(self.unknown.items()))
        super().__init__(
            "cannot KKT-verify the results: the tolerance requested of "
            f"{', '.join(sorted(self.unknown))} is unknown ({lines}). To go on, {REMEDIES}."
        )


@dataclass(frozen=True)
class VerifyPolicy:
    """How results are verified.

    ``rule``: ``"auto"`` (the default) uses the verification a ``bench
    kkt-verify`` copy recorded and otherwise verifies every row by the
    requested-tolerance rule; ``"requested"`` always re-verifies by that rule
    (``--reverify``); ``"fixed"`` holds every row, solutions and certificates
    alike, to ``tol`` (``--verify-tol``). ``slack`` is the factor of the
    requested-tolerance rule. ``promote`` turns inaccurate, iteration- and
    time-limited stops whose point or certificate passes into the accurate
    status. ``missing_is_failure`` demotes claimed solutions without complete
    residuals and claimed certificates without a recorded certificate.
    ``requested_tol`` maps solver ids to their requested tolerance, ahead of
    everything the run records; ``tol_from_solver_id`` reads it from a
    tolerance tag ending the solver id when nothing else gives it. Both
    belong to the requested-tolerance rule, and a fixed rule refuses them.
    """

    rule: str = "auto"
    tol: float | None = None
    slack: float = VERIFY_SLACK
    promote: bool = True
    missing_is_failure: bool = True
    requested_tol: Mapping[str, Any] = field(default_factory=dict)
    tol_from_solver_id: bool = False

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise VerifyInputError(f"verification rule must be one of {list(RULES)} (got {self.rule!r})")
        if self.rule == "fixed":
            if not _positive(self.tol):
                raise VerifyInputError(f"a fixed verification tolerance must be a positive number (got {self.tol!r})")
        elif self.tol is not None:
            raise VerifyInputError("a verification tolerance needs rule='fixed'")
        if not _positive(self.slack):
            raise VerifyInputError(f"the verification slack must be a positive number (got {self.slack!r})")
        overrides = {str(key): _override(key, value) for key, value in dict(self.requested_tol).items()}
        object.__setattr__(self, "requested_tol", overrides)
        if self.rule == "fixed" and (overrides or self.tol_from_solver_id):
            raise VerifyInputError(
                "a fixed verification tolerance holds every row to it; requested tolerances "
                "(--requested-tol, --tol-from-solver-id) would not be used"
            )
        if self.rule == "fixed" and self.slack != VERIFY_SLACK:
            raise VerifyInputError("a fixed verification tolerance holds every row to it; a slack would not be used")

    def describe(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "tol": self.tol,
            "slack": self.slack,
            "promote": self.promote,
            "missing_is_failure": self.missing_is_failure,
            "requested_tol": dict(self.requested_tol),
            "tol_from_solver_id": self.tol_from_solver_id,
        }


def _positive(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _override(key: Any, value: Any) -> float:
    if not _positive(value):
        raise VerifyInputError(f"requested tolerance of {key!r} must be a positive number (got {value!r})")
    return float(value)


def parse_requested_tol(items: Iterable[str]) -> dict[str, float]:
    """``{"scs_1e-4": 1e-4}`` from ``["scs_1e-4=1e-4"]`` (``--requested-tol``)."""
    parsed: dict[str, float] = {}
    for item in items:
        solver_id, sep, value = str(item).rpartition("=")
        if not sep or not solver_id:
            raise VerifyInputError(f"--requested-tol takes SOLVER_ID=TOL (got {item!r})")
        try:
            number = float(value)
        except ValueError as exc:
            raise VerifyInputError(f"--requested-tol takes SOLVER_ID=TOL (got {item!r})") from exc
        parsed[solver_id] = _override(solver_id, number)
    return parsed


# --- the check ----------------------------------------------------------------


@dataclass
class ResidualCheck:
    """What a record's KKT residuals say for a set of wanted fields."""

    worst: float | None = None
    worst_field: str | None = None
    checked: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def check_residuals(record: dict[str, Any], fields: tuple[str, ...] = KKT_FIELDS) -> ResidualCheck:
    """Check the listed KKT residuals of a record, plus the form's extras.

    QP-form records also need the multipliers on absent bounds to vanish
    (``QP_FIELDS``, when the record has them), and cone-form records the
    cone distances (``CONE_FIELDS``, falling back to the absolute distances
    of older records), whatever ``fields`` lists. ``checked`` names every
    field that was looked at, ``missing`` those absent or not finite, and
    ``worst``/``worst_field`` the largest finite one.
    """
    kkt = record.get("kkt") or {}
    wanted = list(fields)
    if kkt.get("form") == "qp":
        wanted += [f for f in QP_FIELDS if f not in wanted and f in kkt]
    if kkt.get("form") == "cone" or any(f in kkt or _CONE_FALLBACK[f] in kkt for f in CONE_FIELDS):
        wanted += [f for f in CONE_FIELDS if f not in wanted]
    result = ResidualCheck(checked=wanted)
    for name in wanted:
        value = kkt.get(name)
        if value is None and name in _CONE_FALLBACK:
            value = kkt.get(_CONE_FALLBACK[name])
        try:
            number = float(value) if value is not None else None
        except (TypeError, ValueError):
            number = None
        if number is None or not math.isfinite(number):
            result.missing.append(name)
            continue
        if result.worst is None or number > result.worst:
            result.worst, result.worst_field = number, name
    return result


def residual_summary(
    record: dict[str, Any], fields: tuple[str, ...] = KKT_FIELDS
) -> tuple[float | None, bool]:
    """Return ``(worst, complete)`` for the listed KKT residuals of a record.

    ``worst`` is the largest finite residual among the wanted fields (None if
    there is none) and ``complete`` says whether every wanted field was
    present and finite (see :func:`check_residuals` for which fields are
    wanted). The two are reported separately so that a residual known to
    fail is never hidden by another one being missing.
    """
    check = check_residuals(record, fields)
    return check.worst, check.complete


def worst_relative_residual(
    record: dict[str, Any], fields: tuple[str, ...] = KKT_FIELDS
) -> float | None:
    """Largest of the listed KKT residuals, or None if any is missing or not finite.

    A missing or non-finite component means the point could not be fully
    checked, and an unchecked point must never count as verified: the caller
    treats None as a failed check.
    """
    worst, complete = residual_summary(record, fields)
    return worst if complete else None


def returned_after_time_limit(row: dict[str, Any]) -> bool:
    """Whether the runner marked the row as returned after the run's time limit.

    ``info.exceeded_time_limit``, or, in runs recorded before that marker, a
    conclusion recorded as time_limit with ``info.status_before_timeout``.
    """
    info = row.get("info")
    if not isinstance(info, dict):
        return False
    return bool(info.get("exceeded_time_limit")) or info.get("status_before_timeout") is not None


@dataclass
class Verdict:
    """What :func:`classify` decided for one row."""

    verdict: str
    # None only for a row that records no status at all (not_checked).
    status: str | None
    worst: float | None = None
    worst_field: str | None = None
    fields: tuple[str, ...] = ()
    reason: str | None = None
    # The objective a promotion gives a row without one, and where it came from.
    objective: float | None = None
    objective_from: str | None = None


# The row's tolerance, asked for only when a residual is compared with it.
Tolerance = Callable[[], float]


def classify(
    view: dict[str, Any],
    *,
    tolerance: Tolerance,
    promote: bool = True,
    missing_is_failure: bool = True,
    fields: tuple[str, ...] = KKT_FIELDS,
) -> Verdict:
    """Decide a row's verified status from its reported one.

    ``view`` is the row as ``{"status": <reported status>, "kkt": {...},
    "info": {...}, "objective_value": ...}``. ``tolerance()`` gives the row's
    tolerance, for solutions and certificates alike; it is asked only when a
    residual is compared with it, so a row without residuals never needs one.

    A claimed solution (``optimal``) keeps its status only if the worst of
    ``fields`` (relative residuals computed by the harness from the returned
    point, independently of the solver's own report; cone- and QP-form records
    always add their cone distances and bound multipliers, see
    :func:`check_residuals`) is at or below the solution tolerance; otherwise
    it becomes ``optimal_inaccurate``. Claimed optima with residuals missing
    are demoted too unless ``missing_is_failure`` is False. A row whose only
    unknown residuals are the distances to cones the harness cannot project
    (``kkt.unsupported_cones``) is kept when the rest pass
    (``unverified_cones``).

    Infeasibility certificates (``primal_infeasible``, ``dual_infeasible``)
    are held to the same tolerance: their largest residual once scaled so
    that the certificate's objective term is −1 (``kkt.cert_res``, rebuilt
    for older records) must be at or below it, or the row becomes the
    matching ``*_inaccurate`` status. A claim whose adapter recorded no
    certificate (cuOpt never does, its API exposing none; the others not
    when the solver produces none, e.g. HiGHS, Gurobi and CPLEX on an
    unbounded QP) cannot be verified and is demoted like a solution without
    residuals (``missing_residuals``), unless ``missing_is_failure`` is
    False. A certificate in which nothing fails but the distance to a cone
    the harness cannot project is unknown is kept whatever
    ``missing_is_failure`` says (``unverified_certificate``),
    as a solution in that case is.

    With ``promote``, the converse also applies: a solve reported as
    inaccurate, iteration-limited or time-limited whose returned point passes
    the check becomes ``optimal`` (its recorded time is what it took to
    produce that point). If it has no ``objective_value`` (adapters report
    one only for a solution), it takes the point's objective that the worker
    recorded as ``kkt.primal_obj_reported``. Likewise a
    ``primal_infeasible_inaccurate`` / ``dual_infeasible_inaccurate`` row
    whose certificate is fully checked and passes becomes accurate, and so
    does an iteration- or time-limited row whose record holds such a
    certificate instead of a point (OSQP records its inaccurate certificates
    so); a missing or partly checked certificate is never promoted. A row
    returned after the run's time limit is never promoted
    (``over_time_limit``; see :func:`returned_after_time_limit`).
    """
    reported = view.get("status")
    kkt = view.get("kkt") or {}
    if promote and (reported in PROMOTABLE or reported in CERTIFICATE_PROMOTION):
        return _classify_promotable(view, kkt, str(reported), tolerance, fields)
    if reported in CERTIFICATE_DEMOTION:
        return _classify_certificate(kkt, str(reported), tolerance, missing_is_failure)
    if reported in status.SOLUTION_PRESENT:
        return _classify_solution(view, kkt, str(reported), tolerance, missing_is_failure, fields)
    return Verdict("not_checked", str(reported) if reported is not None else reported)


def _classify_promotable(
    view: dict[str, Any], kkt: dict[str, Any], reported: str, tolerance: Tolerance, fields: tuple[str, ...]
) -> Verdict:
    if returned_after_time_limit(view):
        return Verdict("over_time_limit", reported, reason="returned after the time limit; not promoted")
    target = CERTIFICATE_PROMOTION.get(reported)
    if reported in LIMIT_STATUSES and kkt.get("certificate") in CERTIFICATE_DEMOTION:
        # A limit stop that returned a certificate (OSQP records its
        # inaccurate certificates so) is judged as one.
        target = kkt["certificate"]
    if target is not None:
        cert, complete = certificate_check(kkt)
        base: dict[str, Any] = {
            "fields": ("cert_res",), "worst": cert, "worst_field": None if cert is None else "cert_res",
        }
        # A missing or partly checked certificate is never promoted, nor one
        # of the other kind.
        if cert is None:
            return Verdict("kept", reported, reason="not promoted: no certificate recorded", **base)
        if not complete:
            return Verdict("kept", reported, reason="not promoted: certificate not fully checked", **base)
        if kkt.get("certificate") != target:
            return Verdict(
                "kept", reported, reason=f"not promoted: the record holds a {kkt.get('certificate')} certificate",
                **base,
            )
        cert_tol = tolerance()
        if cert <= cert_tol:
            return Verdict("promoted", target, reason=f"certificate residual {cert:.3e} <= {cert_tol:.1e}", **base)
        return Verdict(
            "kept", reported, reason=f"not promoted: certificate residual {cert:.3e} > {cert_tol:.1e}", **base
        )
    check = check_residuals(view, fields)
    base = {"fields": tuple(check.checked), "worst": check.worst, "worst_field": check.worst_field}
    if check.worst is None:
        return Verdict("kept", reported, reason="not promoted: no KKT residuals recorded", **base)
    if not check.complete:
        return Verdict("kept", reported, reason="not promoted: incomplete KKT residuals", **base)
    tol = tolerance()
    if check.worst <= tol:
        objective = objective_from = None
        point_objective = kkt.get(POINT_OBJECTIVE)
        if view.get("objective_value") is None and point_objective is not None:
            # Only a solution has an objective_value, so a limit stop has
            # none; the point it returned is now one.
            objective, objective_from = float(point_objective), f"kkt.{POINT_OBJECTIVE}"
        return Verdict(
            "promoted",
            status.OPTIMAL,
            reason=f"worst relative residual {check.worst:.3e} ({check.worst_field}) <= {tol:.1e}",
            objective=objective,
            objective_from=objective_from,
            **base,
        )
    return Verdict(
        "kept",
        reported,
        reason=f"not promoted: worst relative residual {check.worst:.3e} ({check.worst_field}) > {tol:.1e}",
        **base,
    )


def _classify_certificate(
    kkt: dict[str, Any], reported: str, tolerance: Tolerance, missing_is_failure: bool
) -> Verdict:
    cert, complete = certificate_check(kkt)
    base: dict[str, Any] = {
        "fields": ("cert_res",), "worst": cert, "worst_field": None if cert is None else "cert_res",
    }
    if cert is None:
        # The adapter recorded no certificate (cuOpt never does): the claim
        # cannot be verified, so it is not a verified success.
        new_status = CERTIFICATE_DEMOTION[reported] if missing_is_failure else reported
        return Verdict("missing_residuals", new_status, reason="no certificate recorded", **base)
    if math.isinf(cert):
        return Verdict(
            "demoted",
            CERTIFICATE_DEMOTION[reported],
            reason="certificate residual inf (its objective term is not negative)",
            **base,
        )
    cert_tol = tolerance()
    if cert > cert_tol:
        reason = f"certificate residual {cert:.3e} > {cert_tol:.1e}"
        if not complete:
            reason += " (some residuals unknown)"
        return Verdict("demoted", CERTIFICATE_DEMOTION[reported], reason=reason, **base)
    if not complete:
        # Nothing fails, but the distance to a cone the harness cannot project
        # is unknown; kept, like an optimal row with such cones.
        cones = kkt.get("unsupported_cones")
        unknown = f"unsupported cones {sorted(cones)}" if cones else "some residuals unknown"
        return Verdict(
            "unverified_certificate",
            reported,
            reason=f"certificate not fully checked ({unknown}); known residuals <= {cert_tol:.1e}",
            **base,
        )
    return Verdict("kept", reported, reason=f"certificate residual {cert:.3e} <= {cert_tol:.1e}", **base)


def _classify_solution(
    view: dict[str, Any],
    kkt: dict[str, Any],
    reported: str,
    tolerance: Tolerance,
    missing_is_failure: bool,
    fields: tuple[str, ...],
) -> Verdict:
    check = check_residuals(view, fields)
    worst = check.worst
    base: dict[str, Any] = {"fields": tuple(check.checked), "worst": worst, "worst_field": check.worst_field}
    if worst is not None:
        tol = tolerance()
        if worst > tol:
            # A residual that demonstrably fails demotes the row even when
            # other residuals are missing and missing ones are tolerated.
            reason = f"worst relative residual {worst:.3e} ({check.worst_field}) > {tol:.1e}"
            if not check.complete:
                reason += " (some residuals missing)"
            return Verdict("demoted", status.OPTIMAL_INACCURATE, reason=reason, **base)
        if check.complete:
            return Verdict(
                "kept", reported,
                reason=f"worst relative residual {worst:.3e} ({check.worst_field}) <= {tol:.1e}", **base,
            )
        unsupported = kkt.get("unsupported_cones")
        if unsupported and set(check.missing) <= set(CONE_FIELDS):
            # The point passes every residual the harness could compute; the
            # cone distances are unknown, not failed.
            return Verdict(
                "unverified_cones",
                reported,
                reason=f"cone distances not checked for unsupported cones {sorted(unsupported)}; "
                f"other residuals <= {tol:.1e}",
                **base,
            )
    reason = "no KKT residuals recorded" if worst is None else "incomplete KKT residuals"
    new_status = status.OPTIMAL_INACCURATE if missing_is_failure else reported
    return Verdict("missing_residuals", new_status, reason=reason, **base)


# --- requested tolerances -----------------------------------------------------


@dataclass(frozen=True)
class RowTolerance:
    """The tolerance a row is held to, and where it came from."""

    tol: float
    rule: str
    requested: float | None = None
    source: str | None = None


@dataclass(frozen=True)
class RowInfo:
    """What resolving a row's requested tolerance looks at."""

    solver_id: str
    run_id: str | None = None
    solver: str | None = None
    problem_kind: str | None = None
    is_lp: bool | None = None
    versions: tuple[tuple[str, str], ...] = ()
    recorded_optimality: float | None = None
    recorded_unknown: str | None = None


def row_info_from_record(record: Mapping[str, Any]) -> RowInfo:
    """:class:`RowInfo` of a results.jsonl record."""
    metadata = _mapping(record.get("metadata"))
    runtime = _mapping(metadata.get("runtime"))
    versions = _mapping(runtime.get("solver_package_versions"))
    requested = _mapping(metadata.get("requested_tolerance"))
    return RowInfo(
        solver_id=str(record.get("solver_id")),
        run_id=_text(record.get("run_id")),
        solver=_text(record.get("solver")),
        problem_kind=_text(record.get("problem_kind")),
        is_lp=_is_lp(metadata.get("nnz_p")),
        versions=_versions_key(versions),
        recorded_optimality=_number(requested.get("optimality")),
        recorded_unknown=_text(requested.get("unknown")),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str | None:
    value = _clean(value)
    return None if value is None else str(value)


def _number(value: Any) -> float | None:
    value = _clean(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _is_lp(nnz_p: Any) -> bool | None:
    value = _clean(nnz_p)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return None


def _versions_key(versions: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(versions, Mapping):
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in versions.items() if _clean(value) is not None))


SolverSettings = dict[str, list[tuple[str, dict]]]


def manifest_solver_settings(manifest: Mapping[str, Any] | None) -> tuple[SolverSettings, dict[str, SolverSettings]]:
    """Every ``(solver, settings)`` a run's manifests give each solver id.

    Walks the manifest's own config and, for derived runs, the source
    manifests a merge kept (``derived.source_manifests``) and the steps a
    verification wraps (``derived.source_derived``), recursively. Returns
    the entries of every manifest by solver id, identical entries listed
    once, and the same per ``run_id`` of the manifest that lists them: the
    rows of a merge keep the run id of the shard that wrote them, so their
    settings are that shard's even when shards gave one solver id different
    settings.
    """
    found: SolverSettings = {}
    by_run: dict[str, SolverSettings] = {}
    seen: set[tuple[str, str, str, str]] = set()

    def add(target: SolverSettings, key: tuple[str, str, str, str], solver_id: str, entry: tuple[str, dict]) -> None:
        if key not in seen:
            seen.add(key)
            target.setdefault(solver_id, []).append(entry)

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        config = _mapping(node.get("config"))
        run_id = str(node.get("run_id")) if node.get("run_id") is not None else None
        for entry in config.get("solvers") or []:
            if not isinstance(entry, Mapping):
                continue
            solver_id = str(entry.get("id"))
            settings = _mapping(entry.get("settings"))
            text = json.dumps(settings, sort_keys=True, default=str)
            item = (str(entry.get("solver")), dict(settings))
            add(found, ("", solver_id, item[0], text), solver_id, item)
            if run_id is not None:
                add(by_run.setdefault(run_id, {}), (run_id, solver_id, item[0], text), solver_id, item)
        derived = node.get("derived")
        while isinstance(derived, Mapping) and derived:
            for source in derived.get("source_manifests") or []:
                visit(source)
            derived = derived.get("source_derived")

    visit(manifest)
    return found, by_run


def requested_tolerance(
    solver: str, settings: Mapping[str, Any] | None, *, problem_kind: str | None = None,
    is_lp: bool | None = None, versions: Mapping[str, Any] | None = None,
):
    """``SolverAdapter.requested_tolerance`` of the registered solver ``solver``.

    Raises ``ToleranceUnknown`` (a ValueError) naming why it cannot be told:
    an unregistered solver, or settings the adapter cannot interpret.
    """
    from solver_benchmarks.solvers import get_solver
    from solver_benchmarks.solvers.base import SolverConfigError, ToleranceUnknown

    try:
        adapter = get_solver(str(solver))
    except KeyError as exc:
        raise ToleranceUnknown(f"solver {solver!r} is not registered") from exc
    try:
        return adapter.requested_tolerance(
            settings or {}, problem_kind=problem_kind, is_lp=is_lp, versions=versions
        )
    except ToleranceUnknown:
        raise
    except (SolverConfigError, ValueError, TypeError, KeyError) as exc:
        raise ToleranceUnknown(f"its settings cannot be read ({exc})") from exc


class ToleranceResolver:
    """Each row's tolerances under a policy (see :meth:`resolve`), cached."""

    def __init__(self, manifest: Mapping[str, Any] | None, policy: VerifyPolicy) -> None:
        self.policy = policy
        self.has_manifest = bool(manifest)
        self.settings, self.settings_by_run = manifest_solver_settings(manifest)
        self._cache: dict[RowInfo, RowTolerance | str] = {}

    def resolve(self, row: RowInfo) -> RowTolerance | str:
        """The row's :class:`RowTolerance`, or why it has none (a string).

        Under the fixed rule, the policy's tolerance. Otherwise ``slack``
        times the first requested tolerance found: the policy's override for
        the solver id, the row's record, the settings of the run's manifest
        (in a merge, of the input whose ``run_id`` the row carries, else of
        every input, which must then agree), and with ``tol_from_solver_id``
        the solver id's tolerance tag.
        """
        cached = self._cache.get(row)
        if cached is None:
            cached = self._cache[row] = self._resolve(row)
        return cached

    def _resolve(self, row: RowInfo) -> RowTolerance | str:
        policy = self.policy
        if policy.rule == "fixed":
            assert policy.tol is not None
            return RowTolerance(tol=float(policy.tol), rule="fixed", source="fixed")
        reasons: list[str] = []
        found = self._requested(row, reasons)
        if found is None:
            return "; ".join(reasons)
        requested, source = found
        return RowTolerance(tol=policy.slack * requested, rule="requested", requested=requested, source=source)

    def _requested(self, row: RowInfo, reasons: list[str]) -> tuple[float, str] | None:
        override = self.policy.requested_tol.get(row.solver_id)
        if override is not None:
            return override, "override"
        if row.recorded_optimality is not None:
            return row.recorded_optimality, "row"
        if row.recorded_unknown:
            reasons.append(f"the worker could not tell it ({row.recorded_unknown})")
        entries = (self.settings_by_run.get(row.run_id or "") or {}).get(row.solver_id) or self.settings.get(
            row.solver_id
        )
        if not entries:
            reasons.append(
                "the run's manifest lists no settings for it" if self.has_manifest else "the run has no manifest.json"
            )
        else:
            results: set[float] = set()
            for solver, settings in entries:
                try:
                    tolerance = requested_tolerance(
                        solver, settings, problem_kind=row.problem_kind, is_lp=row.is_lp,
                        versions=dict(row.versions),
                    )
                except ValueError as exc:
                    reasons.append(f"its {solver} settings {settings!r}: {exc}")
                    results = set()
                    break
                results.add(tolerance.optimality)
            if len(results) == 1:
                return results.pop(), "manifest"
            if len(results) > 1:
                reasons.append(
                    "the run's manifests give it different settings, requesting "
                    + " and ".join(f"{value:g}" for value in sorted(results))
                )
        if self.policy.tol_from_solver_id:
            match = SOLVER_ID_TOL_TAG.search(row.solver_id)
            if match:
                return float(match.group(1)), "solver_id"
            reasons.append("its id ends in no tolerance tag")
        return None


# --- frames ---------------------------------------------------------------------

# Columns of a results frame that verification reads besides kkt.*.
_INFO_COLUMNS = {"exceeded_time_limit": "info.exceeded_time_limit",
                 "status_before_timeout": "info.status_before_timeout"}
_ORIGINAL_STATUS = "metadata.kkt_verify.original_status"
_OBJECTIVE_FROM = "metadata.kkt_verify.objective_from"
# What verify_frame keeps of a row's report in a frame it verified.
_STATUS_REPORTED = "status_reported"
_VERIFIED_OBJECTIVE_FROM = "verify.objective_from"
# Cone distances, present in a record whenever it is cone-form or names
# cones it cannot project (their NaN is then stored as null).
_CONE_DISTANCES = frozenset({*CONE_FIELDS, *_CONE_FALLBACK.values()})


class _Missing:
    """A value absent from a record (NaN in the frame, null in the jsonl)."""


_ABSENT = _Missing()


def _clean(value: Any) -> Any:
    """``value`` as JSON would hold it: None for NaN, Python scalars and lists."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    module = type(value).__module__
    if module == "numpy":
        if hasattr(value, "tolist"):
            value = value.tolist()
            if isinstance(value, float) and math.isnan(value):
                return None
            if isinstance(value, list):
                return [_clean(item) for item in value]
            return value
    if module.startswith("pandas") and type(value).__name__ in ("NAType", "NaTType"):
        return None
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _decoded(value: Any) -> Any:
    """A value parquet normalization stored as text (a column that mixed
    types, such as lists among nulls), decoded back."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text in ("True", "False"):
        return text == "True"
    if text[:1] in "[{" or text in ("true", "false", "null"):
        try:
            return json.loads(text)
        except ValueError:
            return value
    return value


def reported_statuses(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Each row's reported status and objective: before any earlier verification.

    A ``bench kkt-verify`` copy keeps the status a row was recorded with as
    ``metadata.kkt_verify.original_status``, and a frame already verified
    (by ``verify_frame``) as ``status_reported``; a promotion that gave the
    row an objective says so in ``metadata.kkt_verify.objective_from`` or
    ``verify.objective_from``, and that objective is dropped. So verifying a
    verified frame again, under any policy, starts from what the solvers
    reported.
    """
    import pandas as pd

    statuses = df["status"] if "status" in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)
    for column in (_STATUS_REPORTED, _ORIGINAL_STATUS):
        if column in df.columns:
            earlier = df[column]
            statuses = earlier.where(earlier.notna(), statuses)
    objective = df["objective_value"] if "objective_value" in df.columns else pd.Series(
        [None] * len(df), index=df.index, dtype=float
    )
    for column in (_OBJECTIVE_FROM, _VERIFIED_OBJECTIVE_FROM):
        if column in df.columns:
            objective = objective.where(df[column].isna(), float("nan"))
    return statuses.astype(object), objective


class _FrameRows:
    """Row views and :class:`RowInfo` of a results frame, built on demand."""

    def __init__(self, df: pd.DataFrame, statuses: pd.Series, objective: pd.Series) -> None:
        self.kkt_columns = [(name[4:], df[name].to_numpy(dtype=object)) for name in df.columns if name.startswith("kkt.")]
        self.info = {key: _column_values(df, name) for key, name in _INFO_COLUMNS.items()}
        self.statuses = statuses.to_numpy(dtype=object)
        self.objective = objective.to_numpy(dtype=object)
        self.solver_id = _column_values(df, "solver_id")
        self.run_id = _column_values(df, "run_id")
        self.solver = _column_values(df, "solver")
        self.problem_kind = _column_values(df, "problem_kind")
        self.nnz_p = _column_values(df, "metadata.nnz_p")
        self.recorded = {
            key: _column_values(df, f"metadata.requested_tolerance.{key}")
            for key in ("optimality", "unknown")
        }
        prefix = "metadata.runtime.solver_package_versions."
        self.versions = [(name[len(prefix):], df[name].to_numpy(dtype=object))
                         for name in df.columns if name.startswith(prefix)]

    def view(self, i: int) -> dict[str, Any]:
        kkt: dict[str, Any] = {}
        raw = {name: _decoded(_clean(values[i])) for name, values in self.kkt_columns}
        form = raw.get("form")
        cone_present = form == "cone" or raw.get("unsupported_cones") is not None
        for name, value in raw.items():
            if value is None:
                # Absent in the record, unless the record has the field with a
                # null: cert_res is written with cert_tol, and the cone
                # distances with a cone-form record.
                if name == "cert_res" and raw.get("cert_tol") is not None:
                    pass
                elif name in _CONE_DISTANCES and cone_present:
                    pass
                else:
                    continue
            _set_nested(kkt, name, value)
        info = {key: _decoded(_clean(values[i])) for key, values in self.info.items()}
        objective = _clean(self.objective[i])
        return {
            "status": self.statuses[i],
            "kkt": kkt,
            "info": {key: value for key, value in info.items() if value is not None},
            "objective_value": objective,
        }

    def row_info(self, i: int) -> RowInfo:
        versions = {name: _clean(values[i]) for name, values in self.versions}
        return RowInfo(
            solver_id=str(_clean(self.solver_id[i])),
            run_id=_text(self.run_id[i]),
            solver=_text(self.solver[i]),
            problem_kind=_text(self.problem_kind[i]),
            is_lp=_is_lp(self.nnz_p[i]),
            versions=_versions_key(versions),
            recorded_optimality=_number(self.recorded["optimality"][i]),
            recorded_unknown=_text(self.recorded["unknown"][i]),
        )


def _column_values(df: pd.DataFrame, name: str):
    import numpy as np

    if name in df.columns:
        return df[name].to_numpy(dtype=object)
    return np.full(len(df), None, dtype=object)


def _set_nested(target: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        nested = target.get(part)
        if not isinstance(nested, dict):
            nested = target[part] = {}
        target = nested
    target[parts[-1]] = value


def unverified_frame(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with the statuses and objectives its solvers reported.

    Undoes an earlier verification, a ``bench kkt-verify`` copy's or that of
    a frame ``verify_frame`` returned (see :func:`reported_statuses`), and
    adds ``status_reported``; no ``verify.*`` columns.
    """
    out = df.copy()
    if out.empty:
        out.attrs["verification"] = _attrs("unverified")
        return out
    # A run never verified keeps its columns exactly as read.
    earlier = (_ORIGINAL_STATUS, _OBJECTIVE_FROM, _STATUS_REPORTED, _VERIFIED_OBJECTIVE_FROM)
    if any(column in out.columns for column in earlier):
        statuses, objective = reported_statuses(out)
        out["status"] = _like(statuses.to_numpy(dtype=object), out["status"])
        if "objective_value" in out.columns:
            out["objective_value"] = objective
        out = out.drop(columns=[name for name in VERIFY_COLUMNS if name in out.columns])
    if "status" in out.columns:
        out["status_reported"] = out["status"].copy()
    out.attrs["verification"] = _attrs("unverified")
    return out


def _like(values, template: pd.Series) -> pd.Series:
    """``values`` as a Series with ``template``'s index and, where it can
    hold them, dtype (a string column stays one)."""
    import pandas as pd

    series = pd.Series(values, index=template.index, dtype=object)
    try:
        return series.astype(template.dtype)
    except (TypeError, ValueError):
        return series


def verify_frame(
    df: pd.DataFrame, manifest: Mapping[str, Any] | None, policy: VerifyPolicy | None = None
) -> pd.DataFrame:
    """``df`` with verified statuses (see the module docstring).

    ``status`` becomes the verified status and ``status_reported`` holds the
    reported one; promoted rows without an objective get the point's; the
    ``verify.*`` columns (``VERIFY_COLUMNS``) say what was decided and why,
    and ``df.attrs["verification"]`` sums it up. With ``rule="auto"`` a run
    written by ``bench kkt-verify`` keeps the verification it recorded.
    A frame this function already verified is verified again from its
    reported statuses (``tools/make_figures.py`` re-checks a series it shows
    in another tolerance's figure that way). Raises
    :class:`RequestedToleranceUnknown` when a row needs a requested
    tolerance that nothing gives.
    """
    policy = policy or VerifyPolicy()
    manifest = manifest or {}
    derived = _mapping(manifest.get("derived"))
    if policy.rule == "auto" and derived.get("kind") == "kkt_verify":
        return _recorded_frame(df, derived, policy)
    return _computed_frame(df, manifest, policy)


def _attrs(mode: str, **details: Any) -> dict[str, Any]:
    return {"mode": mode, "warnings": [], **details}


def _computed_frame(df: pd.DataFrame, manifest: Mapping[str, Any], policy: VerifyPolicy) -> pd.DataFrame:
    import numpy as np

    rule = "fixed" if policy.rule == "fixed" else "requested"
    out = df.copy()
    details = {"rule": rule, "slack": policy.slack, "tol": policy.tol, "promote": policy.promote,
               "missing_is_failure": policy.missing_is_failure, "policy": policy.describe()}
    if out.empty:
        out.attrs["verification"] = _attrs("computed", **details)
        return out
    statuses, objective = reported_statuses(out)
    n = len(out)
    columns: dict[str, np.ndarray] = {name: np.full(n, None, dtype=object) for name in VERIFY_COLUMNS}
    new_status = statuses.to_numpy(dtype=object).copy()
    new_objective = objective.to_numpy(dtype=object).copy()
    columns["verify.verdict"][:] = "not_checked"
    columns["verify.rule"][:] = rule
    checked = set(CLAIMS)
    if policy.promote:
        checked |= PROMOTABLE | set(CERTIFICATE_PROMOTION)
    positions = np.flatnonzero(statuses.isin(list(checked)).to_numpy(dtype=bool))
    rows = _FrameRows(out, statuses, objective)
    resolver = ToleranceResolver(manifest, policy)
    unknown: dict[str, str] = {}

    class _Unresolved(Exception):
        pass

    for position in positions:
        i = int(position)
        info = rows.row_info(i)
        resolved = resolver.resolve(info)

        def tolerance(resolved=resolved) -> float:
            if isinstance(resolved, str):
                raise _Unresolved
            return resolved.tol

        try:
            verdict = classify(rows.view(i), tolerance=tolerance, promote=policy.promote,
                               missing_is_failure=policy.missing_is_failure)
        except _Unresolved:
            unknown.setdefault(info.solver_id, str(resolved))
            continue
        new_status[i] = verdict.status
        if verdict.objective is not None:
            new_objective[i] = verdict.objective
        _store(columns, i, verdict, resolved if isinstance(resolved, RowTolerance) else None)
    if unknown:
        raise RequestedToleranceUnknown(unknown)
    warnings = _earlier_verification_warnings(out, policy)
    status_template = out["status"] if "status" in out.columns else statuses
    out["status"] = _like(new_status, status_template)
    if "objective_value" in out.columns or np.any(pd_notna(new_objective)):
        out["objective_value"] = pd_float(new_objective)
    out["status_reported"] = _like(statuses.to_numpy(dtype=object), status_template)
    for name, values in columns.items():
        out[name] = _typed(name, values)
    out.attrs["verification"] = {**_attrs("computed", **details), "warnings": warnings,
                                 "counts": _verdict_counts(out)}
    return out


def pd_notna(values) -> Any:
    import pandas as pd

    return pd.notna(pd.Series(values, dtype=object)).to_numpy(dtype=bool)


def pd_float(values) -> Any:
    import pandas as pd

    return pd.to_numeric(pd.Series(values, dtype=object), errors="coerce").to_numpy(dtype=float)


_FLOAT_COLUMNS = frozenset({"verify.tol", "verify.requested_tol", "verify.worst"})


def _typed(name: str, values):
    return pd_float(values) if name in _FLOAT_COLUMNS else values


def _store(columns: dict[str, Any], i: int, verdict: Verdict, tolerance: RowTolerance | None) -> None:
    columns["verify.verdict"][i] = verdict.verdict
    if verdict.verdict == "not_checked":
        return
    if tolerance is not None:
        columns["verify.tol"][i] = tolerance.tol
        columns["verify.requested_tol"][i] = tolerance.requested
        columns["verify.requested_tol_source"][i] = tolerance.source
    columns["verify.worst"][i] = verdict.worst
    columns["verify.worst_field"][i] = verdict.worst_field
    columns["verify.reason"][i] = verdict.reason
    columns["verify.objective_from"][i] = verdict.objective_from


def _earlier_verification_warnings(df: pd.DataFrame, policy: VerifyPolicy) -> list[str]:
    """A warning per earlier ``bench kkt-verify`` verification whose rows (a
    merge of verified copies) were checked under another policy than the one
    now applied: another rule or tolerance, slack, promotion or treatment of
    missing residuals. Settings a row's annotation does not record (copies
    written before they were) are not compared."""
    names = ("rule", "tol", "slack", "promote", "missing_is_failure")
    if not any(f"metadata.kkt_verify.{name}" in df.columns for name in names):
        return []
    columns = [_column_values(df, f"metadata.kkt_verify.{name}") for name in names]
    seen = {tuple(_clean(value) for value in values) for values in zip(*columns, strict=True)}
    messages = []
    for rule, tol, slack, promote, missing_is_failure in sorted(seen, key=str):
        if rule is None and tol is None:
            continue  # not annotated
        rule = rule or "fixed"
        differences = []
        if rule == "fixed" and (policy.rule != "fixed" or tol is None or float(tol) != float(policy.tol or 0.0)):
            shown = f"{float(tol):g}" if tol is not None else "?"
            differences.append(
                f"at a fixed tolerance {shown} (--verify-tol {shown} reproduces it)"
            )
        elif rule == "requested" and policy.rule == "fixed":
            differences.append("at each solver's requested tolerance (drop --verify-tol to reproduce it)")
        elif rule == "requested" and slack is not None and float(slack) != float(policy.slack):
            differences.append(f"at {float(slack):g} x each solver's requested tolerance")
        if promote is not None and bool(_decoded(promote)) != policy.promote:
            differences.append("with promotion" if bool(_decoded(promote)) else "without promotion")
        if missing_is_failure is not None and bool(_decoded(missing_is_failure)) != policy.missing_is_failure:
            differences.append(
                "demoting claims without residuals" if bool(_decoded(missing_is_failure))
                else "keeping claims without residuals (--allow-missing)"
            )
        if differences:
            messages.append(
                "rows carry an earlier bench kkt-verify verification " + ", ".join(differences)
                + "; they are verified again from their reported statuses by the current rule"
            )
    return messages


def _unused_options_warnings(recorded: Mapping[str, Any], policy: VerifyPolicy) -> list[str]:
    """Warnings for the options a ``bench kkt-verify`` copy read as it
    recorded does not use (``--reverify`` would)."""
    unused = []
    if policy.requested_tol:
        unused.append("--requested-tol")
    if policy.tol_from_solver_id:
        unused.append("--tol-from-solver-id")
    if not policy.missing_is_failure and bool(recorded.get("missing_is_failure", True)):
        unused.append("--allow-missing")
    if not unused:
        return []
    return [
        f"this run was written by bench kkt-verify and is read as it recorded its verification, so "
        f"{', '.join(unused)} {'is' if len(unused) == 1 else 'are'} not applied; add --reverify to verify it "
        "again with them"
    ]


def _recorded_frame(df: pd.DataFrame, derived: Mapping[str, Any], policy: VerifyPolicy) -> pd.DataFrame:
    """A ``bench kkt-verify`` copy, as it recorded its verification."""
    import numpy as np

    out = df.copy()
    recorded_rule = str(derived.get("rule") or "fixed")
    details = {
        "rule": recorded_rule,
        "slack": derived.get("slack"),
        "tol": derived.get("tol"),
        "promote": bool(derived.get("promote")),
        "missing_is_failure": derived.get("missing_is_failure", True),
        "policy": policy.describe(),
    }
    warnings = _unused_options_warnings(details, policy)
    if out.empty:
        out.attrs["verification"] = {**_attrs("recorded", **details), "warnings": warnings}
        return out
    statuses = out["status"].astype(object)
    reported, _ = reported_statuses(out)
    n = len(out)
    annotated = {name: _column_values(out, f"metadata.kkt_verify.{name}") for name in (
        "verdict", "rule", "tol", "requested_tol", "requested_tol_source",
        "worst", "worst_field", "reason", "objective_from", "original_status")}
    columns: dict[str, np.ndarray] = {name: np.full(n, None, dtype=object) for name in VERIFY_COLUMNS}
    all_rows = bool(derived.get("annotates_all_rows"))
    promote = bool(derived.get("promote"))
    old_tol = derived.get("tol")
    status_values = statuses.to_numpy(dtype=object)
    for i in range(n):
        current = status_values[i]
        has_annotation = _clean(annotated["original_status"][i]) is not None
        verdict = _clean(annotated["verdict"][i])
        if verdict is None:
            verdict = _inferred_verdict(current, annotated, i, has_annotation, all_rows, promote)
        columns["verify.verdict"][i] = verdict
        columns["verify.rule"][i] = _clean(annotated["rule"][i]) or recorded_rule
        if verdict == "not_checked":
            continue
        tol = _clean(annotated["tol"][i])
        columns["verify.tol"][i] = tol if tol is not None else old_tol
        for name in ("requested_tol", "requested_tol_source", "worst", "worst_field", "reason", "objective_from"):
            columns[f"verify.{name}"][i] = _clean(annotated[name][i])
    out["status_reported"] = _like(reported.to_numpy(dtype=object), out["status"])
    for name, values in columns.items():
        out[name] = _typed(name, values)
    out.attrs["verification"] = {**_attrs("recorded", **details), "warnings": warnings,
                                 "counts": _verdict_counts(out)}
    return out


def _inferred_verdict(current: Any, annotated: dict[str, Any], i: int, has_annotation: bool,
                      all_rows: bool, promote: bool) -> str:
    """The verdict of a row in a copy written before verdicts were recorded."""
    if not has_annotation:
        if all_rows:
            return "not_checked"
        # Older copies annotated only the rows they changed or kept with a
        # note; an unannotated claim passed, an unannotated promotable row
        # was checked and stayed.
        if current in CLAIMS or (promote and (current in PROMOTABLE or current in CERTIFICATE_PROMOTION)):
            return "kept"
        return "not_checked"
    original = _clean(annotated["original_status"][i])
    reason = str(_clean(annotated["reason"][i]) or "")
    missing = ("no KKT residuals", "incomplete KKT residuals", "no certificate recorded")
    if current != original:
        if current in CLAIMS:
            return "promoted"
        if reason.startswith(missing):
            return "missing_residuals"
        return "demoted"
    if reason.startswith("returned after the time limit"):
        return "over_time_limit"
    if reason.startswith("cone distances not checked"):
        return "unverified_cones"
    if reason.startswith("certificate not fully checked"):
        return "unverified_certificate"
    if reason.startswith(missing):
        # Kept: older copies kept claims without a certificate.
        return "missing_residuals"
    return "kept"


def _verdict_counts(df: pd.DataFrame) -> dict[str, int]:
    """Rows per verdict, and as ``unchecked_kept`` the claims without
    residuals or a certificate whose status stands (missing residuals
    tolerated, or a copy written when such certificates were kept)."""
    if "verify.verdict" not in df.columns:
        return {}
    verdict = df["verify.verdict"]
    counts = {str(key): int(value) for key, value in verdict.value_counts(dropna=True).items()}
    if "status_reported" in df.columns:
        unchanged = df["status"].astype(object).to_numpy() == df["status_reported"].astype(object).to_numpy()
        counts["unchecked_kept"] = int(((verdict == "missing_residuals").to_numpy() & unchanged).sum())
    return counts


# --- summaries ------------------------------------------------------------------


def verification_info(df: pd.DataFrame) -> dict[str, Any]:
    """``df.attrs["verification"]``, or what the frame's columns say when
    the attribute was lost (pandas drops attrs through some operations; see
    :func:`_info_from_columns`)."""
    info = getattr(df, "attrs", {}).get("verification")
    if isinstance(info, dict):
        return info
    if "verify.verdict" in df.columns:
        return _info_from_columns(df)
    return _attrs("unverified")


def _info_from_columns(df: pd.DataFrame) -> dict[str, Any]:
    """The verification summary rebuilt from a frame's ``verify.*`` columns.

    Nothing is assumed about the policy. The rule is ``verify.rule``, a fixed
    tolerance ``verify.tol``, and the slack ``verify.tol`` over
    ``verify.requested_tol``. Promotion is on if the promotable stops were
    checked and off if they were not. Missing residuals counted as failures
    if the claims without them were demoted, and were tolerated if those
    claims stand. A setting that no row shows, or that rows disagree on, is
    ``None``: the header then states nothing about it and says that the
    summary was rebuilt (``reconstructed``).
    """
    import pandas as pd

    verdict = df["verify.verdict"].astype(object)
    checked = (verdict.notna() & (verdict != "not_checked")).to_numpy(dtype=bool)
    rules = {str(value) for value in df["verify.rule"].dropna().unique()} if "verify.rule" in df.columns else set()
    rule = rules.pop() if len(rules) == 1 else None

    def single(values: pd.Series) -> float | None:
        values = pd.to_numeric(values, errors="coerce").dropna()
        if values.empty:
            return None
        first = float(values.iloc[0])
        same = (values - first).abs() <= 1e-9 * abs(first)
        return float(f"{first:.12g}") if bool(same.all()) else None

    tol = slack = None
    if rule == "fixed" and "verify.tol" in df.columns:
        tol = single(df.loc[checked, "verify.tol"])
    if rule == "requested" and {"verify.tol", "verify.requested_tol"} <= set(df.columns):
        slack = single(pd.to_numeric(df["verify.tol"], errors="coerce")
                       / pd.to_numeric(df["verify.requested_tol"], errors="coerce"))

    def setting(rows: Any, on: Any) -> bool | None:
        """True if every row in ``rows`` shows the setting, False if none does,
        None if there are no such rows or they disagree."""
        shown = on[rows]
        if not len(shown):
            return None
        if bool(shown.all()):
            return True
        return False if not bool(shown.any()) else None

    promote = missing_is_failure = None
    if "status_reported" in df.columns:
        reported = df["status_reported"].astype(object)
        promotable = reported.isin(list(PROMOTABLE | set(CERTIFICATE_PROMOTION))).to_numpy(dtype=bool)
        promote = setting(promotable, checked)
        changed = df["status"].astype(object).to_numpy() != reported.to_numpy()
        missing_is_failure = setting((verdict == "missing_residuals").to_numpy(dtype=bool), changed)
    return _attrs("computed", rule=rule, tol=tol, slack=slack, promote=promote,
                  missing_is_failure=missing_is_failure, reconstructed=True, counts=_verdict_counts(df))


def _format_tol(value: Any) -> str:
    return f"{float(value):g}" if _number(value) is not None else "?"


def verification_header(info_or_df: Any) -> str:
    """One paragraph saying how the statuses of a loaded frame were decided."""
    info = info_or_df if isinstance(info_or_df, dict) else verification_info(info_or_df)
    mode = info.get("mode")
    if mode == "unverified":
        return "Statuses as the solvers reported them (not KKT-verified)."
    promotion = (
        " Inaccurate and iteration- or time-limited stops whose point passes count as solved; results "
        "returned after the run's time limit never do."
    ) if info.get("promote") else ""
    caveats = _unchecked_caveats(info.get("counts") or {})
    if mode == "recorded":
        if info.get("rule") == "fixed":
            how = f"fixed tolerance {_format_tol(info.get('tol'))}"
        else:
            # A copy written before the slack was recorded used the default.
            how = f"{float(info.get('slack') or VERIFY_SLACK):g} x each solver's requested tolerance"
        how += ", promote" if info.get("promote") else ", no promote"
        return (
            f"Statuses as verified by `bench kkt-verify` ({how}) when this run was written.{caveats} "
            f"--reverify re-checks at {VERIFY_SLACK:g} x each solver's requested tolerance."
        )
    rule, slack = info.get("rule"), info.get("slack")
    if rule == "fixed" and info.get("tol") is not None:
        how = f"a fixed tolerance {_format_tol(info.get('tol'))} for every solver"
        # A rebuilt summary cannot tell --verify-tol from a bench kkt-verify --tol copy.
        how += "" if info.get("reconstructed") else " (--verify-tol)"
    elif rule == "requested" and slack is not None:
        how = f"{float(slack):g} x the tolerance its solver was asked for"
    else:
        how = "the tolerance in its verify.tol column"
    missing_is_failure = info.get("missing_is_failure")
    if missing_is_failure is None:
        missing = "; a claim that fails counts as unsolved."
    elif missing_is_failure:
        missing = "; a claim that fails, or that recorded no residuals or certificate to check, counts as unsolved."
    else:
        missing = "; a claim that fails counts as unsolved (claims with no residuals or certificate are kept, --allow-missing)."
    rebuilt = (
        " (This summary was rebuilt from the verify.* columns, as the frame lost its record of the policy; a "
        "setting no row shows is not stated.)"
    ) if info.get("reconstructed") else ""
    return (
        "Statuses are KKT-verified: each claimed solution and infeasibility certificate is checked against the "
        f"harness's own KKT residuals at {how}{missing}{promotion}{caveats}{rebuilt} --unverified shows the "
        "solvers' own statuses."
    )


def _unchecked_caveats(counts: Mapping[str, int]) -> str:
    """What the header must admit was not checked, from the verdict counts."""
    text = ""
    cones = int(counts.get("unverified_cones", 0)) + int(counts.get("unverified_certificate", 0))
    if cones:
        text += (
            f" {cones} claim{'s' if cones != 1 else ''} passed every residual the harness can compute but "
            "lie on cones it cannot project (complex PSD); their cone distances are unchecked."
        )
    unchecked = int(counts.get("unchecked_kept", 0))
    if unchecked:
        text += (
            f" {unchecked} claim{'s' if unchecked != 1 else ''} without residuals or a certificate "
            f"{'were' if unchecked != 1 else 'was'} kept unchecked."
        )
    return text


VERIFICATION_TABLE_COLUMNS = [
    "solver_id",
    "requested_tol",
    "source",
    "tol",
    "kept",
    "demoted",
    "promoted",
    "over_limit",
    "no_residuals",
    "unverified",
]


def verification_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per solver: the tolerances its rows were held to and what happened.

    ``requested_tol`` is what its settings asked for (empty under a fixed
    tolerance), ``source`` where that came from (``row``, ``manifest``,
    ``override``, ``solver_id`` or ``fixed``), ``tol`` the tolerance its
    solutions and certificates were checked at (several values are listed
    when its rows differ). Counts are rows: ``kept`` claims that stand
    (``unverified`` ones included), ``demoted`` claims that failed or had no
    residuals or certificate (``no_residuals`` counts the latter on their own,
    demoted or not), ``promoted`` stops that passed, ``over_limit`` stops
    returned after the time limit, and ``unverified`` claims kept without a
    full check (cones the harness cannot project, or no residuals or
    certificate with missing residuals tolerated).
    """
    import pandas as pd

    if df.empty or "verify.verdict" not in df.columns:
        return pd.DataFrame(columns=VERIFICATION_TABLE_COLUMNS)
    rows = []
    for solver_id, group in df.groupby("solver_id", observed=True, sort=True):
        verdict = group["verify.verdict"]
        reported = group["status_reported"] if "status_reported" in group else group["status"]
        changed = group["status"].astype(object) != reported.astype(object)
        missing = verdict == "missing_residuals"
        unverified = verdict.isin(["unverified_cones", "unverified_certificate"]) | (missing & ~changed)
        rows.append(
            {
                "solver_id": solver_id,
                "requested_tol": _distinct(group, "verify.requested_tol"),
                "source": _distinct(group, "verify.requested_tol_source"),
                "tol": _distinct(group, "verify.tol"),
                "kept": int((((verdict == "kept") & reported.isin(list(CLAIMS))) | unverified).sum()),
                "demoted": int(((verdict == "demoted") | (missing & changed)).sum()),
                "promoted": int((verdict == "promoted").sum()),
                "over_limit": int((verdict == "over_time_limit").sum()),
                "no_residuals": int(missing.sum()),
                "unverified": int(unverified.sum()),
            }
        )
    return pd.DataFrame(rows, columns=VERIFICATION_TABLE_COLUMNS)


def _distinct(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    values = group[column].dropna().unique()
    texts = sorted({f"{value:g}" if isinstance(value, float) else str(value) for value in values})
    return ", ".join(texts)


def verification_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per ``(solver_id, status_reported, status)``: what verification changed."""
    import pandas as pd

    columns = ["solver_id", "status_reported", "status", "count"]
    if df.empty or "status_reported" not in df.columns:
        return pd.DataFrame(columns=columns)
    table = (
        df.groupby(["solver_id", "status_reported", "status"], dropna=False, observed=True)
        .size()
        .reset_index(name="count")
    )
    return table.sort_values(["solver_id", "status_reported", "status"]).reset_index(drop=True)[columns]
