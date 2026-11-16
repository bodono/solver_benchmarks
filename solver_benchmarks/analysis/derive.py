"""Derived run directories: merge sharded runs, and re-verify claimed optimality
against the independent KKT residuals recorded with every solve.

Both operations write a new run directory holding ``results.jsonl``, a
``manifest.json`` that records its provenance, and copies of the source
config and events. Per-solve artifacts are not copied; result rows keep
their original ``artifact_dir``. The directory is assembled next to its
destination and moved into place at the end, so a failure leaves neither a
partial directory nor a damaged earlier one. With ``overwrite``, only an
empty directory or an earlier derived run is ever replaced.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from solver_benchmarks.analysis.kkt import certificate_check
from solver_benchmarks.core import status
from solver_benchmarks.core.config import manifest_dataset_entries

logger = logging.getLogger(__name__)

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
PROMOTABLE = {status.OPTIMAL_INACCURATE, status.MAX_ITER_REACHED, status.TIME_LIMIT}
# Certificate statuses that count as a success, and what they become when
# their certificate fails the check (see kkt.certificate_check).
_CERTIFICATE_DEMOTION = {
    status.PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE_INACCURATE,
    status.DUAL_INFEASIBLE: status.DUAL_INFEASIBLE_INACCURATE,
}
# With promote, the converse: inaccurate certificate statuses whose
# certificate is fully checked and passes become the accurate status.
_CERTIFICATE_PROMOTION = {inaccurate: accurate for accurate, inaccurate in _CERTIFICATE_DEMOTION.items()}
# Limit statuses whose record may hold a certificate instead of a point;
# such a row is promoted to the certificate's status under the same rule.
_LIMIT_STATUSES = {status.MAX_ITER_REACHED, status.TIME_LIMIT}
# Where the worker records the returned point's objective in the convention
# of objective_value (constant term and sense applied); a row promoted to
# optimal without an objective takes it from there.
_POINT_OBJECTIVE = "primal_obj_reported"
_COPIED_FILES = ("run_config.yaml", "run_config.json", "events.jsonl")


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Records of a results file and the number of lines that did not parse.

    A run killed mid-write leaves a torn last line; like ``load_results``,
    skip it (with a warning) instead of refusing the whole run. Also like
    ``load_results``, rows a later row replaced (a retried worker error or a
    recovered unavailable skip, which a resume drops only at its end) are left
    out, so a derived run never carries them.
    """
    # Imported here: storage imports pandas, which the workers that import
    # this module (the MOSEK adapter) do not need.
    from solver_benchmarks.core.storage import replaced_positions

    records: list[dict[str, Any]] = []
    skipped = 0
    if not path.exists():
        return records, skipped
    with path.open() as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            if not isinstance(record, dict):
                skipped += 1
                logger.warning("Skipping unparseable line %d in %s", lineno, path)
                continue
            records.append(record)
    replaced = replaced_positions(records)
    if replaced:
        records = [record for i, record in enumerate(records) if i not in replaced]
    return records, skipped


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        text = path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise DeriveInputError(f"cannot read {path}: {exc}") from exc
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeriveInputError(
            f"{path} is not valid JSON ({exc.msg}: line {exc.lineno} column {exc.colno}); was it cut short?"
        ) from exc
    if not isinstance(manifest, dict):
        raise DeriveInputError(f"{path} does not hold a JSON object")
    return manifest


def _check_no_overlap(out_dir: Path, sources: Iterable[Path]) -> None:
    """Refuse an output directory that is, contains, or lies inside a source.

    Without this, ``--overwrite`` would delete the very run it is about to
    read (``bench kkt-verify RUN --output-dir RUN --overwrite``) and report
    success on empty results.
    """
    out = out_dir.resolve()
    for src in sources:
        s = Path(src).resolve()
        if out == s or out in s.parents or s in out.parents:
            raise DeriveInputError(f"output directory {out_dir} overlaps source run {src}")


class OutputExistsError(FileExistsError):
    """The output directory exists and ``overwrite`` is false."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"{path} exists; pass overwrite=True to replace it")
        self.path = path


class DeriveInputError(ValueError):
    """A source or output directory a derivation cannot use."""


def _refuse_existing(out_dir: Path, overwrite: bool) -> None:
    """Refuse an existing ``out_dir`` unless ``overwrite`` may replace it.

    Only an empty directory or a derived run (its manifest has ``derived``)
    is ever replaced: the output path is the run directory itself, and a
    base run or a directory of other files given there by mistake must not
    be deleted.
    """
    if not out_dir.exists():
        return
    if not overwrite:
        raise OutputExistsError(out_dir)
    refusal = "refusing to replace it: only an empty directory or a run written by merge or kkt-verify is replaced"
    if not out_dir.is_dir():
        raise DeriveInputError(f"{out_dir} is not a directory; {refusal}")
    entries = sorted(path.name for path in out_dir.iterdir())
    if not entries:
        return
    if "manifest.json" in entries and _read_manifest(out_dir).get("derived"):
        return
    if "manifest.json" in entries or "results.jsonl" in entries:
        raise DeriveInputError(f"{out_dir} is a run directory, not a derived run; {refusal}")
    shown = ", ".join(entries[:5]) + (", ..." if len(entries) > 5 else "")
    raise DeriveInputError(f"{out_dir} is not a run directory (it holds {shown}); {refusal}")


def _check_is_run(run_dir: Path) -> None:
    """Refuse a source that is not a run, rather than derive an empty run.

    A run that planned no solves has a manifest but no results.jsonl; it is
    accepted, with no rows.
    """
    if not ((run_dir / "results.jsonl").is_file() or (run_dir / "manifest.json").is_file()):
        raise DeriveInputError(
            f"{run_dir} is not a run directory: it has neither results.jsonl nor manifest.json"
        )


@contextmanager
def _staged_out_dir(out_dir: Path, overwrite: bool) -> Iterator[Path]:
    """Yield an empty sibling directory that replaces ``out_dir`` on success.

    On an exception the staging directory is removed and ``out_dir`` (if it
    existed) is left as it was. The staging directory is made with a plain
    ``mkdir`` under a unique name, not ``tempfile.mkdtemp``: it becomes the
    output, so it must get the umask's permissions like any run directory
    rather than mkdtemp's owner-only 0700.
    """
    _refuse_existing(out_dir, overwrite)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.parent / f".{out_dir.name}.{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        yield staging
        if out_dir.exists():
            retired = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.old.", dir=out_dir.parent))
            out_dir.rename(retired / out_dir.name)
            staging.rename(out_dir)
            shutil.rmtree(retired, ignore_errors=True)
        else:
            staging.rename(out_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def merge_runs(
    run_dirs: list[str | Path], out_dir: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Concatenate the results of several run directories into ``out_dir``.

    Intended for runs sharded across machines (one solver or dataset per
    run). Rows are kept as they are; duplicates of the same
    ``(dataset, problem, solver_id)`` across shards are reported, not
    resolved. The manifest of the first shard seeds the merged manifest.
    """
    sources = [Path(d) for d in run_dirs]
    if not sources:
        raise ValueError("merge_runs needs at least one run directory")
    out = Path(out_dir)
    _check_no_overlap(out, sources)
    _refuse_existing(out, overwrite)
    for src in sources:
        _check_is_run(src)
    records: list[dict[str, Any]] = []
    seen: dict[tuple[Any, Any, Any], int] = {}
    per_source: list[dict[str, Any]] = []
    skipped = 0
    for src in sources:
        rows, src_skipped = _read_jsonl(src / "results.jsonl")
        skipped += src_skipped
        for row in rows:
            key = (row.get("dataset"), row.get("problem"), row.get("solver_id"))
            seen[key] = seen.get(key, 0) + 1
        records.extend(rows)
        entry: dict[str, Any] = {"run_dir": str(src), "rows": len(rows)}
        if src_skipped:
            entry["skipped_lines"] = src_skipped
        per_source.append(entry)
    duplicates = sum(1 for count in seen.values() if count > 1)
    mixed_forms = _problems_posed_in_several_forms(records)
    manifests = [_read_manifest(src) for src in sources]
    manifest = json.loads(json.dumps(manifests[0])) if manifests else {}
    # Union the configured datasets and solvers across shards (by id) so
    # completion / missing-results checks see every selection, and keep the
    # full source manifests for provenance (settings, environments).
    cfg = manifest.setdefault("config", {})
    cfg["datasets"] = _union_dataset_entries(m.get("config") or {} for m in manifests)
    # The per-entry selections above already fold in each shard's run-level
    # include / exclude, which must not survive as a global filter.
    for filter_key in ("include", "exclude", "dataset", "dataset_options"):
        cfg.pop(filter_key, None)
    solvers: dict[str, dict] = {}
    for m in manifests:
        for entry in (m.get("config") or {}).get("solvers") or []:
            solvers.setdefault(str(entry.get("id")), entry)
    cfg["solvers"] = list(solvers.values())
    # What each shard actually planned, as (dataset entries, solver ids)
    # groups; completion and missing-results checks union these instead of
    # crossing the merged dataset and solver lists, which are informative
    # only. A merged source contributes its own groups, so nesting is exact.
    selections: list[dict[str, Any]] = []
    for m in manifests:
        nested = (m.get("derived") or {}).get("selections")
        if nested:
            selections.extend(nested)
        else:
            source_cfg = m.get("config") or {}
            selections.append(
                {
                    "datasets": manifest_dataset_entries(source_cfg),
                    "solvers": [str(entry.get("id")) for entry in source_cfg.get("solvers") or []],
                }
            )
    manifest["derived"] = {
        "kind": "merge",
        "sources": per_source,
        "source_manifests": manifests,
        "selections": selections,
        "duplicate_keys": duplicates,
        "mixed_forms": mixed_forms,
    }
    with _staged_out_dir(out, overwrite) as staging:
        _write_jsonl(staging / "results.jsonl", records)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for name in _COPIED_FILES:
            if (sources[0] / name).exists():
                shutil.copy(sources[0] / name, staging / name)
    return {
        "rows": len(records),
        "sources": len(sources),
        "duplicate_keys": duplicates,
        "mixed_forms": mixed_forms,
        "skipped_lines": skipped,
    }


def _problems_posed_in_several_forms(records: Iterable[dict[str, Any]]) -> list[str]:
    """Problems whose rows disagree on the number of constraint rows.

    The rows of one ``(dataset, problem)`` should all come from the same
    problem data. If two shards posed it differently (for example SDPLIB with
    its declared blocks in one and as one merged block in the other), their
    results are not comparable, and the row count recorded with each result
    is where that shows. Rows without one (skips, some failures) are ignored.
    """
    sizes: dict[tuple[str, str], set[int]] = {}
    for row in records:
        metadata = row.get("metadata")
        rows_posed = metadata.get("m") if isinstance(metadata, dict) else None
        if isinstance(rows_posed, int) and not isinstance(rows_posed, bool):
            key = (str(row.get("dataset")), str(row.get("problem")))
            sizes.setdefault(key, set()).add(rows_posed)
    return sorted(f"{dataset}/{problem}" for (dataset, problem), seen in sizes.items() if len(seen) > 1)


def _union_dataset_entries(configs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union the dataset selections of several shard configs, by dataset id.

    Each shard's entries are resolved with ``manifest_dataset_entries`` (which
    applies run-level ``include`` / ``exclude`` and the legacy single-dataset
    shape). A problem is expected from the merged run if any shard expected
    it: an entry with no ``include`` list expects every problem, so it wins
    over an explicit list, and a problem stays excluded only if every shard
    excluded it.
    """
    merged: dict[str, dict[str, Any]] = {}
    for config in configs:
        for entry in manifest_dataset_entries(config):
            current = merged.get(entry["id"])
            if current is None:
                merged[entry["id"]] = {**entry, "include": list(entry["include"]), "exclude": list(entry["exclude"])}
                continue
            if not current["include"] or not entry["include"]:
                current["include"] = []
            else:
                current["include"] = sorted({*current["include"], *entry["include"]})
            current["exclude"] = sorted(set(current["exclude"]) & set(entry["exclude"]))
    return list(merged.values())


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


def _returned_after_time_limit(row: dict[str, Any]) -> bool:
    info = row.get("info")
    if not isinstance(info, dict):
        return False
    return bool(info.get("exceeded_time_limit")) or info.get("status_before_timeout") is not None


def kkt_verify(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    tol: float,
    fields: tuple[str, ...] = KKT_FIELDS,
    missing_is_failure: bool = True,
    promote: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Re-classify claimed-optimal solves whose KKT residuals exceed ``tol``.

    A row whose status says a solution is present keeps that status only if
    the worst of ``fields`` (relative residuals computed by the harness
    from the returned point, independently of the solver's own report;
    cone- and QP-form records always add their cone distances and bound
    multipliers, see :func:`check_residuals`) is at or below ``tol``.
    Otherwise it becomes ``optimal_inaccurate`` and ``metadata.kkt_verify``
    records why, which fields were checked and which one failed. Claimed
    optima with residuals missing are demoted too unless
    ``missing_is_failure`` is False. A row whose only unknown residuals are the distances to cones the
    harness cannot project (``kkt.unsupported_cones``) is kept when the rest
    pass, with the reason recorded and counted under ``unverified_cones``.

    Infeasibility certificates (``primal_infeasible``, ``dual_infeasible``)
    are held to the same ``tol``: their largest residual once scaled so that
    the certificate's objective term is −1 (``kkt.cert_res``, rebuilt for
    older records) must be at or below it, or the row becomes the matching
    ``*_inaccurate`` status. A certificate row in which nothing fails but
    not everything is known (the adapter recorded no certificate, as
    several return none, or the distance to a cone the harness cannot
    project is unknown) is kept whatever ``missing_is_failure`` says: that
    is a gap in the adapter or the checker, not in the solver's accuracy.
    The reason is recorded and the row counted under
    ``unverified_certificates``.

    With ``promote``, the converse also applies: a solve the solver itself
    reported as inaccurate, iteration-limited or time-limited, but whose
    returned point does pass the check, becomes ``optimal`` (its recorded
    time is what it took to produce that point). If it has no
    ``objective_value`` (adapters report one only for a solution), it takes
    the point's objective that the worker recorded as
    ``kkt.primal_obj_reported`` and ``metadata.kkt_verify.objective_from``
    says so; rows of older runs without that field stay without one. Likewise
    a ``primal_infeasible_inaccurate`` / ``dual_infeasible_inaccurate`` row
    whose certificate is fully checked and passes ``tol`` becomes
    ``primal_infeasible`` / ``dual_infeasible``, and so does an iteration- or
    time-limited row whose record holds such a certificate instead of a
    point (OSQP records its inaccurate certificates so); a missing or partly
    checked certificate is never promoted. Success then means one thing for every
    solver: the returned point or certificate satisfies the check to
    ``tol``. A row returned after the run's time limit is never promoted
    (the limit is every solver's budget): the runner marks it with
    ``info.exceeded_time_limit``, or, in older runs, records a conclusion
    reached after the limit as time_limit with ``info.status_before_timeout``.
    It keeps its status, ``metadata.kkt_verify`` says why, and it is counted
    under ``over_time_limit``. A solver stopped by its own time limit
    returns slightly after it, so when that limit equals ``timeout_seconds``
    its time-limit stops are never promoted; set the solver's limit below
    ``timeout_seconds`` if those points should be eligible.

    Every solver in a comparison is held to the same ``tol`` regardless of
    what its own tolerance settings mean, which is what makes timings
    across solvers comparable. Unparseable lines (a torn last line) are
    skipped and counted under ``skipped_lines``. An existing ``out_dir`` is
    replaced only with ``overwrite``, and only if it is empty or a derived
    run.
    """
    src = Path(run_dir)
    out = Path(out_dir)
    _check_no_overlap(out, [src])
    _refuse_existing(out, overwrite)
    _check_is_run(src)
    records, skipped = _read_jsonl(src / "results.jsonl")
    demoted: dict[str, int] = {}
    kept: dict[str, int] = {}
    missing: dict[str, int] = {}
    promoted: dict[str, int] = {}
    over_time_limit: dict[str, int] = {}
    unverified: dict[str, int] = {}
    unverified_certificates: dict[str, int] = {}
    checked_fields: set[str] = set()

    def count(tally: dict[str, int], solver_id: str) -> None:
        tally[solver_id] = tally.get(solver_id, 0) + 1

    def annotate(row: dict[str, Any], **details: Any) -> None:
        metadata = row.setdefault("metadata", {}) or {}
        metadata["kkt_verify"] = {"tol": tol, **details, "original_status": row["status"]}
        row["metadata"] = metadata

    for row in records:
        solver_id = str(row.get("solver_id"))
        row_status = row.get("status")
        if promote and (row_status in PROMOTABLE or row_status in _CERTIFICATE_PROMOTION):
            if _returned_after_time_limit(row):
                annotate(row, reason="returned after the time limit; not promoted")
                count(over_time_limit, solver_id)
                continue
            kkt = row.get("kkt") or {}
            target = _CERTIFICATE_PROMOTION.get(row_status)
            if row_status in _LIMIT_STATUSES and kkt.get("certificate") in _CERTIFICATE_DEMOTION:
                # A limit stop that returned a certificate (OSQP records its
                # inaccurate certificates so) is judged as one.
                target = kkt["certificate"]
            if target is not None:
                cert, complete = certificate_check(kkt)
                checked_fields.add("cert_res")
                # A missing or partly checked certificate is never promoted,
                # nor one of the other kind.
                if complete and cert is not None and cert <= tol and kkt.get("certificate") == target:
                    annotate(row, fields=["cert_res"], worst=cert, worst_field="cert_res",
                             reason=f"certificate residual {cert:.3e} <= {tol:.1e}")
                    row["status"] = target
                    count(promoted, solver_id)
                continue
            check = check_residuals(row, fields)
            checked_fields.update(check.checked)
            if check.complete and check.worst is not None and check.worst <= tol:
                details: dict[str, Any] = {}
                point_objective = (row.get("kkt") or {}).get(_POINT_OBJECTIVE)
                if row.get("objective_value") is None and point_objective is not None:
                    # Only a solution has an objective_value, so a limit
                    # stop has none; the point it returned is now one.
                    row["objective_value"] = point_objective
                    details["objective_from"] = f"kkt.{_POINT_OBJECTIVE}"
                annotate(
                    row,
                    fields=check.checked,
                    worst=check.worst,
                    worst_field=check.worst_field,
                    reason=f"worst relative residual {check.worst:.3e} ({check.worst_field}) <= {tol:.1e}",
                    **details,
                )
                row["status"] = status.OPTIMAL
                count(promoted, solver_id)
            continue
        if row_status in _CERTIFICATE_DEMOTION:
            kkt = row.get("kkt") or {}
            cert, complete = certificate_check(kkt)
            checked_fields.add("cert_res")
            if cert is not None and cert > tol:
                reason = f"certificate residual {cert:.3e} > {tol:.1e}"
                if not complete:
                    reason += " (some residuals unknown)"
                annotate(row, fields=["cert_res"], worst=cert, worst_field="cert_res", reason=reason)
                row["status"] = _CERTIFICATE_DEMOTION[row_status]
                count(demoted, solver_id)
                continue
            if not complete:
                # Nothing fails, but not everything is known: the adapter
                # recorded no certificate (several return none), or the
                # distance to a cone the harness cannot project is unknown.
                # Neither says anything about the solver's accuracy, so the
                # row is kept, like an optimal row with such cones.
                if cert is None:
                    reason = "no certificate recorded"
                else:
                    cones = kkt.get("unsupported_cones")
                    unknown = f"unsupported cones {sorted(cones)}" if cones else "some residuals unknown"
                    reason = f"certificate not fully checked ({unknown}); known residuals <= {tol:.1e}"
                annotate(row, fields=["cert_res"], worst=cert,
                         worst_field=None if cert is None else "cert_res", reason=reason)
                count(unverified_certificates, solver_id)
            count(kept, solver_id)
            continue
        if row_status not in status.SOLUTION_PRESENT:
            continue
        check = check_residuals(row, fields)
        checked_fields.update(check.checked)
        worst = check.worst
        if worst is not None and worst > tol:
            # A residual that demonstrably fails demotes the row even when
            # other residuals are missing and missing ones are tolerated.
            reason = f"worst relative residual {worst:.3e} ({check.worst_field}) > {tol:.1e}"
            if not check.complete:
                reason += " (some residuals missing)"
        elif not check.complete:
            unsupported = (row.get("kkt") or {}).get("unsupported_cones")
            if unsupported and worst is not None and set(check.missing) <= set(CONE_FIELDS):
                # The point passes every residual the harness could compute;
                # the cone distances are unknown, not failed.
                annotate(
                    row,
                    fields=check.checked,
                    worst=worst,
                    worst_field=check.worst_field,
                    reason=f"cone distances not checked for unsupported cones {sorted(unsupported)}; "
                    f"other residuals <= {tol:.1e}",
                )
                count(unverified, solver_id)
                count(kept, solver_id)
                continue
            count(missing, solver_id)
            if not missing_is_failure:
                count(kept, solver_id)
                continue
            reason = "no KKT residuals recorded" if worst is None else "incomplete KKT residuals"
        else:
            count(kept, solver_id)
            continue
        annotate(row, fields=check.checked, worst=worst, worst_field=check.worst_field, reason=reason)
        row["status"] = status.OPTIMAL_INACCURATE
        count(demoted, solver_id)
    manifest = _read_manifest(src)
    # The input run's own provenance (merge sources and their manifests, the
    # per-source selections, earlier verifications) is kept in full: a
    # verified merge must still know each source's settings and time limits.
    prior = manifest.get("derived") or {}
    summary: dict[str, Any] = {
        "kind": "kkt_verify",
        "source": str(src),
        "tol": tol,
        "fields": list(fields),
        "checked_fields": sorted(checked_fields),
        "missing_is_failure": missing_is_failure,
        "promote": promote,
        "kept": kept,
        "demoted": demoted,
        "promoted": promoted,
        "over_time_limit": over_time_limit,
        "missing_residuals": missing,
        "unverified_cones": unverified,
        "unverified_certificates": unverified_certificates,
        "skipped_lines": skipped,
    }
    if prior:
        summary["source_derived"] = prior
        for key in ("selections", "source_manifests"):
            if key in prior:
                summary[key] = prior[key]
    manifest["derived"] = summary
    with _staged_out_dir(out, overwrite) as staging:
        _write_jsonl(staging / "results.jsonl", records)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for name in _COPIED_FILES:
            if (src / name).exists():
                shutil.copy(src / name, staging / name)
    return summary
