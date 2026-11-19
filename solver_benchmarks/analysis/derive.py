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
from pathlib import Path
from typing import Any

# The residual check lives in verify; its names are kept here, where callers
# (the MOSEK adapter, tests) have imported them from.
from solver_benchmarks.analysis.verify import (  # noqa: F401
    CERTIFICATE_DEMOTION,
    CLAIMS,
    CONE_FIELDS,
    KKT_FIELDS,
    PROMOTABLE,
    QP_FIELDS,
    VERIFY_SLACK,
    RequestedToleranceUnknown,
    ResidualCheck,
    RowTolerance,
    ToleranceResolver,
    VerifyPolicy,
    check_residuals,
    classify,
    residual_summary,
    row_info_from_record,
    worst_relative_residual,
)
from solver_benchmarks.core.config import manifest_dataset_entries

logger = logging.getLogger(__name__)

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
    mixed_verification = _mixed_verification(manifests)
    manifest["derived"] = {
        "kind": "merge",
        "sources": per_source,
        "source_manifests": manifests,
        "selections": selections,
        "duplicate_keys": duplicates,
        "mixed_forms": mixed_forms,
    }
    if mixed_verification:
        manifest["derived"]["mixed_verification"] = mixed_verification
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
        "mixed_verification": mixed_verification,
        "skipped_lines": skipped,
    }


def _verification_state(manifest: dict[str, Any]) -> str:
    """How a run's statuses were verified when it was written, in words."""
    derived = manifest.get("derived") or {}
    if derived.get("kind") != "kkt_verify":
        return "as reported (not verified by bench kkt-verify)"
    if derived.get("rule") == "requested":
        how = f"bench kkt-verify at {derived.get('slack') or VERIFY_SLACK:g} x each solver's requested tolerance"
    else:
        how = f"bench kkt-verify at a fixed tolerance {derived.get('tol')}"
    return how + (", promote" if derived.get("promote") else ", no promote")


def _mixed_verification(manifests: list[dict[str, Any]]) -> list[str]:
    """The verification states of a merge's inputs, when they differ.

    A merged run is verified again when it is loaded (the rows keep their
    reported statuses in ``metadata.kkt_verify.original_status``), so this
    only warns that the inputs had been judged differently.
    """
    counts: dict[str, int] = {}
    for manifest in manifests:
        state = _verification_state(manifest)
        counts[state] = counts.get(state, 0) + 1
    if len(counts) < 2:
        return []
    return [f"{count} input(s) {state}" for state, count in counts.items()]


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


def kkt_verify(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    tol: float | None = None,
    slack: float = VERIFY_SLACK,
    fields: tuple[str, ...] = KKT_FIELDS,
    missing_is_failure: bool = True,
    promote: bool = True,
    overwrite: bool = False,
    requested_tol: dict[str, Any] | None = None,
    tol_from_solver_id: bool = False,
) -> dict[str, Any]:
    """Write a copy of a run whose statuses are KKT-verified.

    Every row is judged from its reported status by
    :func:`solver_benchmarks.analysis.verify.classify` (the rule ``bench
    report`` and every analysis command apply when they load a run): claimed
    solutions whose relative KKT residuals exceed the row's tolerance become
    ``optimal_inaccurate``, claimed certificates whose scaled residual exceeds
    it the matching ``*_inaccurate`` status, and with ``promote`` (the
    default) inaccurate, iteration- and time-limited stops whose point or
    certificate passes become accurate, except a row returned after the run's
    time limit. Claimed optima with residuals missing, and claimed
    certificates whose adapter recorded none, are demoted unless
    ``missing_is_failure`` is False; claims whose only unknown residuals are
    distances to cones the harness cannot project are kept and counted under
    ``unverified_cones`` / ``unverified_certificates``.

    The row's tolerance is ``slack`` times the tolerance its configuration
    asked the solver for (``SolverAdapter.requested_tolerance``), for
    solutions and certificates alike, or, with ``tol``, that one value for
    every row (``slack``, ``requested_tol`` and ``tol_from_solver_id`` are
    then refused). ``requested_tol`` and ``tol_from_solver_id`` supply
    requested tolerances the run does not record (see ``VerifyPolicy``); a
    row that needs one and has none raises ``RequestedToleranceUnknown``
    before anything is written.

    Every checked row gets ``metadata.kkt_verify``: its ``verdict``, the
    ``rule``, tolerance and policy (``slack``, ``promote``,
    ``missing_is_failure``), the fields checked and the worst one, the reason,
    and the ``original_status`` it was reported with (kept from an earlier
    verification, so verifying a verified run starts again from the reported
    status and, under the same policy, writes the same rows; a copy of a copy
    written before this version records the status of the step before, which
    it then starts from); a promotion that gave the row an objective
    (``kkt.primal_obj_reported``) names it in ``objective_from``. Unparseable
    lines (a torn last line) are skipped and counted under ``skipped_lines``.
    An existing ``out_dir`` is replaced only with ``overwrite``, and only if
    it is empty or a derived run.
    """
    src = Path(run_dir)
    out = Path(out_dir)
    policy = VerifyPolicy(
        rule="fixed" if tol is not None else "requested",
        tol=tol,
        slack=slack,
        promote=promote,
        missing_is_failure=missing_is_failure,
        requested_tol=requested_tol or {},
        tol_from_solver_id=tol_from_solver_id,
    )
    _check_no_overlap(out, [src])
    _refuse_existing(out, overwrite)
    _check_is_run(src)
    manifest = _read_manifest(src)
    resolver = ToleranceResolver(manifest, policy)
    records, skipped = _read_jsonl(src / "results.jsonl")
    tallies: dict[str, dict[str, int]] = {
        key: {}
        for key in ("kept", "demoted", "promoted", "over_time_limit", "missing_residuals",
                    "unverified_cones", "unverified_certificates")
    }
    tolerances: dict[str, dict[str, set]] = {}
    checked_fields: set[str] = set()
    unknown: dict[str, str] = {}

    class _Unresolved(Exception):
        pass

    for row in records:
        solver_id = str(row.get("solver_id"))
        metadata = row.get("metadata")
        earlier = metadata.get("kkt_verify") if isinstance(metadata, dict) else None
        if not isinstance(earlier, dict):
            earlier = {}
        reported = earlier.get("original_status") or row.get("status")
        objective = None if earlier.get("objective_from") else row.get("objective_value")
        info = row.get("info")
        view = {
            "status": reported,
            "kkt": row.get("kkt") or {},
            "info": info if isinstance(info, dict) else {},
            "objective_value": objective,
        }
        resolved = resolver.resolve(row_info_from_record(row))

        def tolerance(resolved=resolved) -> float:
            if isinstance(resolved, str):
                raise _Unresolved
            return resolved.tol

        try:
            verdict = classify(
                view, tolerance=tolerance, promote=promote, missing_is_failure=missing_is_failure, fields=fields
            )
        except _Unresolved:
            unknown.setdefault(solver_id, str(resolved))
            continue
        row["status"] = verdict.status
        if "objective_value" in row or verdict.objective is not None:
            row["objective_value"] = verdict.objective if verdict.objective is not None else objective
        if verdict.verdict == "not_checked":
            if isinstance(metadata, dict):
                metadata.pop("kkt_verify", None)
            continue
        checked_fields.update(verdict.fields)
        _tally(tallies, solver_id, verdict.verdict, str(reported), verdict.status, missing_is_failure)
        annotation: dict[str, Any] = {"verdict": verdict.verdict, "rule": policy.rule}
        if policy.rule == "requested":
            annotation["slack"] = slack
        annotation.update(promote=promote, missing_is_failure=missing_is_failure)
        if isinstance(resolved, RowTolerance):
            annotation.update(
                tol=resolved.tol,
                requested_tol=resolved.requested,
                requested_tol_source=resolved.source,
            )
            seen = tolerances.setdefault(solver_id, {key: set() for key in (
                "requested_tol", "requested_tol_source", "tol")})
            for key in seen:
                value = annotation.get(key)
                if value is not None:
                    seen[key].add(value)
        annotation.update(
            fields=list(verdict.fields),
            worst=_finite_or_none(verdict.worst),
            worst_field=verdict.worst_field,
            reason=verdict.reason,
            original_status=reported,
        )
        if verdict.objective_from:
            annotation["objective_from"] = verdict.objective_from
        if not isinstance(metadata, dict):
            metadata = row["metadata"] = {}
        metadata["kkt_verify"] = annotation
    if unknown:
        raise RequestedToleranceUnknown(unknown)
    # The input run's own provenance (merge sources and their manifests, the
    # per-source selections, earlier verifications) is kept in full: a
    # verified merge must still know each source's settings and time limits.
    prior = manifest.get("derived") or {}
    summary: dict[str, Any] = {
        "kind": "kkt_verify",
        "source": str(src),
        "rule": policy.rule,
        "tol": tol,
        "slack": slack if tol is None else None,
        "fields": list(fields),
        "checked_fields": sorted(checked_fields),
        "missing_is_failure": missing_is_failure,
        "promote": promote,
        "requested_tol": dict(policy.requested_tol),
        "tol_from_solver_id": tol_from_solver_id,
        # Every checked row carries its verdict (copies written before this
        # field annotated only the rows they changed or kept with a note).
        "annotates_all_rows": True,
        **tallies,
        "tolerances": {
            solver_id: {key: sorted(values, key=str) for key, values in seen.items()}
            for solver_id, seen in sorted(tolerances.items())
        },
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


def _tally(
    tallies: dict[str, dict[str, int]], solver_id: str, verdict: str, reported: str, new_status: str | None,
    missing_is_failure: bool,
) -> None:
    """Count a verdict the way ``bench kkt-verify`` has always reported it:
    ``kept`` claims that stand (unverified ones and tolerated missing
    residuals included), ``demoted`` claims that failed (missing residuals
    included when they are failures), ``missing_residuals`` claims without
    complete residuals either way."""

    def count(key: str) -> None:
        tallies[key][solver_id] = tallies[key].get(solver_id, 0) + 1

    if verdict == "kept" and reported in CLAIMS:
        count("kept")
    elif verdict == "demoted":
        count("demoted")
    elif verdict == "promoted":
        count("promoted")
    elif verdict == "over_time_limit":
        count("over_time_limit")
    elif verdict == "missing_residuals":
        count("missing_residuals")
        count("demoted" if missing_is_failure and new_status != reported else "kept")
    elif verdict == "unverified_cones":
        count("unverified_cones")
        count("kept")
    elif verdict == "unverified_certificate":
        count("unverified_certificates")
        count("kept")


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value
