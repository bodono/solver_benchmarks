"""Derived run directories: merge sharded runs, and re-verify claimed optimality
against the independent KKT residuals recorded with every solve.

Both operations write a new run directory holding ``results.jsonl``, a
``manifest.json`` that records its provenance, and copies of the source
config and events. Per-solve artifacts are not copied; result rows keep
their original ``artifact_dir``.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from solver_benchmarks.core import status
from solver_benchmarks.core.config import manifest_dataset_entries

KKT_FIELDS = ("primal_res_rel", "dual_res_rel", "duality_gap_rel")
# Cone-form records also carry the distance of s and y to their cones; a point
# that satisfies Ax + s = b with s outside K is not feasible, so these are part
# of the check whenever the record has them (QP-form records have none). The
# distances are scaled like the equality residuals (by 1 + the size of the
# primal, respectively dual, data) so the check does not depend on the units
# of the problem or on projection roundoff in a large cone.
CONE_FIELDS = ("primal_cone_res_rel", "dual_cone_res_rel")
# Records written before the relative distances existed only carry the
# absolute ones; they are used in that case.
_CONE_FALLBACK = {"primal_cone_res_rel": "primal_cone_res", "dual_cone_res_rel": "dual_cone_res"}
# Statuses that carry a returned point which may still pass the check.
PROMOTABLE = {status.OPTIMAL_INACCURATE, status.MAX_ITER_REACHED, status.TIME_LIMIT}
_COPIED_FILES = ("run_config.yaml", "run_config.json", "events.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


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
            raise ValueError(f"output directory {out_dir} overlaps source run {src}")


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass overwrite=True to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


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
    _prepare_out_dir(out, overwrite)
    records: list[dict[str, Any]] = []
    seen: dict[tuple[Any, Any, Any], int] = {}
    per_source: list[dict[str, Any]] = []
    for src in sources:
        rows = _read_jsonl(src / "results.jsonl")
        for row in rows:
            key = (row.get("dataset"), row.get("problem"), row.get("solver_id"))
            seen[key] = seen.get(key, 0) + 1
        records.extend(rows)
        per_source.append({"run_dir": str(src), "rows": len(rows)})
    duplicates = sum(1 for count in seen.values() if count > 1)
    _write_jsonl(out / "results.jsonl", records)
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
    manifest["derived"] = {
        "kind": "merge",
        "sources": per_source,
        "source_manifests": manifests,
        "duplicate_keys": duplicates,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in _COPIED_FILES:
        if (sources[0] / name).exists():
            shutil.copy(sources[0] / name, out / name)
    return {"rows": len(records), "sources": len(sources), "duplicate_keys": duplicates}


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


def residual_summary(
    record: dict[str, Any], fields: tuple[str, ...] = KKT_FIELDS
) -> tuple[float | None, bool]:
    """Return ``(worst, complete)`` for the listed KKT residuals of a record.

    ``worst`` is the largest finite residual among the wanted fields (None if
    there is none) and ``complete`` says whether every wanted field was
    present and finite. Cone-form records also need the cone distances
    (``CONE_FIELDS``, falling back to the absolute distances of older
    records). The two are reported separately so that a residual known to
    fail is never hidden by another one being missing.
    """
    kkt = record.get("kkt") or {}
    wanted = list(fields)
    if kkt.get("form") == "cone" or any(f in kkt or _CONE_FALLBACK[f] in kkt for f in CONE_FIELDS):
        wanted += [f for f in CONE_FIELDS if f not in wanted]
    worst: float | None = None
    complete = True
    for field in wanted:
        value = kkt.get(field)
        if value is None and field in _CONE_FALLBACK:
            value = kkt.get(_CONE_FALLBACK[field])
        try:
            number = float(value) if value is not None else None
        except (TypeError, ValueError):
            number = None
        if number is None or not math.isfinite(number):
            complete = False
            continue
        worst = number if worst is None else max(worst, number)
    return worst, complete


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
    from the returned point, independently of the solver's own report) is
    at or below ``tol``. Otherwise it becomes ``optimal_inaccurate`` and
    ``metadata.kkt_verify`` records why. Rows with no residuals recorded
    are demoted too unless ``missing_is_failure`` is False.

    With ``promote``, the converse also applies: a solve the solver itself
    reported as inaccurate, iteration-limited or time-limited, but whose
    returned point does pass the check, becomes ``optimal`` (its recorded
    time is what it took to produce that point). Success then means one
    thing for every solver: the returned point satisfies the KKT conditions
    to ``tol``.

    Every solver in a comparison is held to the same ``tol`` regardless of
    what its own tolerance settings mean, which is what makes timings
    across solvers comparable.
    """
    src = Path(run_dir)
    out = Path(out_dir)
    _check_no_overlap(out, [src])
    _prepare_out_dir(out, overwrite)
    records = _read_jsonl(src / "results.jsonl")
    demoted: dict[str, int] = {}
    kept: dict[str, int] = {}
    missing: dict[str, int] = {}
    promoted: dict[str, int] = {}
    for row in records:
        solver_id = str(row.get("solver_id"))
        if row.get("status") in PROMOTABLE and promote:
            worst = worst_relative_residual(row, fields)
            if worst is not None and worst <= tol:
                metadata = row.setdefault("metadata", {}) or {}
                metadata["kkt_verify"] = {
                    "tol": tol,
                    "fields": list(fields),
                    "worst": worst,
                    "original_status": row["status"],
                    "reason": f"worst relative residual {worst:.3e} <= {tol:.1e}",
                }
                row["metadata"] = metadata
                row["status"] = status.OPTIMAL
                promoted[solver_id] = promoted.get(solver_id, 0) + 1
            continue
        if row.get("status") not in status.SOLUTION_PRESENT:
            continue
        worst, complete = residual_summary(row, fields)
        if worst is not None and worst > tol:
            # A residual that demonstrably fails demotes the row even when
            # other residuals are missing and missing ones are tolerated.
            reason = f"worst relative residual {worst:.3e} > {tol:.1e}"
            if not complete:
                reason += " (some residuals missing)"
        elif not complete:
            missing[solver_id] = missing.get(solver_id, 0) + 1
            if not missing_is_failure:
                kept[solver_id] = kept.get(solver_id, 0) + 1
                continue
            reason = "no KKT residuals recorded" if worst is None else "incomplete KKT residuals"
        else:
            kept[solver_id] = kept.get(solver_id, 0) + 1
            continue
        metadata = row.setdefault("metadata", {}) or {}
        metadata["kkt_verify"] = {
            "tol": tol,
            "fields": list(fields),
            "worst": worst,
            "original_status": row["status"],
            "reason": reason,
        }
        row["metadata"] = metadata
        row["status"] = status.OPTIMAL_INACCURATE
        demoted[solver_id] = demoted.get(solver_id, 0) + 1
    _write_jsonl(out / "results.jsonl", records)
    manifest = _read_manifest(src)
    summary = {
        "kind": "kkt_verify",
        "source": str(src),
        "tol": tol,
        "fields": list(fields),
        "missing_is_failure": missing_is_failure,
        "promote": promote,
        "kept": kept,
        "demoted": demoted,
        "promoted": promoted,
        "missing_residuals": missing,
    }
    manifest["derived"] = summary
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in _COPIED_FILES:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    return summary
