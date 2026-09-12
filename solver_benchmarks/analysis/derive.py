"""Derived run directories: merge sharded runs, and re-verify claimed optimality
against the independent KKT residuals recorded with every solve.

Both operations write a new run directory holding ``results.jsonl``, a
``manifest.json`` that records its provenance, and copies of the source
config and events. Per-solve artifacts are not copied; result rows keep
their original ``artifact_dir``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from solver_benchmarks.core import status

KKT_FIELDS = ("primal_res_rel", "dual_res_rel", "duality_gap_rel")
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
    manifest = _read_manifest(sources[0])
    manifest["derived"] = {
        "kind": "merge",
        "sources": per_source,
        "duplicate_keys": duplicates,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for name in _COPIED_FILES:
        if (sources[0] / name).exists():
            shutil.copy(sources[0] / name, out / name)
    return {"rows": len(records), "sources": len(sources), "duplicate_keys": duplicates}


def worst_relative_residual(
    record: dict[str, Any], fields: tuple[str, ...] = KKT_FIELDS
) -> float | None:
    """Largest of the listed relative KKT residuals, or None if none is present."""
    kkt = record.get("kkt") or {}
    values = []
    for field in fields:
        value = kkt.get(field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


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
        worst = worst_relative_residual(row, fields)
        if worst is None:
            missing[solver_id] = missing.get(solver_id, 0) + 1
            if not missing_is_failure:
                kept[solver_id] = kept.get(solver_id, 0) + 1
                continue
            reason = "no KKT residuals recorded"
        elif worst > tol:
            reason = f"worst relative residual {worst:.3e} > {tol:.1e}"
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
