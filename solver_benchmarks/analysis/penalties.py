"""Resolve fixed comparison time limits from persisted run provenance."""

from __future__ import annotations

import json
import math
from pathlib import Path

from solver_benchmarks.analysis.profiles import TIME_METRICS
from solver_benchmarks.core.config import manifest_dataset_entries


def geomean_time_limits(
    run_dir: str | Path,
    *,
    metric: str,
    max_value: float | None = None,
    penalize_failures: bool = True,
) -> float | dict[str, float] | None:
    """Read limits without depending on results, solver selection, or source files.

    Explicit penalties and success-only/non-time calculations need no manifest
    time limit. Merges retain the complete source manifests; their display
    config inherits one source's timeout and cannot describe mixed limits.
    """
    if max_value is not None or not penalize_failures or metric not in TIME_METRICS:
        return None
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        raise ValueError("No manifest time limit is available; pass --max-value explicitly")
    leaves = _source_configs(json.loads(path.read_text()))
    limits: dict[str, float] = {}
    distinct: set[float] = set()
    unnamed = False
    for config in leaves:
        raw = config.get("timeout_seconds")
        try:
            limit = float(raw)
        except (TypeError, ValueError):
            raise ValueError("A manifest has no valid timeout_seconds; pass --max-value explicitly")
        if isinstance(raw, bool) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("Manifest timeout_seconds must be positive and finite; pass --max-value explicitly")
        distinct.add(limit)
        entries = manifest_dataset_entries(config)
        unnamed |= not bool(entries)
        for entry in entries:
            dataset = entry["id"]
            if dataset in limits and limits[dataset] != limit:
                raise ValueError(
                    f"Conflicting manifest time limits for dataset {dataset!r}; "
                    "pass --max-value explicitly"
                )
            limits[dataset] = limit
    if len(distinct) == 1:
        return distinct.pop()
    if not distinct or unnamed:
        raise ValueError("Cannot assign manifest time limits to datasets; pass --max-value explicitly")
    return limits


def _source_configs(manifest: dict) -> list[dict]:
    derived = manifest.get("derived") or {}
    sources = derived.get("source_manifests")
    if sources:
        return [config for source in sources for config in _source_configs(source)]
    if derived.get("kind") == "merge" or (
        derived.get("kind") == "kkt_verify" and derived.get("selections")
    ):
        # Older verified merges discard embedded source limits. The first
        # source's display timeout is not a safe replacement; archived source
        # paths may also be absent or refer to unrelated current files.
        raise ValueError("Merged run source time limits were not preserved; pass --max-value explicitly")
    return [manifest.get("config") or {}]
