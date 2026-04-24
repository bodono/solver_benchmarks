"""Benchmark configuration loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hashlib
import json


@dataclass(frozen=True)
class SolverConfig:
    id: str
    solver: str
    settings: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    solvers: list[SolverConfig]
    output_dir: Path = Path("runs")
    dataset_options: dict[str, Any] = field(default_factory=dict)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    parallelism: int = 1
    resume: bool = True
    timeout_seconds: float | None = None
    fail_on_unsupported: bool = False

    @property
    def config_hash(self) -> str:
        payload = {
            "dataset": self.dataset,
            "dataset_options": self.dataset_options,
            "solvers": [
                {
                    "id": solver.id,
                    "solver": solver.solver,
                    "settings": solver.settings,
                    "timeout_seconds": solver.timeout_seconds,
                }
                for solver in self.solvers
            ],
            "include": self.include,
            "exclude": self.exclude,
            "timeout_seconds": self.timeout_seconds,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()[:12]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_options": self.dataset_options,
            "output_dir": str(self.output_dir),
            "include": self.include,
            "exclude": self.exclude,
            "parallelism": self.parallelism,
            "resume": self.resume,
            "timeout_seconds": self.timeout_seconds,
            "fail_on_unsupported": self.fail_on_unsupported,
            "config_hash": self.config_hash,
            "solvers": [
                {
                    "id": solver.id,
                    "solver": solver.solver,
                    "settings": solver.settings,
                    "timeout_seconds": solver.timeout_seconds,
                }
                for solver in self.solvers
            ],
        }


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = _load_mapping(path)
    return parse_run_config(raw, base_dir=path.parent)


def parse_run_config(raw: dict[str, Any], base_dir: Path | None = None) -> RunConfig:
    run = raw.get("run", {})
    dataset = raw.get("dataset") or run.get("dataset")
    if not dataset:
        raise ValueError("Config must define run.dataset")

    solvers = []
    for item in raw.get("solvers", []):
        solver_name = item.get("solver")
        solver_id = item.get("id")
        if not solver_name or not solver_id:
            raise ValueError("Every solver entry must define id and solver")
        solvers.append(
            SolverConfig(
                id=str(solver_id),
                solver=str(solver_name),
                settings=dict(item.get("settings", {})),
                timeout_seconds=item.get("timeout_seconds"),
            )
        )
    if not solvers:
        raise ValueError("Config must define at least one solver")

    output_dir = Path(run.get("output_dir", raw.get("output_dir", "runs")))
    if base_dir is not None and not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    include = _listify(run.get("include", raw.get("include", [])))
    exclude = _listify(run.get("exclude", raw.get("exclude", [])))
    dataset_options = dict(run.get("dataset_options", raw.get("dataset_options", {})))

    return RunConfig(
        dataset=str(dataset),
        solvers=solvers,
        output_dir=output_dir,
        dataset_options=dataset_options,
        include=include,
        exclude=exclude,
        parallelism=int(run.get("parallelism", raw.get("parallelism", 1))),
        resume=bool(run.get("resume", raw.get("resume", True))),
        timeout_seconds=run.get("timeout_seconds", raw.get("timeout_seconds")),
        fail_on_unsupported=bool(
            run.get("fail_on_unsupported", raw.get("fail_on_unsupported", False))
        ),
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return json.loads(text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return data


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]
