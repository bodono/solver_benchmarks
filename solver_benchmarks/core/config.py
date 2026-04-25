"""Benchmark configuration loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
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
    auto_prepare_data: bool = False

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
            "auto_prepare_data": self.auto_prepare_data,
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


@dataclass(frozen=True)
class EnvironmentConfig:
    id: str
    python: str = "python"
    install: list[str] = field(default_factory=list)
    solvers: list[SolverConfig] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentRunConfig:
    run: RunConfig
    environments: list[EnvironmentConfig]


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = _load_mapping(path)
    return parse_run_config(raw, base_dir=path.parent)


def load_environment_run_config(path: str | Path) -> EnvironmentRunConfig:
    path = Path(path)
    raw = _load_mapping(path)
    return parse_environment_run_config(raw, base_dir=path.parent)


def parse_run_config(raw: dict[str, Any], base_dir: Path | None = None) -> RunConfig:
    run = raw.get("run", {})
    dataset = raw.get("dataset") or run.get("dataset")
    if not dataset:
        raise ValueError("Config must define run.dataset")

    solvers = _parse_solver_entries(raw.get("solvers", []), context="solver")
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
        auto_prepare_data=bool(
            run.get("auto_prepare_data", raw.get("auto_prepare_data", False))
        ),
    )


def parse_environment_run_config(
    raw: dict[str, Any],
    base_dir: Path | None = None,
) -> EnvironmentRunConfig:
    environments_raw = raw.get("environments")
    if not environments_raw:
        raise ValueError("Environment config must define environments")
    environments = []
    all_solver_ids: set[str] = set()
    for item in environments_raw:
        env_id = item.get("id")
        if not env_id:
            raise ValueError("Every environment entry must define id")
        solvers = _parse_solver_entries(
            item.get("solvers", []),
            context=f"environment {env_id!r} solver",
        )
        if not solvers:
            raise ValueError(f"Environment {env_id!r} must define at least one solver")
        for solver in solvers:
            if solver.id in all_solver_ids:
                raise ValueError(
                    f"Duplicate solver id across environments: {solver.id!r}"
                )
            all_solver_ids.add(solver.id)
        environments.append(
            EnvironmentConfig(
                id=str(env_id),
                python=str(item.get("python", "python")),
                install=[str(command) for command in item.get("install", [])],
                solvers=solvers,
                metadata=dict(item.get("metadata", {})),
            )
        )
    child_raw = dict(raw)
    child_raw.pop("environments", None)
    child_raw["solvers"] = [
        {
            "id": solver.id,
            "solver": solver.solver,
            "settings": solver.settings,
            "timeout_seconds": solver.timeout_seconds,
        }
        for solver in environments[0].solvers
    ]
    return EnvironmentRunConfig(
        run=parse_run_config(child_raw, base_dir=base_dir),
        environments=environments,
    )


def _parse_solver_entries(items: list[dict[str, Any]], *, context: str) -> list[SolverConfig]:
    solvers = []
    seen_ids: set[str] = set()
    for item in items:
        solver_name = item.get("solver")
        solver_id = item.get("id")
        if not solver_name or not solver_id:
            raise ValueError(f"Every {context} entry must define id and solver")
        expanded = _expand_solver_entry(item, context=context)
        for solver in expanded:
            if solver.id in seen_ids:
                raise ValueError(f"Duplicate solver id after sweep expansion: {solver.id!r}")
            seen_ids.add(solver.id)
            solvers.append(solver)
    return solvers


def _expand_solver_entry(item: dict[str, Any], *, context: str) -> list[SolverConfig]:
    solver_name = str(item["solver"])
    base_id = str(item["id"])
    base_settings = dict(item.get("settings", {}))
    timeout_seconds = item.get("timeout_seconds")
    sweep = item.get("sweep")
    if sweep is None:
        return [
            SolverConfig(
                id=base_id,
                solver=solver_name,
                settings=base_settings,
                timeout_seconds=timeout_seconds,
            )
        ]
    if not isinstance(sweep, dict) or not sweep:
        raise ValueError(f"{context} {base_id!r} sweep must be a non-empty mapping")
    keys = [str(key) for key in sweep.keys()]
    value_grid = [_sweep_values(base_id, key, sweep[key]) for key in keys]
    id_template = item.get("id_template")
    expanded = []
    for values in product(*value_grid):
        sweep_settings = dict(zip(keys, values))
        settings = {**base_settings, **sweep_settings}
        solver_id = (
            _render_solver_id_template(str(id_template), base_id, solver_name, settings)
            if id_template
            else _default_sweep_id(base_id, sweep_settings)
        )
        expanded.append(
            SolverConfig(
                id=solver_id,
                solver=solver_name,
                settings=settings,
                timeout_seconds=timeout_seconds,
            )
        )
    return expanded


def _sweep_values(base_id: str, key: str, values: Any) -> list[Any]:
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"Solver {base_id!r} sweep parameter {key!r} must be a non-empty list"
        )
    return values


def _render_solver_id_template(
    template: str,
    base_id: str,
    solver_name: str,
    settings: dict[str, Any],
) -> str:
    fields = {"id": base_id, "solver": solver_name, **settings}
    try:
        return template.format(**fields)
    except KeyError as exc:
        raise ValueError(f"id_template references unknown field {exc.args[0]!r}") from exc


def _default_sweep_id(base_id: str, sweep_settings: dict[str, Any]) -> str:
    parts = [base_id]
    for key, value in sweep_settings.items():
        parts.append(f"{key}={_format_sweep_value(value)}")
    return "__".join(parts)


def _format_sweep_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


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
