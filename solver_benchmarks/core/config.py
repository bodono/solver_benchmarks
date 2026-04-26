"""Benchmark configuration loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
class DatasetConfig:
    name: str
    dataset_options: dict[str, Any] = field(default_factory=dict)
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    id: str | None = None

    def __post_init__(self) -> None:
        # ``id`` is the per-entry identity used for results/resume keys and
        # artifact directories; ``name`` is the registry lookup. Defaulting
        # to ``name`` keeps the common single-entry case unchanged while
        # still allowing two entries that share a registry name to coexist
        # under different ids.
        if self.id is None:
            object.__setattr__(self, "id", self.name)


@dataclass(frozen=True)
class RunConfig:
    datasets: list[DatasetConfig]
    solvers: list[SolverConfig]
    name: str | None = None
    output_dir: Path = Path("results")
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    parallelism: int = 1
    resume: bool = True
    timeout_seconds: float | None = None
    fail_on_unsupported: bool = False
    auto_prepare_data: bool = False

    @property
    def dataset(self) -> str:
        """Single-dataset name; raises when configured for multiple datasets.

        Kept as a convenience for the common case so simple call-sites and
        existing tests do not need to drill into ``datasets[0].name``.
        Multi-dataset callers must iterate ``self.datasets`` directly.
        """
        if len(self.datasets) != 1:
            raise ValueError(
                f"RunConfig has {len(self.datasets)} datasets; use `datasets` instead."
            )
        return self.datasets[0].name

    @property
    def dataset_options(self) -> dict[str, Any]:
        if len(self.datasets) != 1:
            raise ValueError(
                f"RunConfig has {len(self.datasets)} datasets; use `datasets` instead."
            )
        return dict(self.datasets[0].dataset_options)

    def effective_filters(self, dataset: DatasetConfig) -> tuple[list[str], list[str]]:
        """Resolve ``(include, exclude)`` to apply for ``dataset``.

        A dataset-level ``include`` replaces the run-level include for that
        dataset; this lets users say "for netlib only run afiro" while still
        keeping a different include set for another dataset. Excludes are
        unioned: a name listed at either level is dropped.
        """
        include = list(dataset.include) if dataset.include else list(self.include)
        exclude = sorted(set(self.exclude) | set(dataset.exclude))
        return include, exclude

    @property
    def config_hash(self) -> str:
        payload = {
            "datasets": [
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "dataset_options": dataset.dataset_options,
                    "include": dataset.include,
                    "exclude": dataset.exclude,
                }
                for dataset in self.datasets
            ],
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
            "name": self.name,
            "datasets": [
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "dataset_options": dataset.dataset_options,
                    "include": dataset.include,
                    "exclude": dataset.exclude,
                }
                for dataset in self.datasets
            ],
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
    raw = _with_default_run_name(_load_mapping(path), path.stem)
    return parse_run_config(raw, base_dir=path.parent)


def load_environment_run_config(path: str | Path) -> EnvironmentRunConfig:
    path = Path(path)
    raw = _with_default_run_name(_load_mapping(path), path.stem)
    return parse_environment_run_config(raw, base_dir=path.parent)


def parse_run_config(raw: dict[str, Any], base_dir: Path | None = None) -> RunConfig:
    run = raw.get("run", {})
    datasets = _parse_datasets(raw, run)

    solvers = _parse_solver_entries(raw.get("solvers", []), context="solver")
    if not solvers:
        raise ValueError("Config must define at least one solver")

    output_dir = Path(run.get("output_dir", raw.get("output_dir", "results")))

    include = _listify(run.get("include", raw.get("include", [])))
    exclude = _listify(run.get("exclude", raw.get("exclude", [])))

    return RunConfig(
        datasets=datasets,
        solvers=solvers,
        name=_optional_string(run.get("name", raw.get("name"))),
        output_dir=output_dir,
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


def resolve_output_dir(config: RunConfig, root: str | Path) -> RunConfig:
    """Resolve a relative run output directory under the benchmark repo root."""
    if config.output_dir.is_absolute():
        return config
    return replace(
        config,
        output_dir=(Path(root).resolve() / config.output_dir).resolve(),
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


def _parse_datasets(raw: dict[str, Any], run: dict[str, Any]) -> list[DatasetConfig]:
    datasets_raw = run.get("datasets", raw.get("datasets"))
    single_name = run.get("dataset", raw.get("dataset"))

    if datasets_raw is not None and single_name:
        raise ValueError(
            "Config must define either `dataset` (single) or `datasets` (list), not both."
        )

    if datasets_raw is not None:
        if not isinstance(datasets_raw, list) or not datasets_raw:
            raise ValueError("`datasets` must be a non-empty list")
        return _parse_dataset_entries(datasets_raw, defaults=run, root=raw)

    if not single_name:
        raise ValueError("Config must define `dataset` or `datasets`")

    dataset_options = dict(run.get("dataset_options", raw.get("dataset_options", {})))
    return [
        DatasetConfig(
            name=str(single_name),
            dataset_options=dataset_options,
            include=[],
            exclude=[],
        )
    ]


def _parse_dataset_entries(
    items: list[Any],
    *,
    defaults: dict[str, Any],
    root: dict[str, Any],
) -> list[DatasetConfig]:
    default_options = dict(defaults.get("dataset_options", root.get("dataset_options", {})))
    parsed: list[DatasetConfig] = []
    seen_ids: set[str] = set()
    seen_names_no_id: set[str] = set()
    for entry in items:
        explicit_id: str | None = None
        if isinstance(entry, str):
            name = entry
            dataset_options: dict[str, Any] = dict(default_options)
            include: list[str] = []
            exclude: list[str] = []
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("dataset")
            if not name:
                raise ValueError(
                    "Every dataset entry must define `name` (or legacy `dataset`)"
                )
            raw_id = entry.get("id")
            explicit_id = str(raw_id) if raw_id is not None else None
            dataset_options = {
                **default_options,
                **dict(entry.get("dataset_options", {})),
            }
            include = _listify(entry.get("include", []))
            exclude = _listify(entry.get("exclude", []))
        else:
            raise ValueError(
                f"Dataset entry must be a string or mapping, got {type(entry).__name__}"
            )
        name = str(name)
        entry_id = explicit_id if explicit_id is not None else name
        if entry_id in seen_ids:
            if explicit_id is None:
                raise ValueError(
                    f"Duplicate dataset name {name!r}: include each adapter once, "
                    "or give each occurrence an explicit `id` so its results can be "
                    "told apart."
                )
            raise ValueError(f"Duplicate dataset id in datasets list: {entry_id!r}")
        # Surface a clearer message when a user lists the same adapter twice
        # without ids on either entry.
        if explicit_id is None and name in seen_names_no_id:
            raise ValueError(
                f"Duplicate dataset name {name!r}: include each adapter once, "
                "or give each occurrence an explicit `id` so its results can be "
                "told apart."
            )
        seen_ids.add(entry_id)
        if explicit_id is None:
            seen_names_no_id.add(name)
        parsed.append(
            DatasetConfig(
                name=name,
                dataset_options=dataset_options,
                include=include,
                exclude=exclude,
                id=explicit_id,
            )
        )
    return parsed


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


def _with_default_run_name(raw: dict[str, Any], default_name: str) -> dict[str, Any]:
    if raw.get("name") is not None:
        return raw
    run = raw.get("run")
    if isinstance(run, dict) and run.get("name") is not None:
        return raw
    updated = dict(raw)
    updated_run = dict(run) if isinstance(run, dict) else {}
    updated_run["name"] = default_name
    updated["run"] = updated_run
    return updated


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def manifest_dataset_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a manifest's dataset section into a list of entries.

    Reads either the new ``datasets: [...]`` shape or the legacy
    ``dataset: name`` + ``dataset_options`` + ``include`` / ``exclude``
    shape, and returns a uniform list of dicts with keys
    ``id``, ``name``, ``dataset_options``, ``include``, ``exclude``. Each
    entry's ``id`` defaults to its ``name``; the same registry ``name`` may
    appear multiple times when distinct ids are given. Run-level
    ``include`` / ``exclude`` are merged in so callers do not need to
    apply them separately.
    """
    run_include = list(config.get("include") or [])
    run_exclude = list(config.get("exclude") or [])
    datasets = config.get("datasets")
    if isinstance(datasets, list) and datasets:
        out: list[dict[str, Any]] = []
        for entry in datasets:
            include = list(entry.get("include") or []) or list(run_include)
            exclude = sorted({*run_exclude, *(entry.get("exclude") or [])})
            name = str(entry["name"])
            entry_id = str(entry.get("id") or name)
            out.append(
                {
                    "id": entry_id,
                    "name": name,
                    "dataset_options": dict(entry.get("dataset_options") or {}),
                    "include": include,
                    "exclude": exclude,
                }
            )
        return out
    name = config.get("dataset")
    if not name:
        return []
    return [
        {
            "id": str(name),
            "name": str(name),
            "dataset_options": dict(config.get("dataset_options") or {}),
            "include": list(run_include),
            "exclude": list(run_exclude),
        }
    ]
