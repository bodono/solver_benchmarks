"""Benchmark configuration loading and normalization."""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any

# Identifiers flow into manifest fields, parquet columns, slugified
# directory names, and CSV file names. Restrict them to a portable
# charset so the same id round-trips cleanly across all of those.
# `=` is allowed because the default sweep id template formats values
# as ``key=value`` (e.g. ``scs__alpha=1e-6``).
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.+=-]+$")

# Keys each config section accepts. Anything else is a typo, which would
# otherwise be dropped silently (a misspelled ``timeout`` ran with no limit).
_RUN_KEYS = frozenset(
    {
        "name",
        "dataset",
        "datasets",
        "dataset_options",
        "output_dir",
        "include",
        "exclude",
        "parallelism",
        "resume",
        "max_worker_error_retries",
        "worker_memory_limit_mb",
        "timeout_seconds",
        "fail_on_unsupported",
        "auto_prepare_data",
    }
)
# Run keys may also sit at the top level, as in a manifest's config, which
# environment matrices hand to each child run (config_hash included).
_TOP_LEVEL_KEYS = _RUN_KEYS | {"run", "solvers", "config_hash"}
_DATASET_ENTRY_KEYS = frozenset(
    {"name", "dataset", "id", "dataset_options", "include", "exclude"}
)
_SOLVER_ENTRY_KEYS = frozenset(
    {"id", "solver", "settings", "timeout_seconds", "sweep", "id_template"}
)
_ENVIRONMENT_KEYS = frozenset({"id", "python", "install", "solvers", "metadata"})


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
    max_worker_error_retries: int = 2
    worker_memory_limit_mb: int | None = None

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
        return _stable_digest(payload, length=12)

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
            "max_worker_error_retries": self.max_worker_error_retries,
            "worker_memory_limit_mb": self.worker_memory_limit_mb,
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
    return parse_run_config(raw)


def load_environment_run_config(path: str | Path) -> EnvironmentRunConfig:
    path = Path(path)
    raw = _with_default_run_name(_load_mapping(path), path.stem)
    return parse_environment_run_config(raw)


def parse_run_config(raw: dict[str, Any]) -> RunConfig:
    """Parse and validate a run config mapping.

    Unknown keys, unregistered solver or dataset names, dataset options the
    dataset does not read and ids that would share an artifact directory
    raise ValueError. Empty (null) sections
    count as empty. Relative paths stay relative: the runner resolves
    ``output_dir`` and ``data_root`` against the repository root.
    """
    raw = _mapping(raw, "config")
    _check_keys(
        raw,
        _TOP_LEVEL_KEYS,
        "top-level",
        hint=(
            " Environment matrix configs run with `bench env run`."
            if "environments" in raw
            else ""
        ),
    )
    run = _mapping(raw.get("run"), "run")
    _check_keys(run, _RUN_KEYS, "run")
    datasets = _parse_datasets(raw, run)

    solvers = _parse_solver_entries(raw.get("solvers"), context="solver")
    if not solvers:
        raise ValueError("Config must define at least one solver")

    output_dir = _validate_output_dir(run.get("output_dir", raw.get("output_dir", "results")))

    include = _listify(run.get("include", raw.get("include", [])))
    exclude = _listify(run.get("exclude", raw.get("exclude", [])))

    return RunConfig(
        datasets=datasets,
        solvers=solvers,
        name=_optional_string(run.get("name", raw.get("name"))),
        output_dir=output_dir,
        include=include,
        exclude=exclude,
        parallelism=_validate_parallelism(
            run.get("parallelism", raw.get("parallelism", 1))
        ),
        resume=_validate_bool(run.get("resume", raw.get("resume", True)), "run.resume"),
        max_worker_error_retries=_validate_worker_error_retries(
            run.get("max_worker_error_retries", raw.get("max_worker_error_retries", 2))
        ),
        worker_memory_limit_mb=_validate_worker_memory_limit(
            run.get("worker_memory_limit_mb", raw.get("worker_memory_limit_mb"))
        ),
        timeout_seconds=_validate_timeout(
            run.get("timeout_seconds", raw.get("timeout_seconds")),
            context="run.timeout_seconds",
        ),
        fail_on_unsupported=_validate_bool(
            run.get("fail_on_unsupported", raw.get("fail_on_unsupported", False)),
            "run.fail_on_unsupported",
        ),
        auto_prepare_data=_validate_bool(
            run.get("auto_prepare_data", raw.get("auto_prepare_data", False)),
            "run.auto_prepare_data",
        ),
    )


def resolve_data_roots(
    datasets: list[DatasetConfig], base_dir: str | Path
) -> list[DatasetConfig]:
    """Make each relative ``data_root`` dataset option absolute under ``base_dir``.

    The runner passes the repository root, as for ``output_dir``. It lists
    problems in its own working directory but starts each worker in the
    repository root, so a relative path would name two different
    directories.
    """
    resolved = []
    for dataset in datasets:
        data_root = dataset.dataset_options.get("data_root")
        if data_root and not Path(data_root).is_absolute():
            dataset = replace(
                dataset,
                dataset_options={
                    **dataset.dataset_options,
                    "data_root": str((Path(base_dir) / data_root).resolve()),
                },
            )
        resolved.append(dataset)
    return resolved


def resolve_output_dir(config: RunConfig, root: str | Path) -> RunConfig:
    """Resolve a relative run output directory under the benchmark repo root."""
    if config.output_dir.is_absolute():
        return config
    return replace(
        config,
        output_dir=(Path(root).resolve() / config.output_dir).resolve(),
    )


def parse_environment_run_config(raw: dict[str, Any]) -> EnvironmentRunConfig:
    raw = _mapping(raw, "config")
    if "solvers" in raw:
        raise ValueError(
            "An environment matrix config takes its solvers from each environment: "
            "move the top-level `solvers` into environments[].solvers."
        )
    environments_raw = _entries(raw.get("environments"), "environments")
    if not environments_raw:
        raise ValueError("Environment config must define environments")
    environments = []
    all_solver_ids: set[str] = set()
    solver_slugs: dict[str, str] = {}
    environment_ids: set[str] = set()
    environment_slugs: dict[str, str] = {}
    for item in environments_raw:
        if not isinstance(item, dict):
            raise ValueError(
                f"Every environment entry must be a mapping, got {type(item).__name__}"
            )
        env_id = item.get("id")
        if not env_id:
            raise ValueError("Every environment entry must define id")
        _check_keys(item, _ENVIRONMENT_KEYS, f"environment {env_id!r}")
        # The id names the child run's config file in the run directory and
        # is recorded on every result row.
        _validate_identifier(str(env_id), context="environment id")
        if str(env_id) in environment_ids:
            raise ValueError(f"Duplicate environment id: {str(env_id)!r}")
        environment_ids.add(str(env_id))
        _check_slug_unique(str(env_id), environment_slugs, "environment")
        solvers = _parse_solver_entries(
            item.get("solvers"),
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
            _check_slug_unique(solver.id, solver_slugs, "solver")
        environments.append(
            EnvironmentConfig(
                id=str(env_id),
                python=str(item.get("python") or "python"),
                install=[str(command) for command in _entries(item.get("install"), "install")],
                solvers=solvers,
                metadata=dict(_mapping(item.get("metadata"), "metadata")),
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
        run=parse_run_config(child_raw),
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

    dataset_options = _canonicalize(
        dict(_mapping(run.get("dataset_options", raw.get("dataset_options")), "dataset_options"))
    )
    _check_registered_dataset(str(single_name))
    _check_dataset_options(str(single_name), dataset_options)
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
    # Canonicalize dataset_options at parse time so values like Path or
    # set survive the manifest's json.dumps round-trip. Without this,
    # programmatic configs with ``data_root=Path(...)`` would parse and
    # hash successfully but later crash ``ResultStore.create()`` with
    # ``TypeError: Object of type PosixPath is not JSON serializable``.
    default_options = _canonicalize(
        dict(
            _mapping(
                defaults.get("dataset_options", root.get("dataset_options")),
                "dataset_options",
            )
        )
    )
    parsed: list[DatasetConfig] = []
    seen_ids: set[str] = set()
    seen_names_no_id: set[str] = set()
    seen_slugs: dict[str, str] = {}
    # Run-level defaults go to the listed datasets that read them; one that
    # no listed dataset reads is an error (below).
    defaults_taken: set[str] = set()
    for entry in items:
        explicit_id: str | None = None
        entry_options: dict[str, Any] = {}
        if isinstance(entry, str):
            name = entry
            include: list[str] = []
            exclude: list[str] = []
        elif isinstance(entry, dict):
            name = entry.get("name") or entry.get("dataset")
            if not name:
                raise ValueError(
                    "Every dataset entry must define `name` (or legacy `dataset`)"
                )
            _check_keys(entry, _DATASET_ENTRY_KEYS, f"dataset entry {name!r}")
            raw_id = entry.get("id")
            explicit_id = str(raw_id) if raw_id is not None else None
            entry_options = _canonicalize(
                dict(_mapping(entry.get("dataset_options"), f"dataset {name!r} dataset_options"))
            )
            include = _listify(entry.get("include", []))
            exclude = _listify(entry.get("exclude", []))
        else:
            raise ValueError(
                f"Dataset entry must be a string or mapping, got {type(entry).__name__}"
            )
        name = str(name)
        _check_registered_dataset(name)
        defaults = _defaults_read_by(name, default_options)
        defaults_taken |= set(defaults)
        dataset_options = {**defaults, **entry_options}
        _check_dataset_options(name, dataset_options)
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
        _check_slug_unique(entry_id, seen_slugs, "dataset")
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
    unused = sorted(set(default_options) - defaults_taken)
    if unused:
        names = {str(entry.name) for entry in parsed}
        readable = set().union(*(_all_options(name) for name in names))
        described = []
        for key in unused:
            close = close_names(key, readable)
            described.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            f"run.dataset_options: {', '.join(described)} "
            f"{'is' if len(unused) == 1 else 'are'} not an option of any listed dataset "
            f"({', '.join(sorted(names))}). Run-level dataset options are defaults for "
            "the listed datasets that read them."
        )
    return parsed


def _all_options(name: str) -> set[str]:
    from solver_benchmarks.datasets.base import GENERIC_OPTIONS
    from solver_benchmarks.datasets.registry import DATASETS

    return set(GENERIC_OPTIONS | (getattr(DATASETS[name], "option_names", None) or frozenset()))


def _defaults_read_by(name: str, default_options: dict[str, Any]) -> dict[str, Any]:
    """The run-level default options that dataset ``name`` reads."""
    from solver_benchmarks.datasets.registry import DATASETS

    reads = getattr(DATASETS[name], "options_it_reads", None)
    return dict(default_options) if reads is None else reads(default_options)


def _parse_solver_entries(items: Any, *, context: str) -> list[SolverConfig]:
    solvers = []
    seen_ids: set[str] = set()
    seen_slugs: dict[str, str] = {}
    for item in _entries(items, "solvers"):
        if not isinstance(item, dict):
            raise ValueError(
                f"Every {context} entry must be a mapping with id and solver, "
                f"got {type(item).__name__}: {item!r}"
            )
        solver_name = item.get("solver")
        solver_id = item.get("id")
        if not solver_name or not solver_id:
            raise ValueError(f"Every {context} entry must define id and solver")
        _check_keys(item, _SOLVER_ENTRY_KEYS, f"{context} {solver_id!r}")
        _validate_identifier(str(solver_id), context=f"{context} id")
        _check_registered_solver(str(solver_name))
        expanded = _expand_solver_entry(item, context=context)
        for solver in expanded:
            _validate_identifier(solver.id, context=f"{context} id")
            if solver.id in seen_ids:
                raise ValueError(f"Duplicate solver id after sweep expansion: {solver.id!r}")
            seen_ids.add(solver.id)
            _check_slug_unique(solver.id, seen_slugs, "solver")
            solvers.append(solver)
    return solvers


def _expand_solver_entry(item: dict[str, Any], *, context: str) -> list[SolverConfig]:
    solver_name = str(item["solver"])
    base_id = str(item["id"])
    # Deep-copy nested settings so two SolverConfigs produced from the same
    # entry (e.g. via sweep expansion) do not share mutable nested dicts.
    base_settings_raw = copy.deepcopy(item.get("settings") or {})
    if not isinstance(base_settings_raw, dict):
        raise ValueError(f"{context} {base_id!r} settings must be a mapping")
    # Canonicalize settings (Path -> str, set -> sorted list, etc.) at
    # parse time so they survive the manifest's json.dumps round-trip.
    # config_hash already runs the same canonicalizer before hashing, so
    # this also ensures the SolverConfig.settings dict that flows into
    # adapters is the same shape as the one we hash on.
    base_settings = _canonicalize(base_settings_raw)
    timeout_seconds = _validate_timeout(
        item.get("timeout_seconds"),
        context=f"{context} {base_id!r} timeout_seconds",
    )
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
    id_template = item.get("id_template")
    value_grid = [
        _sweep_values(base_id, key, sweep[key], require_scalar=id_template is None)
        for key in keys
    ]
    expanded = []
    for values in product(*value_grid):
        sweep_settings = _canonicalize(dict(zip(keys, values)))
        # Re-deepcopy on each expansion so the underlying SolverConfig
        # entries cannot share mutable nested settings.
        settings = {**copy.deepcopy(base_settings), **copy.deepcopy(sweep_settings)}
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


def _sweep_values(
    base_id: str,
    key: str,
    values: Any,
    *,
    require_scalar: bool,
) -> list[Any]:
    if not isinstance(values, list) or not values:
        raise ValueError(
            f"Solver {base_id!r} sweep parameter {key!r} must be a non-empty list"
        )
    if require_scalar:
        # When the user is relying on the default _format_sweep_value-based
        # id, every sweep value must produce a unique scalar slug. Reject
        # nested values up front rather than silently producing
        # "key=[1, 2]"-style ids that flow into directory names.
        for value in values:
            if not _is_scalar_sweep_value(value):
                raise ValueError(
                    f"Solver {base_id!r} sweep parameter {key!r} contains a "
                    f"non-scalar value {value!r}; provide an `id_template` "
                    "to use non-scalar sweep values."
                )
    # Validate JSON-serializability so manifest writes never silently
    # truncate or lose information about the sweep settings.
    try:
        json.dumps(values, default=_json_default)
    except TypeError as exc:
        raise ValueError(
            f"Solver {base_id!r} sweep parameter {key!r} contains a "
            f"non-JSON-serializable value: {exc}"
        ) from exc
    return values


def _is_scalar_sweep_value(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
    except (IndexError, AttributeError, TypeError, ValueError) as exc:
        # A positional field ({0}), an attribute or index of a value
        # ({x.y}, {x[0]}), or a format spec the value does not take.
        raise ValueError(
            f"id_template {template!r} cannot be rendered: {exc}. Use only named "
            "fields: id, solver and the settings."
        ) from exc


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
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            data = json.loads(text)
        else:
            try:
                data = yaml.load(text, Loader=_config_loader())
            except yaml.YAMLError as exc:
                raise ValueError(f"Could not parse config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return data


_YAML_LOADER: Any = None


def _config_loader() -> Any:
    """A SafeLoader that reads floats the YAML 1.2 way.

    PyYAML follows YAML 1.1, where a float needs a dot and a signed
    exponent: ``1e-6`` and ``1.0e6`` load as strings and reach the solver
    as such, failing every solve. YAML 1.2 (and JSON) read them as floats.
    """
    global _YAML_LOADER
    if _YAML_LOADER is None:
        import yaml  # type: ignore

        class ConfigLoader(yaml.SafeLoader):
            pass

        ConfigLoader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        _YAML_LOADER = ConfigLoader
    return _YAML_LOADER


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


def _mapping(value: Any, context: str) -> dict[str, Any]:
    """A config section that must be a mapping; null (an empty section) is ``{}``."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"`{context}` must be a mapping, got {type(value).__name__}: {value!r}"
        )
    return value


def _entries(value: Any, context: str) -> list[Any]:
    """A config section that must be a list; null (an empty section) is ``[]``."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"`{context}` must be a list of entries, got {type(value).__name__}: {value!r}"
        )
    return value


def _check_keys(
    mapping: dict[str, Any], allowed: frozenset[str], context: str, *, hint: str = ""
) -> None:
    unknown = sorted(str(key) for key in mapping if str(key) not in allowed)
    if not unknown:
        return
    described = []
    for key in unknown:
        close = difflib.get_close_matches(key, sorted(allowed), n=1)
        described.append(f"{key!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
    raise ValueError(
        f"Unknown {context} key{'s' if len(unknown) > 1 else ''}: "
        f"{', '.join(described)}. Known keys: {', '.join(sorted(allowed))}.{hint}"
    )


def _check_registered_solver(name: str) -> None:
    # Imported here: the solver adapters import this package's core modules.
    from solver_benchmarks.solvers.registry import SOLVERS

    if name.lower() not in SOLVERS:
        raise ValueError(_unknown_name_message("solver", name, SOLVERS))


def _check_registered_dataset(name: str) -> None:
    from solver_benchmarks.datasets.registry import DATASETS

    if name not in DATASETS:
        raise ValueError(_unknown_name_message("dataset", name, DATASETS))


def close_names(name: str, known: Iterable[str], n: int = 1) -> list[str]:
    """The names in ``known`` that ``name`` was probably meant to be.

    Names equal to it but for case come first (problem names differ in case
    between datasets: NETLIB is lower case, Maros-Meszaros upper case), then
    ``difflib``'s close matches, compared case-insensitively.
    """
    candidates = sorted(set(known))
    same = [candidate for candidate in candidates if candidate.lower() == name.lower()]
    if same:
        return same[:n]
    by_lower: dict[str, str] = {}
    for candidate in candidates:
        by_lower.setdefault(candidate.lower(), candidate)
    return [by_lower[match] for match in difflib.get_close_matches(name.lower(), list(by_lower), n=n)]


def unknown_name_message(kind: str, name: str, known: Iterable[str]) -> str:
    """``Unknown <kind> <name>.`` with a suggestion and the known names."""
    return _unknown_name_message(kind, name, {key: None for key in known})


def _check_dataset_options(name: str, options: dict[str, Any]) -> None:
    """Raise ValueError for an option the dataset does not read (see Dataset.validate_options)."""
    from solver_benchmarks.datasets.registry import DATASETS

    validate = getattr(DATASETS[name], "validate_options", None)
    if validate is not None:
        validate(options)


def _unknown_name_message(kind: str, name: str, known: Iterable[str]) -> str:
    close = close_names(name, known)
    suggestion = f" Did you mean {close[0]!r}?" if close else ""
    return (
        f"Unknown {kind} {name!r}.{suggestion} Available: {', '.join(sorted(known))} "
        f"(see `bench list {kind}s`)."
    )


def slugify(value: str) -> str:
    """The directory name for a run, dataset, problem or solver id.

    Ids may use ``+`` and ``=`` (sweep ids such as ``scs__scale=1e+06``),
    and both are safe in file names, so they are kept: mapping them to ``-``
    made ``scale=1e-06`` and ``scale=1e+06`` share one directory.
    """
    value = re.sub(r"[^A-Za-z0-9_.+=-]+", "-", value.strip())
    return value.strip("-") or "run"


def _check_slug_unique(entry_id: str, seen: dict[str, str], kind: str) -> None:
    """Reject ids whose artifact directories would coincide.

    Compared case-insensitively, since macOS and Windows file systems are.
    """
    slug = slugify(entry_id).lower()
    other = seen.get(slug)
    if other is not None and other != entry_id:
        raise ValueError(
            f"{kind.capitalize()} ids {other!r} and {entry_id!r} would share the "
            f"artifact directory {slugify(entry_id)!r}; rename one of them."
        )
    seen[slug] = entry_id


def _listify(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise ValueError(
        f"Expected a string or list, got {type(value).__name__}: {value!r}"
    )


def _validate_identifier(value: str, *, context: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"{context} {value!r} contains characters outside [A-Za-z0-9_.+=-]; "
            "use only those characters so the id round-trips through "
            "filesystem paths and CSV column names."
        )


def _validate_output_dir(value: Any) -> Path:
    if isinstance(value, Path):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"run.output_dir must be a directory path, got {value!r}; "
            "leave it out to use the default 'results'."
        )
    return Path(value)


def _validate_worker_error_retries(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("run.max_worker_error_retries must be an integer >= 0")
    return value


def _validate_parallelism(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"run.parallelism must be an integer >= 1, got bool: {value!r}")
    if isinstance(value, int):
        coerced = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"run.parallelism must be an integer >= 1, got {value!r}")
        coerced = int(value)
    else:
        try:
            coerced = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"run.parallelism must be an integer >= 1, got {value!r}"
            ) from exc
    if coerced < 1:
        raise ValueError(f"run.parallelism must be >= 1, got {coerced!r}")
    return coerced


def _validate_worker_memory_limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"run.worker_memory_limit_mb must be an integer >= 1 or null, got {value!r}"
        )
    return value


def _validate_bool(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{context} must be a boolean, got {value!r}")


def _validate_timeout(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    # Reject bool explicitly: ``True`` / ``False`` would otherwise pass
    # the float() coercion as 1.0 / 0.0 and silently disable timeout
    # behavior on a typo.
    if isinstance(value, bool):
        raise ValueError(
            f"{context} must be a number or null, got bool: {value!r}"
        )
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} must be a number or null, got {type(value).__name__}: {value!r}"
        ) from exc
    # nan / +inf / -inf are accepted by float() but would silently
    # disable the timeout (compares as not-< 0 but later math.isfinite
    # gates would treat them as "no limit"). Reject them up front.
    if not math.isfinite(coerced):
        raise ValueError(f"{context} must be finite (got {coerced!r})")
    if coerced < 0:
        raise ValueError(f"{context} must be >= 0 (got {coerced!r})")
    return coerced


def _stable_digest(payload: Any, *, length: int) -> str:
    encoded = json.dumps(
        _canonicalize(payload), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def _canonicalize(value: Any) -> Any:
    """Recursively normalize a config payload for stable hashing.

    - dict -> dict with sorted, str keys; each value canonicalized.
    - set/frozenset -> sorted list (deterministic order).
    - list/tuple -> list of canonicalized items.
    - Path -> str.
    - other primitives unchanged.
    """
    if isinstance(value, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_canonicalize(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


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


# Dataset options that only say where a dataset's files are or which of its
# problems are listed. They never change a problem that is solved, so they
# stay out of the resume signature: pointing --data-root at a copy of the data
# or raising a size cap does not invalidate results already recorded. Each
# dataset adds the options that only select its problems
# (``Dataset.selection_options``, such as ``subset``), so widening a subset
# does not invalidate them either.
_LOCATION_OPTIONS = frozenset({"data_root", "max_size_mb"})


def solve_signature(
    config: RunConfig,
    dataset: DatasetConfig,
    solver: SolverConfig,
    *,
    legacy: bool = False,
    legacy_selection: bool = False,
) -> str:
    """Return the resume identity for one dataset/solver solve family.

    The signature intentionally ignores run bookkeeping fields such as
    name, output_dir, parallelism, resume, and reporting flags, the
    dataset options in ``_LOCATION_OPTIONS`` and the dataset's selection
    options. It covers the pieces that can change the actual mathematical
    problem or solver behavior for an already-planned
    ``(dataset, problem, solver_id)`` row.

    Earlier releases computed it in two other forms, and rows they wrote
    carry those, so resume accepts all three. ``legacy=True`` returns the
    first form, which hashed every dataset option, the location options
    included. ``legacy_selection=True`` returns the second, which left out
    the location options but hashed the selection options.
    """
    return _solve_signature(
        dataset_id=str(dataset.id),
        dataset_name=dataset.name,
        dataset_options=dataset.dataset_options,
        solver_id=solver.id,
        solver_name=solver.solver,
        settings=solver.settings,
        effective_timeout=_effective_timeout(config, solver),
        legacy=legacy,
        legacy_selection=legacy_selection,
    )


def solve_signatures(
    config: RunConfig, *, legacy: bool = False, legacy_selection: bool = False
) -> dict[tuple[str, str], str]:
    return {
        (str(dataset.id), solver.id): solve_signature(
            config, dataset, solver, legacy=legacy, legacy_selection=legacy_selection
        )
        for dataset in config.datasets
        for solver in config.solvers
    }


def manifest_solve_signatures(
    config: dict[str, Any], *, legacy: bool = False, legacy_selection: bool = False
) -> dict[tuple[str, str], str]:
    """Return solve signatures for a manifest ``config`` payload.

    The same signatures ``solve_signatures`` gives for the parsed config.
    """
    run_timeout = config.get("timeout_seconds")
    signatures: dict[tuple[str, str], str] = {}
    for dataset in manifest_dataset_entries(config):
        dataset_id = str(dataset["id"])
        for solver in config.get("solvers", []) or []:
            solver_id = str(solver["id"])
            solver_timeout = solver.get("timeout_seconds")
            signatures[(dataset_id, solver_id)] = _solve_signature(
                dataset_id=dataset_id,
                dataset_name=str(dataset["name"]),
                dataset_options=dict(dataset.get("dataset_options") or {}),
                solver_id=solver_id,
                solver_name=str(solver["solver"]),
                settings=dict(solver.get("settings") or {}),
                effective_timeout=(
                    solver_timeout if solver_timeout is not None else run_timeout
                ),
                legacy=legacy,
                legacy_selection=legacy_selection,
            )
    return signatures


def _solve_signature(
    *,
    dataset_id: str,
    dataset_name: str,
    dataset_options: dict[str, Any],
    solver_id: str,
    solver_name: str,
    settings: dict[str, Any],
    effective_timeout: Any,
    legacy: bool,
    legacy_selection: bool = False,
) -> str:
    return _stable_digest(
        {
            "dataset": {
                "id": dataset_id,
                "name": dataset_name,
                "dataset_options": _signature_dataset_options(
                    dataset_name,
                    dataset_options,
                    legacy=legacy,
                    legacy_selection=legacy_selection,
                ),
            },
            "solver": {
                "id": solver_id,
                "solver": solver_name,
                "settings": settings,
                "effective_timeout_seconds": effective_timeout,
            },
        },
        length=24,
    )


def _signature_dataset_options(
    name: str,
    options: dict[str, Any],
    *,
    legacy: bool = False,
    legacy_selection: bool = False,
) -> dict[str, Any]:
    """Dataset options as the dataset wants them hashed (see Dataset.signature_options)."""
    # Imported here: the dataset adapters import this package's core modules.
    from solver_benchmarks.datasets.registry import DATASETS, get_dataset

    # An error importing the loader (a KeyError included) is raised as it is.
    dataset_cls: type[Any] | None = get_dataset(name) if name in DATASETS else None
    hashed = dict(options)
    if not legacy:
        left_out = set(_LOCATION_OPTIONS)
        if not legacy_selection:
            left_out |= set(getattr(dataset_cls, "selection_options", ()))
            # Names the dataset does not read. A config cannot hold them any
            # more, but a run directory written before options were checked
            # can, and the dataset ignored them: the rows its manifest maps to
            # this form then stay complete once the option is removed.
            read = getattr(dataset_cls, "option_names", None)
            if read is not None:
                left_out |= {key for key in hashed if key not in read}
        for key in left_out:
            hashed.pop(key, None)
    if dataset_cls is None:
        return hashed
    # Registered adapters need not derive from Dataset; one without the hook
    # hashes its options as given.
    hook = getattr(dataset_cls, "signature_options", None)
    return hashed if hook is None else hook(hashed)


def _effective_timeout(config: RunConfig, solver: SolverConfig) -> float | None:
    if solver.timeout_seconds is not None:
        return solver.timeout_seconds
    return config.timeout_seconds
