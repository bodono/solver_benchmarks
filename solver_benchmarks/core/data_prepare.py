"""Helpers for data-preparation command selection and messages."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from solver_benchmarks.core.config import DatasetConfig, RunConfig


def data_prepare_selection(
    config: RunConfig,
    dataset_config: DatasetConfig,
) -> tuple[list[str] | None, bool]:
    """Return ``(problem_names, all_problems)`` for auto-preparing a dataset."""
    include, _ = config.effective_filters(dataset_config)
    if include:
        return list(include), False

    subset = dataset_config.dataset_options.get("subset")
    if isinstance(subset, str):
        normalized = subset.strip()
        if normalized.lower() == "all":
            return None, True
        if "," in normalized:
            return [item.strip() for item in normalized.split(",") if item.strip()], False
    if isinstance(subset, (list, tuple, set)):
        return [str(item) for item in subset], False
    return None, False


def data_prepare_command(
    dataset_config: DatasetConfig,
    *,
    problem_names: list[str] | None = None,
    all_problems: bool = False,
    repo_root: str | Path | None = None,
) -> str:
    parts: list[str] = ["bench", "data", "prepare", dataset_config.name]
    if repo_root is not None:
        parts.extend(["--repo-root", str(repo_root)])
    for key, value in sorted(dataset_config.dataset_options.items()):
        parts.extend(["--option", f"{key}={_format_option_value(value)}"])
    if all_problems:
        parts.append("--all")
    for problem in problem_names or []:
        parts.extend(["--problem", problem])
    return shell_join(parts)


def run_with_prepare_command(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> str:
    parts = ["bench", "run", str(config_path), "--prepare-data"]
    if run_dir is not None:
        parts.extend(["--run-dir", str(run_dir)])
    if repo_root is not None:
        parts.extend(["--repo-root", str(repo_root)])
    return shell_join(parts)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _format_option_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)
