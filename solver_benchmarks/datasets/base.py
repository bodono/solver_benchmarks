"""Dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solver_benchmarks.core.problem import ProblemData, ProblemSpec


@dataclass(frozen=True)
class DatasetDataStatus:
    dataset: str
    available: bool
    problem_count: int
    data_dir: Path | None
    source: str
    prepare_command: str | None
    message: str


class Dataset(ABC):
    dataset_id: str
    description: str
    data_source: str = "bundled in the repository"
    data_patterns: tuple[str, ...] = ()
    prepare_command: str | None = None

    def __init__(self, repo_root: str | Path | None = None, **options: Any):
        self.repo_root = Path(repo_root).resolve() if repo_root else _default_repo_root()
        self.options = options

    @abstractmethod
    def list_problems(self) -> list[ProblemSpec]:
        raise NotImplementedError

    @abstractmethod
    def load_problem(self, name: str) -> ProblemData:
        raise NotImplementedError

    def visible_problems(self) -> list[ProblemSpec]:
        """Problems visible after generic dataset options are applied.

        Dataset-specific options such as ``subset`` are handled inside each
        adapter's ``list_problems`` implementation. Generic options shared
        across file-backed datasets, currently ``max_size_mb``, are applied
        here so the runner, CLI listing/status commands, and analysis
        completion logic all agree on the same expected problem set.
        """
        return filter_problem_specs_by_size(
            self.list_problems(),
            self.options.get("max_size_mb"),
        )

    def problem_by_name(self, name: str) -> ProblemSpec:
        for spec in self.visible_problems():
            if spec.name == name:
                return spec
        raise KeyError(f"Problem {name!r} not found in dataset {self.dataset_id!r}")

    @property
    def data_dir(self) -> Path | None:
        return None

    def data_status(self) -> DatasetDataStatus:
        listed = self.list_problems()
        visible = filter_problem_specs_by_size(
            listed,
            self.options.get("max_size_mb"),
        )
        problem_count = len(visible)
        available = bool(listed) or self.data_dir is None
        if available:
            if problem_count == len(listed):
                message = f"{problem_count} problems available."
            else:
                message = (
                    f"{problem_count} of {len(listed)} problems visible after filters."
                )
        else:
            message = self.missing_data_message()
        return DatasetDataStatus(
            dataset=self.dataset_id,
            available=available,
            problem_count=problem_count,
            data_dir=self.data_dir,
            source=self.data_source,
            prepare_command=self.prepare_command,
            message=message,
        )

    def missing_data_message(self) -> str:
        if self.prepare_command:
            return f"No local data found. Run `{self.prepare_command}`."
        return "No local data found and no automatic preparation command is registered."

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        if self.data_status().available:
            return
        raise RuntimeError(self.missing_data_message())

    @property
    def problem_classes_dir(self) -> Path:
        explicit = self.options.get("data_root")
        if explicit:
            return Path(explicit).resolve()
        return self.repo_root / "problem_classes"


def filter_problem_specs_by_size(
    problems: list[ProblemSpec], max_size_mb: Any
) -> list[ProblemSpec]:
    """Drop specs whose backing file exceeds ``max_size_mb``.

    The size used for comparison is ``metadata["size_bytes"]`` when present
    (for archive-backed datasets such as SDPLIB), otherwise the on-disk size
    of ``ProblemSpec.path``. Specs without a known size pass through.
    """
    if max_size_mb is None:
        return list(problems)
    threshold_bytes = float(max_size_mb) * 1.0e6
    return [
        problem
        for problem in problems
        if (size := problem_spec_size_bytes(problem)) is None
        or size <= threshold_bytes
    ]


def problem_spec_size_bytes(problem: ProblemSpec) -> int | None:
    metadata_size = problem.metadata.get("size_bytes") if problem.metadata else None
    if metadata_size is not None:
        try:
            return int(metadata_size)
        except (TypeError, ValueError):
            return None
    if problem.path is None:
        return None
    try:
        return int(problem.path.stat().st_size)
    except OSError:
        return None


def _default_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "problem_classes").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]
