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

    def problem_by_name(self, name: str) -> ProblemSpec:
        for spec in self.list_problems():
            if spec.name == name:
                return spec
        raise KeyError(f"Problem {name!r} not found in dataset {self.dataset_id!r}")

    @property
    def data_dir(self) -> Path | None:
        return None

    def data_status(self) -> DatasetDataStatus:
        problem_count = len(self.list_problems())
        available = problem_count > 0 or self.data_dir is None
        if available:
            message = f"{problem_count} problems available."
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


def _default_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "problem_classes").is_dir():
        return cwd
    return Path(__file__).resolve().parents[2]
