"""Dataset adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from solver_benchmarks.core.problem import ProblemData, ProblemSpec


class Dataset(ABC):
    dataset_id: str
    description: str

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
