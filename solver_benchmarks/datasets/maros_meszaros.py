"""Maros-Meszaros QP dataset."""

from __future__ import annotations

from pathlib import Path

from problem_classes.maros_meszaros import MarosMeszaros

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from .base import Dataset


class MarosMeszarosDataset(Dataset):
    dataset_id = "maros_meszaros"
    description = "Maros-Meszaros convex QP collection."

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "maros_meszaros_data"

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        specs = []
        for path in sorted(self.folder.glob("*.mat")):
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=path.stem,
                    kind=QP,
                    path=path,
                    metadata={"source": str(path), "format": "mat"},
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        instance = MarosMeszaros(str(spec.path.with_suffix("")), name)
        qp = dict(instance.qp_problem)
        qp["obj_type"] = "min"
        return ProblemData(self.dataset_id, name, QP, qp, metadata=dict(spec.metadata))
