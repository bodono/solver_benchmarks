"""QPLIB QP dataset."""

from __future__ import annotations

from pathlib import Path

from problem_classes.qplib import QPLIB

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from .base import Dataset


class QPLIBDataset(Dataset):
    dataset_id = "qplib"
    description = "Convex QPLIB subset present in problem_classes/qplib_data."
    data_patterns = ("QPLIB_*.qplib",)
    prepare_command = "cd problem_classes/qplib_data && python download.py"

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "qplib_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        specs = []
        for path in sorted(self.folder.glob("QPLIB_*.qplib")):
            name = path.name[len("QPLIB_") : -len(".qplib")]
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=QP,
                    path=path,
                    metadata={"source": str(path), "format": "qplib"},
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        instance = QPLIB(str(spec.path), name)
        qp = dict(instance.qp_problem)
        qp["obj_type"] = instance.obj_type
        return ProblemData(self.dataset_id, name, QP, qp, metadata=dict(spec.metadata))
