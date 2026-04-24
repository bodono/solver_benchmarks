"""MPS-backed LP datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

import problem_classes.qpsreader as qpsreader

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from .base import Dataset


class MPSLPDataset(Dataset):
    problem_folder: str
    dataset_id: str
    description: str

    def __init__(self, repo_root: str | Path | None = None, **options: Any):
        super().__init__(repo_root=repo_root, **options)
        self.max_size_mb = float(options.get("max_size_mb", np.inf))

    @property
    def folder(self) -> Path:
        folder = self.options.get("folder", self.problem_folder)
        return self.problem_classes_dir / str(folder)

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        paths: dict[str, Path] = {}
        for path in sorted(self.folder.iterdir()):
            name = _mps_name(path)
            if name is None:
                continue
            if path.stat().st_size > self.max_size_mb * 1.0e6:
                continue
            existing = paths.get(name)
            if existing is None or (existing.suffix == ".gz" and path.suffix != ".gz"):
                paths[name] = path
        return [
            ProblemSpec(
                dataset_id=self.dataset_id,
                name=name,
                kind=QP,
                path=path,
                metadata={"source": str(path), "format": "mps"},
            )
            for name, path in sorted(paths.items())
        ]

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        a, c, l, u = qpsreader.readMpsLp(str(spec.path))
        a = sp.csc_matrix(a)
        n = a.shape[1]
        qp = {
            "P": sp.csc_matrix((n, n)),
            "q": np.asarray(c, dtype=float),
            "r": 0.0,
            "A": a,
            "l": np.asarray(l, dtype=float),
            "u": np.asarray(u, dtype=float),
            "n": int(n),
            "m": int(a.shape[0]),
            "obj_type": "min",
        }
        if self.options.get("add_quadratic", False):
            qp["P"] = sp.eye(n, format="csc")
        return ProblemData(self.dataset_id, name, QP, qp, metadata=dict(spec.metadata))


class NetlibDataset(MPSLPDataset):
    dataset_id = "netlib"
    description = "NETLIB LP dataset."
    problem_folder = "netlib_data/feasible"

    @property
    def folder(self) -> Path:
        subset = self.options.get("subset")
        if subset is None:
            subset = "infeasible" if self.options.get("infeasible", False) else "feasible"
        return self.problem_classes_dir / "netlib_data" / str(subset)


class MiplibDataset(MPSLPDataset):
    dataset_id = "miplib"
    description = "MIPLIB root-node LP relaxation dataset."
    problem_folder = "miplib_data"


class MittelmannDataset(MPSLPDataset):
    dataset_id = "mittelmann"
    description = "Mittelmann LP/QP dataset."
    problem_folder = "mittelmann"


def _mps_name(path: Path) -> str | None:
    if path.name.endswith(".mps.gz"):
        return path.name[: -len(".mps.gz")]
    if path.name.endswith(".mps"):
        return path.name[: -len(".mps")]
    return None
