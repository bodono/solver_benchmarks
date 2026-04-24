"""SDPLIB dataset adapter."""

from __future__ import annotations

from pathlib import Path

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec
from solver_benchmarks.transforms.sdplib import (
    extract_from_tar,
    list_sdplib_tar,
    read_sdplib_jld2,
)
from .base import Dataset


class SDPLIBDataset(Dataset):
    dataset_id = "sdplib"
    description = "SDPLIB SDP benchmark dataset."
    data_patterns = ("*.jld2", "sdplib.tar")

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "sdplib_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    @property
    def tar_path(self) -> Path:
        return self.folder / "sdplib.tar"

    def list_problems(self) -> list[ProblemSpec]:
        specs = []
        if self.folder.is_dir():
            for path in sorted(self.folder.glob("*.jld2")):
                specs.append(
                    ProblemSpec(
                        dataset_id=self.dataset_id,
                        name=path.stem,
                        kind=CONE,
                        path=path,
                        metadata={"source": str(path), "format": "jld2"},
                    )
                )
        existing = {spec.name for spec in specs}
        for name in list_sdplib_tar(self.tar_path):
            if name in existing:
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=CONE,
                    path=self.tar_path,
                    metadata={"source": str(self.tar_path), "format": "tar:jld2"},
                )
            )
        return sorted(specs, key=lambda spec: spec.name)

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        path = spec.path
        if path.suffix == ".tar":
            path = extract_from_tar(path, name, self.folder / ".cache")
        problem = read_sdplib_jld2(path)
        return ProblemData(self.dataset_id, name, CONE, problem, metadata=dict(spec.metadata))
