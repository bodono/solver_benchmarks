"""SDPLIB dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec
from solver_benchmarks.transforms.sdplib import (
    extract_from_tar,
    list_sdplib_tar,
    read_sdplib_jld2,
)
from .base import Dataset


SDPLIB_DEFAULT_SUBSET = ("arch0", "control1", "theta1")


class SDPLIBDataset(Dataset):
    dataset_id = "sdplib"
    description = "SDPLIB SDP benchmark dataset."
    data_source = (
        "bundled converted JLD2 archive; original SDPLIB is documented at "
        "https://vlsicad.eecs.umich.edu/BK/Slots/cache/www.nmt.edu/~borchers/sdplib.html"
    )
    data_patterns = ("*.jld2", "sdplib.tar")
    prepare_command = "python scripts/prepare_sdplib.py"

    def __init__(self, repo_root: str | Path | None = None, **options: Any):
        super().__init__(repo_root=repo_root, **options)
        self.max_size_mb = float(options.get("max_size_mb", float("inf")))

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
        # Tar members share a single ProblemSpec.path (the archive itself), so
        # the runner-level max_size_mb filter cannot see per-member sizes.
        # We must filter inside the archive here.
        for name in list_sdplib_tar(self.tar_path, max_size_mb=self.max_size_mb):
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

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        if not self.tar_path.exists():
            if self.data_status().available:
                return
            raise RuntimeError(
                "SDPLIB data is missing. This repository expects the converted "
                f"JLD2 archive at {self.tar_path}. The original SDPLIB files are "
                "not loaded directly; convert them to the expected JLD2 archive "
                "or restore problem_classes/sdplib_data/sdplib.tar."
            )
        names = list_sdplib_tar(self.tar_path) if all_problems else list(problem_names or SDPLIB_DEFAULT_SUBSET)
        missing = [name for name in names if name not in set(list_sdplib_tar(self.tar_path))]
        if missing:
            raise RuntimeError(f"Unknown SDPLIB problem(s): {', '.join(missing)}")
        for name in names:
            extract_from_tar(self.tar_path, name, self.folder)
