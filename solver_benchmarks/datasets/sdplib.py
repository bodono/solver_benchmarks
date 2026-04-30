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
        # Tar members share ProblemSpec.path (the archive itself). Surface
        # the per-member size via metadata["size_bytes"] so the runner-level
        # size filter can compare against the member, not the whole archive.
        for name, size_bytes in sorted(list_sdplib_tar(self.tar_path).items()):
            if name in existing:
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=CONE,
                    path=self.tar_path,
                    metadata={
                        "source": str(self.tar_path),
                        "format": "tar:jld2",
                        "size_bytes": size_bytes,
                    },
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
        members = list_sdplib_tar(self.tar_path)
        names = list(members) if all_problems else list(problem_names or SDPLIB_DEFAULT_SUBSET)
        missing = [name for name in names if name not in members]
        if missing:
            raise RuntimeError(f"Unknown SDPLIB problem(s): {', '.join(missing)}")
        for name in names:
            extract_from_tar(self.tar_path, name, self.folder)

    def missing_data_message(self) -> str:
        return (
            "SDPLIB data is missing and cannot be downloaded automatically. "
            f"Restore the converted archive at {self.tar_path}, or place converted "
            f"`.jld2` files in {self.folder}."
        )
