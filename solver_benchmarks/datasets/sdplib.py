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
# The four SDPLIB instances that are infeasible by construction. They test
# infeasibility detection, not solve time, so they form their own subset, as
# the NETLIB infeasible LPs do.
SDPLIB_INFEASIBLE = ("infd1", "infd2", "infp1", "infp2")
SDPLIB_SUBSETS = ("feasible", "infeasible", "all")


class SDPLIBDataset(Dataset):
    dataset_id = "sdplib"
    description = "SDPLIB SDP benchmark dataset."
    data_source = (
        "bundled converted JLD2 archive; original SDPLIB is documented at "
        "https://vlsicad.eecs.umich.edu/BK/Slots/cache/www.nmt.edu/~borchers/sdplib.html"
    )
    data_patterns = ("*.jld2", "*.dat-s", "*.dat-s.gz", "sdplib.tar")
    prepare_command = "python scripts/prepare_sdplib.py"

    @property
    def subset(self) -> str:
        """``feasible`` (default), ``infeasible`` or ``all``, from ``dataset_options.subset``."""
        subset = str(self.options.get("subset", "feasible"))
        if subset not in SDPLIB_SUBSETS:
            raise ValueError(f"Unknown SDPLIB subset {subset!r}; expected one of {', '.join(SDPLIB_SUBSETS)}")
        return subset

    def _in_subset(self, name: str) -> bool:
        if self.subset == "all":
            return True
        return (name in SDPLIB_INFEASIBLE) == (self.subset == "infeasible")

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
        """One spec per problem name in the selected subset, with explicit precedence.

        An original SDPA-S file (``.dat-s`` then ``.dat-s.gz``) placed in the
        data folder replaces both an extracted ``.jld2`` and the tar member of
        the same name, which is how corrupt archive entries (maxG55, maxG60)
        are overridden; otherwise ``.jld2`` beats the tar member.
        """
        specs: dict[str, ProblemSpec] = {}
        rank: dict[str, int] = {}

        def offer(name: str, spec: ProblemSpec, priority: int) -> None:
            if name not in specs or priority < rank[name]:
                specs[name] = spec
                rank[name] = priority

        if self.folder.is_dir():
            for path in sorted(self.folder.iterdir()):
                if path.name.endswith(".dat-s.gz"):
                    stem, prio = path.name[: -len(".dat-s.gz")], 1
                elif path.name.endswith(".dat-s"):
                    stem, prio = path.name[: -len(".dat-s")], 0
                elif path.suffix == ".jld2":
                    stem, prio = path.stem, 2
                else:
                    continue
                fmt = "sdpa-s" if prio < 2 else "jld2"
                offer(stem, ProblemSpec(dataset_id=self.dataset_id, name=stem, kind=CONE, path=path,
                                        metadata={"source": str(path), "format": fmt}), prio)
        # Tar members share ProblemSpec.path (the archive itself). Surface
        # the per-member size via metadata["size_bytes"] so the runner-level
        # size filter can compare against the member, not the whole archive.
        if self.tar_path.exists():
            for name, size_bytes in sorted(list_sdplib_tar(self.tar_path).items()):
                offer(name, ProblemSpec(
                    dataset_id=self.dataset_id, name=name, kind=CONE, path=self.tar_path,
                    metadata={"source": str(self.tar_path), "format": "tar:jld2", "size_bytes": int(size_bytes)},
                ), 3)
        return [specs[k] for k in sorted(specs) if self._in_subset(k)]

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        path = spec.path
        if spec.metadata.get("format") == "sdpa-s":
            from solver_benchmarks.transforms.sdpa import parse_sdpa_s_file, sdpa_to_cone_problem
            primal = parse_sdpa_s_file(path)
            return ProblemData(
                self.dataset_id, name, CONE, sdpa_to_cone_problem(primal),
                metadata={**dict(spec.metadata), "num_constraints_primal": int(primal.m),
                          "num_blocks": len(primal.blocks), "block_orders": [blk.order for blk in primal.blocks]},
            )
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
