"""SDPLIB dataset adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec
from solver_benchmarks.transforms.sdplib import (
    extract_from_tar,
    list_sdplib_tar,
    read_sdplib_block_sizes,
    read_sdplib_jld2,
    split_into_declared_blocks,
)

from .base import Dataset

SDPLIB_DEFAULT_SUBSET = ("arch0", "control1", "theta1")
# The four SDPLIB instances that are infeasible by construction. They test
# infeasibility detection, not solve time, so they form their own subset, as
# the NETLIB infeasible LPs do.
SDPLIB_INFEASIBLE = ("infd1", "infd2", "infp1", "infp2")
SDPLIB_SUBSETS = ("feasible", "infeasible", "all")
# How a problem read from the converted archive is posed. ``declared``: with
# the block structure of the original SDPLIB file, which the converted file
# records. ``merged``: as one semidefinite block of the full order, which is
# how the matrices are stored and how this adapter posed every problem
# before it gave the block-diagonal problems their declared blocks.
SDPLIB_BLOCKS = ("declared", "merged")


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

    @property
    def blocks(self) -> str:
        """``declared`` (default) or ``merged``, from ``dataset_options.blocks``."""
        blocks = str(self.options.get("blocks", "declared"))
        if blocks not in SDPLIB_BLOCKS:
            raise ValueError(f"Unknown SDPLIB blocks option {blocks!r}; expected one of {', '.join(SDPLIB_BLOCKS)}")
        return blocks

    @classmethod
    def signature_options(cls, options: dict[str, Any]) -> dict[str, Any]:
        """Results recorded before problems got their declared blocks are merged-form results.

        Their signatures were computed from options with no ``blocks`` key, so
        that is what ``blocks: merged`` hashes to, and the declared form
        (the default now) hashes to something else. A run directory from
        before the change then resumes only under ``blocks: merged``; under
        the default its SDPLIB rows are run again instead of being mixed
        with rows for the other form.
        """
        hashed = dict(options)
        blocks = str(hashed.pop("blocks", "declared"))
        if blocks != "merged":
            hashed["blocks"] = blocks
        return hashed

    def _check_options(self) -> None:
        """Raise on a bad ``subset`` or ``blocks`` value at planning time, not in each worker."""
        _ = self.subset, self.blocks

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
        the same name; otherwise ``.jld2`` beats the tar member. Three archive
        members are damaged and cannot be read (maxG55, maxG60, thetaG51:
        stretches of each file are zeros), so their original files are
        bundled beside the archive and are what these three are read from.

        For the size filter (``max_size_mb``) a problem has the size of its
        converted file, whichever file it is read from: an original that
        replaces an archive member, or an extracted ``.jld2``, is filtered by
        that file's size, and only an original with no converted counterpart
        by its own. An original is a far smaller file (the bundled maxG60
        original is 0.15 MB, its member 395 MB), and the three bundled ones
        are the largest problems of the set; filtered by their own size they
        would pass any limit.
        """
        self._check_options()
        specs: dict[str, ProblemSpec] = {}
        rank: dict[str, int] = {}
        member_sizes = list_sdplib_tar(self.tar_path) if self.tar_path.exists() else {}
        entries = sorted(self.folder.iterdir()) if self.folder.is_dir() else []
        # Size of the converted file of each name: the archive member, or an
        # extracted .jld2 where there is no archive.
        converted_sizes = {path.stem: int(path.stat().st_size) for path in entries if path.suffix == ".jld2"}
        converted_sizes.update(member_sizes)

        def offer(name: str, spec: ProblemSpec, priority: int) -> None:
            if name not in specs or priority < rank[name]:
                specs[name] = spec
                rank[name] = priority

        for path in entries:
            stem = _original_stem(path)
            if stem is not None:
                prio = 1 if path.name.endswith(".gz") else 0
            elif path.suffix == ".jld2":
                stem, prio = path.stem, 2
            else:
                continue
            fmt = "sdpa-s" if prio < 2 else "jld2"
            metadata: dict = {"source": str(path), "format": fmt}
            if fmt == "sdpa-s" and stem in converted_sizes:
                metadata["size_bytes"] = converted_sizes[stem]
            offer(stem, ProblemSpec(dataset_id=self.dataset_id, name=stem, kind=CONE, path=path,
                                    metadata=metadata), prio)
        # Tar members share ProblemSpec.path (the archive itself). Surface
        # the per-member size via metadata["size_bytes"] so the runner-level
        # size filter can compare against the member, not the whole archive.
        if member_sizes:
            for name, size_bytes in sorted(member_sizes.items()):
                offer(name, ProblemSpec(
                    dataset_id=self.dataset_id, name=name, kind=CONE, path=self.tar_path,
                    metadata={"source": str(self.tar_path), "format": "tar:jld2", "size_bytes": int(size_bytes)},
                ), 3)
        return [_with_expected(specs[k]) for k in sorted(specs) if self._in_subset(k)]

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        path = spec.path
        if spec.metadata.get("format") == "sdpa-s":
            from solver_benchmarks.transforms.sdpa import parse_sdpa_s_file, sdpa_to_cone_problem
            primal = parse_sdpa_s_file(path)
            if self.blocks == "merged" and len(primal.blocks) > 1:
                raise ValueError(
                    f"SDPLIB problem {name!r} is read from an original SDPA file, which has "
                    "no merged form; blocks: merged applies to the converted archive only"
                )
            return ProblemData(
                self.dataset_id, name, CONE, sdpa_to_cone_problem(primal),
                metadata={**dict(spec.metadata), "num_constraints_primal": int(primal.m),
                          "block_structure": "declared",
                          "num_blocks": len(primal.blocks), "block_orders": [blk.order for blk in primal.blocks]},
            )
        if path.suffix == ".tar":
            path = extract_from_tar(path, name, self.folder / ".cache")
        problem = read_sdplib_jld2(path)
        metadata = dict(spec.metadata)
        # The matrices are stored whole; the file records the blocks apart.
        # A converted file without that record stays the single block it
        # is stored as.
        sizes = read_sdplib_block_sizes(path) if self.blocks == "declared" else None
        if sizes is None:
            order = int(problem["cone"]["s"][0])
            metadata.update({"block_structure": "merged", "num_blocks": 1, "block_orders": [order]})
        else:
            problem = split_into_declared_blocks(problem, sizes)
            metadata.update({
                "block_structure": "declared",
                "num_blocks": len(sizes),
                "block_orders": [abs(size) for size in sizes],
            })
        return ProblemData(self.dataset_id, name, CONE, problem, metadata=metadata)

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
                f"JLD2 archive at {self.tar_path}: restore "
                "problem_classes/sdplib_data/sdplib.tar, or place converted "
                f"`.jld2` files or original `.dat-s` files in {self.folder}."
            )
        members = list_sdplib_tar(self.tar_path)
        names = list(members) if all_problems else list(problem_names or SDPLIB_DEFAULT_SUBSET)
        missing = [name for name in names if name not in members]
        if missing:
            raise RuntimeError(f"Unknown SDPLIB problem(s): {', '.join(missing)}")
        # A member that has an original beside the archive is never read
        # (and the three bundled originals replace members that cannot be).
        originals = {_original_stem(path) for path in self.folder.iterdir()} - {None}
        for name in names:
            if name not in originals:
                extract_from_tar(self.tar_path, name, self.folder)

    def missing_data_message(self) -> str:
        return (
            "SDPLIB data is missing and cannot be downloaded automatically. "
            f"Restore the converted archive at {self.tar_path}, or place converted "
            f"`.jld2` files or original `.dat-s` files in {self.folder}."
        )


def _original_stem(path: Path) -> str | None:
    """The problem name if ``path`` is an original SDPLIB file, else None."""
    for suffix in (".dat-s.gz", ".dat-s"):
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return None


def _with_expected(spec: ProblemSpec) -> ProblemSpec:
    """Record the known outcome: the four infeasible instances have no optimal
    pair (either kind of certificate is accepted, so the SDPA-to-cone sign
    convention does not matter); every other SDPLIB instance is solvable."""
    outcome = status.EXPECT_INFEASIBLE if spec.name in SDPLIB_INFEASIBLE else status.EXPECT_OPTIMAL
    return replace(spec, metadata={**spec.metadata, "expected_status": outcome})
