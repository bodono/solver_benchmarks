"""MPS-backed LP datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import bz2
import re
import urllib.error
import urllib.request

import numpy as np
import scipy.sparse as sp

import problem_classes.qpsreader as qpsreader

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from .base import Dataset


class MPSLPDataset(Dataset):
    problem_folder: str
    dataset_id: str
    description: str
    data_patterns = ("*.mps", "*.mps.gz")

    def __init__(self, repo_root: str | Path | None = None, **options: Any):
        super().__init__(repo_root=repo_root, **options)
        self.max_size_mb = float(options.get("max_size_mb", np.inf))

    @property
    def folder(self) -> Path:
        folder = self.options.get("folder", self.problem_folder)
        return self.problem_classes_dir / str(folder)

    @property
    def data_dir(self) -> Path:
        return self.folder

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
    data_source = "external download from https://plato.asu.edu/ftp/lptestset/"
    prepare_command = "python scripts/prepare_mittelmann.py --problem qap15"

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        names = list(problem_names or [])
        if all_problems:
            names = _mittelmann_remote_problem_names()
        if not names:
            if self.data_status().available:
                return
            raise RuntimeError(
                "Mittelmann data is large and is not downloaded implicitly. "
                "Use `bench data prepare mittelmann --problem qap15` for specific "
                "problems or `bench data prepare mittelmann --all` for the full "
                "root lptestset index."
            )
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            _download_mittelmann_problem(name, self.folder)


def _mps_name(path: Path) -> str | None:
    if path.name.endswith(".mps.gz"):
        return path.name[: -len(".mps.gz")]
    if path.name.endswith(".mps"):
        return path.name[: -len(".mps")]
    return None


def _mittelmann_remote_problem_names() -> list[str]:
    html = urllib.request.urlopen("https://plato.asu.edu/ftp/lptestset/", timeout=30).read()
    names = []
    for match in re.findall(rb'href="([^"]+\.bz2)"', html):
        filename = match.decode("utf-8")
        names.append(_strip_mittelmann_suffix(filename))
    return sorted(set(names))


def _download_mittelmann_problem(name: str, folder: Path) -> None:
    stem = _strip_mittelmann_suffix(Path(name).name)
    target = folder / f"{stem}.mps"
    if target.exists():
        return
    candidates = [f"{stem}.mps.bz2", f"{stem}.bz2"]
    last_error = None
    for filename in candidates:
        url = f"https://plato.asu.edu/ftp/lptestset/{filename}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                compressed = response.read()
            target.write_bytes(bz2.decompress(compressed))
            return
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not download Mittelmann problem {name!r}: {last_error}")


def _strip_mittelmann_suffix(name: str) -> str:
    for suffix in (".mps.bz2", ".bz2", ".mps.gz", ".mps"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name
