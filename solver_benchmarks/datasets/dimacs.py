"""DIMACS conic benchmark dataset."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
import gzip
import re
import shutil
import urllib.request

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec
from .base import Dataset


DIMACS_BASE_URL = "https://archive.dimacs.rutgers.edu/Challenges/Seventh/Instances"
DIMACS_DEFAULT_SUBSET = ("nb", "filter48_socp", "qssp30")


class DIMACSDataset(Dataset):
    dataset_id = "dimacs"
    description = "DIMACS conic benchmark dataset."
    data_source = "bundled data; official source https://archive.dimacs.rutgers.edu/Challenges/Seventh/Instances/"
    data_patterns = ("*.mat", "*.mat.gz")
    prepare_command = "python scripts/prepare_dimacs.py"

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "dimacs_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        paths: dict[str, Path] = {}
        for path in sorted(self.folder.iterdir()):
            if path.name.endswith(".mat.gz"):
                name = path.name[: -len(".mat.gz")]
            elif path.name.endswith(".mat"):
                name = path.stem
            else:
                continue
            existing = paths.get(name)
            if existing is None or (existing.name.endswith(".gz") and not path.name.endswith(".gz")):
                paths[name] = path
        specs = []
        for name, path in sorted(paths.items()):
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=CONE,
                    path=path,
                    metadata={"source": str(path), "format": "mat"},
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        problem = _read_dimacs(spec.path)
        return ProblemData(self.dataset_id, name, CONE, problem, metadata=dict(spec.metadata))

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        names = dimacs_remote_problem_names() if all_problems else list(problem_names or DIMACS_DEFAULT_SUBSET)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            _download_dimacs_problem(name, self.folder)


def _read_dimacs(path: Path) -> dict:
    data = _loadmat_maybe_gzip(path)
    c = -_dense_vector(data["b"])
    b = _dense_vector(data["c"])
    k = data["K"]

    try:
        raw_a = data["A"].T
    except KeyError:
        raw_a = data["At"]
    raw_a = sp.csc_matrix(raw_a)
    raw_a.data = raw_a.data.astype(float)

    if b.ndim == 0:
        b = np.zeros(raw_a.shape[0])
    if c.ndim == 0:
        c = np.zeros(raw_a.shape[1])

    cone, rows_to_keep, free_rows, dropped_rows, b = _parse_dimacs_cone(k, raw_a.shape[0], b)
    if free_rows:
        a_free = raw_a[free_rows, :] - raw_a[dropped_rows, :]
        b_free = b[free_rows] - b[dropped_rows]
        if np.linalg.norm(b_free) > 0 or sp.linalg.norm(a_free) > 0:
            a = sp.vstack((a_free, raw_a[rows_to_keep, :]), format="csc")
            b = np.hstack((b_free, b[rows_to_keep]))
            cone["f"] = int(cone.get("f", 0) + len(b_free))
        else:
            a = sp.csc_matrix(raw_a[rows_to_keep, :])
            b = b[rows_to_keep]
    else:
        a = sp.csc_matrix(raw_a[rows_to_keep, :])
        b = b[rows_to_keep]

    a = a.sorted_indices()
    return {
        "P": None,
        "A": a,
        "b": np.asarray(b, dtype=float),
        "q": np.asarray(c, dtype=float),
        "r": 0.0,
        "n": int(a.shape[1]),
        "m": int(a.shape[0]),
        "cone": cone,
        "obj_type": "min",
    }


def _loadmat_maybe_gzip(path: Path) -> dict:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as handle:
            return scipy.io.loadmat(BytesIO(handle.read()))
    return scipy.io.loadmat(path)


def _dense_vector(value):
    if not isinstance(value, np.ndarray):
        value = value.toarray()
    elif sp.issparse(value):
        value = value.toarray()
    return np.squeeze(np.asarray(value, dtype=float))


def _parse_dimacs_cone(k, n_rows: int, b: np.ndarray):
    names = k.dtype.names or ()
    if "r" in names and np.sum(k["r"][0][0]) > 0:
        raise ValueError("DIMACS rotated Lorentz cones are not supported yet")

    current = 0
    dropped_rows: list[int] = []
    free_rows: list[int] = []
    cone: dict = {}

    if "f" in names:
        cone["f"] = int(np.sum(k["f"][0][0]))
        current += cone["f"]
    if "q" in names:
        q = np.squeeze(k["q"][0][0].astype(int)).tolist()
        if isinstance(q, int):
            q = [q]
        cone["q"] = q
        current += int(np.sum(q))
    if "l" in names:
        cone["l"] = int(np.sum(k["l"][0][0]))
        current += cone["l"]
    if "s" in names:
        s_blocks = np.squeeze(k["s"][0][0].astype(int)).tolist()
        if isinstance(s_blocks, int):
            s_blocks = [s_blocks]
        cone["s"] = s_blocks
        for block in s_blocks:
            matrix_indices = np.arange(block * block).reshape(block, block).T
            diag = current + np.diag(matrix_indices)
            triu = np.triu_indices(block, 1)
            upper = current + matrix_indices[triu]
            lower = current + matrix_indices[triu[1], triu[0]]
            b[diag] /= np.sqrt(2.0)
            dropped_rows += upper.tolist()
            free_rows += lower.tolist()
            current += block * block

    rows_to_keep = [row for row in range(n_rows) if row not in dropped_rows]
    return cone, rows_to_keep, free_rows, dropped_rows, b


def dimacs_remote_problem_names() -> list[str]:
    with urllib.request.urlopen(f"{DIMACS_BASE_URL}/", timeout=30) as response:
        html = response.read().decode("utf-8", "replace")
    filenames = re.findall(r'href="([^"]+\.mat\.gz)"', html)
    return sorted({filename[: -len(".mat.gz")] for filename in filenames})


def _download_dimacs_problem(name: str, folder: Path) -> Path:
    stem = Path(name).name
    for suffix in (".mat.gz", ".mat"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    existing_mat = folder / f"{stem}.mat"
    target = folder / f"{stem}.mat.gz"
    if existing_mat.exists() or target.exists():
        return target if target.exists() else existing_mat
    bundled = _bundled_dimacs_problem(stem)
    if bundled is not None:
        shutil.copyfile(bundled, target)
        return target
    last_error = None
    url = f"{DIMACS_BASE_URL}/{stem}.mat.gz"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        gzip.decompress(content)
        target.write_bytes(content)
        return target
    except OSError as exc:
        last_error = exc
    raise RuntimeError(f"Could not download DIMACS problem {name!r}: {last_error}")


def _bundled_dimacs_problem(stem: str) -> Path | None:
    bundled = (
        Path(__file__).resolve().parents[2]
        / "problem_classes"
        / "dimacs_data"
        / f"{stem}.mat.gz"
    )
    return bundled if bundled.exists() else None
