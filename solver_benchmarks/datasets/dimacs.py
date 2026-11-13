"""DIMACS conic benchmark dataset."""

from __future__ import annotations

import gzip
import re
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.sparse as sp

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec

from .base import Dataset, atomic_write_bytes, validate_gzip_payload

DIMACS_BASE_URL = "https://archive.dimacs.rutgers.edu/Challenges/Seventh/Instances"
DIMACS_DEFAULT_SUBSET = ("nb", "filter48_socp", "qssp30")


class DIMACSDataset(Dataset):
    dataset_id = "dimacs"
    description = "DIMACS conic benchmark dataset."
    data_source = "bundled data; official source https://archive.dimacs.rutgers.edu/Challenges/Seventh/Instances/"
    data_patterns = ("*.mat", "*.mat.gz")
    prepare_command = "python scripts/prepare_dimacs.py"

    @classmethod
    def signature_options(cls, options: dict[str, Any]) -> dict[str, Any]:
        """Results recorded before the conversion was corrected are not resumed.

        Until then every PSD block was scaled wrongly, so the problems with one
        were not the published problems, and the KKT check read the rows of a
        problem with linear and second-order cones in the wrong order. Those
        signatures were computed from the options as given; the marker makes
        the signatures differ, so a run directory from before the change
        solves its DIMACS problems again instead of keeping those rows.
        """
        return {**options, "conversion": 2}

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
    """Read a DIMACS problem in SeDuMi format as a cone problem.

    The file holds ``A``, ``b``, ``c`` and the cone ``K`` of the pair

        minimize c'x  subject to  Ax = b, x in K
        maximize b'y  subject to  c - A'y in K*

    and the problem returned is the second one, as ``minimize -b'y`` with one
    row of ``c - A'y`` per cone entry. SeDuMi stacks the cones as free, linear,
    second-order, PSD, whatever order the fields of ``K`` are stored in, and
    that is also the row layout used here.

    A PSD block of order ``d`` takes ``d * d`` rows in the file, the whole
    matrix column by column. Here it takes the ``d * (d + 1) / 2`` rows of its
    lower triangle, column by column, with the off-diagonal entries times
    ``sqrt(2)``. SeDuMi asks only that the symmetric part of ``c - A'y`` be
    PSD, so the entry used is the mean of the two mirrored rows. That matters
    for a file whose constraint matrices are not symmetric as stored: in
    ``fap09`` and ``fap25`` each one is a single entry ``E_ij``.

    The objective recorded for a run is that of the minimisation, ``-b'y``:
    minus the optimum published for the challenge.
    """
    data = _loadmat_maybe_gzip(path)
    raw_a = data["A"].T if "A" in data else data["At"]
    raw_a = sp.csr_matrix(raw_a, dtype=float)
    n_rows, n_vars = raw_a.shape
    objective = -_vector(data["b"], n_vars, "b")
    rhs = _vector(data["c"], n_rows, "c")

    free, linear, soc, psd = _cone_sizes(data["K"])
    first_psd_row = free + linear + sum(soc)
    if first_psd_row + sum(order * order for order in psd) != n_rows:
        raise ValueError(
            f"{path.name}: K accounts for "
            f"{first_psd_row + sum(order * order for order in psd)} rows but A has {n_rows}"
        )

    a_parts = [raw_a[:first_psd_row, :]]
    b_parts = [rhs[:first_psd_row]]
    if psd:
        lower, upper, weight = _psd_triangle_rows(psd, first_psd_row)
        mean_rows = 0.5 * (raw_a[lower, :] + raw_a[upper, :])
        a_parts.append(sp.diags(weight) @ mean_rows)
        b_parts.append(weight * (0.5 * (rhs[lower] + rhs[upper])))

    a = sp.csc_matrix(sp.vstack(a_parts, format="csc"))
    a.eliminate_zeros()
    a.sort_indices()
    b = np.concatenate(b_parts)

    # Keys in row order; an empty cone is left out.
    cone: dict = {}
    if free:
        cone["z"] = free
    if linear:
        cone["l"] = linear
    if soc:
        cone["q"] = soc
    if psd:
        cone["s"] = psd
    return {
        "P": None,
        "A": a,
        "b": np.asarray(b, dtype=float),
        "q": np.asarray(objective, dtype=float),
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


def _vector(value, size: int, label: str) -> np.ndarray:
    if sp.issparse(value):
        value = value.toarray()
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size == size:
        return vector
    if vector.size <= 1 and not np.any(vector):
        # A zero vector stored as a single zero, or as nothing.
        return np.zeros(size)
    raise ValueError(f"DIMACS vector {label} has {vector.size} entries, expected {size}")


def _cone_sizes(k) -> tuple[int, int, list[int], list[int]]:
    """Sizes of the free, linear, second-order and PSD cones of a SeDuMi ``K``."""
    names = k.dtype.names or ()

    def sizes(field: str) -> list[int]:
        if field not in names:
            return []
        value = k[field][0][0]
        if sp.issparse(value):
            value = value.toarray()
        entries = np.asarray(value, dtype=float).reshape(-1)
        if np.any(entries != np.round(entries)) or np.any(entries < 0):
            raise ValueError(f"DIMACS cone field K.{field} is not a list of sizes")
        return [int(entry) for entry in entries if entry > 0]

    if sizes("r"):
        raise ValueError("DIMACS rotated Lorentz cones are not supported yet")
    return sum(sizes("f")), sum(sizes("l")), sizes("q"), sizes("s")


def _psd_triangle_rows(
    orders: list[int], first_row: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For PSD blocks stored as whole matrices, column by column, from
    ``first_row``: the row of each lower-triangle entry (column by column), the
    row of its mirror image, and the weight of the entry (1 on the diagonal,
    ``sqrt(2)`` off it).
    """
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    weight: list[np.ndarray] = []
    row = first_row
    for order in orders:
        # Pairs (col, i) with col <= i, col varying slowest: the lower
        # triangle column by column. Entry (i, col) is at row + col * order + i.
        col, i = np.triu_indices(order)
        lower.append(row + col * order + i)
        upper.append(row + i * order + col)
        weight.append(np.where(i == col, 1.0, np.sqrt(2.0)))
        row += order * order
    return np.concatenate(lower), np.concatenate(upper), np.concatenate(weight)


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
    folder.mkdir(parents=True, exist_ok=True)
    bundled = _bundled_dimacs_problem(stem)
    if bundled is not None:
        # Atomically place the bundled copy so a partial copyfile cannot
        # leave a corrupt file at `target`.
        atomic_write_bytes(target, bundled.read_bytes())
        return target
    last_error = None
    url = f"{DIMACS_BASE_URL}/{stem}.mat.gz"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        # Validate the full gzip stream (CRC + EOF marker), not just
        # the header — see datasets/base.py for the rationale.
        validate_gzip_payload(content)
        atomic_write_bytes(target, content)
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
