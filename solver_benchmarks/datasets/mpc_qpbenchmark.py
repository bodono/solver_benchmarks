"""MPC QP Benchmark dataset adapter."""

from __future__ import annotations

import json
import pickle
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from solver_benchmarks.transforms.cones import INF_BOUND

from .base import Dataset, atomic_write_bytes, normalize_name_subset

# The data are read from a fixed commit, as the DC OPF cases are, so that
# two preparations at different times fetch the same problems under the
# same names. (This is the head of ``main`` when the pin was added.)
MPC_QPBENCHMARK_REVISION = "954906ac1f48b77e84a5bc314b7c2f00fb0d4ebc"
MPC_QPBENCHMARK_API_URL = (
    "https://api.github.com/repos/qpsolvers/mpc_qpbenchmark/contents/data"
    f"?ref={MPC_QPBENCHMARK_REVISION}"
)
MPC_QPBENCHMARK_RAW_URL = (
    "https://raw.githubusercontent.com/qpsolvers/mpc_qpbenchmark/"
    f"{MPC_QPBENCHMARK_REVISION}/data"
)
MPC_QPBENCHMARK_DEFAULT_SUBSET = ("LIPMWALK0", "WHLIPBAL0", "QUADCMPC1")
MPC_ZERO_ROW_BOUND_TOL = 1.0e-16  # Absolute data-cleanup threshold


class MPCQPBenchmarkDataset(Dataset):
    dataset_id = "mpc_qpbenchmark"
    description = "Structured MPC QPs from qpsolvers/mpc_qpbenchmark."
    data_source = "external download from https://github.com/qpsolvers/mpc_qpbenchmark"
    prepare_command = "python scripts/prepare_mpc_qpbenchmark.py"
    automatic_download = True
    subset_lists_problems = True
    option_names = frozenset({"subset"})
    selection_options = frozenset({"subset"})

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "mpc_qpbenchmark_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        subset = _normalize_subset(self.options.get("subset"))
        specs = []
        for path in sorted(self.folder.glob("*.npz")):
            name = path.stem
            if subset is not None and name not in subset:
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=name,
                    kind=QP,
                    path=path,
                    metadata={
                        "source": str(path),
                        "format": "mpc_qpbenchmark_npz",
                        "family": _family(name),
                    },
                )
            )
        return specs

    def requested_problem_names(self) -> set[str] | None:
        return _normalize_subset(self.options.get("subset"))

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        qp, metadata = read_mpc_qpbenchmark_npz(spec.path)
        # Some LIPMWALK files contain 0 <= -O(1e-17) from input roundoff.
        clipped = np.flatnonzero(
            (qp["A"].getnnz(axis=1) == 0)
            & (qp["l"] <= -INF_BOUND)
            & (qp["u"] < 0.0)
            & (qp["u"] >= -MPC_ZERO_ROW_BOUND_TOL)
        )
        qp["u"][clipped] = 0.0
        return ProblemData(
            self.dataset_id,
            name,
            QP,
            qp,
            metadata={**dict(spec.metadata), **metadata},
        )

    def prepare_data(
        self,
        problem_names: list[str] | None = None,
        *,
        all_problems: bool = False,
    ) -> None:
        subset = _normalize_subset(self.options.get("subset"))
        if all_problems:
            names = mpc_qpbenchmark_remote_problem_names()
        elif problem_names:
            names = list(problem_names)
        elif subset is not None:
            names = sorted(subset)
        else:
            names = list(MPC_QPBENCHMARK_DEFAULT_SUBSET)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_mpc_qpbenchmark_problem(name, self.folder)


def read_mpc_qpbenchmark_npz(path: Path) -> tuple[dict, dict]:
    """Read a QP from the .npz schema of qpsolvers/mpc_qpbenchmark.

    ``P`` is the Hessian of ``0.5 x'Px``: the whole symmetric matrix, or
    one triangle of it, which is mirrored (see ``_symmetric_hessian``).
    ``G``, ``h``, ``A``, ``b``, ``lb`` and ``ub`` are optional; a missing
    key and a ``None`` placeholder mean the same. Arrays are numeric; the
    file is read without unpickling anything but ``None`` placeholders.

    The arrays must fit together: with ``n`` the length of ``q``, ``P`` is
    ``n x n``, ``G`` and ``A`` have ``n`` columns (a 1-D array is one row),
    ``h`` and ``b`` are present with one entry per row of ``G`` and ``A``,
    and ``lb`` and ``ub`` have ``n`` entries. A ValueError names the file
    and the array that does not fit.
    """
    data = _load_npz_arrays(path)
    p = _symmetric_hessian(sp.csc_matrix(np.asarray(data["P"], dtype=float)), path)
    q = np.asarray(data["q"], dtype=float).reshape(-1)
    n = q.size
    if p.shape != (n, n):
        raise ValueError(f"{path.name}: P has shape {p.shape} but q has {n} entries")

    rows = []
    lower = []
    upper = []

    g = _optional_array(data, "G")
    h = _optional_array(data, "h")
    if g is not None:
        g_rows = _constraint_rows(g, "G", n, path)
        rows.append(g_rows)
        upper.append(_right_hand_side(h, "h", "G", g_rows.shape[0], path))
        lower.append(np.full(g_rows.shape[0], -INF_BOUND))

    aeq = _optional_array(data, "A")
    beq = _optional_array(data, "b")
    if aeq is not None:
        a_rows = _constraint_rows(aeq, "A", n, path)
        rows.append(a_rows)
        eq_rhs = _right_hand_side(beq, "b", "A", a_rows.shape[0], path)
        lower.append(eq_rhs)
        upper.append(eq_rhs)

    lb = _optional_array(data, "lb")
    ub = _optional_array(data, "ub")
    if lb is not None or ub is not None:
        rows.append(sp.eye(n, format="csc"))
        lower.append(
            _variable_bound(lb, "lb", n, path) if lb is not None else np.full(n, -INF_BOUND)
        )
        upper.append(
            _variable_bound(ub, "ub", n, path) if ub is not None else np.full(n, INF_BOUND)
        )

    if rows:
        a = sp.vstack(rows, format="csc")
        l = np.concatenate(lower).astype(float)
        u = np.concatenate(upper).astype(float)
    else:
        a = sp.csc_matrix((0, n))
        l = np.array([], dtype=float)
        u = np.array([], dtype=float)

    qp = {
        "P": p,
        "q": q,
        "r": 0.0,
        "A": a,
        "l": l,
        "u": u,
        "n": n,
        "m": int(a.shape[0]),
        "obj_type": "min",
    }
    metadata = {
        "num_variables": n,
        "num_constraints": int(a.shape[0]),
        "nnz_p": int(p.nnz),
        "nnz_a": int(a.nnz),
    }
    return qp, metadata


def mpc_qpbenchmark_remote_problem_names() -> list[str]:
    with urllib.request.urlopen(MPC_QPBENCHMARK_API_URL, timeout=30) as response:
        items = json.loads(response.read().decode("utf-8"))
    return sorted(Path(item["name"]).stem for item in items if item["name"].endswith(".npz"))


def download_mpc_qpbenchmark_problem(name: str, folder: Path) -> Path:
    stem = Path(name).name.removesuffix(".npz")
    target = folder / f"{stem}.npz"
    if target.exists():
        return target
    folder.mkdir(parents=True, exist_ok=True)
    url = f"{MPC_QPBENCHMARK_RAW_URL}/{stem}.npz"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    # Validate that the server returned a real numpy archive before
    # committing to the cache; an HTML 404 page would otherwise silently
    # be saved as foo.npz and fail at parse time on subsequent runs.
    try:
        with np.load(BytesIO(content), allow_pickle=False) as probe:
            _ = list(probe.files)
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            f"Could not download MPC qpbenchmark problem {name!r}: "
            f"response was not a valid .npz archive ({exc})"
        ) from exc
    atomic_write_bytes(target, content)
    return target


def _optional_array(data: dict, key: str):
    return data.get(key)


def _constraint_rows(value, key: str, n: int, path: Path) -> sp.csc_matrix:
    matrix = sp.csc_matrix(np.asarray(value, dtype=float))
    if matrix.shape[1] != n:
        raise ValueError(
            f"{path.name}: {key} has shape {matrix.shape} but q has {n} entries "
            f"({key} needs {n} columns)"
        )
    return matrix


def _right_hand_side(value, key: str, matrix_key: str, rows: int, path: Path) -> np.ndarray:
    if value is None:
        raise ValueError(f"{path.name}: {matrix_key} is given without {key}")
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != rows:
        raise ValueError(
            f"{path.name}: {key} has {vector.size} entries but {matrix_key} has {rows} rows"
        )
    return vector


def _variable_bound(value, key: str, n: int, path: Path) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != n:
        raise ValueError(f"{path.name}: {key} has {vector.size} entries but q has {n}")
    return vector


def _symmetric_hessian(p: sp.csc_matrix, path: Path) -> sp.csc_matrix:
    """``P`` as the whole symmetric matrix of the objective ``0.5 x'Px``.

    A file may store the whole matrix or one triangle of it (CUTEst's
    coordinate Hessian routines return one). Adapters read ``P``
    differently (some its upper triangle, some its lower, some all of it),
    so a triangle passed on as it is would be a different problem for each
    of them. A triangle is mirrored, as the QPLIB loader mirrors the
    triangle QPLIB stores. A matrix that is symmetric to 1e-12 is kept as
    stored, and one whose asymmetry is rounding-sized (at most 1e-8 of its
    largest entry, as a numerically computed export can carry) is replaced
    by its symmetric part (P + P')/2, which defines the same objective. One
    that is neither symmetric nor triangular does not say which problem it
    means, and is rejected.
    """
    if p.shape[0] != p.shape[1]:
        raise ValueError(f"{path.name}: P must be square, got shape {p.shape}")
    p = sp.csc_matrix(p)
    p.eliminate_zeros()
    scale = max(1.0, float(abs(p).max())) if p.nnz else 1.0
    asymmetry = abs(p - p.T)
    largest = float(asymmetry.max()) if asymmetry.nnz else 0.0
    if largest <= 1e-12 * scale:
        return p
    lower = sp.tril(p, -1)
    upper = sp.triu(p, 1)
    if upper.nnz == 0:
        return sp.csc_matrix(p + lower.T)
    if lower.nnz == 0:
        return sp.csc_matrix(p + upper.T)
    if largest <= 1e-8 * scale:
        return sp.csc_matrix((p + p.T) * 0.5)
    raise ValueError(
        f"{path.name}: P is neither symmetric nor triangular; store the full "
        "symmetric Hessian of 0.5 x'Px or one triangle of it"
    )


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    """The arrays of an .npz file, without unpickling anything but ``None``.

    ``np.savez(..., G=None)`` stores a 0-d object array, which numpy can
    only read by unpickling, and unpickling a downloaded file can run code.
    Such an entry is read with an unpickler that knows nothing but numpy's
    array reconstruction, and is left out (as a missing key is) when it
    holds ``None``. Any other object array, a scipy.sparse matrix saved
    with ``np.savez`` among them, is an error.
    """
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            try:
                arrays[key] = data[key]
            except ValueError:
                placeholder = _is_none_placeholder(path, key)
                if placeholder is None:  # not an object array: a damaged file
                    raise
                if not placeholder:
                    raise ValueError(
                        f"{path.name}: {key!r} is an object array; only numeric arrays "
                        "(a sparse matrix stored dense) and None placeholders are read"
                    ) from None
    return arrays


def _is_none_placeholder(path: Path, key: str) -> bool | None:
    """Whether the object array ``key`` is a 0-d array holding None; None if it is no object array."""
    with zipfile.ZipFile(path) as archive, archive.open(f"{key}.npy") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            return None
        if not dtype.hasobject:
            return None
        if shape != ():
            return False
        try:
            value = _ArrayOnlyUnpickler(handle).load()
        except (pickle.UnpicklingError, ValueError, EOFError, TypeError, AttributeError):
            return False
    return isinstance(value, np.ndarray) and value.shape == () and value.item() is None


class _ArrayOnlyUnpickler(pickle.Unpickler):
    """Unpickles numpy arrays of plain Python values and nothing else."""

    def find_class(self, module: str, name: str):
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if name == "_reconstruct" and module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            return _ndarray_reconstruct()
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed in an .npz object array")


def _ndarray_reconstruct():
    try:
        from numpy._core.multiarray import _reconstruct  # numpy >= 2
    except ImportError:  # pragma: no cover - numpy 1.x
        from numpy.core.multiarray import _reconstruct  # type: ignore[no-redef]
    return _reconstruct


def _normalize_subset(value) -> set[str] | None:
    return normalize_name_subset(
        value, default=MPC_QPBENCHMARK_DEFAULT_SUBSET, strip_suffixes=(".npz",)
    )


def _family(name: str) -> str:
    for prefix in ("LIPMWALK", "WHLIPBAL", "QUADCMPC"):
        if name.startswith(prefix):
            return prefix
    return "unknown"
