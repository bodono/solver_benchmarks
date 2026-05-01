"""Liu-Pataki semidefinite benchmark dataset."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec

from .base import Dataset

_NAME_RE = re.compile(
    r"^(?P<classification>infeas|weak)_"
    r"(?P<conditioning>clean|messy)_"
    r"(?P<constraints>\d+)_"
    r"(?P<block_dim>\d+)_"
    r"(?P<sample>\d+)$"
)


class LiuPatakiDataset(Dataset):
    dataset_id = "liu_pataki"
    description = "Liu-Pataki infeasible and weakly infeasible SDP collection."
    data_patterns = ("*.mat",)

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "liu_pataki_data"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []

        specs = []
        for path in sorted(self.folder.glob("*.mat"), key=_problem_sort_key):
            metadata = _metadata_from_name(path.stem)
            if not _passes_filters(metadata, self.options):
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=path.stem,
                    kind=CONE,
                    path=path,
                    metadata={
                        "source": str(path),
                        "format": "sedumi_mat",
                        "size_bytes": path.stat().st_size,
                        **metadata,
                    },
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        problem = _read_liu_pataki_mat(spec.path)
        return ProblemData(
            self.dataset_id,
            name,
            CONE,
            problem,
            metadata=dict(spec.metadata),
        )


def _read_liu_pataki_mat(path: Path) -> dict:
    data = scipy.io.loadmat(path)
    a_full = sp.csc_matrix(data["A"], dtype=float)
    b = np.asarray(data["b"], dtype=float).reshape(-1)
    c_full = np.asarray(data["c"], dtype=float).reshape(-1)
    s_blocks = _sedumi_psd_blocks(data["K"])

    a_tri, q_tri = _full_sedumi_psd_to_triangle(a_full, c_full, s_blocks)
    cone_rows = a_tri.shape[1]
    a = sp.vstack((a_tri, -sp.eye(cone_rows, format="csc")), format="csc")
    rhs = np.concatenate((b, np.zeros(cone_rows)))
    cone = {"z": int(b.size), "s": s_blocks}
    return {
        "P": None,
        "A": a,
        "b": rhs,
        "q": q_tri,
        "r": 0.0,
        "n": int(a.shape[1]),
        "m": int(a.shape[0]),
        "cone": cone,
        "obj_type": "min",
    }


def _full_sedumi_psd_to_triangle(
    a_full: sp.csc_matrix,
    c_full: np.ndarray,
    s_blocks: list[int],
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Convert SeDuMi full PSD blocks to canonical triangle-vector columns.

    SeDuMi stores an ``n x n`` PSD block as ``vec(X)`` with all ``n*n`` matrix
    entries present. The cone itself is symmetric, so off-diagonal entries
    ``X[i, j]`` and ``X[j, i]`` represent the same decision variable. The
    benchmark's canonical PSD representation stores the lower triangle in
    column-major order with sqrt(2) scaling on off-diagonal entries.
    """
    columns = []
    q_parts = []
    offset = 0
    root_two = np.sqrt(2.0)
    for block_dim in s_blocks:
        width = block_dim * block_dim
        block_a = a_full[:, offset : offset + width]
        block_c = c_full[offset : offset + width]
        for col in range(block_dim):
            for row in range(col, block_dim):
                lower = row + col * block_dim
                if row == col:
                    columns.append(block_a[:, lower])
                    q_parts.append(float(block_c[lower]))
                else:
                    upper = col + row * block_dim
                    columns.append((block_a[:, lower] + block_a[:, upper]) / root_two)
                    q_parts.append(
                        float((block_c[lower] + block_c[upper]) / root_two)
                    )
        offset += width

    if offset != a_full.shape[1] or offset != c_full.size:
        raise ValueError(
            "Liu-Pataki SeDuMi dimensions do not match PSD block sizes: "
            f"K.s implies {offset} variables, A has {a_full.shape[1]}, "
            f"c has {c_full.size}"
        )
    if not columns:
        return sp.csc_matrix((a_full.shape[0], 0)), np.zeros(0)
    return sp.hstack(columns, format="csc"), np.asarray(q_parts, dtype=float)


def _sedumi_psd_blocks(k) -> list[int]:
    names = k.dtype.names or ()
    unsupported = sorted(
        name for name in names if name != "s" and _field_sum(k, name) != 0
    )
    if unsupported:
        raise ValueError(
            "Liu-Pataki adapter only supports PSD SeDuMi cones; "
            f"found {unsupported}"
        )
    if "s" not in names:
        raise ValueError("Liu-Pataki problem does not define K.s PSD blocks")
    values = np.asarray(k["s"][0][0], dtype=int).reshape(-1)
    return [int(value) for value in values if int(value) > 0]


def _field_sum(k, name: str) -> float:
    return float(np.asarray(k[name][0][0], dtype=float).sum())


def _metadata_from_name(name: str) -> dict:
    match = _NAME_RE.match(name)
    if match is None:
        return {}
    return {
        "classification": match.group("classification"),
        "conditioning": match.group("conditioning"),
        "constraint_count": int(match.group("constraints")),
        "block_dim": int(match.group("block_dim")),
        "sample_index": int(match.group("sample")),
    }


def _passes_filters(metadata: dict, options: dict) -> bool:
    return (
        _matches_option(metadata.get("classification"), options.get("classification"))
        and _matches_option(metadata.get("conditioning"), options.get("conditioning"))
        and _matches_option(
            metadata.get("constraint_count"),
            options.get("constraint_count"),
        )
        and _matches_option(metadata.get("block_dim"), options.get("block_dim"))
    )


def _matches_option(value, option) -> bool:
    allowed = _option_values(option)
    if allowed is None:
        return True
    return str(value) in allowed


def _option_values(option) -> set[str] | None:
    if option is None:
        return None
    if isinstance(option, (list, tuple, set)):
        return {str(value) for value in option}
    return {part.strip() for part in str(option).split(",") if part.strip()}


def _problem_sort_key(path: Path):
    metadata = _metadata_from_name(path.stem)
    return (
        metadata.get("classification", ""),
        metadata.get("conditioning", ""),
        metadata.get("constraint_count", 0),
        metadata.get("block_dim", 0),
        metadata.get("sample_index", 0),
        path.stem,
    )
