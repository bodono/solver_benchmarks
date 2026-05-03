"""Maros-Meszaros QP dataset (qpkit-data HDF5 form, native cone)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import CONE, ProblemData, ProblemSpec

from .base import Dataset

# TODO(maros_meszaros_v2): hardcoded for now. Move under problem_classes/
# (symlink) or add a generic qpkit-data root resolver (env var + option)
# once the qpkit-data R2 fetch story is in place.
_DATA_DIR = Path("/Users/matteosantamaria/qpkit-data/processed/maros_meszaros")


class MarosMeszarosV2Dataset(Dataset):
    dataset_id = "maros_meszaros_v2"
    description = "Maros-Meszaros convex QP collection (qpkit-data HDF5, native cone form)."
    data_patterns = ("*.h5",)

    @property
    def folder(self) -> Path:
        return _DATA_DIR

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []
        specs = []
        for path in sorted(self.folder.glob("*.h5")):
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=path.stem,
                    kind=CONE,
                    path=path,
                    metadata={
                        "source": str(path),
                        "format": "qpkit_h5",
                        "size_bytes": path.stat().st_size,
                    },
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        data = _read_qpkit_h5(spec.path)
        return ProblemData(self.dataset_id, name, CONE, data, metadata=dict(spec.metadata))


def _read_qpkit_h5(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        m = int(f.attrs["m"])
        n = int(f.attrs["n"])
        P = sp.csc_matrix(
            (f["P.data"][:], f["P.indices"][:], f["P.indptr"][:]),
            shape=(n, n),
        )
        A = sp.csc_matrix(
            (f["A.data"][:], f["A.indices"][:], f["A.indptr"][:]),
            shape=(m, n),
        )
        b = np.asarray(f["b"][:], dtype=float)
        c = np.asarray(f["c"][:], dtype=float)
        j = int(f["j"][()])

    # qpkit form maps directly to (zero, nonneg) cones with rows already
    # in (z, l) order: A x + s = b, s[:j] == 0, s[j:] >= 0. The qpkit_h5
    # schema does not preserve the objective constant, so r = 0.0 — fine
    # for relative comparisons but absolute objectives shift by a per-
    # problem constant vs. the original .mat data.
    return {
        "P": P,
        "q": c,
        "r": 0.0,
        "A": A,
        "b": b,
        "cone": {"z": j, "l": m - j},
        "n": n,
        "m": m,
        "obj_type": "min",
    }
