"""MPC QP Benchmark dataset adapter."""

from __future__ import annotations

from pathlib import Path
import json
import urllib.request

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from solver_benchmarks.transforms.cones import INF_BOUND
from .base import Dataset


MPC_QPBENCHMARK_API_URL = (
    "https://api.github.com/repos/qpsolvers/mpc_qpbenchmark/contents/data"
)
MPC_QPBENCHMARK_RAW_URL = (
    "https://raw.githubusercontent.com/qpsolvers/mpc_qpbenchmark/main/data"
)
MPC_QPBENCHMARK_DEFAULT_SUBSET = ("LIPMWALK0", "WHLIPBAL0", "QUADCMPC1")


class MPCQPBenchmarkDataset(Dataset):
    dataset_id = "mpc_qpbenchmark"
    description = "Structured MPC QPs from qpsolvers/mpc_qpbenchmark."
    data_source = "external download from https://github.com/qpsolvers/mpc_qpbenchmark"
    data_patterns = ("*.npz",)
    prepare_command = "python scripts/prepare_mpc_qpbenchmark.py"

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

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        qp, metadata = read_mpc_qpbenchmark_npz(spec.path)
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
        if all_problems:
            names = mpc_qpbenchmark_remote_problem_names()
        elif problem_names:
            names = list(problem_names)
        else:
            names = list(MPC_QPBENCHMARK_DEFAULT_SUBSET)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            download_mpc_qpbenchmark_problem(name, self.folder)


def read_mpc_qpbenchmark_npz(path: Path) -> tuple[dict, dict]:
    data = np.load(path, allow_pickle=True)
    p = sp.csc_matrix(np.asarray(data["P"], dtype=float))
    q = np.asarray(data["q"], dtype=float).reshape(-1)
    n = q.size

    rows = []
    lower = []
    upper = []

    g = _optional_array(data, "G")
    h = _optional_array(data, "h")
    if g is not None:
        rows.append(sp.csc_matrix(np.asarray(g, dtype=float)))
        upper.append(np.asarray(h, dtype=float).reshape(-1))
        lower.append(np.full(rows[-1].shape[0], -INF_BOUND))

    aeq = _optional_array(data, "A")
    beq = _optional_array(data, "b")
    if aeq is not None:
        rows.append(sp.csc_matrix(np.asarray(aeq, dtype=float)))
        eq_rhs = np.asarray(beq, dtype=float).reshape(-1)
        lower.append(eq_rhs)
        upper.append(eq_rhs)

    lb = _optional_array(data, "lb")
    ub = _optional_array(data, "ub")
    if lb is not None or ub is not None:
        rows.append(sp.eye(n, format="csc"))
        lower.append(
            np.asarray(lb, dtype=float).reshape(-1)
            if lb is not None
            else np.full(n, -INF_BOUND)
        )
        upper.append(
            np.asarray(ub, dtype=float).reshape(-1)
            if ub is not None
            else np.full(n, INF_BOUND)
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
    url = f"{MPC_QPBENCHMARK_RAW_URL}/{stem}.npz"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    target.write_bytes(content)
    return target


def _optional_array(data, key: str):
    if key not in data.files:
        return None
    value = data[key]
    if value.shape == () and value.dtype == object and value.item() is None:
        return None
    return value


def _normalize_subset(value) -> set[str] | None:
    if value is None or value == "all":
        return None
    if value == "default":
        return set(MPC_QPBENCHMARK_DEFAULT_SUBSET)
    if isinstance(value, str):
        return {item.strip().removesuffix(".npz") for item in value.split(",") if item.strip()}
    return {str(item).removesuffix(".npz") for item in value}


def _family(name: str) -> str:
    for prefix in ("LIPMWALK", "WHLIPBAL", "QUADCMPC"):
        if name.startswith(prefix):
            return prefix
    return "unknown"
