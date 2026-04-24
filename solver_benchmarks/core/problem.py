"""Problem and dataset data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import scipy.sparse as sp

QP = "qp"
CONE = "cone"


@dataclass(frozen=True)
class ProblemSpec:
    dataset_id: str
    name: str
    kind: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemData:
    dataset_id: str
    name: str
    kind: str
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def qp(self) -> dict[str, Any]:
        if self.kind != QP:
            raise TypeError(f"Problem {self.name!r} is {self.kind}, not qp")
        return self.data

    @property
    def cone(self) -> dict[str, Any]:
        if self.kind != CONE:
            raise TypeError(f"Problem {self.name!r} is {self.kind}, not cone")
        return self.data


def qp_dimensions(qp: dict[str, Any]) -> dict[str, int]:
    p = qp["P"]
    a = qp["A"]
    return {
        "n": int(qp.get("n", p.shape[0])),
        "m": int(qp.get("m", a.shape[0])),
        "nnz_p": int(p.nnz if sp.issparse(p) else (p != 0).sum()),
        "nnz_a": int(a.nnz if sp.issparse(a) else (a != 0).sum()),
    }


def cone_dimensions(cone_problem: dict[str, Any]) -> dict[str, int]:
    a = cone_problem["A"]
    return {
        "n": int(cone_problem.get("n", a.shape[1])),
        "m": int(cone_problem.get("m", a.shape[0])),
        "nnz_a": int(a.nnz if sp.issparse(a) else (a != 0).sum()),
    }
