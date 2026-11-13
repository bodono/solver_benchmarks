"""Problem and dataset data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import scipy.sparse as sp

QP = "qp"
CONE = "cone"

# Row layout of a cone problem. The rows of ``A`` and ``b`` are stacked cone
# by cone in one fixed order, the one SCS uses: zero cone, nonnegative cone,
# box cone, second-order cones, PSD cones, complex PSD cones, exponential
# cones, dual exponential cones, power cones. ``f`` is an older name for
# ``z``. The order in which the ``cone`` dictionary lists its keys does not
# set the layout, so code that needs the rows of a cone must walk the keys
# with ``cone_items_in_row_order`` and never in dictionary order.
_CONE_KEY_RANK = {
    "f": 0,
    "z": 0,
    "l": 1,
    "bl": 2,
    "bu": 2,
    "q": 3,
    "s": 4,
    "cs": 5,
    "ep": 6,
    "ed": 7,
    "p": 8,
}


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

    def __post_init__(self) -> None:
        if self.kind == CONE:
            check_cone_layout(self.name, self.data)

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


def cone_items_in_row_order(cone: dict[str, Any]) -> list[tuple[str, Any]]:
    """The ``(key, value)`` pairs of a cone dictionary in the order their rows
    are stacked in ``A`` and ``b``.

    Keys this module does not know come last, in the order given.
    """
    unknown = len(_CONE_KEY_RANK)
    return sorted(cone.items(), key=lambda item: _CONE_KEY_RANK.get(item[0], unknown))


def cone_row_count(cone: dict[str, Any]) -> int | None:
    """Number of rows a cone dictionary accounts for.

    Returns ``None`` when the dictionary has a key this module cannot size.
    """
    rows = 0
    box_counted = False
    for key, value in cone.items():
        if key in ("f", "z", "l"):
            rows += int(value)
        elif key == "q":
            rows += sum(int(dim) for dim in _as_list(value))
        elif key == "s":
            rows += sum(int(dim) * (int(dim) + 1) // 2 for dim in _as_list(value))
        elif key == "cs":
            rows += sum(int(dim) * int(dim) for dim in _as_list(value))
        elif key in ("bl", "bu"):
            if not box_counted:
                # One row for the scaling variable and one per bound; an
                # empty box takes no rows at all, as in SCS.
                width = max(len(_as_list(cone.get("bl", []))), len(_as_list(cone.get("bu", []))))
                rows += 1 + width if width else 0
                box_counted = True
        elif key in ("ep", "ed"):
            rows += 3 * int(value)
        elif key == "p":
            rows += 3 * len(_as_list(value))
        else:
            return None
    return rows


def check_cone_layout(name: str, cone_problem: dict[str, Any]) -> None:
    """Reject a cone problem whose ``cone`` dictionary cannot describe its rows.

    The keys must be listed in row order. A loader that lists them in another
    order may also have stacked its rows in that order, and the two cannot be
    told apart from the dictionary, so it is refused here instead of being
    posed wrongly to the solvers that follow the fixed layout. When every key
    can be sized, the sizes must add up to the number of rows of ``A``, and
    ``b`` must have one entry per row.

    This catches a dictionary that is listed out of order or does not account
    for the rows. It cannot catch rows stacked in another order under a
    dictionary that is listed correctly; only a known optimum shows that.
    """
    cone = cone_problem.get("cone")
    if not isinstance(cone, dict):
        return
    keys = list(cone)
    in_row_order = [key for key, _ in cone_items_in_row_order(cone)]
    if keys != in_row_order:
        raise ValueError(
            f"Problem {name!r}: cone keys {keys} are not in row order {in_row_order}. "
            "Rows are stacked zero, nonnegative, box, second-order, PSD, "
            "exponential, dual exponential, power; list the keys the same way."
        )
    shape = getattr(cone_problem.get("A"), "shape", None)
    if shape is None:
        return
    rows = cone_row_count(cone)
    if rows is not None and rows != int(shape[0]):
        raise ValueError(
            f"Problem {name!r}: the cone dictionary accounts for {rows} rows "
            f"but A has {int(shape[0])}"
        )
    b_size = getattr(cone_problem.get("b"), "size", None)
    if b_size is not None and int(b_size) != int(shape[0]):
        raise ValueError(
            f"Problem {name!r}: b has {int(b_size)} entries but A has {int(shape[0])} rows"
        )


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        listed = value.tolist()
        return listed if isinstance(listed, list) else [listed]
    return [value]
