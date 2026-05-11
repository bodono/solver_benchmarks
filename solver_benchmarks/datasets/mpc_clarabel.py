"""MPC Benchmarking Collection targets exported for Clarabel.

The source MATLAB benchmark structs describe a finite-horizon linear MPC
problem. This adapter initializes the same default fields as the MATLAB
``Benchmark`` class and emits the suite's sparse QP schema:

    minimize 0.5 z' P z + q' z + r
    subject to l <= A z <= u

Variables are ordered as all predicted states ``x_1, ..., x_N`` followed by
the control variables. If a target defines ``uIdx`` move-blocking, one control
variable block is used for each listed move; otherwise every stage has its own
control variable.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.sparse as sp

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from solver_benchmarks.transforms.cones import INF_BOUND

from .base import Dataset

MPC_CLARABEL_DEFAULT_SUBSET = ("toyExample_1", "dcMotor_1", "quadcopter_1")

_NAME_RE = re.compile(r"^(?P<family>.+)_(?P<variant>\d+)$")


class MPCClarabelDataset(Dataset):
    dataset_id = "mpc_clarabel"
    description = "CAESAR MPC Benchmarking Collection targets exported as QPs."
    data_source = (
        "bundled compiled MATLAB targets under problem_classes/mpc_clarabel/targets; "
        "source definitions from the MPC Benchmarking Collection"
    )
    data_patterns = ("targets/*.mat",)
    prepare_command = "python scripts/prepare_mpc_clarabel.py"

    @property
    def folder(self) -> Path:
        return self.problem_classes_dir / "mpc_clarabel" / "targets"

    @property
    def data_dir(self) -> Path:
        return self.folder

    def list_problems(self) -> list[ProblemSpec]:
        if not self.folder.is_dir():
            return []

        subset = _normalize_subset(self.options.get("subset"))
        families = _option_values(self.options.get("family"))
        variants = _option_values(self.options.get("variant"))
        specs: list[ProblemSpec] = []
        for path in sorted(self.folder.glob("*.mat"), key=_problem_sort_key):
            metadata = _metadata_from_name(path.stem)
            if subset is not None and path.stem not in subset:
                continue
            if families is not None and metadata["family"] not in families:
                continue
            if variants is not None and str(metadata["variant"]) not in variants:
                continue
            specs.append(
                ProblemSpec(
                    dataset_id=self.dataset_id,
                    name=path.stem,
                    kind=QP,
                    path=path,
                    metadata={
                        "source": str(path),
                        "format": "mpc_clarabel_mat",
                        "size_bytes": path.stat().st_size,
                        **metadata,
                    },
                )
            )
        return specs

    def load_problem(self, name: str) -> ProblemData:
        spec = self.problem_by_name(name)
        assert spec.path is not None
        qp, metadata = read_mpc_clarabel_mat(spec.path)
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
        if not self.folder.is_dir() or not any(self.folder.glob("*.mat")):
            raise RuntimeError(
                "MPC Clarabel target data is missing. Restore the compiled "
                f"MATLAB targets under {self.folder}, or regenerate them with "
                "problem_classes/mpc_clarabel/make_targets.m from MATLAB/Octave."
            )
        if problem_names:
            available = {path.stem for path in self.folder.glob("*.mat")}
            missing = [name for name in problem_names if Path(name).stem not in available]
            if missing:
                raise RuntimeError(
                    "Unknown MPC Clarabel problem(s): " + ", ".join(sorted(missing))
                )
        _ = all_problems


def read_mpc_clarabel_mat(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    try:
        benchmark = data["data"]
    except KeyError as exc:
        raise ValueError(f"{path} does not contain the expected 'data' struct") from exc

    qp, dimensions = _benchmark_to_qp(benchmark)
    metadata = {
        **_metadata_from_name(path.stem),
        **_benchmark_info(benchmark),
        **dimensions,
        "num_variables": int(qp["n"]),
        "num_constraints": int(qp["m"]),
        "nnz_p": int(qp["P"].nnz),
        "nnz_a": int(qp["A"].nnz),
    }
    return qp, metadata


def _benchmark_to_qp(data) -> tuple[dict[str, Any], dict[str, int]]:
    horizon = int(_scalar(getattr(data, "ni")))
    if horizon <= 0:
        raise ValueError("MPC Clarabel benchmark horizon must be positive")

    nx, nu, ny = _dimensions(data)
    x0 = _initial_state(getattr(data, "x0", np.array([])), nx)

    a_seq = _stage_sequence(data.A, horizon, lambda value: _matrix(value, nx, nx), None)
    b_seq = _stage_sequence(data.B, horizon, lambda value: _matrix(value, nx, nu), None)
    f_seq = _stage_sequence(
        _field(data, "f"),
        horizon,
        lambda value: _vector(value, nx, np.zeros(nx)),
        np.zeros(nx),
    )
    c_seq = _stage_sequence(
        _field(data, "C"),
        horizon,
        lambda value: _matrix(value, ny, nx),
        np.eye(nx) if ny == nx else np.zeros((ny, nx)),
    )
    d_seq = _stage_sequence(
        _field(data, "D"),
        horizon,
        lambda value: _matrix(value, ny, nu),
        np.zeros((ny, nu)),
    )
    e_seq = _stage_sequence(
        _field(data, "e"),
        horizon,
        lambda value: _vector(value, ny, np.zeros(ny)),
        np.zeros(ny),
    )

    q_seq = _stage_sequence(
        _field(data, "Q"),
        horizon,
        lambda value: _square_matrix(value, ny, np.eye(ny)),
        np.eye(ny),
    )
    r_seq = _stage_sequence(
        _field(data, "R"),
        horizon,
        lambda value: _square_matrix(value, nu, np.zeros((nu, nu))),
        np.zeros((nu, nu)),
    )
    s_seq = _stage_sequence(
        _field(data, "S"),
        horizon,
        lambda value: _matrix(value, ny, nu),
        np.zeros((ny, nu)),
    )
    gy_seq = _stage_sequence(
        _field(data, "gy"),
        horizon,
        lambda value: _vector(value, ny, np.zeros(ny)),
        np.zeros(ny),
    )
    gu_seq = _stage_sequence(
        _field(data, "gu"),
        horizon,
        lambda value: _vector(value, nu, np.zeros(nu)),
        np.zeros(nu),
    )
    yr_seq = _stage_sequence(
        _field(data, "yr"),
        horizon,
        lambda value: _vector(value, ny, np.zeros(ny)),
        np.zeros(ny),
    )
    ur_seq = _stage_sequence(
        _field(data, "ur"),
        horizon,
        lambda value: _vector(value, nu, np.zeros(nu)),
        np.zeros(nu),
    )

    y_min_seq = _stage_sequence(
        _field(data, "ymin"),
        horizon,
        lambda value: _vector(value, ny, np.full(ny, -INF_BOUND)),
        np.full(ny, -INF_BOUND),
    )
    y_max_seq = _stage_sequence(
        _field(data, "ymax"),
        horizon,
        lambda value: _vector(value, ny, np.full(ny, INF_BOUND)),
        np.full(ny, INF_BOUND),
    )
    u_min_seq = _stage_sequence(
        _field(data, "umin"),
        horizon,
        lambda value: _vector(value, nu, np.full(nu, -INF_BOUND)),
        np.full(nu, -INF_BOUND),
    )
    u_max_seq = _stage_sequence(
        _field(data, "umax"),
        horizon,
        lambda value: _vector(value, nu, np.full(nu, INF_BOUND)),
        np.full(nu, INF_BOUND),
    )

    poly_rows = _poly_row_count(_field(data, "M"), _field(data, "N"))
    m_seq = _stage_sequence(
        _field(data, "M"),
        horizon,
        lambda value: _matrix(value, poly_rows, ny),
        np.zeros((poly_rows, ny)),
    )
    n_seq = _stage_sequence(
        _field(data, "N"),
        horizon,
        lambda value: _matrix(value, poly_rows, nu),
        np.zeros((poly_rows, nu)),
    )
    d_min_seq = _stage_sequence(
        _field(data, "dmin"),
        horizon,
        lambda value: _vector(value, poly_rows, np.full(poly_rows, -INF_BOUND)),
        np.full(poly_rows, -INF_BOUND),
    )
    d_max_seq = _stage_sequence(
        _field(data, "dmax"),
        horizon,
        lambda value: _vector(value, poly_rows, np.full(poly_rows, INF_BOUND)),
        np.full(poly_rows, INF_BOUND),
    )

    control_blocks, control_block_count = _control_blocks(_field(data, "uIdx"), horizon)
    state_count = horizon * nx
    nvar = state_count + control_block_count * nu

    p = sp.csc_matrix((nvar, nvar), dtype=float)
    q = np.zeros(nvar, dtype=float)
    constant = 0.0

    for stage in range(horizon):
        y_mat, y_const = _output_expression(
            stage,
            nvar,
            nx,
            nu,
            state_count,
            control_blocks,
            c_seq[stage],
            d_seq[stage],
            e_seq[stage],
            x0,
        )
        u_mat = _input_expression(stage, nvar, nu, state_count, control_blocks)
        y_offset = y_const - yr_seq[stage]
        u_offset = -ur_seq[stage]
        p, q, const = _add_square_cost(p, q, y_mat, q_seq[stage], y_offset)
        constant += const
        p, q, const = _add_square_cost(p, q, u_mat, r_seq[stage], u_offset)
        constant += const
        p, q, const = _add_cross_cost(
            p,
            q,
            y_mat,
            u_mat,
            s_seq[stage],
            y_offset,
            u_offset,
        )
        constant += const
        q += _linear_cost(y_mat, gy_seq[stage])
        q += _linear_cost(u_mat, gu_seq[stage])
        constant += float(gy_seq[stage] @ y_offset + gu_seq[stage] @ u_offset)

    p_term = _square_matrix(_field(data, "P"), nx, np.zeros((nx, nx)))
    x_ref = _vector(_field(data, "xNr"), nx, np.zeros(nx))
    x_terminal = _state_expression(horizon, nvar, nx)
    p, q, const = _add_square_cost(p, q, x_terminal, p_term, -x_ref)
    constant += const

    rows: list[sp.csc_matrix] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []

    for stage in range(horizon):
        dyn, rhs = _dynamics_expression(
            stage,
            nvar,
            nx,
            nu,
            state_count,
            control_blocks,
            a_seq[stage],
            b_seq[stage],
            f_seq[stage],
            x0,
        )
        rows.append(dyn)
        lower.append(rhs)
        upper.append(rhs)

        y_mat, y_const = _output_expression(
            stage,
            nvar,
            nx,
            nu,
            state_count,
            control_blocks,
            c_seq[stage],
            d_seq[stage],
            e_seq[stage],
            x0,
        )
        _append_bounded_rows(
            rows,
            lower,
            upper,
            y_mat,
            y_min_seq[stage] - y_const,
            y_max_seq[stage] - y_const,
        )

        u_mat = _input_expression(stage, nvar, nu, state_count, control_blocks)
        _append_bounded_rows(
            rows,
            lower,
            upper,
            u_mat,
            u_min_seq[stage],
            u_max_seq[stage],
        )

        if poly_rows:
            poly_mat = m_seq[stage] @ y_mat + n_seq[stage] @ u_mat
            poly_const = m_seq[stage] @ y_const
            _append_bounded_rows(
                rows,
                lower,
                upper,
                sp.csc_matrix(poly_mat),
                d_min_seq[stage] - poly_const,
                d_max_seq[stage] - poly_const,
            )

    terminal_matrix = _optional_matrix(_field(data, "T"), nx)
    if terminal_matrix is not None:
        terminal_rows = terminal_matrix.shape[0]
        d_n_min = _vector(
            _field(data, "dNmin"),
            terminal_rows,
            np.full(terminal_rows, -INF_BOUND),
        )
        d_n_max = _vector(
            _field(data, "dNmax"),
            terminal_rows,
            np.full(terminal_rows, INF_BOUND),
        )
        terminal_expr = terminal_matrix @ x_terminal
        _append_bounded_rows(
            rows,
            lower,
            upper,
            sp.csc_matrix(terminal_expr),
            d_n_min,
            d_n_max,
        )

    if rows:
        a = sp.vstack(rows, format="csc")
        l = _canonical_bounds(np.concatenate(lower))
        u = _canonical_bounds(np.concatenate(upper))
    else:
        a = sp.csc_matrix((0, nvar))
        l = np.array([], dtype=float)
        u = np.array([], dtype=float)

    p = ((p + p.T) * 0.5).tocsc()
    p.eliminate_zeros()
    a.eliminate_zeros()
    qp = {
        "P": p,
        "q": np.asarray(q, dtype=float),
        "r": float(constant),
        "A": a,
        "l": l,
        "u": u,
        "n": int(nvar),
        "m": int(a.shape[0]),
        "obj_type": "min",
    }
    dimensions = {
        "horizon": int(horizon),
        "state_dim": int(nx),
        "input_dim": int(nu),
        "output_dim": int(ny),
        "control_blocks": int(control_block_count),
    }
    return qp, dimensions


def _dimensions(data) -> tuple[int, int, int]:
    a = np.asarray(data.A, dtype=float)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("MPC Clarabel A matrix must be square")
    nx = int(a.shape[0])

    b = np.asarray(data.B, dtype=float)
    if b.ndim == 0:
        b = b.reshape(1, 1)
    if b.ndim == 1:
        if b.size != nx:
            raise ValueError("MPC Clarabel B vector must have one entry per state")
        nu = 1
    elif b.ndim == 2 and b.shape[0] == nx:
        nu = int(b.shape[1])
    else:
        raise ValueError("MPC Clarabel B matrix has inconsistent dimensions")

    if _is_empty(_field(data, "C")):
        ny = nx
    else:
        c = np.asarray(_field(data, "C"), dtype=float)
        if c.ndim == 0:
            c = c.reshape(1, 1)
        if c.ndim == 1:
            c = c.reshape(1, -1)
        if c.ndim != 2 or c.shape[1] != nx:
            raise ValueError("MPC Clarabel C matrix has inconsistent dimensions")
        ny = int(c.shape[0])
    return nx, nu, ny


def _output_expression(
    stage: int,
    nvar: int,
    nx: int,
    nu: int,
    state_count: int,
    control_blocks: list[int],
    c_matrix: np.ndarray,
    d_matrix: np.ndarray,
    e_vector: np.ndarray,
    x0: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    ny = c_matrix.shape[0]
    mat = sp.lil_matrix((ny, nvar), dtype=float)
    const = np.array(e_vector, dtype=float, copy=True)
    if stage == 0:
        const = const + c_matrix @ x0
    else:
        mat[:, _state_slice(stage, nx)] = c_matrix
    mat[:, _control_slice(stage, nu, state_count, control_blocks)] = d_matrix
    return mat.tocsc(), const


def _input_expression(
    stage: int,
    nvar: int,
    nu: int,
    state_count: int,
    control_blocks: list[int],
) -> sp.csc_matrix:
    mat = sp.lil_matrix((nu, nvar), dtype=float)
    mat[:, _control_slice(stage, nu, state_count, control_blocks)] = np.eye(nu)
    return mat.tocsc()


def _state_expression(stage: int, nvar: int, nx: int) -> sp.csc_matrix:
    mat = sp.lil_matrix((nx, nvar), dtype=float)
    mat[:, _state_slice(stage, nx)] = np.eye(nx)
    return mat.tocsc()


def _dynamics_expression(
    stage: int,
    nvar: int,
    nx: int,
    nu: int,
    state_count: int,
    control_blocks: list[int],
    a_matrix: np.ndarray,
    b_matrix: np.ndarray,
    f_vector: np.ndarray,
    x0: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    mat = sp.lil_matrix((nx, nvar), dtype=float)
    mat[:, _state_slice(stage + 1, nx)] = np.eye(nx)
    if stage == 0:
        rhs = a_matrix @ x0 + f_vector
    else:
        mat[:, _state_slice(stage, nx)] = -a_matrix
        rhs = f_vector
    mat[:, _control_slice(stage, nu, state_count, control_blocks)] = -b_matrix
    return mat.tocsc(), np.asarray(rhs, dtype=float)


def _add_square_cost(
    p: sp.csc_matrix,
    q: np.ndarray,
    expr: sp.csc_matrix,
    weight: np.ndarray,
    offset: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray, float]:
    weight_sparse = sp.csc_matrix(weight)
    p = p + 2.0 * (expr.T @ weight_sparse @ expr)
    q = q + np.asarray(2.0 * (expr.T @ (weight @ offset))).reshape(-1)
    const = float(offset @ weight @ offset)
    return p.tocsc(), q, const


def _add_cross_cost(
    p: sp.csc_matrix,
    q: np.ndarray,
    y_expr: sp.csc_matrix,
    u_expr: sp.csc_matrix,
    weight: np.ndarray,
    y_offset: np.ndarray,
    u_offset: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray, float]:
    if not np.any(weight):
        return p, q, 0.0
    weight_sparse = sp.csc_matrix(weight)
    cross = y_expr.T @ weight_sparse @ u_expr
    p = p + cross + cross.T
    q = q + np.asarray(y_expr.T @ (weight @ u_offset)).reshape(-1)
    q = q + np.asarray(u_expr.T @ (weight.T @ y_offset)).reshape(-1)
    const = float(y_offset @ weight @ u_offset)
    return p.tocsc(), q, const


def _linear_cost(expr: sp.csc_matrix, coeff: np.ndarray) -> np.ndarray:
    if not np.any(coeff):
        return np.zeros(expr.shape[1])
    return np.asarray(expr.T @ coeff).reshape(-1)


def _append_bounded_rows(
    rows: list[sp.csc_matrix],
    lower: list[np.ndarray],
    upper: list[np.ndarray],
    matrix: sp.csc_matrix,
    row_lower: np.ndarray,
    row_upper: np.ndarray,
) -> None:
    row_lower = _canonical_bounds(row_lower)
    row_upper = _canonical_bounds(row_upper)
    keep = (row_lower > -INF_BOUND) | (row_upper < INF_BOUND)
    if not np.any(keep):
        return
    rows.append(sp.csc_matrix(matrix[keep, :]))
    lower.append(row_lower[keep])
    upper.append(row_upper[keep])


def _state_slice(stage: int, nx: int) -> slice:
    if stage <= 0:
        raise ValueError("Predicted state slices are defined for stages 1..N")
    start = (stage - 1) * nx
    return slice(start, start + nx)


def _control_slice(
    stage: int,
    nu: int,
    state_count: int,
    control_blocks: list[int],
) -> slice:
    start = state_count + control_blocks[stage] * nu
    return slice(start, start + nu)


def _control_blocks(raw_u_idx, horizon: int) -> tuple[list[int], int]:
    if _is_empty(raw_u_idx):
        return list(range(horizon)), horizon

    moves = [int(value) for value in np.asarray(raw_u_idx).reshape(-1) if int(value) <= horizon]
    if not moves:
        return list(range(horizon)), horizon
    if moves[0] != 1:
        raise ValueError("MPC Clarabel move-blocking uIdx must start at 1")

    stage_blocks: list[int] = []
    current = 0
    for stage in range(1, horizon + 1):
        while current + 1 < len(moves) and moves[current + 1] <= stage:
            current += 1
        stage_blocks.append(current)
    return stage_blocks, len(moves)


def _poly_row_count(raw_m, raw_n) -> int:
    if not _is_empty(raw_m):
        m = np.asarray(raw_m)
        if m.ndim == 1:
            return int(m.size)
        return int(m.shape[0])
    if not _is_empty(raw_n):
        n = np.asarray(raw_n)
        if n.ndim == 1:
            return int(n.size)
        return int(n.shape[0])
    return 0


def _stage_sequence(
    value: Any,
    length: int,
    convert: Callable[[Any], np.ndarray],
    default: np.ndarray | None,
) -> list[np.ndarray]:
    if _is_object_array(value):
        raw_values = list(np.asarray(value, dtype=object).ravel(order="F"))
    else:
        raw_values = [value]

    result: list[np.ndarray] = []
    previous = default
    for index in range(length):
        raw = raw_values[index] if index < len(raw_values) else None
        if raw is None or _is_empty(raw):
            if previous is None:
                raise ValueError("Missing required MPC Clarabel stage data")
            converted = np.array(previous, dtype=float, copy=True)
        else:
            converted = convert(raw)
        result.append(converted)
        previous = converted
    return result


def _matrix(value: Any, rows: int, cols: int) -> np.ndarray:
    if _is_empty(value):
        raise ValueError(f"Expected a {rows} x {cols} matrix, got empty data")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        if cols == 1 and arr.size == rows:
            arr = arr.reshape(rows, 1)
        elif rows == 1 and arr.size == cols:
            arr = arr.reshape(1, cols)
    if arr.shape != (rows, cols):
        raise ValueError(
            f"Expected a {rows} x {cols} matrix, got shape {tuple(arr.shape)}"
        )
    return np.asarray(arr, dtype=float)


def _square_matrix(value: Any, dim: int, default: np.ndarray) -> np.ndarray:
    if _is_empty(value):
        return np.array(default, dtype=float, copy=True)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        if dim != 1:
            raise ValueError(f"Expected a {dim} x {dim} matrix, got a scalar")
        arr = arr.reshape(1, 1)
    if arr.ndim == 1 and arr.size == dim:
        arr = np.diag(arr)
    if arr.shape != (dim, dim):
        raise ValueError(
            f"Expected a {dim} x {dim} matrix, got shape {tuple(arr.shape)}"
        )
    return np.asarray(arr, dtype=float)


def _optional_matrix(value: Any, cols: int) -> np.ndarray | None:
    if _is_empty(value):
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != cols:
        raise ValueError(f"Expected terminal matrix with {cols} columns")
    return arr


def _vector(value: Any, length: int, default: np.ndarray) -> np.ndarray:
    if _is_empty(value):
        return np.array(default, dtype=float, copy=True)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(length, float(arr))
    else:
        arr = arr.reshape(-1)
        if arr.size == 1 and length != 1:
            arr = np.full(length, float(arr[0]))
    if arr.size != length:
        raise ValueError(f"Expected vector of length {length}, got length {arr.size}")
    return _canonical_bounds(arr.astype(float))


def _initial_state(value: Any, nx: int) -> np.ndarray:
    if _is_object_array(value):
        value = np.asarray(value, dtype=object).ravel(order="F")[0]
    if _is_empty(value):
        return np.zeros(nx)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == nx:
        arr = arr[:, 0]
    return _vector(arr, nx, np.zeros(nx))


def _canonical_bounds(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[np.isneginf(out)] = -INF_BOUND
    out[np.isposinf(out)] = INF_BOUND
    out[out < -INF_BOUND] = -INF_BOUND
    out[out > INF_BOUND] = INF_BOUND
    return out


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"Expected scalar value, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def _is_empty(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.size == 0


def _is_object_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.dtype == object


def _field(data, name: str) -> Any:
    return getattr(data, name, np.array([]))


def _metadata_from_name(name: str) -> dict[str, Any]:
    match = _NAME_RE.match(name)
    if match is None:
        return {"family": name, "variant": 0}
    return {"family": match.group("family"), "variant": int(match.group("variant"))}


def _benchmark_info(data) -> dict[str, Any]:
    info = getattr(data, "info", None)
    if info is None:
        return {}
    result: dict[str, Any] = {}
    for attr, key in (
        ("ID", "benchmark_id"),
        ("name", "benchmark_name"),
        ("description", "description"),
        ("reference", "reference"),
    ):
        value = getattr(info, attr, None)
        if value is None or _is_empty(value):
            continue
        if isinstance(value, str):
            result[key] = value
        else:
            arr = np.asarray(value)
            result[key] = int(arr) if arr.size == 1 else arr.tolist()
    return result


def _normalize_subset(value: Any) -> set[str] | None:
    if value is None or value == "all":
        return None
    if value == "default":
        return set(MPC_CLARABEL_DEFAULT_SUBSET)
    if isinstance(value, str):
        return {Path(item.strip()).stem for item in value.split(",") if item.strip()}
    return {Path(str(item)).stem for item in value}


def _option_values(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item) for item in value}


def _problem_sort_key(path: Path) -> tuple[str, int, str]:
    metadata = _metadata_from_name(path.stem)
    return (str(metadata["family"]), int(metadata["variant"]), path.stem)
