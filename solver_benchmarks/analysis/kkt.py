"""Independent KKT residual computation for QP and cone problems.

All residuals are computed from the primal/dual vectors supplied by the
adapter, using the original problem data — nothing here trusts
solver-reported quantities. Returned dictionaries contain absolute
residuals under their bare names and the SCS-style normalized variants
under a ``_rel`` suffix.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import scipy.sparse as sp

_EPS = 1.0e-16


def qp_residuals(
    P,
    q,
    A,
    l,
    u,
    x,
    y,
) -> dict[str, Any]:
    """KKT residuals for ``min ½ x'Px + q'x  s.t.  l ≤ Ax ≤ u``.

    ``y`` follows the sign convention ``y = λ_u − λ_l`` with
    ``λ_u, λ_l ≥ 0`` (i.e. positive on upper bound, negative on lower).
    """
    x = _as_dense(x)
    y = _as_dense(y)
    P = _as_sparse(P)
    A = _as_sparse(A)
    q = np.asarray(q, dtype=float).ravel()
    l = np.asarray(l, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()

    Ax = A @ x
    Px = P @ x
    Aty = A.T @ y

    primal_violation = np.maximum(l - Ax, 0.0) + np.maximum(Ax - u, 0.0)
    r_pri = float(np.linalg.norm(primal_violation, ord=np.inf))
    r_pri_l2 = float(np.linalg.norm(primal_violation))

    stationarity = Px + q + Aty
    r_dual = float(np.linalg.norm(stationarity, ord=np.inf))
    r_dual_l2 = float(np.linalg.norm(stationarity))

    y_pos = np.maximum(y, 0.0)
    y_neg = np.maximum(-y, 0.0)
    comp_upper = y_pos * np.where(np.isfinite(u), u - Ax, 0.0)
    comp_lower = y_neg * np.where(np.isfinite(l), Ax - l, 0.0)
    comp = np.concatenate([comp_upper, comp_lower])
    r_comp = float(np.linalg.norm(comp, ord=np.inf))

    primal_obj = float(0.5 * x @ Px + q @ x)
    dual_obj = float(
        -0.5 * x @ Px
        + np.where(np.isfinite(l), l, 0.0) @ y_neg
        - np.where(np.isfinite(u), u, 0.0) @ y_pos
    )
    gap = primal_obj - dual_obj
    gap_norm = 1.0 + abs(primal_obj) + abs(dual_obj)

    norm_pri = 1.0 + max(
        _inf_norm(Ax),
        _inf_norm(np.where(np.isfinite(l), l, 0.0)),
        _inf_norm(np.where(np.isfinite(u), u, 0.0)),
    )
    norm_dual = 1.0 + max(_inf_norm(Px), _inf_norm(Aty), _inf_norm(q))

    return {
        "form": "qp",
        "primal_res": r_pri,
        "primal_res_l2": r_pri_l2,
        "primal_res_rel": r_pri / norm_pri,
        "dual_res": r_dual,
        "dual_res_l2": r_dual_l2,
        "dual_res_rel": r_dual / norm_dual,
        "comp_slack": r_comp,
        "primal_obj": primal_obj,
        "dual_obj": dual_obj,
        "duality_gap": gap,
        "duality_gap_rel": abs(gap) / gap_norm,
    }


def cone_residuals(
    P,
    q,
    A,
    b,
    cone: dict[str, Any],
    x,
    y,
    s,
) -> dict[str, Any]:
    """KKT residuals for ``min ½ x'Px + q'x  s.t.  Ax + s = b, s ∈ K``.

    Supported cone keys: ``z``/``f`` (zero), ``l`` (nonnegative),
    ``q`` (second-order), ``s`` (PSD triangle). Other keys (exp, power,
    box) are passed through without projection and reported back under
    ``unsupported_cones``.
    """
    x = _as_dense(x)
    y = _as_dense(y)
    s = _as_dense(s)
    P = _as_sparse(P)
    A = _as_sparse(A)
    q = np.asarray(q, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    Ax = A @ x
    Px = P @ x
    Aty = A.T @ y

    equality = Ax + s - b
    r_eq = float(np.linalg.norm(equality, ord=np.inf))
    r_eq_l2 = float(np.linalg.norm(equality))

    stationarity = Px + q + Aty
    r_dual = float(np.linalg.norm(stationarity, ord=np.inf))
    r_dual_l2 = float(np.linalg.norm(stationarity))

    s_proj, y_proj, unsupported = _project_cones(cone, s, y)
    # If any cone block was unsupported, the projection equals the input
    # for that block by construction (see _project_cones), which would
    # otherwise produce a misleadingly clean cone-residual reading.
    # Surface NaN so the summary reports it as "unknown" instead of
    # silently rolling unsupported blocks into a per-row "all good".
    if unsupported:
        r_s_cone = float("nan")
        r_y_cone = float("nan")
    else:
        r_s_cone = float(np.linalg.norm(s - s_proj, ord=np.inf)) if s_proj is not None else 0.0
        r_y_cone = float(np.linalg.norm(y - y_proj, ord=np.inf)) if y_proj is not None else 0.0

    # comp_slack is reported as |s.y|; the inner product should be 0 at
    # complementarity but can be slightly negative from numerical noise.
    # Downstream code (plots, aggregates) treats it as a magnitude, so
    # report the absolute value here and keep the signed copy as
    # comp_slack_signed for users who want to detect a duality leak.
    comp_signed = float(s @ y)
    comp = abs(comp_signed)
    primal_obj = float(0.5 * x @ Px + q @ x)
    dual_obj = float(-0.5 * x @ Px - b @ y)
    gap = primal_obj - dual_obj
    gap_norm = 1.0 + abs(primal_obj) + abs(dual_obj)

    norm_pri = 1.0 + max(_inf_norm(Ax), _inf_norm(s), _inf_norm(b))
    norm_dual = 1.0 + max(_inf_norm(Px), _inf_norm(Aty), _inf_norm(q))

    out: dict[str, Any] = {
        "form": "cone",
        "primal_res": r_eq,
        "primal_res_l2": r_eq_l2,
        "primal_res_rel": r_eq / norm_pri,
        "dual_res": r_dual,
        "dual_res_l2": r_dual_l2,
        "dual_res_rel": r_dual / norm_dual,
        "primal_cone_res": r_s_cone,
        "dual_cone_res": r_y_cone,
        "comp_slack": comp,
        "comp_slack_signed": comp_signed,
        "primal_obj": primal_obj,
        "dual_obj": dual_obj,
        "duality_gap": gap,
        "duality_gap_rel": abs(gap) / gap_norm,
    }
    if unsupported:
        out["unsupported_cones"] = sorted(set(unsupported))
    return out


def cone_primal_infeasibility_cert(
    A,
    b,
    cone: dict[str, Any],
    y,
) -> dict[str, Any]:
    """Check ``y ∈ K*, A'y = 0, b'y < 0`` for primal infeasibility."""
    y = _as_dense(y)
    A = _as_sparse(A)
    b = np.asarray(b, dtype=float).ravel()
    Aty = A.T @ y
    _, y_proj, unsupported = _project_cones(cone, None, y)
    bty = float(b @ y)
    y_scale = max(_inf_norm(y), _EPS)
    y_cone_res = _inf_norm(y - y_proj) if y_proj is not None else 0.0
    out = {
        "certificate": "primal_infeasible",
        "Aty_inf": _inf_norm(Aty),
        "Aty_rel": _inf_norm(Aty) / y_scale,
        "bty": bty,
        "dual_cone_res": y_cone_res,
        "valid": bty < 0.0,
    }
    if unsupported:
        out["unsupported_cones"] = sorted(set(unsupported))
    return out


def cone_dual_infeasibility_cert(
    P,
    q,
    A,
    cone: dict[str, Any],
    x,
) -> dict[str, Any]:
    """Check ``Px = 0, Ax ∈ -K, q'x < 0`` for dual infeasibility (unbounded)."""
    x = _as_dense(x)
    P = _as_sparse(P)
    A = _as_sparse(A)
    q = np.asarray(q, dtype=float).ravel()
    Px = P @ x
    Ax = A @ x
    neg_Ax = -Ax
    s_proj, _, unsupported = _project_cones(cone, neg_Ax, None)
    cone_res = _inf_norm(neg_Ax - s_proj) if s_proj is not None else 0.0
    qtx = float(q @ x)
    x_scale = max(_inf_norm(x), _EPS)
    out = {
        "certificate": "dual_infeasible",
        "Px_inf": _inf_norm(Px),
        "Px_rel": _inf_norm(Px) / x_scale,
        "qtx": qtx,
        "primal_cone_res": cone_res,
        "valid": qtx < 0.0,
    }
    if unsupported:
        out["unsupported_cones"] = sorted(set(unsupported))
    return out


def qp_primal_infeasibility_cert(
    A,
    l,
    u,
    y,
) -> dict[str, Any]:
    """``A'y = 0`` with ``u'y⁺ − l'y⁻ < 0`` certifies primal infeasibility."""
    y = _as_dense(y)
    A = _as_sparse(A)
    l = np.asarray(l, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    Aty = A.T @ y
    y_pos = np.maximum(y, 0.0)
    y_neg = np.maximum(-y, 0.0)
    support = (
        float(np.where(np.isfinite(u), u, 0.0) @ y_pos)
        - float(np.where(np.isfinite(l), l, 0.0) @ y_neg)
    )
    y_scale = max(_inf_norm(y), _EPS)
    return {
        "certificate": "primal_infeasible",
        "Aty_inf": _inf_norm(Aty),
        "Aty_rel": _inf_norm(Aty) / y_scale,
        "support": support,
        "valid": support < 0.0,
    }


def qp_dual_infeasibility_cert(
    P,
    q,
    A,
    l,
    u,
    x,
) -> dict[str, Any]:
    """``Px = 0``, ``q'x < 0`` with ``Ax`` in the recession cone of ``l ≤ · ≤ u``."""
    x = _as_dense(x)
    P = _as_sparse(P)
    A = _as_sparse(A)
    q = np.asarray(q, dtype=float).ravel()
    l = np.asarray(l, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    Px = P @ x
    Ax = A @ x
    finite_l = np.isfinite(l)
    finite_u = np.isfinite(u)
    both_finite = finite_l & finite_u
    upper_only = finite_u & ~finite_l
    lower_only = finite_l & ~finite_u

    violations = np.zeros_like(Ax)
    violations[both_finite] = np.abs(Ax[both_finite])
    violations[upper_only] = np.maximum(Ax[upper_only], 0.0)
    violations[lower_only] = np.maximum(-Ax[lower_only], 0.0)

    qtx = float(q @ x)
    x_scale = max(_inf_norm(x), _EPS)
    return {
        "certificate": "dual_infeasible",
        "Px_inf": _inf_norm(Px),
        "Px_rel": _inf_norm(Px) / x_scale,
        "Ax_cone_res": _inf_norm(violations),
        "qtx": qtx,
        "valid": qtx < 0.0,
    }


def _project_cones(
    cone: dict[str, Any] | None,
    s,
    y,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    unsupported: list[str] = []
    if cone is None:
        return (None if s is None else np.zeros_like(s),
                None if y is None else np.zeros_like(y), unsupported)

    s_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    row = 0

    def _take(vec, width):
        return vec[row:row + width] if vec is not None else None

    for name, value in cone.items():
        if name in ("f", "z"):
            dim = int(value)
            if s is not None:
                s_parts.append(np.zeros(dim))
            if y is not None:
                y_parts.append(_take(y, dim))
            row += dim
        elif name == "l":
            dim = int(value)
            if s is not None:
                s_parts.append(np.maximum(_take(s, dim), 0.0))
            if y is not None:
                y_parts.append(np.maximum(_take(y, dim), 0.0))
            row += dim
        elif name == "q":
            for dim in _as_list(value):
                dim = int(dim)
                if s is not None:
                    s_parts.append(_project_soc(_take(s, dim)))
                if y is not None:
                    y_parts.append(_project_soc(_take(y, dim)))
                row += dim
        elif name == "s":
            for dim in _as_list(value):
                dim = int(dim)
                triangle = dim * (dim + 1) // 2
                if s is not None:
                    s_parts.append(_project_psd_triangle(_take(s, triangle), dim))
                if y is not None:
                    y_parts.append(_project_psd_triangle(_take(y, triangle), dim))
                row += triangle
        elif name in ("bl", "bu"):
            # SCS box cone: width is 1 (the leading t scalar) plus the
            # length of the box vector. ``bl`` and ``bu`` must agree on
            # length; if the user supplied only one side the other is
            # treated as length-zero.
            bl = np.asarray(cone.get("bl", []), dtype=float)
            bu = np.asarray(cone.get("bu", []), dtype=float)
            if bl.size and bu.size and bl.size != bu.size:
                # Malformed cone — record but do not advance the row
                # cursor in a way that misaligns downstream reads.
                unsupported.append("box")
                continue
            box_width = 1 + max(bl.size, bu.size)
            # Track which key has already consumed the slice so iteration
            # order doesn't matter (whichever of bl/bu we see first
            # consumes the slot; the other is a no-op).
            if "box" not in unsupported:
                unsupported.append("box")
                if s is not None:
                    s_parts.append(_take(s, box_width))
                if y is not None:
                    y_parts.append(_take(y, box_width))
                row += box_width
            # The matching bl/bu key is consumed jointly; skip it.
        elif name in ("ep", "ed"):
            dim = 3 * int(value)
            unsupported.append(name)
            if s is not None:
                s_parts.append(_take(s, dim))
            if y is not None:
                y_parts.append(_take(y, dim))
            row += dim
        elif name == "p":
            blocks = list(_as_list(value))
            dim = 3 * len(blocks)
            unsupported.append("p")
            if s is not None:
                s_parts.append(_take(s, dim))
            if y is not None:
                y_parts.append(_take(y, dim))
            row += dim
        else:
            unsupported.append(name)
            # Unknown dimension — stop advancing to avoid misaligned reads.

    s_out = np.concatenate(s_parts) if s is not None and s_parts else (
        None if s is None else np.zeros(0)
    )
    y_out = np.concatenate(y_parts) if y is not None and y_parts else (
        None if y is None else np.zeros(0)
    )
    return s_out, y_out, unsupported


def _project_soc(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    t, x = v[0], v[1:]
    norm_x = float(np.linalg.norm(x))
    if norm_x <= -t:
        return np.zeros_like(v)
    if norm_x <= t:
        return v.copy()
    out = np.empty_like(v)
    out[0] = 0.5 * (t + norm_x)
    out[1:] = (out[0] / max(norm_x, _EPS)) * x
    return out


def _project_psd_triangle(v: np.ndarray, n: int) -> np.ndarray:
    """Project a vectorized lower-triangular matrix onto PSD.

    Uses SCS convention: off-diagonal entries are scaled by ``√2`` in the
    vectorization. We unscale, symmetrize, eigen-clip, and rescale.
    """
    if v.size == 0:
        return v
    mat = np.zeros((n, n))
    idx = 0
    scale = np.sqrt(2.0)
    for col in range(n):
        for row in range(col, n):
            val = v[idx]
            if row != col:
                val = val / scale
            mat[row, col] = val
            mat[col, row] = val
            idx += 1
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.maximum(eigvals, 0.0)
    mat_proj = (eigvecs * eigvals) @ eigvecs.T
    out = np.empty_like(v)
    idx = 0
    for col in range(n):
        for row in range(col, n):
            val = mat_proj[row, col]
            if row != col:
                val = val * scale
            out[idx] = val
            idx += 1
    return out


def _as_dense(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float).ravel()


def _as_sparse(M):
    return M.tocsr() if sp.issparse(M) else sp.csr_matrix(M)


def _as_list(value) -> Iterable:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _inf_norm(v) -> float:
    if v is None:
        return 0.0
    return float(np.linalg.norm(v, ord=np.inf))
