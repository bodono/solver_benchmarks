"""Independent KKT residual computation for QP and cone problems.

All residuals are computed from the primal/dual vectors supplied by the
adapter, using the original problem data — nothing here trusts
solver-reported quantities. Returned dictionaries contain absolute
residuals under their bare names and the SCS-style normalized variants
under a ``_rel`` suffix.

Infeasibility certificates are checked the way SCS and Clarabel check them:
the certificate is scaled so that its objective term (``b'y``, the support
``u'y⁺ − l'y⁻`` or ``q'x``) equals -1, and every remaining residual of the
certificate must then be at most a tolerance (``CERT_TOL`` by default).
``cert_res`` records that largest scaled residual and ``valid`` the verdict;
``valid`` is None when a cone of the problem cannot be checked.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core.problem import cone_items_in_row_order

_EPS = 1.0e-16
# Bounds at or beyond this magnitude are treated as infinite. The datasets
# document +-1e20 as their infinity sentinel and the transform layer drops
# anything >= 1e19 (see transforms/cones.py); the checker cuts a decade
# lower still so a sentinel that slips through any reader cannot leak an
# O(1e20) term into the dual objective, complementarity, or the residual
# normalizations (a 1e20 sentinel u against a harmless y ~ 1e-11 used to
# report duality_gap_rel ~ 1.0 on half of Maros-Meszaros).
_INF_BOUND = 1.0e18
# Largest residual an infeasibility certificate may have once it is scaled so
# that its objective term is -1, for ``valid``. Solvers stop on certificates
# by their own measures in their own scaled spaces (OSQP's defaults leave
# 1e-5 to 1e-4 in these units on NETLIB, SCS, Clarabel and ECOS 1e-6 or
# less), so the default only rejects what is clearly not a certificate;
# ``bench kkt-verify --tol`` holds certificates to a stricter standard.
CERT_TOL = 1.0e-3
_SQRT2 = math.sqrt(2.0)


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

    finite_l = _finite_bounds(l)
    finite_u = _finite_bounds(u)
    y_pos = np.maximum(y, 0.0)
    y_neg = np.maximum(-y, 0.0)
    # Multipliers on absent bounds must be zero.
    dual_bound_violation = np.where(finite_u, 0.0, y_pos) + np.where(finite_l, 0.0, y_neg)
    r_dual_bound = _inf_norm(dual_bound_violation)
    comp_upper = y_pos * np.where(finite_u, u - Ax, 0.0)
    comp_lower = y_neg * np.where(finite_l, Ax - l, 0.0)
    comp = np.concatenate([comp_upper, comp_lower])
    r_comp = float(np.linalg.norm(comp, ord=np.inf))

    primal_obj = float(0.5 * x @ Px + q @ x)
    dual_obj = float(
        -0.5 * x @ Px
        + np.where(finite_l, l, 0.0) @ y_neg
        - np.where(finite_u, u, 0.0) @ y_pos
    )
    gap = primal_obj - dual_obj
    gap_norm = 1.0 + abs(primal_obj) + abs(dual_obj)

    norm_pri = 1.0 + max(
        _inf_norm(Ax),
        _inf_norm(np.where(finite_l, l, 0.0)),
        _inf_norm(np.where(finite_u, u, 0.0)),
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
        "dual_bound_res": r_dual_bound,
        "dual_bound_res_rel": r_dual_bound / (1.0 + _inf_norm(y)),
        # largest product of a multiplier and its bound's slack, in objective
        # units; ``_rel`` scales it like the duality gap
        "comp_slack": r_comp,
        "comp_slack_rel": r_comp / gap_norm,
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

    Supported cone keys: ``z``/``f`` (zero), ``l`` (nonnegative), ``bl``/``bu``
    (box), ``q`` (second-order), ``s`` (PSD triangle), ``ep``/``ed``
    (exponential and dual exponential) and ``p`` (power, a negative exponent
    meaning the dual power cone). Other keys (complex PSD) cannot be
    projected; they are reported under ``unsupported_cones`` and make the cone
    distances NaN.

    ``primal_res`` is the larger of the equality residual ``Ax + s − b``
    (also reported as ``primal_eq_res``) and the distance of ``s`` to ``K``
    (``primal_cone_res``). Adapters that build ``s`` as ``b − Ax`` have no
    equality residual by construction, so their primal infeasibility shows up
    as the cone distance, and both kinds of adapter report it in
    ``primal_res``.
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
        r_s_cone = r_s_cone_l2 = float("nan")
        r_y_cone = float("nan")
        r_pri, r_pri_l2 = r_eq, r_eq_l2
    else:
        r_s_cone = _inf_norm(s - s_proj)
        r_s_cone_l2 = float(np.linalg.norm(s - s_proj))
        r_y_cone = _inf_norm(y - y_proj)
        # np.max keeps a NaN from a non-finite iterate instead of dropping it.
        r_pri = float(np.max([r_eq, r_s_cone]))
        r_pri_l2 = float(np.max([r_eq_l2, r_s_cone_l2]))

    # comp_slack is reported as |s.y|; the inner product should be 0 at
    # complementarity but can be slightly negative from numerical noise.
    # Downstream code (plots, aggregates) treats it as a magnitude, so
    # report the absolute value here and keep the signed copy as
    # comp_slack_signed for users who want to detect a duality leak. It is
    # in objective units (QP form reports the largest per-row product
    # instead); comp_slack_rel scales it like the duality gap so it reads on
    # the same axis as the other relative residuals.
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
        "primal_res": r_pri,
        "primal_res_l2": r_pri_l2,
        "primal_res_rel": r_pri / norm_pri,
        "primal_eq_res": r_eq,
        "primal_eq_res_rel": r_eq / norm_pri,
        "dual_res": r_dual,
        "dual_res_l2": r_dual_l2,
        "dual_res_rel": r_dual / norm_dual,
        "primal_cone_res": r_s_cone,
        "dual_cone_res": r_y_cone,
        # the distance of s to K scaled like the primal equality residual (s is
        # in the units of b and Ax), and the distance of y to K* in the units
        # of y itself: the stationarity scale involves A'y and q, so using it
        # would let large coefficients hide an invalid dual point
        "primal_cone_res_rel": r_s_cone / norm_pri,
        "dual_cone_res_rel": r_y_cone / (1.0 + _inf_norm(y)),
        "comp_slack": comp,
        "comp_slack_rel": comp / gap_norm,
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
    *,
    tol: float = CERT_TOL,
) -> dict[str, Any]:
    """Check ``y ∈ K*, A'y = 0, b'y < 0`` for primal infeasibility.

    Scaled so that ``b'y = −1``, ``‖A'y‖∞`` and the distance of ``y`` to
    ``K*`` must be at most ``tol``.
    """
    y = _as_dense(y)
    A = _as_sparse(A)
    b = np.asarray(b, dtype=float).ravel()
    Aty = A.T @ y
    _, y_proj, unsupported = _project_cones(cone, None, y)
    bty = float(b @ y)
    y_scale = max(_inf_norm(y), _EPS)
    # A block that cannot be projected would otherwise read as a clean zero.
    y_cone_res = float("nan") if unsupported else _inf_norm(y - y_proj)
    out = {
        "certificate": "primal_infeasible",
        "Aty_inf": _inf_norm(Aty),
        "Aty_rel": _inf_norm(Aty) / y_scale,
        "bty": bty,
        "dual_cone_res": y_cone_res,
        **_certificate_verdict(bty, (_inf_norm(Aty), y_cone_res), tol),
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
    *,
    tol: float = CERT_TOL,
) -> dict[str, Any]:
    """Check ``Px = 0, Ax ∈ -K, q'x < 0`` for dual infeasibility (unbounded).

    Scaled so that ``q'x = −1``, ``‖Px‖∞`` and the distance of ``−Ax`` to
    ``K`` must be at most ``tol``.
    """
    x = _as_dense(x)
    P = _as_sparse(P)
    A = _as_sparse(A)
    q = np.asarray(q, dtype=float).ravel()
    Px = P @ x
    Ax = A @ x
    neg_Ax = -Ax
    s_proj, _, unsupported = _project_cones(cone, neg_Ax, None)
    cone_res = float("nan") if unsupported else _inf_norm(neg_Ax - s_proj)
    qtx = float(q @ x)
    x_scale = max(_inf_norm(x), _EPS)
    out = {
        "certificate": "dual_infeasible",
        "Px_inf": _inf_norm(Px),
        "Px_rel": _inf_norm(Px) / x_scale,
        "qtx": qtx,
        "primal_cone_res": cone_res,
        **_certificate_verdict(qtx, (_inf_norm(Px), cone_res), tol),
    }
    if unsupported:
        out["unsupported_cones"] = sorted(set(unsupported))
    return out


def qp_primal_infeasibility_cert(
    A,
    l,
    u,
    y,
    *,
    tol: float = CERT_TOL,
) -> dict[str, Any]:
    """``A'y = 0`` with ``u'y⁺ − l'y⁻ < 0`` certifies primal infeasibility.

    The support function ``u'y⁺ − l'y⁻`` is ``+∞`` as soon as ``y`` puts
    weight on a bound that is absent (``y_i > 0`` with ``u_i = +∞``, or
    ``y_i < 0`` with ``l_i = −∞``), so that weight must vanish:
    ``dual_bound_res`` is its size. ``support`` is computed on the finite
    bounds. Scaled so that the support is −1, ``‖A'y‖∞`` and
    ``dual_bound_res`` must be at most ``tol``.
    """
    y = _as_dense(y)
    A = _as_sparse(A)
    l = np.asarray(l, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    Aty = A.T @ y
    finite_l = _finite_bounds(l)
    finite_u = _finite_bounds(u)
    y_pos = np.maximum(y, 0.0)
    y_neg = np.maximum(-y, 0.0)
    support = (
        float(np.where(finite_u, u, 0.0) @ y_pos)
        - float(np.where(finite_l, l, 0.0) @ y_neg)
    )
    dual_bound_res = _inf_norm(np.where(finite_u, 0.0, y_pos) + np.where(finite_l, 0.0, y_neg))
    y_scale = max(_inf_norm(y), _EPS)
    return {
        "certificate": "primal_infeasible",
        "Aty_inf": _inf_norm(Aty),
        "Aty_rel": _inf_norm(Aty) / y_scale,
        "support": support,
        "dual_bound_res": dual_bound_res,
        **_certificate_verdict(support, (_inf_norm(Aty), dual_bound_res), tol),
    }


def qp_dual_infeasibility_cert(
    P,
    q,
    A,
    l,
    u,
    x,
    *,
    tol: float = CERT_TOL,
) -> dict[str, Any]:
    """``Px = 0``, ``q'x < 0`` with ``Ax`` in the recession cone of ``l ≤ · ≤ u``.

    Scaled so that ``q'x = −1``, ``‖Px‖∞`` and the distance of ``Ax`` to
    the recession cone must be at most ``tol``.
    """
    x = _as_dense(x)
    P = _as_sparse(P)
    A = _as_sparse(A)
    q = np.asarray(q, dtype=float).ravel()
    l = np.asarray(l, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    Px = P @ x
    Ax = A @ x
    finite_l = _finite_bounds(l)
    finite_u = _finite_bounds(u)
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
        **_certificate_verdict(qtx, (_inf_norm(Px), _inf_norm(violations)), tol),
    }


def certificate_residual(kkt: dict[str, Any]) -> float | None:
    """Largest residual of a recorded certificate scaled to objective term −1.

    Uses ``cert_res`` when the record has it and otherwise rebuilds it from
    the absolute fields that records written before it carry (QP records
    of that age have no ``dual_bound_res`` and are checked without it).
    Returns None when the record holds no certificate, or a needed residual
    is missing or unknown; a certificate whose objective term is not
    negative gets ``inf``.
    """
    if "cert_res" in kkt:
        try:
            number = float(kkt["cert_res"])
        except (TypeError, ValueError):
            return None
        return None if math.isnan(number) else number
    kind = kkt.get("certificate")
    if kind == "primal_infeasible":
        sign_key = "bty" if "bty" in kkt else "support"
        residual_keys = ["Aty_inf", "dual_cone_res" if sign_key == "bty" else "dual_bound_res"]
    elif kind == "dual_infeasible":
        sign_key = "qtx"
        residual_keys = ["Px_inf", "primal_cone_res" if "primal_cone_res" in kkt else "Ax_cone_res"]
    else:
        return None
    if kkt.get("unsupported_cones"):
        # Older records reported a clean zero for blocks they could not check.
        return None
    if "dual_bound_res" in residual_keys and "dual_bound_res" not in kkt:
        residual_keys.remove("dual_bound_res")
    sign_value = _finite_or_none(kkt.get(sign_key))
    residuals = [_finite_or_none(kkt.get(key)) for key in residual_keys]
    if sign_value is None or any(value is None for value in residuals):
        return None
    verdict = _certificate_verdict(sign_value, tuple(float(v) for v in residuals if v is not None), CERT_TOL)
    return float(verdict["cert_res"])


def _certificate_verdict(sign_value: float, residuals: tuple[float, ...], tol: float) -> dict[str, Any]:
    """``cert_res`` (residuals scaled by ``|sign_value|``) and ``valid``.

    ``valid`` is False when the objective term is not negative or a known
    residual exceeds ``tol``, True when all are within it, and None when a
    residual is unknown (NaN) and none of the known ones fails.
    """
    if math.isnan(sign_value):
        return {"cert_res": float("nan"), "cert_tol": tol, "valid": None}
    if sign_value >= 0.0:
        return {"cert_res": float("inf"), "cert_tol": tol, "valid": False}
    scaled = [value / -sign_value for value in residuals]
    known = [value for value in scaled if not math.isnan(value)]
    worst = max(known, default=0.0)
    if worst > tol:
        return {"cert_res": worst, "cert_tol": tol, "valid": False}
    if len(known) < len(scaled):
        return {"cert_res": float("nan"), "cert_tol": tol, "valid": None}
    return {"cert_res": worst, "cert_tol": tol, "valid": True}


def _finite_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _project_cones(
    cone: dict[str, Any] | None,
    s,
    y,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    """Project ``s`` onto ``K`` and ``y`` onto ``K*``, block by block.

    Blocks that cannot be projected (complex PSD, unknown keys, a malformed
    box) are passed through unchanged and named in the returned list.
    """
    unsupported: list[str] = []
    if cone is None:
        return (None if s is None else np.zeros_like(s),
                None if y is None else np.zeros_like(y), unsupported)

    s_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    row = 0

    def _take(vec, width):
        return vec[row:row + width] if vec is not None else None

    def _add(width, project_s, project_y):
        if s is not None:
            s_parts.append(project_s(_take(s, width)))
        if y is not None:
            y_parts.append(project_y(_take(y, width)))

    # Rows follow the fixed layout, not the order of the dictionary's keys.
    for name, value in cone_items_in_row_order(cone):
        if name in ("f", "z"):
            dim = int(value)
            # K = {0}, K* = everything.
            _add(dim, np.zeros_like, np.copy)
            row += dim
        elif name == "l":
            dim = int(value)
            _add(dim, _project_nonneg, _project_nonneg)
            row += dim
        elif name == "q":
            for dim in _as_list(value):
                dim = int(dim)
                _add(dim, _project_soc, _project_soc)
                row += dim
        elif name == "s":
            for dim in _as_list(value):
                dim = int(dim)
                triangle = dim * (dim + 1) // 2
                project = functools.partial(_project_psd_triangle, n=dim)
                _add(triangle, project, project)
                row += triangle
        elif name in ("bl", "bu"):
            # SCS box cone: width is 1 (the leading t scalar) plus the
            # length of the box vector. ``bl`` and ``bu`` must agree on
            # length; if the user supplied only one side the other is
            # unbounded.
            bl = np.asarray(cone.get("bl", []), dtype=float)
            bu = np.asarray(cone.get("bu", []), dtype=float)
            if bl.size and bu.size and bl.size != bu.size:
                # Malformed cone — record but do not advance the row
                # cursor in a way that misaligns downstream reads.
                if "box" not in unsupported:
                    unsupported.append("box")
                continue
            if not bl.size and not bu.size:
                # An empty box takes no rows, as in SCS.
                continue
            width = max(bl.size, bu.size)
            if name == ("bl" if bl.size else "bu") and (name == "bl" or not bl.size):
                # ``bl`` and ``bu`` describe one block: consume it once.
                lo = bl if bl.size else np.full(width, -np.inf)
                hi = bu if bu.size else np.full(width, np.inf)
                _add(
                    1 + width,
                    functools.partial(_project_box, lo=lo, hi=hi),
                    functools.partial(_project_box_dual, lo=lo, hi=hi),
                )
                row += 1 + width
        elif name in ("ep", "ed"):
            dim = 3 * int(value)
            primal, dual = _project_exp, _project_exp_dual
            if name == "ed":
                primal, dual = dual, primal
            _add(dim, _rowwise(primal), _rowwise(dual))
            row += dim
        elif name == "p":
            for power in _as_list(value):
                a = float(power)
                if not 0.0 < abs(a) < 1.0:
                    unsupported.append("p")
                    _add(3, np.copy, np.copy)
                elif a > 0:
                    _add(3, _rowwise(_project_pow, a), _rowwise(_project_pow_dual, a))
                else:
                    _add(3, _rowwise(_project_pow_dual, -a), _rowwise(_project_pow, -a))
                row += 3
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


def _rowwise(project, *args):
    """Apply a projection of ``(k, 3)`` rows to a flat block of 3-tuples."""

    def apply(v: np.ndarray) -> np.ndarray:
        return project(np.asarray(v, dtype=float).reshape(-1, 3), *args).ravel()

    return apply


def _project_nonneg(v: np.ndarray) -> np.ndarray:
    return np.maximum(v, 0.0)


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


@functools.lru_cache(maxsize=64)
def _triangle_layout(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row and column of each entry of the column-major lower triangle.

    ``np.triu_indices`` walks the upper triangle row by row, which read
    transposed is the lower triangle column by column, the SCS order.
    """
    cols, rows = np.triu_indices(n)
    for array in (cols, rows):
        array.setflags(write=False)
    off_diagonal = rows != cols
    off_diagonal.setflags(write=False)
    return rows, cols, off_diagonal


def _project_psd_triangle(v: np.ndarray, n: int) -> np.ndarray:
    """Project a vectorized lower-triangular matrix onto PSD.

    Uses SCS convention: off-diagonal entries are scaled by ``√2`` in the
    vectorization. We unscale, symmetrize, eigen-clip, and rescale. A block
    with a non-finite entry has no meaningful projection and comes back as
    NaN, so the residual reads as unknown instead of raising.
    """
    if v.size == 0:
        return v
    if not np.all(np.isfinite(v)):
        return np.full_like(v, np.nan, dtype=float)
    rows, cols, off_diagonal = _triangle_layout(n)
    values = np.where(off_diagonal, v / _SQRT2, v)
    mat = np.zeros((n, n))
    mat[rows, cols] = values
    mat[cols, rows] = values
    try:
        eigvals, eigvecs = np.linalg.eigh(mat)
    except np.linalg.LinAlgError:
        return np.full_like(v, np.nan, dtype=float)
    mat_proj = (eigvecs * np.maximum(eigvals, 0.0)) @ eigvecs.T
    out = mat_proj[rows, cols]
    return np.where(off_diagonal, out * _SQRT2, out)


# Exponential cone K = cl{(r, s, t): s > 0, s exp(r / s) <= t}, the SCS
# orientation, and its dual K* = cl{(u, v, w): u < 0, -u exp(v / u) <= e w}.
_EXP_BISECTIONS = 200
_EXP_NEWTON_STEPS = 60


def _project_exp(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of each row of ``v`` (shape ``(k, 3)``) onto K_exp.

    Points in the cone, in its polar, or with ``r, s <= 0`` have closed
    forms. The others solve ``min ½‖x − v‖²  s.t.  x₁ + x₂ log(x₂ / x₃) <= 0``
    by bisection on the multiplier ``μ`` of that convex constraint: for a
    fixed ``μ`` the minimizer has ``x₁ = r − μ`` and ``x₂, x₃`` from a
    monotone scalar equation (Newton), and the constraint value at the
    minimizer decreases in ``μ`` (the method of SCS 2, after Parikh & Boyd,
    Proximal Algorithms).
    """
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    out = np.full_like(v, np.nan)
    finite = np.all(np.isfinite(v), axis=1)
    r, s, t = v[:, 0], v[:, 1], v[:, 2]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        in_cone = finite & (
            ((s > 0) & (s * np.exp(r / s) <= t)) | ((r <= 0) & (s == 0) & (t >= 0))
        )
        in_polar = finite & ~in_cone & (
            ((r > 0) & (r * np.exp(s / r) + np.e * t <= 0)) | ((r == 0) & (s <= 0) & (t <= 0))
        )
    face = finite & ~in_cone & ~in_polar & (r <= 0) & (s <= 0)
    general = finite & ~in_cone & ~in_polar & ~face
    out[in_cone] = v[in_cone]
    out[in_polar] = 0.0
    # (r, 0, max(t, 0)) lies on the face {s = 0, r <= 0, t >= 0}; the
    # remainder (0, s, min(t, 0)) is in the polar and orthogonal to it.
    out[face] = np.column_stack([r[face], np.zeros(int(face.sum())), np.maximum(t[face], 0.0)])
    if np.any(general):
        out[general] = _project_exp_general(v[general])
    return out


def _project_exp_general(v: np.ndarray) -> np.ndarray:
    r, s, t = v[:, 0], v[:, 1], v[:, 2]
    scale = np.maximum(1.0, np.max(np.abs(v), axis=1))
    lower = np.zeros_like(r)
    upper = 0.125 * scale
    # Grow the bracket until the constraint holds at the upper multiplier.
    for _ in range(200):
        grow = _exp_constraint(v, upper) > 0
        if not np.any(grow):
            break
        lower = np.where(grow, upper, lower)
        upper = np.where(grow, 2.0 * upper, upper)
    for _ in range(_EXP_BISECTIONS):
        mid = 0.5 * (lower + upper)
        positive = _exp_constraint(v, mid) > 0
        lower = np.where(positive, mid, lower)
        upper = np.where(positive, upper, mid)
        if np.all(upper - lower <= 1e-15 * np.maximum(upper, _EPS)):
            break
    mu = 0.5 * (lower + upper)
    x2, x3 = _exp_inner(s, t, mu)
    # The multiplier form loses digits when the projection sits next to the
    # face s = 0 (x₂ = τ x₃ / μ with both factors tiny). Points of the cone
    # obtained by moving one coordinate of v bound the distance from above
    # and are near-optimal exactly there, so keep whichever point is closest
    # (as Friberg's heuristic solutions do for the same reason).
    candidates = [np.column_stack([r - mu, x2, x3])]
    candidates.append(np.column_stack([np.minimum(r, 0.0), np.zeros_like(r), np.maximum(t, 0.0)]))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        positive = s > 0
        raised_t = np.where(positive, np.maximum(t, s * np.exp(r / s)), np.inf)
        candidates.append(np.column_stack([r, np.where(positive, s, 0.0), raised_t]))
        lowered_r = np.where(positive & (t > 0), np.minimum(r, s * np.log(t / s)), np.inf)
        candidates.append(np.column_stack([lowered_r, np.where(positive, s, 0.0), np.maximum(t, 0.0)]))
    stacked = np.stack(candidates)
    with np.errstate(invalid="ignore", over="ignore"):
        distances = np.sum((stacked - v[None, :, :]) ** 2, axis=2)
    distances = np.where(np.isfinite(distances), distances, np.inf)
    best = np.argmin(distances, axis=0)
    return stacked[best, np.arange(v.shape[0])]


def _exp_constraint(v: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """``x₁ + x₂ log(x₂ / x₃)`` at the minimizer for multiplier ``mu``."""
    x2, x3 = _exp_inner(v[:, 1], v[:, 2], mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = np.where(x2 > 0, x2 * np.log(x2 / x3), 0.0)
    return v[:, 0] - mu + entropy


def _exp_inner(s: np.ndarray, t: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimize ``½(x₂−s)² + ½(x₃−t)² + μ x₂ log(x₂/x₃)`` over ``x₂, x₃ >= 0``.

    Stationarity gives ``x₂ = τ x₃ / μ`` with ``τ = x₃ − t > 0`` and the
    scalar equation ``τ(τ + t)/μ² − s/μ + log(τ/μ) + 1 = 0`` on
    ``τ > max(0, −t)``. Without a root there the minimizer is
    ``x₂ = x₃ = 0``. In ``u = log τ`` the left side is increasing and convex
    on that domain, so Newton steps from a point where it is nonnegative
    decrease monotonically to the root, however small (often ~1e-14) the
    root is.
    """
    lo = np.maximum(0.0, -t)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_mu = np.log(mu)
        # The left side at the left end of the domain: -inf when lo = 0.
        g_lo = np.where(lo > 0, -s / mu + np.log(lo) - log_mu + 1.0, -np.inf)
        has_root = g_lo < 0
        # It is nonnegative once τ(τ + t) >= μ s and τ >= μ / e.
        hi = np.maximum(np.maximum(0.5 * (-t + np.sqrt(t * t + 4.0 * np.maximum(mu * s, 0.0))), mu), lo)
        u = np.log(hi)
        u_floor = np.log(lo)
        done = ~has_root
        for _ in range(_EXP_NEWTON_STEPS):
            tau = np.exp(u)
            g = tau * (tau + t) / mu**2 - s / mu + u - log_mu + 1.0
            dg = tau * (2.0 * tau + t) / mu**2 + 1.0
            step = g / dg
            noise = 1e-14 * (tau * (tau + np.abs(t)) / mu**2 + np.abs(s) / mu + np.abs(u) + np.abs(log_mu) + 1.0)
            converged = (np.abs(g) <= noise) | (np.abs(step) <= 1e-15 * np.maximum(1.0, np.abs(u))) | ~np.isfinite(step)
            u = np.where(done | converged, u, np.maximum(u - step, u_floor))
            done |= converged
            if np.all(done):
                break
        tau = np.exp(u)
        x3 = np.where(has_root, tau + t, 0.0)
        x2 = np.where(has_root, tau * x3 / mu, 0.0)
    return x2, x3


def _project_exp_dual(v: np.ndarray) -> np.ndarray:
    """Projection onto K_exp* by Moreau: ``Π_K*(v) = v + Π_K(−v)``."""
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    return v + _project_exp(-v)


# Power cone K_a = {(x, y, z): x^a y^(1-a) >= |z|, x, y >= 0} for 0 < a < 1,
# and its dual K_a* = {(u, v, w): (u/a)^a (v/(1-a))^(1-a) >= |w|, u, v >= 0}.
_POW_NEWTON_STEPS = 100


def _project_pow(v: np.ndarray, a: float) -> np.ndarray:
    """Projection of each row onto K_a: Newton on the radius ``r = |z|`` of
    the projection, as in SCS (Hien, 2015)."""
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    out = np.full_like(v, np.nan)
    finite = np.all(np.isfinite(v), axis=1)
    xh, yh, zh = v[:, 0], v[:, 1], v[:, 2]
    rh = np.abs(zh)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        in_cone = finite & (xh >= 0) & (yh >= 0) & (xh**a * yh ** (1 - a) >= rh)
        in_polar = finite & ~in_cone & (xh <= 0) & (yh <= 0) & (
            (-xh) ** a * (-yh) ** (1 - a) >= rh * a**a * (1 - a) ** (1 - a)
        )
    general = finite & ~in_cone & ~in_polar
    out[in_cone] = v[in_cone]
    out[in_polar] = 0.0
    if np.any(general):
        x0, y0, r0 = xh[general], yh[general], rh[general]
        lo, hi = np.zeros_like(r0), r0.copy()
        r = 0.5 * r0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for _ in range(_POW_NEWTON_STEPS):
                x = _pow_coordinate(r, x0, r0, a)
                y = _pow_coordinate(r, y0, r0, 1 - a)
                f = x**a * y ** (1 - a) - r
                lo = np.where(f > 0, r, lo)
                hi = np.where(f > 0, hi, r)
                dx = a * (r0 - 2 * r) / (2 * x - x0)
                dy = (1 - a) * (r0 - 2 * r) / (2 * y - y0)
                fp = x**a * y ** (1 - a) * (a * dx / x + (1 - a) * dy / y) - 1
                candidate = r - f / fp
                outside = ~((candidate > lo) & (candidate < hi)) | ~np.isfinite(candidate)
                new_r = np.where(outside, 0.5 * (lo + hi), candidate)
                done = np.abs(new_r - r) <= 4.0 * np.finfo(float).eps * np.maximum(r0, _EPS)
                r = new_r
                if np.all(done):
                    break
            x = _pow_coordinate(r, x0, r0, a)
            y = _pow_coordinate(r, y0, r0, 1 - a)
        out[general] = np.column_stack([x, y, np.sign(zh[general]) * r])
    return out


def _pow_coordinate(r, value, r0, weight):
    # Nonnegative root of x² − value·x − weight·(r0 − r)·r = 0. For a negative
    # ``value`` the textbook form cancels; use the conjugate instead.
    c = 4.0 * weight * np.maximum(r0 - r, 0.0) * r
    d = np.sqrt(value * value + c)
    with np.errstate(divide="ignore", invalid="ignore"):
        conjugate = np.where(d - value > 0, 0.5 * c / (d - value), 0.0)
    return np.where(value >= 0, 0.5 * (value + d), conjugate)


def _project_pow_dual(v: np.ndarray, a: float) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1, 3)
    return v + _project_pow(-v, a)


# Box cone K = cl{(t, s): t >= 0, t lo <= s <= t hi} (SCS); an infinite
# bound leaves that side open for every t, including t = 0.
def _project_box(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Projection onto the box cone: minimize the convex function
    ``φ(t) = (t − t₀)² + dist(s₀, [t lo, t hi])²`` over ``t >= 0`` (the
    inner projection is a clip), by bisection on ``φ'`` followed by an exact
    solve of the piecewise-linear ``φ' = 0`` on the final active set."""
    v = np.asarray(v, dtype=float).ravel()
    if not np.all(np.isfinite(v)):
        return np.full_like(v, np.nan)
    t0, s0 = v[0], v[1:]
    lo_f = np.isfinite(lo)
    hi_f = np.isfinite(hi)
    lo_v = np.where(lo_f, lo, 0.0)
    hi_v = np.where(hi_f, hi, 0.0)

    def active(t):
        below = lo_f & (s0 < t * lo_v)
        above = hi_f & (s0 > t * hi_v)
        return below, above

    def slope(t):
        below, above = active(t)
        return (t - t0) + np.sum(np.where(below, lo_v * (t * lo_v - s0), 0.0)) + np.sum(
            np.where(above, hi_v * (t * hi_v - s0), 0.0)
        )

    if slope(0.0) >= 0:
        t_star = 0.0
    else:
        left, right = 0.0, max(1.0, abs(t0), float(np.max(np.abs(s0), initial=0.0)))
        while slope(right) < 0:
            left, right = right, 2.0 * right
        for _ in range(200):
            mid = 0.5 * (left + right)
            if slope(mid) < 0:
                left = mid
            else:
                right = mid
            if right - left <= 4.0 * np.finfo(float).eps * right:
                break
        below, above = active(0.5 * (left + right))
        weights = 1.0 + np.sum(lo_v[below] ** 2) + np.sum(hi_v[above] ** 2)
        exact = (t0 + np.sum(lo_v[below] * s0[below]) + np.sum(hi_v[above] * s0[above])) / weights
        t_star = float(np.clip(exact, left, right))
    lower = np.where(lo_f, t_star * lo_v, -np.inf)
    upper = np.where(hi_f, t_star * hi_v, np.inf)
    return np.concatenate([[t_star], np.clip(s0, lower, upper)])


def _project_box_dual(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    return v + _project_box(-v, lo, hi)


def _finite_bounds(v: np.ndarray) -> np.ndarray:
    """Mask of genuine finite bounds: excludes inf, nan, and inf sentinels."""
    return np.isfinite(v) & (np.abs(v) < _INF_BOUND)


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
