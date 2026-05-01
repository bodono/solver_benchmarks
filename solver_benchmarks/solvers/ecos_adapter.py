"""ECOS adapter.

ECOS is an interior-point solver for linear, second-order, and
exponential cone problems. It does not natively support quadratic
objectives — the adapter accepts QPs only when ``P`` is zero (so the
problem is really an LP) and passes them through after the standard
QP→nonneg-cone transform; QPs with non-zero ``P`` return
``SKIPPED_UNSUPPORTED`` rather than being secretly reformulated, so
the benchmark stays honest about which solvers can take which
problem shapes.

ECOS' native data form is::

    minimize   c' x
    subject to A x = b           (equality / zero cone)
               h - G x in K      (K = R^{l}_+ × Q^{q_1} × ... × Exp × ...)

with ``dims = {"l": l, "q": [q_1, ..., q_p], "e": e}``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

from .base import (
    SolverAdapter,
    SolverUnavailable,
    mark_threads_ignored,
    pop_threads,
    pop_time_limit,
    settings_with_defaults,
)


class ECOSSolverAdapter(SolverAdapter):
    solver_name = "ecos"
    supported_problem_kinds = {QP, CONE}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import ecos  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import ecos
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the ecos extra to use ECOS") from exc

        settings = settings_with_defaults(self.settings)
        # ECOS exposes neither a native time-limit knob nor a thread
        # count (the reference build is single-threaded). Record any
        # configured values on info as ignored markers so the contract
        # matches the rest of the adapters.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)

        if problem.kind == QP:
            qp = problem.qp
            if _qp_has_nonzero_p(qp):
                return SolverResult(
                    status=status.SKIPPED_UNSUPPORTED,
                    info={
                        "reason": (
                            "ECOS does not support quadratic objectives natively; "
                            "use SCS, Clarabel, or a QP-capable solver for QPs with "
                            "non-zero P."
                        )
                    },
                )
            data, dims, cone_dict = _qp_lp_to_ecos(qp)
        else:
            data, dims, cone_dict = _cone_to_ecos(problem.cone)
            if isinstance(data, SolverResult):
                # Unsupported cone type; bail out cleanly.
                return data

        kwargs = {**settings}
        verbose = bool(kwargs.pop("verbose", False))
        kwargs["verbose"] = verbose

        start = time.perf_counter()
        raw = ecos.solve(
            data["c"], data["G"], data["h"], dims, A=data["A"], b=data["b"], **kwargs
        )
        elapsed = time.perf_counter() - start

        info = dict(raw.get("info", {}))
        mark_time_limit_ignored = _mark_time_limit_ignored
        mark_time_limit_ignored(info, time_limit)
        mark_threads_ignored(info, threads)

        mapped = _map_ecos_status(info)
        # ECOS reports ``pcost`` (primal objective). For infeasibility-
        # certificate statuses ``pcost`` is the certificate normalizer,
        # not an objective value, so suppress it there (mirrors SCS).
        objective_present = (
            mapped in status.SOLUTION_PRESENT or mapped == status.OPTIMAL_INACCURATE
        )
        objective = (
            _maybe_float(info.get("pcost")) if objective_present else None
        )
        kkt_dict = _compute_kkt(problem, mapped, raw, dims, cone_dict)

        return SolverResult(
            status=mapped,
            objective_value=objective,
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_maybe_float(info.get("timing", {}).get("tsetup")),
            solve_time_seconds=_maybe_float(info.get("timing", {}).get("tsolve")),
            info=info,
            kkt=kkt_dict,
        )


def _qp_has_nonzero_p(qp: dict) -> bool:
    p = qp.get("P")
    if p is None:
        return False
    p_sparse = sp.csc_matrix(p)
    return p_sparse.nnz > 0 and float(np.abs(p_sparse).sum()) > 0.0


def _qp_lp_to_ecos(qp: dict):
    """Convert a P=0 QP to ECOS LP form via the nonneg-cone transform.

    The transform produces ``a_cone x + s = b_cone`` with the first
    ``z`` rows in the zero cone (equalities) and the rest in the
    nonneg cone (inequalities). ECOS wants equalities in (A, b) and
    inequalities in (G, h) with ``h - Gx in K``. Since ``s = b - ax``
    we have ``b - ax in K``, which matches ECOS' ``h - Gx in K`` form
    with ``G = a`` and ``h = b``.
    """
    a_cone, b_cone, z = qp_to_nonnegative_cone(qp)
    a_cone = sp.csc_matrix(a_cone)
    c = np.asarray(qp["q"], dtype=float)
    a_eq = a_cone[:z, :]
    b_eq = b_cone[:z]
    a_ineq = a_cone[z:, :]
    b_ineq = b_cone[z:]
    dims = {"l": int(a_ineq.shape[0]), "q": [], "e": 0}
    cone_dict: dict = {}
    if z:
        cone_dict["z"] = int(z)
    if a_ineq.shape[0]:
        cone_dict["l"] = int(a_ineq.shape[0])
    return (
        {
            "c": c,
            "G": sp.csc_matrix(a_ineq),
            "h": b_ineq,
            "A": sp.csc_matrix(a_eq),
            "b": b_eq,
        },
        dims,
        cone_dict,
    )


def _cone_to_ecos(cone_problem: dict):
    """Convert a CONE-shape problem into ECOS' (c, G, h, dims, A, b).

    The CONE problem is::

        minimize   q' x   (P assumed 0; ECOS does not solve QPs)
        subject to A x + s = b, s in K

    where K is described by ``cone_problem["cone"]`` with keys
    ``z`` (or legacy ``f``), ``l``, ``q``, ``e``. ECOS supports
    ``l``, ``q``, and ``e`` directly. Equality rows (zero cone) move
    to ECOS' (A, b). Other cones are unsupported here.
    """
    p = cone_problem.get("P")
    if p is not None and sp.csc_matrix(p).nnz > 0:
        return (
            SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": "ECOS does not solve QPs (CONE problem with non-zero P)"},
            ),
            None,
            None,
        )
    a = sp.csc_matrix(cone_problem["A"])
    b = np.asarray(cone_problem["b"], dtype=float)
    q = np.asarray(cone_problem["q"], dtype=float)
    cone = dict(cone_problem["cone"])
    # Merge legacy free-variable cone key into the zero cone.
    z_count = int(cone.pop("f", 0)) + int(cone.pop("z", 0))
    l_count = int(cone.pop("l", 0))
    q_list = list(cone.pop("q", []))
    e_count = int(cone.pop("ep", 0)) + int(cone.pop("e", 0))
    unsupported = sorted(cone.keys())
    if unsupported:
        return (
            SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={
                    "reason": (
                        f"ECOS does not support cone keys {unsupported!r} (only "
                        "z/f, l, q, ep/e are handled)."
                    )
                },
            ),
            None,
            None,
        )

    # Reorder rows so the zero cone (equalities) come first, then
    # NN, then SOCP, then exp, matching the cone_dict layout that
    # CONE-form problems are expected to follow. We trust the caller
    # already laid out rows in (z, l, q*, e*) order — that's the
    # convention in the synthetic / SDPLIB / DIMACS datasets.
    a_eq = a[:z_count, :]
    b_eq = b[:z_count]
    a_ineq = a[z_count:, :]
    b_ineq = b[z_count:]

    dims = {"l": l_count, "q": [int(d) for d in q_list], "e": e_count}
    cone_dict: dict = {}
    if z_count:
        cone_dict["z"] = z_count
    if l_count:
        cone_dict["l"] = l_count
    if q_list:
        cone_dict["q"] = [int(d) for d in q_list]
    if e_count:
        cone_dict["ep"] = e_count

    return (
        {
            "c": q,
            "G": sp.csc_matrix(a_ineq),
            "h": b_ineq,
            "A": sp.csc_matrix(a_eq),
            "b": b_eq,
        },
        dims,
        cone_dict,
    )


def _mark_time_limit_ignored(info, time_limit):
    # Inlined so the import surface in this file stays small; keeps the
    # adapter import section visually identical to mark_threads_ignored
    # without pulling in a one-shot helper from base.
    if time_limit is not None:
        info["time_limit_ignored"] = True
        info["time_limit_seconds"] = float(time_limit)


_ECOS_EXIT_FLAG_MAP = {
    0: status.OPTIMAL,                  # ECOS_OPTIMAL
    1: status.PRIMAL_INFEASIBLE,        # ECOS_PINF
    2: status.DUAL_INFEASIBLE,          # ECOS_DINF
    10: status.OPTIMAL_INACCURATE,      # ECOS_OPTIMAL + INACC
    11: status.PRIMAL_INFEASIBLE_INACCURATE,
    12: status.DUAL_INFEASIBLE_INACCURATE,
    -1: status.MAX_ITER_REACHED,        # ECOS_MAXIT
    -2: status.SOLVER_ERROR,            # ECOS_NUMERICS
    -3: status.SOLVER_ERROR,            # ECOS_OUTCONE
    -4: status.SOLVER_ERROR,            # ECOS_SIGINT (interrupted)
    -7: status.SOLVER_ERROR,            # ECOS_FATAL
}


def _map_ecos_status(info: dict) -> str:
    flag = info.get("exitFlag")
    if flag is None:
        return status.SOLVER_ERROR
    try:
        flag_int = int(flag)
    except (TypeError, ValueError):
        return status.SOLVER_ERROR
    return _ECOS_EXIT_FLAG_MAP.get(flag_int, status.SOLVER_ERROR)


def _compute_kkt(problem, mapped_status, raw, dims, cone_dict):
    """Forward to the canonical KKT helpers in the same shape as SCS.

    For QP-as-LP solves we call ``kkt.qp_residuals`` with P=0 and the
    original l/u bounds; for CONE problems we call the conic helpers.
    """
    x = raw.get("x")
    if x is None:
        return None
    x = np.asarray(x, dtype=float)
    # ECOS returns z (dual for inequalities) and y (dual for equalities)
    # plus s (slack). The KKT helpers expect a single combined dual y.
    z_dual = raw.get("z")
    y_dual = raw.get("y")
    s_slack = raw.get("s")

    if problem.kind == QP:
        qp = problem.qp
        n_eq = int(cone_dict.get("z", 0))
        if mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
            # Stack equality and inequality duals back into the original
            # row order produced by qp_to_nonnegative_cone: [eq, finite_u, finite_l].
            y_full = _qp_dual_in_native_order(qp, y_dual, z_dual, n_eq)
            if y_full is None:
                return None
            return kkt.qp_residuals(
                qp["P"] if qp.get("P") is not None else sp.csc_matrix((x.size, x.size)),
                qp["q"],
                qp["A"],
                qp["l"],
                qp["u"],
                x,
                y_full,
            )
        if mapped_status in {
            status.PRIMAL_INFEASIBLE,
            status.PRIMAL_INFEASIBLE_INACCURATE,
        }:
            y_full = _qp_dual_in_native_order(qp, y_dual, z_dual, n_eq)
            if y_full is None:
                return None
            return kkt.qp_primal_infeasibility_cert(qp["A"], qp["l"], qp["u"], y_full)
        if mapped_status in {
            status.DUAL_INFEASIBLE,
            status.DUAL_INFEASIBLE_INACCURATE,
        }:
            return kkt.qp_dual_infeasibility_cert(
                qp["P"] if qp.get("P") is not None else sp.csc_matrix((x.size, x.size)),
                qp["q"],
                qp["A"],
                qp["l"],
                qp["u"],
                x,
            )
        return None

    # CONE branch.
    cone_problem = problem.cone
    p = cone_problem.get("P")
    if p is None:
        p = sp.csc_matrix((sp.csc_matrix(cone_problem["A"]).shape[1],) * 2)
    if (
        mapped_status in {status.OPTIMAL, status.OPTIMAL_INACCURATE}
        and z_dual is not None
        and s_slack is not None
        and y_dual is not None
    ):
        y_full = _cone_dual_combined(y_dual, z_dual, cone_dict)
        s_full = _cone_slack_combined(b_or_zero=None, s_slack=s_slack, cone_dict=cone_dict)
        return kkt.cone_residuals(
            p,
            cone_problem["q"],
            cone_problem["A"],
            cone_problem["b"],
            _cone_dict_for_kkt(cone_dict),
            x,
            y_full,
            s_full,
        )
    if (
        mapped_status
        in {status.PRIMAL_INFEASIBLE, status.PRIMAL_INFEASIBLE_INACCURATE}
        and (z_dual is not None or y_dual is not None)
    ):
        y_full = _cone_dual_combined(y_dual, z_dual, cone_dict)
        return kkt.cone_primal_infeasibility_cert(
            cone_problem["A"], cone_problem["b"], _cone_dict_for_kkt(cone_dict), y_full
        )
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE}:
        return kkt.cone_dual_infeasibility_cert(
            p,
            cone_problem["q"],
            cone_problem["A"],
            _cone_dict_for_kkt(cone_dict),
            x,
        )
    return None


def _qp_dual_in_native_order(qp, y_dual, z_dual, n_eq):
    """Reassemble the ECOS duals into the QP's native (eq, finite_u,
    finite_l) row order, matching what ``qp_residuals`` expects.
    """
    a_native = sp.csc_matrix(qp["A"])
    n_rows = a_native.shape[0]
    if y_dual is None and z_dual is None:
        return None
    y_eq = np.asarray(y_dual, dtype=float) if y_dual is not None else np.zeros(n_eq)
    z_ineq = np.asarray(z_dual, dtype=float) if z_dual is not None else np.zeros(0)

    # Reconstruct the (eq, finite_u, finite_l) row layout used by
    # qp_to_nonnegative_cone so we can fold the ECOS duals back into a
    # length-n_rows vector aligned with the original A.
    from solver_benchmarks.transforms.cones import split_qp_bounds

    _a, _l, _u, eq, finite_l, finite_u = split_qp_bounds(qp)
    eq_rows = np.flatnonzero(eq)
    fu_rows = np.flatnonzero(finite_u)
    fl_rows = np.flatnonzero(finite_l)

    if eq_rows.size + fu_rows.size + fl_rows.size != (
        eq_rows.size + z_ineq.size
    ):
        # Layouts disagree (shouldn't happen given our own transform).
        return None

    # ECOS form: min q^T x s.t. h - Gx in NN_+ with multiplier z >= 0,
    # giving stationarity q + G^T z = 0. With G = [A_eq; A_finite_u;
    # -A_finite_l] (the qp_to_nonnegative_cone layout), the QP-side
    # convention y_qp = λ_u - λ_l requires:
    #   y_qp[eq] = y_eq
    #   y_qp[finite_u] = +z_u  (positive on upper bound active)
    #   y_qp[finite_l] = -z_l  (negative on lower bound active)
    y = np.zeros(n_rows)
    if eq_rows.size:
        y[eq_rows] += y_eq[: eq_rows.size]
    cursor = 0
    if fu_rows.size:
        y[fu_rows] += z_ineq[cursor : cursor + fu_rows.size]
        cursor += fu_rows.size
    if fl_rows.size:
        y[fl_rows] += -z_ineq[cursor : cursor + fl_rows.size]
        cursor += fl_rows.size
    return y


def _cone_dual_combined(y_eq, z_ineq, cone_dict):
    """Stack equality and inequality duals back into the canonical
    [z, l, q*, e*] row layout the KKT helpers expect."""
    parts = []
    if cone_dict.get("z"):
        parts.append(np.asarray(y_eq, dtype=float) if y_eq is not None else np.zeros(int(cone_dict["z"])))
    if z_ineq is not None and len(z_ineq):
        parts.append(np.asarray(z_ineq, dtype=float))
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def _cone_slack_combined(b_or_zero, s_slack, cone_dict):
    """ECOS reports s only for the inequality block (matching G, h);
    pad with zeros for the equality block so the combined slack
    aligns with the full row layout."""
    n_eq = int(cone_dict.get("z", 0))
    s_arr = np.asarray(s_slack, dtype=float)
    if n_eq:
        return np.concatenate([np.zeros(n_eq), s_arr])
    return s_arr


def _cone_dict_for_kkt(cone_dict: dict) -> dict:
    """Translate the ECOS-side cone dict into the schema the KKT
    helpers expect (which uses 'ep' for exponential)."""
    out: dict = {}
    if cone_dict.get("z"):
        out["z"] = int(cone_dict["z"])
    if cone_dict.get("l"):
        out["l"] = int(cone_dict["l"])
    if cone_dict.get("q"):
        out["q"] = list(cone_dict["q"])
    if cone_dict.get("ep"):
        out["ep"] = int(cone_dict["ep"])
    return out


def _maybe_float(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _maybe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
