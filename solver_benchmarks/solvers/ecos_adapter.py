"""ECOS adapter.

ECOS is an interior-point solver for linear, second-order, and
exponential cone problems. ECOS does not natively support quadratic
objectives — but every convex QP with PSD ``P`` admits a standard
SOCP epigraph reformulation, which the adapter applies automatically
so users can run ECOS against QP datasets like Maros-Meszaros.

QP → SOCP reformulation (epigraph trick):

    QP: minimize  ½ x'Px + q'x  subject to  l ≤ Ax ≤ u

becomes (with ``P = R'R`` via Cholesky / eigendecomposition for
rank-deficient ``P``):

    SOCP: minimize  t + q'x  subject to  l ≤ Ax ≤ u
                                          (u, v, Rx) in std SOC of dim k+2

where ``u = t/√2 + 1/√2``, ``v = t/√2 - 1/√2``. The std SOC means
``u² ≥ v² + ‖Rx‖²``, which after the substitution reduces to
``2t ≥ ‖Rx‖² = x'Px`` — exactly the QP epigraph constraint
``t ≥ ½ x'Px``. At the optimum ``t = ½ x'Px``, so we report the
actual ``½ x'Px + q'x`` as the objective rather than the surrogate
``t + q'x`` (numerically these agree to within solver tolerance,
but reporting the actual avoids drift when comparing to other
solvers).

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

        # QP-as-SOCP marker: when set, ``socp_state`` carries the
        # data needed to recover the original-x and report the actual
        # ½ x'Px + q'x objective rather than the surrogate t + q'x.
        socp_state: dict | None = None
        if problem.kind == QP:
            qp = problem.qp
            if _qp_has_nonzero_p(qp):
                try:
                    data, dims, cone_dict, socp_state = _qp_to_ecos_via_socp(qp)
                except _SOCPReformulationError as exc:
                    return SolverResult(
                        status=status.SKIPPED_UNSUPPORTED,
                        info={"reason": str(exc)},
                    )
            else:
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
        if socp_state is not None:
            # Augmented variables x_full = [x_orig, t]; recover x_orig
            # and report ½ x'Px + q'x as the actual QP objective rather
            # than the surrogate t + q'x. Restore the dropped trailing
            # ``t`` from ``raw['x']`` so KKT / dual reconstruction sees
            # only the original variables.
            socp_state["raw_full"] = raw
            raw = _strip_socp_aux_from_solution(raw, socp_state["n_x"])
            info["socp_reformulation"] = True
            info["socp_t_value"] = _maybe_float(socp_state["raw_full"]["x"][socp_state["n_x"]])
            objective = (
                _qp_objective_value(problem.qp, np.asarray(raw["x"], dtype=float))
                if objective_present
                else None
            )
        else:
            objective = (
                _maybe_float(info.get("pcost")) if objective_present else None
            )
        kkt_dict = _compute_kkt(problem, mapped, raw, dims, cone_dict, socp_state=socp_state)

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


class _SOCPReformulationError(ValueError):
    """Raised when a QP cannot be cleanly reformulated as an SOCP for
    ECOS — typically because ``P`` is not symmetric / PSD within
    numerical tolerance, or the dimension is too large for the dense
    eigendecomposition path."""


def _qp_to_ecos_via_socp(qp: dict):
    """Reformulate ``min ½ x'Px + q'x s.t. l ≤ Ax ≤ u`` as an SOCP
    over augmented variables ``[x, t]``.

    Returns ``(data, dims, cone_dict, socp_state)`` where ``data`` is
    the ECOS input tuple and ``socp_state`` records the metadata the
    post-solve code needs to recover ``x`` and report the actual
    ``½ x'Px + q'x`` objective.

    Builds the std SOC of dim ``k+2`` over ``(u, v, R x)`` where
    ``u = (t+1)/√2``, ``v = (t-1)/√2``, and ``R`` is a square root of
    ``P`` from Cholesky (positive-definite case) or eigendecomposition
    (rank-deficient PSD case). Zero eigenvalues are dropped, so the
    SOC dim is ``rank(P) + 2``.
    """
    n = int(np.asarray(qp["q"], dtype=float).size)
    p = sp.csc_matrix(qp["P"])
    if p.shape != (n, n):
        raise _SOCPReformulationError(
            f"QP P matrix has shape {p.shape}, expected {(n, n)}; "
            "cannot reformulate to SOCP."
        )
    r_factor = _psd_square_root(p)
    if r_factor is None:
        raise _SOCPReformulationError(
            "QP P matrix is not symmetric PSD within tolerance; "
            "cannot reformulate to SOCP for ECOS."
        )
    k = int(r_factor.shape[0])

    # Original (non-Q) constraints via the nonneg-cone transform.
    a_cone, b_cone, z_count = qp_to_nonnegative_cone(qp)
    a_cone = sp.csc_matrix(a_cone)
    b_eq = b_cone[:z_count]
    b_ineq = b_cone[z_count:]

    # Pad equality / inequality rows with a zero column for the new
    # ``t`` variable (it does not appear in any of the original
    # constraints).
    zero_col = sp.csc_matrix((a_cone.shape[0], 1))
    a_full = sp.hstack([a_cone, zero_col], format="csc")
    a_eq_full = a_full[:z_count, :]
    a_ineq_full = a_full[z_count:, :]

    # Std SOC of dim k+2 over (u, v, R x):
    #   u = t/√2 + 1/√2, v = t/√2 − 1/√2, plus the linear part R x.
    # ECOS encodes the cone as h − G [x; t] in std SOC.
    sqrt2 = float(np.sqrt(2.0))
    inv_sqrt2 = 1.0 / sqrt2
    # Row 0 (u): s_u = h_u − G_u [x; t] = 1/√2 − (-1/√2) t = 1/√2 + t/√2.
    g_u = sp.csc_matrix(
        ([- inv_sqrt2], ([0], [n])), shape=(1, n + 1)
    )
    # Row 1 (v): s_v = -1/√2 − (-1/√2) t = -1/√2 + t/√2.
    g_v = sp.csc_matrix(
        ([- inv_sqrt2], ([0], [n])), shape=(1, n + 1)
    )
    # Rows 2..k+1: s_y = 0 − (-R_full) [x; t] = R x. R_full is R
    # padded with a zero column for the t variable.
    r_padded = sp.hstack(
        [sp.csc_matrix(-r_factor), sp.csc_matrix((k, 1))], format="csc"
    )
    g_soc = sp.vstack([g_u, g_v, r_padded], format="csc")
    h_soc = np.concatenate([[inv_sqrt2, -inv_sqrt2], np.zeros(k)])

    # Stack inequality blocks: linear (NN) first, then the SOC rows.
    if a_ineq_full.shape[0]:
        g_full = sp.vstack([a_ineq_full, g_soc], format="csc")
        h_full = np.concatenate([b_ineq, h_soc])
    else:
        g_full = g_soc
        h_full = h_soc

    c = np.concatenate([np.asarray(qp["q"], dtype=float), [1.0]])

    dims = {"l": int(a_ineq_full.shape[0]), "q": [k + 2], "e": 0}
    cone_dict: dict = {}
    if z_count:
        cone_dict["z"] = int(z_count)
    if a_ineq_full.shape[0]:
        cone_dict["l"] = int(a_ineq_full.shape[0])
    cone_dict["q"] = [k + 2]

    socp_state = {
        "n_x": n,
        "rank_p": k,
        "r_factor": r_factor,
        "z_count": z_count,
        "n_lin": int(a_ineq_full.shape[0]),
    }
    return (
        {
            "c": c,
            "G": g_full,
            "h": h_full,
            "A": sp.csc_matrix(a_eq_full),
            "b": b_eq,
        },
        dims,
        cone_dict,
        socp_state,
    )


def _psd_square_root(p: sp.csc_matrix) -> np.ndarray | None:
    """Return ``R`` such that ``P = R'R``. ``R`` is square (rank=n) for
    positive-definite ``P`` and tall (rows = rank ≤ n) when ``P`` is
    rank-deficient. Returns ``None`` if ``P`` is not symmetric PSD
    within numerical tolerance.

    Tries Cholesky first (cheap, only works for SPD); falls back to
    a symmetric eigendecomposition that drops eigenvalues below a
    relative tolerance.
    """
    p_dense = p.toarray() if hasattr(p, "toarray") else np.asarray(p, dtype=float)
    if p_dense.ndim != 2 or p_dense.shape[0] != p_dense.shape[1]:
        return None
    # Symmetrize to absorb tiny numerical asymmetry.
    p_sym = 0.5 * (p_dense + p_dense.T)
    if not np.allclose(p_dense, p_sym, atol=1e-8 * max(1.0, np.abs(p_dense).max())):
        return None
    try:
        import scipy.linalg as la

        return la.cholesky(p_sym, lower=False)
    except (np.linalg.LinAlgError, ValueError):
        # P is rank-deficient PSD; fall through to eigendecomposition.
        pass
    try:
        import scipy.linalg as la

        eigvals, eigvecs = la.eigh(p_sym)
    except (np.linalg.LinAlgError, ValueError):
        return None
    # Drop eigenvalues below a relative threshold; treat them as zero
    # (the QP epigraph constraint t ≥ ½ x'Px ignores those directions).
    max_eig = float(eigvals.max()) if eigvals.size else 0.0
    if max_eig <= 0.0:
        # P is identically zero — caller should have used the LP path.
        return np.zeros((0, p_sym.shape[0]))
    threshold = max(1e-12, max_eig * 1e-12)
    keep = eigvals > threshold
    if not keep.any():
        return np.zeros((0, p_sym.shape[0]))
    sqrt_eigvals = np.sqrt(eigvals[keep])
    return (sqrt_eigvals[:, None] * eigvecs[:, keep].T).astype(float)


def _strip_socp_aux_from_solution(raw: dict, n_x: int) -> dict:
    """Drop the trailing ``t`` from ``raw['x']`` so the post-solve
    code (KKT, dual reconstruction) sees only the original-x. The
    SOC dual ``z`` and slack ``s`` blocks are also trimmed: KKT for
    the original QP only cares about the linear (NN cone) duals.
    """
    new_raw = dict(raw)
    if raw.get("x") is not None:
        new_raw["x"] = np.asarray(raw["x"], dtype=float)[:n_x]
    # Keep y (eq dual) as-is; trim z and s to the linear block only,
    # discarding the SOC block that came from the SOCP reformulation.
    # The linear block size lives at ``cone_dict["l"]`` but we don't
    # have it here; the solver-side ``raw`` is rebuilt by the caller.
    return new_raw


def _qp_objective_value(qp: dict, x: np.ndarray) -> float:
    p = qp.get("P")
    q = np.asarray(qp["q"], dtype=float)
    if p is None:
        return float(q @ x)
    px = sp.csc_matrix(p) @ x
    return float(0.5 * x @ px + q @ x)


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


def _compute_kkt(problem, mapped_status, raw, dims, cone_dict, *, socp_state=None):
    """Forward to the canonical KKT helpers in the same shape as SCS.

    For QP-as-LP solves we call ``kkt.qp_residuals`` with P=0 and the
    original l/u bounds; for CONE problems we call the conic helpers.

    The ``socp_state`` argument carries the metadata for QP-as-SOCP
    solves. When present, ``raw`` already has the augmented ``t``
    variable trimmed off; we trim ``z``/``s`` to the linear block
    (``cone_dict["l"]``) so the KKT helpers see the original-QP
    layout, and we use the actual ``P`` (not zero) for residuals.
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
    if socp_state is not None and z_dual is not None:
        # Drop the SOC dual block (not relevant to original-QP KKT);
        # keep only the linear block.
        n_lin = int(socp_state.get("n_lin", 0))
        z_dual = np.asarray(z_dual, dtype=float)[:n_lin]
    if socp_state is not None and s_slack is not None:
        n_lin = int(socp_state.get("n_lin", 0))
        s_slack = np.asarray(s_slack, dtype=float)[:n_lin]

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
