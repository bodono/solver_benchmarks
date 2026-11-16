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

where ``u = t/√2 + 1/√2``, ``v = t/√2 - 1/√2``. ``R`` is built without
densifying ``P``: a row/column whose only entry is its diagonal gives one
row ``√p_ii e_i'`` (so a diagonal ``P`` gives ``R = diag(√p_ii)`` at any
size), and only the rows/columns coupled by off-diagonal entries are
factored densely, one connected block at a time. Those dense factors may
together hold at most ``ECOS_DENSE_FACTOR_MAX_DIM ** 2`` entries (one
2000 × 2000 block); above that the problem is skipped as unsupported rather
than allocating a huge dense matrix and handing ECOS an almost dense SOC
block (which runs out of memory or measures a far larger artificial
problem). The reformulation is part of ``run_time_seconds``
(it is work a QP solver does in its own factorization) and is also recorded
as ``info["socp_reformulation_seconds"]``. The std SOC means
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

Settings are ECOS's own keyword arguments (``feastol``, ``abstol``,
``reltol``, ``max_iters``, ...); an unknown name or a value ECOS refuses
raises ``SolverConfigError`` naming it. The harness aliases ``max_iter``
(-> ``max_iters``), ``eps`` / ``eps_abs`` (-> ``feastol`` and ``abstol``) and
``eps_rel`` (-> ``reltol``) are translated, recorded in
``info["settings_translated"]``. ECOS has no time limit or thread count;
those are recorded as ignored.

ECOS answers "close to optimal" (or close to a certificate, exit flags 10-12)
when it stops at ``max_iters`` with an iterate that meets its reduced
tolerances; such a stop is ``max_iter_reached`` like every other adapter's,
with ECOS's text kept in ``info.raw_status``. The same flags for an earlier
stop (numerical trouble) stay ``*_inaccurate``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

# Imported here, not in the timed reformulation: the runner starts a fresh
# worker per solve, so a first-use import there would be charged to every
# ECOS solve (about 0.15 s, more than the solve on small QPs).
from scipy.sparse.csgraph import connected_components

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, QP, ProblemData, as_list
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.transforms.cones import qp_to_nonnegative_cone

from .base import (
    IMPORT_ERRORS,
    OBJECTIVE_STATUSES,
    SolverAdapter,
    SolverConfigError,
    coerce_int_setting,
    finite_point,
    is_setting_type_error,
    mark_threads_ignored,
    mark_time_limit_ignored,
    module_available,
    pop_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    translate_tolerance_aliases,
)
from .cone_layout import qp_dual_in_native_order

# The QP -> SOCP reformulation factors densely only the rows/columns of P
# coupled by off-diagonal entries, one connected block at a time; the dense
# factors together may hold at most ECOS_DENSE_FACTOR_MAX_DIM ** 2 entries
# (one block of this order). Diagonal-only rows never need a dense factor.
ECOS_DENSE_FACTOR_MAX_DIM = 2000


class ECOSSolverAdapter(SolverAdapter):
    solver_name = "ecos"
    supported_problem_kinds = {QP, CONE}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("ecos")

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import ecos
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "ecos", "Install with the ecos extra to use ECOS", exc
            ) from exc

        settings = settings_with_defaults(self.settings)
        # ECOS exposes neither a native time-limit knob nor a thread
        # count (the reference build is single-threaded). Record any
        # configured values on info as ignored markers so the contract
        # matches the rest of the adapters.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        translated = translate_settings(settings)

        # QP-as-SOCP marker: when set, ``socp_state`` carries the
        # data needed to recover the original-x and report the actual
        # ½ x'Px + q'x objective rather than the surrogate t + q'x.
        socp_state: dict | None = None
        reformulation_seconds = 0.0
        if problem.kind == QP:
            qp = problem.qp
            if _qp_has_nonzero_p(qp):
                # Factoring P stands in for the factorization a QP solver
                # does itself, so it is timed with the solve.
                reformulation_start = time.perf_counter()
                try:
                    data, dims, cone_dict, socp_state = _qp_to_ecos_via_socp(qp)
                except _SOCPReformulationError as exc:
                    return SolverResult(
                        status=status.SKIPPED_UNSUPPORTED,
                        info={"reason": str(exc)},
                    )
                reformulation_seconds = time.perf_counter() - reformulation_start
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

        # ECOS takes float64 arrays only (an integer-typed G or A is refused
        # with a TypeError); the other adapters convert as well.
        g_matrix = sp.csc_matrix(data["G"], dtype=float)
        a_matrix = sp.csc_matrix(data["A"], dtype=float)
        c_vec = np.asarray(data["c"], dtype=float)
        h_vec = np.asarray(data["h"], dtype=float)
        b_vec = np.asarray(data["b"], dtype=float)
        start = time.perf_counter()
        try:
            raw = ecos.solve(c_vec, g_matrix, h_vec, dims, A=a_matrix, b=b_vec, **kwargs)
        except (TypeError, ValueError) as exc:
            # ECOS checks its keyword arguments before solving: an unknown
            # name or a value of the wrong type is a TypeError about the
            # argument, a bad value a ValueError naming the setting
            # ("'feastol' must be ..."). Anything else is ECOS refusing the
            # problem data.
            if _is_ecos_setting_error(exc):
                raise SolverConfigError(
                    f"ECOS rejected the settings {sorted(kwargs)}: {exc}"
                ) from exc
            error_info = {
                "raw_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "threads": None,
            }
            mark_time_limit_ignored(error_info, time_limit)
            mark_threads_ignored(error_info, threads)
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=time.perf_counter() - start + reformulation_seconds,
                info=error_info,
            )
        elapsed = time.perf_counter() - start + reformulation_seconds
        ineq_perm = data.get("ineq_perm")
        if ineq_perm is not None:
            # Inequality duals and slacks back to the canonical cone order.
            raw = dict(raw)
            for key in ("z", "s"):
                if raw.get(key) is not None:
                    raw[key] = np.asarray(raw[key], dtype=float)[ineq_perm]

        info = dict(raw.get("info", {}))
        if "infostring" in info:
            info["raw_status"] = info["infostring"]
        info["threads"] = None
        mark_time_limit_ignored(info, time_limit)
        mark_threads_ignored(info, threads)
        if translated:
            info["settings_translated"] = translated

        mapped = _map_ecos_status(info, max_iters=kwargs.get("max_iters", _ECOS_DEFAULT_MAX_ITERS))
        # ECOS reports ``pcost`` (primal objective). For infeasibility-
        # certificate statuses ``pcost`` is the certificate normalizer,
        # not an objective value, so only a solution reports it (mirrors SCS).
        if socp_state is not None:
            # Augmented variables x_full = [x_orig, t]. ECOS may
            # return without a primal vector (max-iter, numerical
            # error, infeasibility certificate); guard every read of
            # ``raw['x']`` so non-solution statuses don't crash the
            # adapter.
            info["socp_reformulation"] = True
            info["socp_reformulation_seconds"] = reformulation_seconds
            full_x = raw.get("x")
            full_x_arr = (
                np.asarray(full_x, dtype=float) if full_x is not None else None
            )
            n_x = int(socp_state["n_x"])
            if full_x_arr is not None and full_x_arr.size > n_x:
                info["socp_t_value"] = _maybe_float(full_x_arr[n_x])
            raw = _strip_socp_aux_from_solution(raw, n_x)
            stripped_x = raw.get("x")
            if (
                mapped in OBJECTIVE_STATUSES
                and stripped_x is not None
                and np.asarray(stripped_x).size == n_x
            ):
                objective = _qp_objective_value(
                    problem.qp, np.asarray(stripped_x, dtype=float)
                )
            else:
                objective = None
        else:
            objective = _maybe_float(info.get("pcost"))
        kkt_dict = _compute_kkt(problem, mapped, raw, cone_dict, socp_state=socp_state)

        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, objective),
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_maybe_float(info.get("timing", {}).get("tsetup")),
            solve_time_seconds=_maybe_float(info.get("timing", {}).get("tsolve")),
            info=info,
            kkt=kkt_dict,
        )


# ECOS's internal names for its keyword arguments, as its value checks name
# them ("'maxit' must be a nonnegative integer" for max_iters).
_ECOS_SETTING_NAMES = frozenset({
    "feastol", "abstol", "reltol", "feastol_inacc", "abstol_inacc", "reltol_inacc",
    "maxit", "max_iters", "nitref", "verbose", "mi_max_iters", "mi_abs_eps", "mi_rel_eps",
})


def _is_ecos_setting_error(exc: Exception) -> bool:
    """Whether ECOS's TypeError / ValueError is about a keyword argument."""
    if isinstance(exc, TypeError) and is_setting_type_error(exc):
        return True
    text = str(exc)
    quoted = text[1:].split("'", 1)[0] if text.startswith("'") else ""
    return quoted in _ECOS_SETTING_NAMES


def _qp_has_nonzero_p(qp: dict) -> bool:
    p = qp.get("P")
    if p is None:
        return False
    p_sparse = sp.csc_matrix(p)
    return p_sparse.nnz > 0 and float(np.abs(p_sparse).sum()) > 0.0


class _SOCPReformulationError(ValueError):
    """Raised when a QP cannot be cleanly reformulated as an SOCP for
    ECOS — because ``P`` is not symmetric / PSD within numerical
    tolerance, or the blocks of ``P`` coupled by off-diagonal entries are
    too large to factor densely (more than ``ECOS_DENSE_FACTOR_MAX_DIM ** 2``
    entries together)."""


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


def _psd_square_root(p: sp.csc_matrix) -> sp.csc_matrix | None:
    """Return a sparse ``R`` such that ``P = R'R``. ``R`` is square (rank=n)
    for positive-definite ``P`` and has ``rank(P)`` rows when ``P`` is
    rank-deficient. Returns ``None`` if ``P`` is not symmetric PSD within
    numerical tolerance — including the **indefinite** case (any
    eigenvalue meaningfully below zero) so a nonconvex QP is rejected
    up-front rather than silently solved as if its negative curvature
    didn't exist.

    ``P`` is never densified as a whole. A row/column whose only entry is
    its diagonal ``p_ii`` gives one row ``√p_ii e_i'`` of ``R`` (zero entries
    dropped), so a diagonal ``P`` gives ``R = diag(√p_ii)`` at any size. The
    rows/columns coupled by off-diagonal entries split into the connected
    components of that coupling graph; each component's block (its
    diagonal included) is factored densely on its own — Cholesky first
    (cheap, only works for SPD), then a symmetric eigendecomposition that
    drops eigenvalues of round-off size but rejects negative curvature
    (see :func:`_dense_psd_square_root`) — so ``R`` is block diagonal.
    ``R'R`` reproduces ``P`` to round-off, so ECOS solves the posed QP even
    when ``P``'s spectrum spans many orders of magnitude. The dense factors
    together may hold at most ``ECOS_DENSE_FACTOR_MAX_DIM ** 2`` entries (one
    2000 x 2000 block, or e.g. four 1000 x 1000 ones); above that
    ``_SOCPReformulationError`` is raised before anything is factored.
    """
    p = sp.csc_matrix(p, dtype=float)
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        return None
    n = p.shape[0]
    p.eliminate_zeros()
    # Symmetrize to absorb tiny numerical asymmetry; anything larger than
    # np.allclose(P, P_sym, atol=1e-8 * max(1, |P|_max)) would accept is
    # refused, as when P was densified.
    p_sym = sp.csc_matrix(0.5 * (p + p.T))
    p_sym.eliminate_zeros()
    if not _close_to_symmetric(p, p_sym):
        return None
    if p_sym.nnz == 0:
        return sp.csc_matrix((0, n))

    diagonal = p_sym.diagonal()
    # Negative curvature is refused against the scale of the whole of P
    # (diagonal-only rows and coupled blocks alike), allowing for the
    # precision P's data was stored with; what is kept is decided per block.
    scale = float(np.abs(diagonal).max())
    negative_tol = _negative_curvature_tol(scale)
    off_diagonal = sp.csr_matrix(p_sym - sp.diags(diagonal))
    off_diagonal.eliminate_zeros()
    coupled_mask = np.diff(off_diagonal.indptr) > 0
    coupled = np.flatnonzero(coupled_mask)
    single = np.flatnonzero(~coupled_mask)

    single_diagonal = diagonal[single]
    if single.size and float(single_diagonal.min()) < -negative_tol:
        return None
    # A diagonal-only row is its own 1 x 1 block whose eigenvalue p_ii is
    # exact, so every positive one is kept, however small next to the rest.
    keep = single[single_diagonal > 0.0]
    row_parts = [np.arange(keep.size)]
    col_parts = [keep]
    val_parts = [np.sqrt(diagonal[keep])]
    n_rows = int(keep.size)

    if coupled.size:
        coupling = off_diagonal[coupled][:, coupled]
        n_components, labels = connected_components(coupling, directed=False)
        sizes = np.bincount(labels, minlength=n_components)
        dense_entries = int(np.sum(sizes.astype(np.int64) ** 2))
        if dense_entries > ECOS_DENSE_FACTOR_MAX_DIM**2:
            largest = int(np.max(sizes))
            raise _SOCPReformulationError(
                f"QP P matrix has off-diagonal entries coupling {coupled.size} "
                f"rows/columns in {n_components} connected block(s), the largest "
                f"{largest} x {largest}; their dense factors would "
                f"hold {dense_entries} entries, more than the "
                f"{ECOS_DENSE_FACTOR_MAX_DIM} x {ECOS_DENSE_FACTOR_MAX_DIM} the ECOS "
                "SOCP reformulation factors, and its SOC block would be nearly dense."
            )
        # Group the coupled rows by component: the permuted block of P is
        # block diagonal with contiguous components.
        order = np.argsort(labels, kind="stable")
        members = coupled[order]
        block_matrix = sp.csr_matrix(p_sym[members][:, members])
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            block = block_matrix[start:stop, start:stop].toarray()
            r_block = _dense_psd_square_root(block, scale=scale)
            if r_block is None:
                return None
            rows, cols = np.nonzero(r_block)
            row_parts.append(rows + n_rows)
            col_parts.append(members[start + cols])
            val_parts.append(r_block[rows, cols])
            n_rows += int(r_block.shape[0])

    return sp.csc_matrix(
        (np.concatenate(val_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(n_rows, n),
    )


def _close_to_symmetric(p: sp.csc_matrix, p_sym: sp.csc_matrix) -> bool:
    atol = 1e-8 * max(1.0, float(abs(p).max()) if p.nnz else 0.0)
    diff = sp.csr_matrix(abs(p - p_sym))
    diff.eliminate_zeros()
    if not diff.nnz:
        return True
    pattern = diff.copy()
    pattern.data[:] = 1.0
    excess = diff - 1e-5 * sp.csr_matrix(abs(p_sym)).multiply(pattern)
    return not (excess.nnz and float(excess.max()) > atol)


def _negative_curvature_tol(scale: float) -> float:
    """How far below zero an eigenvalue of ``P`` may be and still count as
    zero (dropped) rather than negative curvature (``P`` refused).

    ``1e-12`` relative to ``scale`` (at least ``1e-12``): larger than the
    round-off of the eigendecomposition, to allow for ``P`` data stored with
    fewer than full digits.
    """
    return max(1e-12, float(scale) * 1e-12)


def _dense_psd_square_root(p_sym: np.ndarray, *, scale: float = 0.0) -> np.ndarray | None:
    """Dense ``R`` with ``R'R = p_sym``, or None when ``p_sym`` is indefinite.

    Eigenvalues up to the eigendecomposition's round-off, ``k * eps * |λ|max``
    for this ``k x k`` block (``numpy.linalg.matrix_rank``'s rule), count as
    zero and are dropped; every larger one is kept, so ``R'R`` reproduces
    ``p_sym`` to round-off whatever the spread of its spectrum. An eigenvalue
    below minus :func:`_negative_curvature_tol` of ``max(λmax, scale)`` is
    negative curvature and ``p_sym`` is refused; ``scale`` lets a block of a
    larger ``P`` use the scale of the whole matrix for that test.
    """
    try:
        return la.cholesky(p_sym, lower=False)
    except (np.linalg.LinAlgError, ValueError):
        # P is rank-deficient PSD or indefinite; fall through to
        # eigendecomposition to disambiguate.
        pass
    try:
        eigvals, eigvecs = la.eigh(p_sym)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if eigvals.size == 0:
        return np.zeros((0, p_sym.shape[0]))
    max_eig = float(eigvals.max())
    min_eig = float(eigvals.min())
    # Reject if any eigenvalue is meaningfully negative (indefinite
    # P): pre-fix this branch silently dropped negative eigenvalues
    # alongside the near-zero ones, so a nonconvex QP like
    # ``diag([1, -1])`` was sent to ECOS as a rank-1 PSD relaxation.
    # An all-non-positive block is either zero within the cut-off (empty
    # factor below) or negative definite (refused here).
    if min_eig < -_negative_curvature_tol(max(max_eig, float(scale))):
        return None
    # Drop the eigenvalues that are zero up to round-off (the QP epigraph
    # constraint t ≥ ½ x'Px ignores zero-curvature directions). A cut-off
    # relative to λmax but above round-off would drop genuine curvature of a
    # wide spectrum and have ECOS solve a different problem.
    cutoff = p_sym.shape[0] * np.finfo(float).eps * max(abs(max_eig), abs(min_eig))
    keep = eigvals > cutoff
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
    # A second-order cone of dimension 0 has no rows; ECOS refuses it.
    q_list = [int(dim) for dim in as_list(cone.pop("q", [])) if int(dim)]
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

    # Rows are in the fixed layout of a cone problem: the zero cone
    # (equalities) first, then NN, then SOCP, then exp. ProblemData
    # refuses a cone dictionary listed in another order.
    a_eq = a[:z_count, :]
    b_eq = b[:z_count]
    a_ineq = a[z_count:, :]
    b_ineq = b[z_count:]
    # The canonical (SCS/Clarabel) exponential cone is (x, y, z) with
    # y exp(x / y) <= z; ECOS's is (x, y, z) with z exp(x / z) <= y. Swap
    # the last two rows of each 3-row block; the swap is its own inverse,
    # so the same permutation maps ECOS's duals and slacks back.
    ineq_perm = _ecos_exp_row_perm(a_ineq.shape[0], e_count)
    if ineq_perm is not None:
        a_ineq = a_ineq[ineq_perm, :]
        b_ineq = b_ineq[ineq_perm]

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
            "ineq_perm": ineq_perm,
        },
        dims,
        cone_dict,
    )


def _ecos_exp_row_perm(n_ineq: int, e_count: int) -> np.ndarray | None:
    """Row permutation of the inequality block swapping (y, z) in every
    exponential cone; the exp cones are the last ``3 * e_count`` rows."""
    if not e_count:
        return None
    perm = np.arange(n_ineq)
    start = n_ineq - 3 * e_count
    for k in range(e_count):
        i = start + 3 * k
        perm[i + 1], perm[i + 2] = i + 2, i + 1
    return perm


def translate_settings(settings: dict) -> dict:
    """Map the harness aliases onto ECOS's keyword arguments, in place.

    ``max_iter`` becomes ``max_iters``; ``eps_abs`` (or ``eps``) supplies
    ``feastol`` and ``abstol`` and ``eps_rel`` (or ``eps``) supplies
    ``reltol``. An explicit native value always wins, and only settings
    actually inserted are reported. ``max_iters`` and ``nitref`` follow the
    shared integer rule (``coerce_int_setting``).
    """
    translated: dict = {}
    # ECOS takes max_iters 0 (it then reports its starting point).
    max_iter = pop_max_iter(settings, minimum=0)
    if max_iter is not None and settings.get("max_iters") is None:
        settings["max_iters"] = max_iter
        translated["max_iters"] = max_iter
    for key in ("max_iters", "nitref"):
        if settings.get(key) is not None:
            settings[key] = coerce_int_setting(key, settings[key], minimum=0)
    translated.update(
        translate_tolerance_aliases(settings, ("feastol", "abstol"), ("reltol",))
    )
    return translated


# ECOS's iteration cap when ``max_iters`` is not given.
_ECOS_DEFAULT_MAX_ITERS = 100

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
# "Close to" a solution or certificate: the reduced-tolerance verdicts.
_ECOS_INACCURATE_FLAGS = frozenset({10, 11, 12})


def _map_ecos_status(info: dict, *, max_iters=None) -> str:
    """Canonical status for ECOS's ``info``.

    An inaccurate verdict (exit flag 10-12) from a solve that used up
    ``max_iters`` is ``max_iter_reached``: ECOS stopped on its limit and
    labelled the last iterate by its reduced tolerances.
    """
    flag = info.get("exitFlag")
    if flag is None:
        return status.SOLVER_ERROR
    try:
        flag_int = int(flag)
    except (TypeError, ValueError):
        return status.SOLVER_ERROR
    if flag_int in _ECOS_INACCURATE_FLAGS and max_iters is not None:
        iterations = _maybe_int(info.get("iter"))
        if iterations is not None and iterations >= int(max_iters):
            return status.MAX_ITER_REACHED
    return _ECOS_EXIT_FLAG_MAP.get(flag_int, status.SOLVER_ERROR)


def _compute_kkt(problem, mapped_status, raw, cone_dict, *, socp_state=None):
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
        # Every status that may carry a usable point (including max-iter)
        # gets solution residuals, so kkt-verify --promote treats ECOS like
        # every other solver.
        if mapped_status in status.ANY_FEASIBLE:
            # Stack equality and inequality duals back into the original
            # row order produced by qp_to_nonnegative_cone: [eq, finite_u, finite_l].
            y_full = qp_dual_in_native_order(qp, y_dual, z_dual, n_eq)
            if not finite_point(x, y_full):
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
            y_full = qp_dual_in_native_order(qp, y_dual, z_dual, n_eq)
            if not finite_point(y_full):
                return None
            return kkt.qp_primal_infeasibility_cert(qp["A"], qp["l"], qp["u"], y_full)
        if mapped_status in {
            status.DUAL_INFEASIBLE,
            status.DUAL_INFEASIBLE_INACCURATE,
        } and finite_point(x):
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
        mapped_status in status.ANY_FEASIBLE
        and z_dual is not None
        and s_slack is not None
        and y_dual is not None
    ):
        y_full = _cone_dual_combined(y_dual, z_dual, cone_dict)
        s_full = _cone_slack_combined(s_slack, cone_dict)
        if not finite_point(x, y_full, s_full):
            return None
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
        if not finite_point(y_full):
            return None
        return kkt.cone_primal_infeasibility_cert(
            cone_problem["A"], cone_problem["b"], _cone_dict_for_kkt(cone_dict), y_full
        )
    if mapped_status in {status.DUAL_INFEASIBLE, status.DUAL_INFEASIBLE_INACCURATE} and finite_point(x):
        return kkt.cone_dual_infeasibility_cert(
            p,
            cone_problem["q"],
            cone_problem["A"],
            _cone_dict_for_kkt(cone_dict),
            x,
        )
    return None


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


def _cone_slack_combined(s_slack, cone_dict):
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
