"""MOSEK adapter.

Settings are MOSEK parameter names (``MSK_DPAR_*``, ``MSK_IPAR_*``,
``MSK_SPAR_*`` or the ``mosek.dparam`` / ``iparam`` / ``sparam`` members); a
name or value MOSEK refuses raises ``SolverConfigError`` naming it. The shared
``max_iter`` sets ``MSK_IPAR_INTPNT_MAX_ITERATIONS`` and
``MSK_IPAR_SIM_MAX_ITERATIONS`` (a native name given alongside wins; recorded
in ``info.settings_translated``), ``time_limit`` sets the optimizer time limit
and ``threads`` ``MSK_IPAR_NUM_THREADS``, one unless ``threads`` or the native
name is given (0: MOSEK's own choice). The ``eps`` aliases are refused: MOSEK's
tolerances are not an absolute/relative pair. Bounds go through the shared
infinity cutoff (``|bound| >= 1e19`` on the open side is infinite).

An infeasibility claim carries MOSEK's certificate (the dual solution for a
primal infeasibility certificate, the primal solution for a dual one),
checked like every other adapter's; see :func:`_certificate_kkt`. Not run
against a real MOSEK here (no licence on the development machine); the
conversion follows MOSEK's documentation and is pinned by the fake in
``tests/test_mosek_adapter_fake.py``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.analysis.derive import worst_relative_residual
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import (
    EPS_ALIASES,
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    SolverUnavailable,
    finite_point,
    loosest_tolerance,
    module_available,
    pop_max_iter,
    pop_threads,
    pop_time_limit,
    refuse_settings,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    threads_or_default,
)
from .qp_split import (
    DUAL_CERTIFICATE_STATUSES,
    PRIMAL_CERTIFICATE_STATUSES,
    normalize_qp_bounds,
    qp_certificate_kkt,
)

# MOSEK's interior-point termination tolerances (documented defaults; not
# checked here, MOSEK needs a license): the linear ones apply to LPs, the
# QO ones to QPs.
MOSEK_LP_TOLERANCES = {
    "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,
    "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,
    "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
}
MOSEK_QP_TOLERANCES = {
    "MSK_DPAR_INTPNT_QO_TOL_PFEAS": 1e-8,
    "MSK_DPAR_INTPNT_QO_TOL_DFEAS": 1e-8,
    "MSK_DPAR_INTPNT_QO_TOL_REL_GAP": 1e-8,
}


class MosekSolverAdapter(SolverAdapter):
    solver_name = "mosek"
    supported_problem_kinds = {QP}
    solver_modules = ("mosek",)

    @classmethod
    def is_available(cls) -> bool:
        return module_available("mosek")

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        refuse_settings(settings, EPS_ALIASES, "MOSEK", "set its own tolerances instead")
        given = {_param_name(key): value for key, value in settings.items()}
        # A QP whose P is zero is an LP to MOSEK; when that is not known, the
        # looser of the two sets applies.
        if is_lp is None:
            knobs = {**MOSEK_LP_TOLERANCES, **MOSEK_QP_TOLERANCES}
        else:
            knobs = MOSEK_LP_TOLERANCES if is_lp else MOSEK_QP_TOLERANCES
        return loosest_tolerance(given, knobs)

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import mosek
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "mosek", "Install with the mosek extra to use MOSEK", exc
            ) from exc

        qp = problem.qp
        q = np.asarray(qp["q"], dtype=float)
        p_mat = sp.tril(sp.coo_matrix(qp["P"]), format="coo")
        a_mat = sp.coo_matrix(qp["A"])
        n = int(qp.get("n", int(q.shape[0])))
        m = int(qp.get("m", a_mat.shape[0]))
        lower, upper = normalize_qp_bounds(qp["l"], qp["u"])

        # Wrap MOSEK Env / Task in `with` blocks so the C-level resources
        # and the streamprinter callback are released deterministically.
        # Otherwise long batch runs leak license tokens until GC fires.
        with mosek.Env() as env, env.Task() as task:
            translated, threads = _configure_mosek(task, env, self.settings, mosek)

            task.appendcons(m)
            task.appendvars(n)
            # Set linear costs and free variable bounds in bulk; the
            # previous per-variable putvarbound loop was O(n) Python calls.
            if n:
                task.putclist(list(range(n)), q.tolist())
                task.putvarboundsliceconst(0, n, mosek.boundkey.fr, -np.inf, np.inf)

            task.putaijlist(a_mat.row, a_mat.col, a_mat.data)
            for row in range(m):
                bound_key, l_val, u_val = _mosek_bound(lower[row], upper[row], mosek)
                task.putconbound(row, bound_key, l_val, u_val)

            if p_mat.nnz:
                task.putqobj(p_mat.row, p_mat.col, p_mat.data)
            task.putobjsense(mosek.objsense.minimize)

            start = time.perf_counter()
            try:
                termination_code = task.optimize()
            except Exception as exc:
                if _is_license_error(exc, mosek):
                    # MOSEK checks out its license in optimize(). A missing,
                    # expired or exhausted license is a broken environment,
                    # not MOSEK failing this problem.
                    raise SolverUnavailable(f"MOSEK license error: {exc}") from exc
                return SolverResult(
                    status=status.SOLVER_ERROR,
                    run_time_seconds=time.perf_counter() - start,
                    info={"raw_status": "error", "error": str(exc), "threads": threads},
                )
            elapsed = time.perf_counter() - start

            soltype, kkt_dict, unavailable = _select_mosek_solution(task, mosek, qp)
            raw_status = task.getsolsta(soltype) if task.solutiondef(soltype) else mosek.solsta.unknown
            mapped = _map_mosek_status(raw_status, termination_code, mosek)
            solver_reported_runtime = task.getdouinf(mosek.dinfitem.optimizer_time)
            iterations = task.getintinf(mosek.iinfitem.intpnt_iter)
            # Only a claimed solution has an objective; a limit-hit point
            # keeps its residuals (and primal_obj) in the KKT record, and a
            # certificate has none.
            has_point = kkt_dict is not None and mapped not in status.ANY_INFEASIBLE
            objective = task.getprimalobj(soltype) if has_point else None
            certificate_info: dict[str, str] = {}
            if mapped in status.ANY_INFEASIBLE and unavailable is not None:
                certificate_info["certificate_unavailable"] = unavailable
            return SolverResult(
                status=mapped,
                kkt=kkt_dict,
                objective_value=reported_objective(mapped, objective),
                iterations=int(iterations) if iterations is not None else None,
                run_time_seconds=elapsed,
                info={
                    "raw_status": str(raw_status),
                    "termination_code": str(termination_code),
                    "solver": "mosek",
                    "solver_reported_runtime": float(solver_reported_runtime),
                    "soltype": str(soltype),
                    "threads": threads,
                    **certificate_info,
                    **({"settings_translated": translated} if translated else {}),
                },
            )


# MOSEK's iteration caps, set by the shared ``max_iter``.
_MAX_ITER_PARAMS = ("MSK_IPAR_INTPNT_MAX_ITERATIONS", "MSK_IPAR_SIM_MAX_ITERATIONS")
_NUM_THREADS_PARAM = "MSK_IPAR_NUM_THREADS"


def _configure_mosek(task, env, settings: dict, mosek) -> tuple[dict, int]:
    """Apply ``settings`` to ``task``; return the aliases it translated and
    the thread count."""
    settings = settings_with_defaults(settings)
    refuse_settings(
        settings,
        EPS_ALIASES,
        "MOSEK",
        "set its own tolerances instead (MSK_DPAR_INTPNT_QO_TOL_REL_GAP, "
        "MSK_DPAR_INTPNT_QO_TOL_PFEAS, MSK_DPAR_INTPNT_QO_TOL_DFEAS, or the "
        "MSK_DPAR_INTPNT_TOL_* ones for LPs)",
    )
    if settings.pop("verbose", False):
        def streamprinter(text):
            print(text, end="")

        env.set_Stream(mosek.streamtype.log, streamprinter)
        task.set_Stream(mosek.streamtype.log, streamprinter)
    else:
        task.putintparam(mosek.iparam.log, 0)

    time_limit = pop_time_limit(settings)
    if time_limit is not None:
        task.putdouparam(mosek.dparam.optimizer_max_time, float(time_limit))
    threads = pop_threads(settings)
    given = {_param_name(key) for key in settings}
    translated: dict = {}
    max_iter = pop_max_iter(settings, minimum=0)
    if max_iter is not None:
        for name in _MAX_ITER_PARAMS:
            if name not in given:
                settings[name] = max_iter
                translated[name] = max_iter
    native_threads = [key for key in settings if _param_name(key) == _NUM_THREADS_PARAM]
    if native_threads:
        threads = settings[native_threads[0]]
    else:
        threads = threads_or_default(threads)
        task.putintparam(mosek.iparam.num_threads, threads)

    settings.pop("high_accuracy", None)
    for key, value in settings.items():
        try:
            if isinstance(key, str):
                _handle_mosek_str_param(task, key.strip(), value)
            else:
                _handle_mosek_enum_param(task, key, value, mosek)
        except getattr(mosek, "Error", ()) as exc:
            raise SolverConfigError(f"Invalid MOSEK parameter {key!r}={value!r}: {exc}") from exc
    return translated, threads


def _param_name(key) -> str:
    """``MSK_IPAR_NUM_THREADS`` for that string or for ``mosek.iparam.num_threads``."""
    if isinstance(key, str):
        return key.strip().upper()
    prefix = {"iparam": "MSK_IPAR_", "dparam": "MSK_DPAR_", "sparam": "MSK_SPAR_"}.get(
        type(key).__name__, ""
    )
    return f"{prefix}{getattr(key, 'name', key)}".upper()


def _is_license_error(exc: Exception, mosek) -> bool:
    """True for a ``mosek.Error`` whose response code is a license failure
    (the ``err_license*`` family, ``err_missing_license_file``,
    ``err_*_license``, ``err_flexlm``, ``err_platform_not_licensed``)."""
    error_type = getattr(mosek, "Error", None)
    if error_type is None or not isinstance(exc, error_type):
        return False
    code = getattr(exc, "errno", None)
    name = str(getattr(code, "name", code)).lower()
    return "licen" in name or "flexlm" in name


def _mosek_bound(lower: float, upper: float, mosek):
    """Bound key and values for a row whose bounds went through
    ``normalize_qp_bounds`` (an infinite side is +-inf)."""
    l_val = float(lower)
    u_val = float(upper)
    if abs(l_val - u_val) <= 1.0e-10:
        return mosek.boundkey.fx, l_val, u_val
    if l_val == -np.inf and u_val == np.inf:
        return mosek.boundkey.fr, l_val, u_val
    if l_val == -np.inf:
        return mosek.boundkey.up, l_val, u_val
    if u_val == np.inf:
        return mosek.boundkey.lo, l_val, u_val
    return mosek.boundkey.ra, l_val, u_val


def _map_mosek_status(raw_status, termination_code, mosek) -> str:
    """Map MOSEK's (solsta, termination_code) pair to the canonical status.

    Solsta is checked first: if the optimizer found an optimum *before*
    the time/iteration limit fired, we should report OPTIMAL rather than
    discarding it as TIME_LIMIT.
    """
    solsta = mosek.solsta
    solsta_mapping = {
        solsta.optimal: status.OPTIMAL,
        solsta.integer_optimal: status.OPTIMAL,
        solsta.prim_and_dual_feas: status.OPTIMAL_INACCURATE,
        solsta.prim_feas: status.OPTIMAL_INACCURATE,
        solsta.prim_infeas_cer: status.PRIMAL_INFEASIBLE,
        solsta.dual_infeas_cer: status.DUAL_INFEASIBLE,
        solsta.unknown: status.SOLVER_ERROR,
    }
    canonical_solsta = solsta_mapping.get(raw_status)
    # If MOSEK actually concluded with a useful solsta, keep it even if
    # the optimizer also raised a time/iter termination code.
    if canonical_solsta in {status.OPTIMAL, status.PRIMAL_INFEASIBLE, status.DUAL_INFEASIBLE}:
        return canonical_solsta
    if termination_code == mosek.rescode.trm_max_time:
        return status.TIME_LIMIT
    if termination_code == mosek.rescode.trm_max_iterations:
        return status.MAX_ITER_REACHED
    if canonical_solsta is not None:
        return canonical_solsta
    return status.SOLVER_ERROR


def _select_mosek_solution(task, mosek, qp):
    """Select the point with the smallest worst relative KKT residual.

    A solution that is an infeasibility certificate is taken as it is, with
    its certificate check (see :func:`_certificate_kkt`). Returns ``(soltype,
    kkt_record, unavailable)``; ``unavailable`` says why a certificate has no
    record, and is None otherwise.
    """
    candidates = []
    # min() keeps the first candidate on ties, so prefer the interior point.
    for soltype in (mosek.soltype.itr, mosek.soltype.bas):
        if not task.solutiondef(soltype):
            continue
        mapped = _map_mosek_status(task.getsolsta(soltype), None, mosek)
        if mapped in status.ANY_INFEASIBLE:
            return (soltype, *_certificate_kkt(task, soltype, mapped, qp))
        x = np.asarray(task.getxx(soltype), dtype=float)
        y = -np.asarray(task.gety(soltype), dtype=float)
        if not finite_point(x, y):
            continue
        # Use the full original Hessian, not MOSEK's lower triangle.
        residuals = kkt.qp_residuals(
            qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], x, y,
        )
        worst = worst_relative_residual({"kkt": residuals})
        candidates.append((float("inf") if worst is None else worst, soltype, residuals))
    if not candidates:
        return mosek.soltype.itr, None, None
    _, soltype, residuals = min(candidates, key=lambda candidate: candidate[0])
    return soltype, residuals, None


def _certificate_kkt(task, soltype, mapped: str, qp):
    """The certificate check of the solution ``soltype`` holding a certificate.

    MOSEK reports a primal infeasibility certificate as the dual solution:
    ``y`` with ``A'y = 0`` (the variables are free), ``y = s_l^c - s_u^c``,
    ``s >= 0`` and ``(l^c)'s_l^c - (u^c)'s_u^c > 0`` (MOSEK's "Infeasibility
    for linear optimization"). Its sign is that of MOSEK's duals, which the
    adapter negates for solutions too: ``-y`` puts ``s_u^c`` on the upper
    bounds and ``s_l^c`` on the lower ones, the harness's support. A dual
    infeasibility certificate is the primal solution ``x`` (``c'x < 0``,
    ``Ax`` in the recession cone of the bounds), the harness's ray as it is.
    MOSEK computes both by default, for LPs and QPs alike, so no parameter
    is needed. Rows with no bound (free) have zero multipliers. An
    inaccurate claim (``*_infeasible_inaccurate``) is read the same way.

    Returns ``(kkt_record, unavailable)``. A vector that is not finite, or
    one MOSEK fails to return, gives no record, and ``unavailable`` says why
    (the caller puts it on ``info``): the claim is already made, so an error
    reading its certificate costs the certificate, not the row.
    """
    lower, upper = normalize_qp_bounds(qp["l"], qp["u"])
    farkas = ray = None
    try:
        if mapped in PRIMAL_CERTIFICATE_STATUSES:
            farkas = -np.asarray(task.gety(soltype), dtype=float)
        elif mapped in DUAL_CERTIFICATE_STATUSES:
            ray = np.asarray(task.getxx(soltype), dtype=float)
        else:
            return None, None  # no certificate is claimed
    except Exception as exc:
        return None, f"no certificate from MOSEK: {exc}"
    record = qp_certificate_kkt(mapped, qp["P"], qp["q"], qp["A"], lower, upper, farkas=farkas, ray=ray)
    return record, None if record is not None else "MOSEK's certificate is not finite"


def _handle_mosek_str_param(task, param: str, value) -> None:
    if param.startswith("MSK_DPAR_"):
        task.putnadouparam(param, value)
    elif param.startswith("MSK_IPAR_"):
        task.putnaintparam(param, value)
    elif param.startswith("MSK_SPAR_"):
        task.putnastrparam(param, value)
    else:
        raise SolverConfigError(f"Invalid MOSEK parameter {param!r}")


def _handle_mosek_enum_param(task, param, value, mosek) -> None:
    if isinstance(param, mosek.dparam):
        task.putdouparam(param, value)
    elif isinstance(param, mosek.iparam):
        task.putintparam(param, value)
    elif isinstance(param, mosek.sparam):
        task.putstrparam(param, value)
    else:
        raise SolverConfigError(f"Invalid MOSEK parameter {param!r}")
