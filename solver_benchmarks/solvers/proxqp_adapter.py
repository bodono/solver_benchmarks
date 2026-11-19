"""ProxQP adapter via ProxSuite."""

from __future__ import annotations

import math
import numbers
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.core import status
from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult

from .base import (
    IMPORT_ERRORS,
    RequestedTolerance,
    SolverAdapter,
    SolverConfigError,
    loosest_tolerance,
    mark_threads_ignored,
    mark_time_limit_ignored,
    module_available,
    normalize_max_iter,
    pop_threads,
    pop_time_limit,
    reported_objective,
    settings_with_defaults,
    solver_import_failed,
    translate_eps_alias,
)
from .qp_split import (
    combine_qp_duals,
    dense_backend_requested,
    split_qp_for_range_constraints,
    split_qp_kkt,
)

# ProxQP's infeasibility tolerances (checked on proxsuite 0.7.3); the
# optimality defaults are _PROXQP_DEFAULT_EPS_ABS / _REL below. ProxQP records
# no certificate the harness can check, so these matter only if one does.
PROXQP_DEFAULT_CERTIFICATE_TOLERANCES = {"eps_primal_inf": 1e-4, "eps_dual_inf": 1e-4}
# A stand-in for the proxsuite module: _duality_gap_settings then uses the
# documented defaults instead of importing ProxSuite.
_NO_PROXSUITE = SimpleNamespace(proxqp=SimpleNamespace())


class ProxQPSolverAdapter(SolverAdapter):
    solver_name = "proxqp"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("proxsuite")

    @classmethod
    def requested_tolerance(cls, settings, *, problem_kind=None, is_lp=None, versions=None) -> RequestedTolerance:
        settings = dict(settings or {})
        translate_eps_alias(settings)
        gap = _duality_gap_settings(settings, _NO_PROXSUITE)
        # eps_rel 0 (the default) makes the residual test absolute only.
        knobs = {"eps_abs": _PROXQP_DEFAULT_EPS_ABS, "eps_rel": _PROXQP_DEFAULT_EPS_REL}
        if gap["check_duality_gap"]:
            # Filled from eps_abs / eps_rel by _duality_gap_settings when not given.
            knobs.update({"eps_duality_gap_abs": gap["eps_duality_gap_abs"],
                          "eps_duality_gap_rel": gap["eps_duality_gap_rel"]})
        return loosest_tolerance(settings, knobs, PROXQP_DEFAULT_CERTIFICATE_TOLERANCES)

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import proxsuite
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "proxsuite", "Install with the proxqp extra to use ProxQP", exc
            ) from exc

        qp = problem.qp
        p, q, aeq, b, g, h_l, h_u, eq_idx, ineq_idx = split_qp_for_range_constraints(qp)
        settings = settings_with_defaults(self.settings)
        use_dense = dense_backend_requested(settings, "ProxQP")
        # ProxQP has no native time-limit / threads knobs; record the
        # configured values on info so callers know they were ignored.
        time_limit = pop_time_limit(settings)
        threads = pop_threads(settings)
        normalize_max_iter(settings)
        translated = translate_eps_alias(settings)
        settings.setdefault("compute_timings", True)
        backend = proxsuite.proxqp.dense if use_dense else proxsuite.proxqp.sparse
        _check_settings(settings, backend.solve)
        _check_setting_values(settings)
        gap_settings = _duality_gap_settings(settings, proxsuite)
        start = time.perf_counter()
        # 1-D float64 vectors and float64 matrices, so that the binding's
        # TypeError ("incompatible function arguments") is about a setting.
        p, aeq, g = (_float_matrix(m) for m in (p, aeq, g))
        q, b, h_l, h_u = (_float_vector(v) for v in (q, b, h_l, h_u))
        if use_dense:
            data = (
                p.toarray(),
                q,
                None if aeq is None else aeq.toarray(),
                b,
                None if g is None else g.toarray(),
                h_l,
                h_u,
            )
        else:
            data = (p, q, aeq, b, g, h_l, h_u)
        try:
            result = backend.solve(*data, **settings)
        except TypeError as exc:
            if not _settings_refused(backend, settings, use_dense):
                # The settings work on a one-variable problem: the binding
                # refused this problem's data (e.g. complex entries).
                return SolverResult(
                    status=status.SOLVER_ERROR,
                    run_time_seconds=time.perf_counter() - start,
                    info={"raw_status": "error", "error": f"TypeError: {exc}", "threads": None},
                )
            # The binding refuses a setting of the wrong type (the names were
            # checked above) before solving.
            raise SolverConfigError(
                f"ProxQP rejected a setting value in {sorted(settings)}: {exc}"
            ) from exc
        elapsed = time.perf_counter() - start

        mapped = _map_proxqp_status(result.info.status, proxsuite)
        y = combine_qp_duals(
            int(qp["A"].shape[0]),
            eq_idx,
            getattr(result, "y", None),
            ineq_idx,
            getattr(result, "z", None),
        )
        kkt_dict = split_qp_kkt(mapped, qp, result.x, y)
        info = _info_dict(result.info)
        info["raw_status"] = info.get("status", str(result.info.status))
        info["threads"] = None
        mark_time_limit_ignored(info, time_limit)
        mark_threads_ignored(info, threads)
        info.update(gap_settings)
        if translated:
            info["settings_translated"] = translated
        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, info.get("objValue")),
            iterations=_maybe_int(info.get("iter")),
            run_time_seconds=elapsed,
            setup_time_seconds=_proxqp_seconds(info.get("setup_time")),
            solve_time_seconds=_proxqp_seconds(info.get("solve_time")),
            info=info,
            kkt=kkt_dict,
        )


# ProxSuite's documented defaults, used if its Settings class is unavailable.
_PROXQP_DEFAULT_EPS_ABS = 1e-5
_PROXQP_DEFAULT_EPS_REL = 0.0


def _duality_gap_settings(settings: dict, proxsuite) -> dict:
    """Turn on ProxQP's duality-gap stopping test unless the config says not to.

    ProxSuite's default convergence test checks only the primal and dual
    residuals (``check_duality_gap=False``), so it stops at points that are
    feasible and stationary but not complementary and calls them solved
    (min x s.t. 1 <= x <= 2 came back at x = 1.1; Maros-Meszaros QPTEST 28%
    off). The adapter enables the check and, unless given, sets its
    tolerances to the run's ``eps_abs`` / ``eps_rel`` (ProxQP's own defaults,
    1e-5 and 0, when those are not set), so a ProxQP "solved" means the gap
    is closed to the same accuracy as the residuals.

    The gap is a difference of objective-sized terms, so an absolute-only
    test (``eps_rel`` 0, ProxQP's default) cannot be met in double precision
    once the objective is large: at ``eps_abs`` 1e-9, QSCAGR25 (objective
    2e8) ran to ``max_iter`` and QPCBOEI1 ended in a false infeasibility. With
    ``eps_rel`` 0, ``eps_abs`` is therefore also the gap's relative tolerance.
    Returns what was used, for ``info``.
    """
    check = bool(settings.setdefault("check_duality_gap", True))
    if not check:
        return {"check_duality_gap": False}
    settings_class = getattr(proxsuite.proxqp, "Settings", None)
    defaults = settings_class() if settings_class is not None else None
    default_abs = getattr(defaults, "eps_abs", _PROXQP_DEFAULT_EPS_ABS)
    default_rel = getattr(defaults, "eps_rel", _PROXQP_DEFAULT_EPS_REL)
    eps_abs = float(_given(settings, "eps_abs", default_abs))
    eps_rel = float(_given(settings, "eps_rel", default_rel))
    settings.setdefault("eps_duality_gap_abs", eps_abs)
    settings.setdefault("eps_duality_gap_rel", eps_rel if eps_rel > 0 else eps_abs)
    return {
        "check_duality_gap": True,
        "eps_duality_gap_abs": float(settings["eps_duality_gap_abs"]),
        "eps_duality_gap_rel": float(settings["eps_duality_gap_rel"]),
    }


def _given(settings: dict, key: str, default):
    """``settings[key]``, or ``default`` if it is absent or None (ProxQP's "unset")."""
    value = settings.get(key)
    return default if value is None else value


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _is_integer(value) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


# Value ranges for ProxQP's numeric settings. ProxSuite does not check them: a
# negative eps_abs or max_iter is accepted and the solve runs to max_iter.
_SETTING_RANGES = {
    **{
        key: ("a finite number >= 0", lambda v: _is_number(v) and 0 <= v < math.inf)
        for key in ("eps_abs", "eps_rel", "eps_duality_gap_abs", "eps_duality_gap_rel")
    },
    **{
        key: ("a finite number > 0", lambda v: _is_number(v) and 0 < v < math.inf)
        for key in ("rho", "mu_eq", "mu_in")
    },
    "max_iter": ("an integer >= 1", lambda v: _is_integer(v) and v >= 1),
}


def _check_setting_values(settings: dict) -> None:
    """Raise SolverConfigError naming a setting whose value is out of range (None means unset)."""
    for key, value in settings.items():
        if key in _SETTING_RANGES and value is not None:
            description, test = _SETTING_RANGES[key]
            if not test(value):
                raise SolverConfigError(
                    f"Invalid ProxQP setting {key!r}={value!r}: must be {description}"
                )


# Positional problem data in ProxQP's ``solve`` signatures, not settings.
_DATA_ARGUMENTS = {"H", "g", "A", "b", "C", "l", "u", "l_box", "u_box", "x", "y", "z"}


def _check_settings(settings: dict, solve) -> None:
    """Raise SolverConfigError naming any setting ProxQP's ``solve`` does not take.

    The nanobind binding would raise a TypeError that lists every passed
    keyword without saying which one is wrong. The accepted names are read
    from the binding's signature docstring (all overloads); if it cannot be
    parsed, the settings are passed through unchecked.
    """
    accepted: set[str] = set()
    for line in (getattr(solve, "__doc__", None) or "").splitlines():
        if line.startswith("solve(") and ") ->" in line:
            inner = line[len("solve(") : line.rindex(") ->")]
            accepted.update(re.findall(r"(?:^|,\s)(\w+):", inner))
    accepted -= _DATA_ARGUMENTS
    if not accepted:
        return
    unknown = sorted(key for key in settings if key not in accepted)
    if unknown:
        raise SolverConfigError(
            f"Unrecognized ProxQP settings {unknown}; accepted: {sorted(accepted)}"
        )


def _map_proxqp_status(raw_status, proxsuite) -> str:
    proxqp = proxsuite.proxqp
    mapping = {
        proxqp.PROXQP_SOLVED: status.OPTIMAL,
        # Only with primal_infeasibility_solving. ProxSuite documents it as
        # "the closest (in L2 sense) feasible problem is solved", and its
        # sparse backend returns it for feasible problems too, with their
        # solution; it is not an infeasibility verdict. The KKT check of the
        # returned point tells the two cases apart for kkt-verify.
        proxqp.PROXQP_SOLVED_CLOSEST_PRIMAL_FEASIBLE: status.OPTIMAL_INACCURATE,
        proxqp.PROXQP_MAX_ITER_REACHED: status.MAX_ITER_REACHED,
        proxqp.PROXQP_PRIMAL_INFEASIBLE: status.PRIMAL_INFEASIBLE,
        proxqp.PROXQP_DUAL_INFEASIBLE: status.DUAL_INFEASIBLE,
    }
    # Newer ProxSuite versions added inaccurate variants of the
    # infeasibility statuses. getattr keeps us forward-compatible.
    primal_inaccurate = getattr(proxqp, "PROXQP_PRIMAL_INFEASIBLE_INACCURATE", None)
    dual_inaccurate = getattr(proxqp, "PROXQP_DUAL_INFEASIBLE_INACCURATE", None)
    not_run = getattr(proxqp, "PROXQP_NOT_RUN", None)
    if primal_inaccurate is not None:
        mapping[primal_inaccurate] = status.PRIMAL_INFEASIBLE_INACCURATE
    if dual_inaccurate is not None:
        mapping[dual_inaccurate] = status.DUAL_INFEASIBLE_INACCURATE
    if not_run is not None:
        mapping[not_run] = status.SOLVER_ERROR
    return mapping.get(raw_status, status.SOLVER_ERROR)


def _info_dict(info) -> dict:
    out = {}
    for key in [
        "status",
        "iter",
        "iter_ext",
        "objValue",
        "pri_res",
        "dua_res",
        "duality_gap",
        "run_time",
        "setup_time",
        "solve_time",
        "rho",
        "mu_eq",
        "mu_in",
    ]:
        if hasattr(info, key):
            value = getattr(info, key)
            out[key] = str(value) if key == "status" else value
    return out


def _float_vector(value):
    """A 1-D float64 copy of ``value`` (None and complex data pass unchanged)."""
    if value is None:
        return None
    array = np.asarray(value)
    if np.iscomplexobj(array):
        return array
    return array.astype(float).reshape(-1)


def _float_matrix(value):
    if value is None or np.iscomplexobj(value.data if sp.issparse(value) else value):
        return value
    return sp.csc_matrix(value, dtype=float)


def _settings_refused(backend, settings: dict, use_dense: bool) -> bool:
    """Whether ProxQP refuses ``settings`` on a one-variable problem too."""
    h = np.eye(1) if use_dense else sp.csc_matrix(np.eye(1))
    try:
        backend.solve(h, np.ones(1), None, None, None, None, None, **settings)
    except TypeError:
        return True
    except Exception:  # noqa: BLE001 - any other failure says nothing about types
        return False
    return False


def _proxqp_seconds(value):
    return None if value is None else float(value) / 1.0e6


def _maybe_int(value):
    return None if value is None else int(value)
