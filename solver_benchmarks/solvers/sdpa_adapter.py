"""SDPA adapter via sdpa-python."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from solver_benchmarks.analysis import kkt
from solver_benchmarks.core import status
from solver_benchmarks.core.problem import CONE, ProblemData
from solver_benchmarks.core.result import SolverResult
from .base import SolverAdapter, SolverUnavailable, settings_with_defaults


class SDPASolverAdapter(SolverAdapter):
    solver_name = "sdpa"
    supported_problem_kinds = {CONE}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import sdpap  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import sdpap
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install with the sdpa extra to use SDPA") from exc

        cone_problem = problem.cone
        p = cone_problem.get("P")
        if p is not None and sp.csc_matrix(p).nnz:
            return SolverResult(
                status=status.SKIPPED_UNSUPPORTED,
                info={"reason": "SDPA supports linear cone objectives, not quadratic objectives"},
            )

        try:
            prepared = _prepare_sdpap_problem(cone_problem, sdpap)
        except ValueError as exc:
            return SolverResult(status=status.SKIPPED_UNSUPPORTED, info={"reason": str(exc)})

        settings = settings_with_defaults(self.settings)
        optimality_tolerance = float(settings.pop("optimality_tolerance", 1.0e-6))
        option = _sdpa_options(settings, artifacts_dir)
        start = time.perf_counter()
        result = sdpap.solve(prepared.a, prepared.b, prepared.c, prepared.k, prepared.j, option)
        elapsed = time.perf_counter() - start
        if result is None:
            return SolverResult(
                status=status.SOLVER_ERROR,
                run_time_seconds=elapsed,
                info={"raw_status": "no_result"},
            )
        x_raw, y_raw, sdpap_info, time_info, sdpa_info = result
        x = _dense(x_raw)
        y_sdpap = _dense(y_raw)
        y = prepared.dual_to_original(y_sdpap)
        mapped = _map_sdpa_status(sdpap_info, sdpa_info, option, optimality_tolerance)
        slack = np.asarray(cone_problem["b"], dtype=float) - sp.csc_matrix(cone_problem["A"]) @ x
        kkt_dict = None
        if mapped in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
            p_mat = sp.csc_matrix((sp.csc_matrix(cone_problem["A"]).shape[1],) * 2)
            kkt_dict = kkt.cone_residuals(
                p_mat,
                cone_problem["q"],
                cone_problem["A"],
                cone_problem["b"],
                cone_problem["cone"],
                x,
                y,
                slack,
            )
        return SolverResult(
            status=mapped,
            objective_value=_maybe_float(sdpap_info.get("primalObj")),
            iterations=_maybe_int(sdpa_info.get("iteration")),
            run_time_seconds=elapsed,
            solve_time_seconds=_maybe_float(time_info.get("sdpa")),
            info={
                "phasevalue": sdpap_info.get("phasevalue"),
                "primalObj": sdpap_info.get("primalObj"),
                "dualObj": sdpap_info.get("dualObj"),
                "dualityGap": sdpap_info.get("dualityGap"),
                "primalError": sdpap_info.get("primalError"),
                "dualError": sdpap_info.get("dualError"),
                "time": time_info,
                "sdpa": sdpa_info,
            },
            kkt=kkt_dict,
        )


@dataclass
class _PreparedSDPAProblem:
    a: sp.csc_matrix
    b: np.ndarray
    c: np.ndarray
    k: object
    j: object
    dual_blocks: list[tuple[int, int, int, int, sp.csc_matrix | None]]

    def dual_to_original(self, y_sdpap: np.ndarray) -> np.ndarray:
        if not self.dual_blocks:
            return np.array([], dtype=float)
        original_size = max(start + size for start, size, _, _, _ in self.dual_blocks)
        y = np.zeros(original_size)
        for orig_start, orig_size, sdpa_start, sdpa_size, transform in self.dual_blocks:
            block = y_sdpap[sdpa_start : sdpa_start + sdpa_size]
            if transform is not None:
                block = transform.T @ block
            y[orig_start : orig_start + orig_size] = np.asarray(block, dtype=float).reshape(-1)
        return y


def _prepare_sdpap_problem(cone_problem: dict, sdpap) -> _PreparedSDPAProblem:
    raw_a = -sp.csc_matrix(cone_problem["A"])
    raw_b = -np.asarray(cone_problem["b"], dtype=float)
    c = np.asarray(cone_problem["q"], dtype=float)
    cone = dict(cone_problem["cone"])
    sdpa_row = 0
    a_parts = []
    b_parts = []
    dual_blocks: list[tuple[int, int, int, int, sp.csc_matrix | None]] = []
    unsupported = set(cone) - {"f", "z", "l", "q", "s"}
    if unsupported:
        raise ValueError(f"SDPA adapter does not support cone keys: {', '.join(sorted(unsupported))}")
    blocks = _cone_blocks(cone)
    row = sum(block[2] for block in blocks)
    if row != raw_a.shape[0]:
        raise ValueError(f"SDPA cone dimensions ({row}) do not match A rows ({raw_a.shape[0]})")
    by_kind: dict[str, list[tuple[int, int, int]]] = {"f": [], "l": [], "q": [], "s": []}
    for kind, start, dim, order in blocks:
        by_kind[kind].append((start, dim, order))

    for kind in ("f", "l", "q"):
        for start, dim, _ in by_kind[kind]:
            _append_identity_block(raw_a, raw_b, start, dim, sdpa_row, a_parts, b_parts, dual_blocks)
            sdpa_row += dim
    for start, dim, psd_order in by_kind["s"]:
        transform = _psd_triangle_to_full(psd_order)
        a_parts.append(transform @ raw_a[start : start + dim, :])
        b_parts.append(transform @ raw_b[start : start + dim])
        dual_blocks.append((start, dim, sdpa_row, psd_order * psd_order, transform))
        sdpa_row += psd_order * psd_order

    j_f = sum(dim for _, dim, _ in by_kind["f"])
    j_l = sum(dim for _, dim, _ in by_kind["l"])
    j_q = tuple(dim for _, dim, _ in by_kind["q"])
    j_s = tuple(order for _, _, order in by_kind["s"])
    a = sp.vstack(a_parts, format="csc") if a_parts else sp.csc_matrix((0, raw_a.shape[1]))
    b = np.concatenate([np.asarray(part).reshape(-1) for part in b_parts]) if b_parts else np.array([])
    k = sdpap.SymCone(f=int(raw_a.shape[1]))
    j = sdpap.SymCone(f=j_f, l=j_l, q=j_q, s=j_s)
    return _PreparedSDPAProblem(a=a, b=b, c=c, k=k, j=j, dual_blocks=dual_blocks)


def _cone_blocks(cone: dict) -> list[tuple[str, int, int, int]]:
    blocks = []
    row = 0
    for key, value in cone.items():
        if key in {"f", "z"}:
            dim = int(value)
            if dim:
                blocks.append(("f", row, dim, dim))
            row += dim
        elif key == "l":
            dim = int(value)
            if dim:
                blocks.append(("l", row, dim, dim))
            row += dim
        elif key == "q":
            for dim in _as_tuple(value):
                dim = int(dim)
                if dim:
                    blocks.append(("q", row, dim, dim))
                row += dim
        elif key == "s":
            for order in _as_tuple(value):
                order = int(order)
                dim = order * (order + 1) // 2
                if dim:
                    blocks.append(("s", row, dim, order))
                row += dim
    return blocks


def _append_identity_block(
    raw_a,
    raw_b,
    row: int,
    dim: int,
    sdpa_row: int,
    a_parts: list,
    b_parts: list,
    dual_blocks: list,
) -> None:
    a_parts.append(raw_a[row : row + dim, :])
    b_parts.append(raw_b[row : row + dim])
    dual_blocks.append((row, dim, sdpa_row, dim, None))


def _psd_triangle_to_full(dim: int) -> sp.csc_matrix:
    rows = []
    cols = []
    data = []
    col = 0
    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                rows.append(i + j * dim)
                cols.append(col)
                data.append(1.0)
            else:
                scale = 1.0 / np.sqrt(2.0)
                rows.extend([i + j * dim, j + i * dim])
                cols.extend([col, col])
                data.extend([scale, scale])
            col += 1
    return sp.csc_matrix((data, (rows, cols)), shape=(dim * dim, dim * (dim + 1) // 2))


def _sdpa_options(settings: dict, artifacts_dir: Path) -> dict:
    options = {}
    verbose = bool(settings.pop("verbose", True))
    options["print"] = settings.pop("print", "display" if verbose else "no")
    if settings.pop("result_file", False):
        options["resultFile"] = str(artifacts_dir / "sdpa_result.txt")
    if "max_iter" in settings:
        options["maxIteration"] = int(settings.pop("max_iter"))
    if "maxIteration" in settings:
        options["maxIteration"] = int(settings.pop("maxIteration"))
    if "eps_abs" in settings:
        options["epsilonStar"] = float(settings.pop("eps_abs"))
    if "eps_rel" in settings:
        options["epsilonDash"] = float(settings.pop("eps_rel"))
    ignored = {"time_limit", "time_limit_sec"}
    for key in list(settings):
        if key in ignored:
            settings.pop(key)
    options.update(settings)
    return options


def _map_sdpa_status(sdpap_info, sdpa_info, options: dict, tol: float) -> str:
    phase = str(sdpap_info.get("phasevalue", sdpa_info.get("phasevalue", "")))
    primal_error = _safe_float(sdpap_info.get("primalError"))
    dual_error = _safe_float(sdpap_info.get("dualError"))
    gap = abs(_safe_float(sdpap_info.get("dualityGap")))
    if phase in {"pdOPT", "pdFEAS"}:
        if max(primal_error, dual_error, gap) <= tol:
            return status.OPTIMAL
        return status.OPTIMAL_INACCURATE
    # CLP-form (sdpapinfo) phasevalue convention. By LP duality, primal unbounded
    # (pUNBD) implies dual infeasible, and dual unbounded (dUNBD) implies primal
    # infeasible. sdpap.py flips SDPA's internal labels into this CLP convention
    # (see GET_PRIMAL_STATUS_FROM_DUAL in the sdpap source).
    if phase in {"pINF_dFEAS", "dUNBD"}:
        return status.PRIMAL_INFEASIBLE
    if phase in {"pFEAS_dINF", "pUNBD"}:
        return status.DUAL_INFEASIBLE
    if phase == "pdINF":
        return status.PRIMAL_OR_DUAL_INFEASIBLE
    if int(sdpa_info.get("iteration", 0)) >= int(options.get("maxIteration", 100)):
        return status.MAX_ITER_REACHED
    return status.SOLVER_ERROR


def _dense(value) -> np.ndarray:
    if sp.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=float).reshape(-1)


def _as_tuple(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _safe_float(value) -> float:
    if value is None:
        return np.inf
    return float(value)


def _maybe_float(value):
    return None if value is None else float(value)


def _maybe_int(value):
    return None if value is None else int(value)
