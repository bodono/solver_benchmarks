"""Build DC OPF QPs with per-unit generation, bus angles, and PWL cost variables."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def dc_opf_lp(case: dict) -> tuple[dict, dict]:
    """Return the QP and metadata for an evaluated MATPOWER case."""
    base_mva = float(case["baseMVA"])
    bus = np.asarray(case["bus"], dtype=float)
    branch = np.asarray(case["branch"], dtype=float)
    gen = np.asarray(case["gen"], dtype=float)
    gencost = np.asarray(case.get("gencost", np.empty((0, 0))), dtype=float)[: len(gen)]
    if bus.ndim != 2 or len(bus) == 0:
        raise ValueError("MATPOWER case has no buses.")
    if len(gen) == 0:
        raise ValueError("MATPOWER case has no generators.")

    n_bus, n_gen = len(bus), len(gen)
    n_primary = n_bus + n_gen
    bus_index = {int(row[0]): i for i, row in enumerate(bus)}
    ref = int(np.argmax(bus[:, 1] == 3))  # Defaults to bus 0 if no reference is specified.
    branch = branch[(branch[:, 10] != 0) & (branch[:, 3] != 0)]
    incidence, flow, shift = _network(branch, bus_index)
    generators = sp.csc_matrix(
        (np.ones(n_gen), ([bus_index[int(g)] for g in gen[:, 0]], np.arange(n_gen))),
        shape=(n_bus, n_gen),
    )

    # Flow is Bf @ theta - shift; shunt conductance adds real-power demand.
    balance = sp.hstack([generators, -incidence.T @ flow], format="csc")
    demand = (bus[:, 2] + bus[:, 4]) / base_mva - incidence.T @ shift
    refs = np.flatnonzero(bus[:, 1] == 3)
    if not len(refs):
        refs = np.array([ref])
    reference = sp.csc_matrix(
        (np.ones(len(refs)), (np.arange(len(refs)), n_gen + refs)),
        shape=(len(refs), n_primary),
    )
    reference_angles = np.deg2rad(bus[refs, 8] - bus[ref, 8])

    limited = branch[:, 5] > 0  # A zero rating means unlimited flow.
    limits = branch[limited, 5] / base_mva
    line_bounds = sp.hstack([sp.csc_matrix((sum(limited), n_gen)), flow[limited]])
    gen_bounds = sp.hstack([sp.eye(n_gen), sp.csc_matrix((n_gen, n_bus))])
    gen_min = np.where(gen[:, 7].astype(int) > 0, gen[:, 9] / base_mva, 0.0)
    gen_max = np.where(gen[:, 7].astype(int) > 0, gen[:, 8] / base_mva, 0.0)

    angle_min, angle_max = branch[:, 11], branch[:, 12]
    angle_limited = (
        ((angle_min != 0) & (angle_min > -360))
        | ((angle_max != 0) & (angle_max < 360))
        | ((angle_min != 0) & (angle_max == 0))
        | ((angle_min == 0) & (angle_max != 0))
    )
    angle_bounds = sp.hstack([sp.csc_matrix((sum(angle_limited), n_gen)), incidence[angle_limited]])
    angle_lower = np.deg2rad(np.where(angle_min < -360, -np.inf, angle_min))[angle_limited]
    angle_upper = np.deg2rad(np.where(angle_max > 360, np.inf, angle_max))[angle_limited]
    a = sp.vstack([balance, reference, line_bounds, angle_bounds, gen_bounds], format="csc")
    l = np.concatenate([demand, reference_angles, shift[limited] - limits, angle_lower, gen_min])
    u = np.concatenate([demand, reference_angles, shift[limited] + limits, angle_upper, gen_max])

    pwl_a, pwl_b = _piecewise_linear_costs(gencost, n_primary, base_mva)
    n_vars = pwl_a.shape[1]
    n_cost_vars = n_vars - n_primary
    if n_cost_vars:
        a = sp.vstack(
            [
                sp.hstack([a, sp.csc_matrix((a.shape[0], n_cost_vars))]),
                pwl_a,
            ],
            format="csc",
        )
        l = np.concatenate([l, np.full(len(pwl_b), -np.inf)])
        u = np.concatenate([u, pwl_b])
    p, q, r, dropped = _polynomial_costs(gencost, n_vars, base_mva)
    q[n_primary:] = 1.0

    problem = {
        "P": p,
        "q": q,
        "r": r,
        "A": a,
        "l": l,
        "u": u,
        "n": n_vars,
        "m": a.shape[0],
        "obj_type": "min",
    }
    metadata = {
        "num_buses": n_bus,
        "num_generators": n_gen,
        "num_branches_active": len(branch),
        "num_lines_with_flow_limit": int(sum(limited)),
        "reference_bus_index": ref,
        "base_mva": base_mva,
        "has_transformer_taps": bool(np.any((branch[:, 8] != 0) & (branch[:, 8] != 1))),
        "has_phase_shifts": bool(np.any(branch[:, 9] != 0)),
        "num_piecewise_linear_costs": n_cost_vars,
        "missing_gencost": "gencost" not in case,
        "dropped_cost_rows": dropped,
    }
    return problem, metadata


def _network(branch: np.ndarray, bus_index: dict[int, int]):
    """Branch incidence C, angle-to-flow matrix diag(b) C, and shift offsets."""
    n = len(branch)
    endpoints = [bus_index[int(bus)] for bus in branch[:, :2].ravel()]
    incidence = sp.csc_matrix(
        (np.tile([1.0, -1.0], n), (np.repeat(np.arange(n), 2), endpoints)),
        shape=(n, len(bus_index)),
    )
    tap = np.where(branch[:, 8] == 0, 1.0, branch[:, 8])
    susceptance = 1.0 / (branch[:, 3] * tap)
    flow = incidence.multiply(susceptance[:, None]).tocsc()
    shift = susceptance * np.deg2rad(branch[:, 9])
    return incidence, flow, shift


def _polynomial_costs(gencost: np.ndarray, n_vars: int, base_mva: float):
    """Scale MW costs for 0.5 * x.T @ P @ x + q.T @ x + r in per-unit."""
    diagonal, q = np.zeros(n_vars), np.zeros(n_vars)
    r = 0.0
    dropped = {"higher_order": [], "unknown_model": []}
    for g, row in enumerate(gencost):
        model = row[0]
        if model == 1:
            continue
        if model != 2:
            dropped["unknown_model"].append((g, float(model)))
            continue
        coefficients = row[4 : 4 + int(row[3])][::-1]
        if len(coefficients) > 0:
            r += coefficients[0]
        if len(coefficients) > 1:
            q[g] = coefficients[1] * base_mva
        if len(coefficients) > 2:
            diagonal[g] = 2 * coefficients[2] * base_mva**2
        if np.any(coefficients[3:] != 0):
            dropped["higher_order"].append(g)
    return sp.diags(diagonal, format="csc"), q, float(r), dropped


def _piecewise_linear_costs(
    gencost: np.ndarray,
    n_vars: int,
    base_mva: float,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Encode each segment as slope * Pg - cost <= -intercept."""
    generators = np.flatnonzero(gencost[:, 0] == 1) if gencost.size else []
    rows, cols, values, bounds = [], [], [], []
    for i, g in enumerate(generators):
        count = int(gencost[g, 3])
        x, y = gencost[g, 4 : 4 + 2 * count].reshape(-1, 2).T
        if count < 2 or np.any(np.diff(x) <= 0):
            raise ValueError(f"Invalid piecewise-linear cost points for generator {g}")
        slopes = np.diff(y) / np.diff(x)
        intercepts = y[:-1] - slopes * x[:-1]
        for slope, intercept in zip(slopes, intercepts):
            row = len(bounds)
            rows.extend([row, row])
            cols.extend([g, n_vars + i])
            values.extend([slope * base_mva, -1.0])
            bounds.append(-intercept)
    matrix = sp.csc_matrix(
        (values, (rows, cols)),
        shape=(len(bounds), n_vars + len(generators)),
    )
    return matrix, np.asarray(bounds)
