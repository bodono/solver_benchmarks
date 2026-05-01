"""DC Optimal Power Flow LP construction.

Given a parsed MATPOWER case (buses, generators, branches, generator
costs), build the DC OPF LP::

    minimize    sum_g (c1_g * Pg_g + c0_g)
    subject to  power balance at each bus:
                    sum_{g at bus i} Pg_g - Pd_i  =  sum_j B[i,j] * theta_j
                reference bus angle:
                    theta_ref = 0
                generator limits:
                    Pmin_g <= Pg_g <= Pmax_g
                line flow limits:
                    -rateA_l <= (theta_i - theta_j) / x_l <= rateA_l
                                (only when rateA_l > 0; rateA_l = 0 means
                                 unlimited per MATPOWER convention)

where ``B`` is the bus susceptance matrix from the line reactances.
This is a clean LP with structure typical of power-systems workloads:
sparse equality constraints (power balance), sparse inequality
constraints (line flows), and box bounds (generator limits).

The variable layout is ``x = [Pg_1, ..., Pg_G, theta_1, ..., theta_N]``
with ``G`` generators and ``N`` buses. Costs are read from
``gencost`` rows assuming the polynomial cost model (``model = 2``);
quadratic terms are ignored (DC OPF in this benchmark is the linear
relaxation, leaving QP-form OPF for a follow-up).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def dc_opf_lp(case: dict) -> tuple[dict, dict]:
    """Build the DC OPF LP from a parsed MATPOWER case.

    Parameters
    ----------
    case:
        Output of ``parse_matpower_case`` — a dict with keys
        ``baseMVA`` (float), ``bus`` (np.ndarray), ``branch``
        (np.ndarray), ``gen`` (np.ndarray), ``gencost`` (np.ndarray).

    Returns
    -------
    (problem_dict, metadata)
        ``problem_dict`` is the canonical QP-form dict (P=0 since LP).
        ``metadata`` carries diagnostic counts.
    """
    base_mva = float(case["baseMVA"])
    bus = np.asarray(case["bus"], dtype=float)
    branch = np.asarray(case["branch"], dtype=float)
    gen = np.asarray(case["gen"], dtype=float)
    gencost = np.asarray(case["gencost"], dtype=float)

    if bus.ndim != 2 or bus.shape[0] == 0:
        raise ValueError("MATPOWER case has no buses.")
    if gen.shape[0] == 0:
        raise ValueError("MATPOWER case has no generators.")

    n_bus = int(bus.shape[0])
    n_gen = int(gen.shape[0])
    n_branch = int(branch.shape[0])

    # MATPOWER bus IDs may be arbitrary (not 1..N); build a remap.
    bus_id = bus[:, 0].astype(int)
    bus_index = {bus_id[i]: i for i in range(n_bus)}

    # Reference (slack) bus: bus type column index 1 (0-indexed); type 3 = REF.
    ref_idx = int(np.argmax(bus[:, 1] == 3.0))
    if bus[ref_idx, 1] != 3.0:
        # No explicit REF bus; pick bus 0 as the reference.
        ref_idx = 0

    # Bus-level demand (Pd) at column 2.
    pd_pu = bus[:, 2] / base_mva

    # Build the bus susceptance matrix B (DC). Only active branches
    # (status = column 10, 1.0 = active) contribute.
    b_rows: list[int] = []
    b_cols: list[int] = []
    b_data: list[float] = []
    line_susceptance = np.zeros(n_branch, dtype=float)
    line_status = np.ones(n_branch, dtype=bool)
    for k in range(n_branch):
        status = float(branch[k, 10]) if branch.shape[1] > 10 else 1.0
        if status == 0.0:
            line_status[k] = False
            continue
        f = bus_index[int(branch[k, 0])]
        t = bus_index[int(branch[k, 1])]
        # Reactance in column 3; resistance ignored in DC OPF.
        x = float(branch[k, 3])
        if x == 0.0:
            # Avoid divide-by-zero; flag and skip.
            line_status[k] = False
            continue
        susceptance = 1.0 / x
        line_susceptance[k] = susceptance
        # B[f, t] -= susceptance; B[t, f] -= susceptance; B[f, f] += susceptance; B[t, t] += susceptance.
        b_rows.extend([f, t, f, t])
        b_cols.extend([t, f, f, t])
        b_data.extend([-susceptance, -susceptance, susceptance, susceptance])
    b_matrix = sp.csc_matrix(
        (b_data, (b_rows, b_cols)), shape=(n_bus, n_bus)
    )

    # Generator-to-bus selector ``M``: M[bus, g] = 1 iff generator g
    # is connected to that bus.
    m_rows: list[int] = []
    m_cols: list[int] = []
    m_data: list[float] = []
    for g in range(n_gen):
        bus_pos = bus_index[int(gen[g, 0])]
        m_rows.append(bus_pos)
        m_cols.append(g)
        m_data.append(1.0)
    m_matrix = sp.csc_matrix(
        (m_data, (m_rows, m_cols)), shape=(n_bus, n_gen)
    )

    # Variable ordering: x = [Pg (n_gen), theta (n_bus)].
    n_vars = n_gen + n_bus

    # Power balance: M Pg - Pd = B theta  ⇔  M Pg - B theta = Pd.
    eq_a = sp.hstack([m_matrix, -b_matrix], format="csc")
    eq_b = pd_pu

    # Reference bus angle: theta_ref = 0.
    ref_row = sp.csc_matrix(
        ([1.0], ([0], [n_gen + ref_idx])), shape=(1, n_vars)
    )

    # Line flow limits: rateA in column 5, in MW; convert to p.u.
    # flow_l = (theta_f - theta_t) / x_l, |flow_l| <= rateA / baseMVA.
    flow_rows: list[sp.csc_matrix] = []
    flow_l: list[float] = []
    flow_u: list[float] = []
    for k in range(n_branch):
        if not line_status[k]:
            continue
        rate_mw = float(branch[k, 5]) if branch.shape[1] > 5 else 0.0
        if rate_mw <= 0.0:
            # MATPOWER convention: rateA = 0 means unlimited.
            continue
        rate_pu = rate_mw / base_mva
        f = bus_index[int(branch[k, 0])]
        t = bus_index[int(branch[k, 1])]
        susceptance = line_susceptance[k]
        # Flow: susceptance * (theta_f - theta_t).
        row_data = [susceptance, -susceptance]
        row_cols = [n_gen + f, n_gen + t]
        flow_rows.append(
            sp.csc_matrix(
                (row_data, ([0, 0], row_cols)), shape=(1, n_vars)
            )
        )
        flow_l.append(-rate_pu)
        flow_u.append(rate_pu)

    # Generator bounds (in p.u.): cols 8 (Pmax) and 9 (Pmin), MW.
    gen_pmin = gen[:, 9] / base_mva
    gen_pmax = gen[:, 8] / base_mva
    # Status column at 7; deactivated generators clamped to zero.
    gen_status = (
        gen[:, 7].astype(int) if gen.shape[1] > 7 else np.ones(n_gen, dtype=int)
    )
    gen_pmin = np.where(gen_status > 0, gen_pmin, 0.0)
    gen_pmax = np.where(gen_status > 0, gen_pmax, 0.0)

    # Stack constraints: equality (power balance + reference) at the
    # top, then line flows, then generator bounds, then theta box
    # (free angles, [-inf, inf]).
    constraint_blocks = [eq_a, ref_row]
    l_blocks = [eq_b, np.array([0.0])]
    u_blocks = [eq_b, np.array([0.0])]
    if flow_rows:
        constraint_blocks.append(sp.vstack(flow_rows, format="csc"))
        l_blocks.append(np.array(flow_l, dtype=float))
        u_blocks.append(np.array(flow_u, dtype=float))
    # Generator box: rows of identity selecting Pg_g.
    gen_box_rows = sp.hstack(
        [sp.eye(n_gen, format="csc"), sp.csc_matrix((n_gen, n_bus))],
        format="csc",
    )
    constraint_blocks.append(gen_box_rows)
    l_blocks.append(gen_pmin)
    u_blocks.append(gen_pmax)

    a_full = sp.vstack(constraint_blocks, format="csc")
    l_full = np.concatenate(l_blocks)
    u_full = np.concatenate(u_blocks)

    # Linear cost from gencost. MATPOWER polynomial model (model=2):
    # cost(P) = c_n * P^n + ... + c_1 * P + c_0. We take c_1 (linear)
    # only for the LP relaxation; quadratic terms are ignored. The
    # last n columns of gencost are the polynomial coefficients in
    # decreasing-degree order; row col 3 (0-indexed) is the polynomial
    # degree count ``n``.
    q_vec = np.zeros(n_vars, dtype=float)
    r_const = 0.0
    for g in range(n_gen):
        if g >= gencost.shape[0]:
            continue
        model = float(gencost[g, 0])
        if model != 2.0:
            # Piecewise-linear (model=1) would need extra variables;
            # treat unsupported cost model as linear-zero so the LP
            # is well-formed but not meaningful for the cost. Caller
            # can detect via metadata.
            continue
        n_coef = int(gencost[g, 3])
        # Coefficients are in cols 4..(4+n_coef-1), in decreasing
        # degree order: c_{n-1}, c_{n-2}, ..., c_1, c_0.
        coefs = gencost[g, 4 : 4 + n_coef]
        if n_coef >= 2:
            # Linear term is the second-to-last coefficient.
            q_vec[g] += float(coefs[-2])
        if n_coef >= 1:
            r_const += float(coefs[-1])
        # Costs are in $/MWh per MW; the per-unit objective gets a
        # factor of baseMVA.
        q_vec[g] *= base_mva

    problem = {
        "P": sp.csc_matrix((n_vars, n_vars)),
        "q": q_vec,
        "r": float(r_const),
        "A": a_full,
        "l": l_full,
        "u": u_full,
        "n": n_vars,
        "m": int(a_full.shape[0]),
        "obj_type": "min",
    }
    metadata = {
        "num_buses": n_bus,
        "num_generators": n_gen,
        "num_branches_active": int(line_status.sum()),
        "num_lines_with_flow_limit": len(flow_rows),
        "reference_bus_index": ref_idx,
        "base_mva": base_mva,
    }
    return problem, metadata
