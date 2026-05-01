"""DC Optimal Power Flow LP construction.

Given a parsed MATPOWER case (buses, generators, branches, generator
costs), build the DC OPF LP. Implements MATPOWER's ``makeBdc``
convention so cases with off-nominal transformer taps and phase
shifters produce the LP MATPOWER itself would build.

Per-branch susceptance::

    b_l = 1 / (x_l * tau_l),    where tau_l = branch[8] if non-zero else 1
    shift_l_rad = branch[9] * pi / 180   (MATPOWER ships shifts in degrees)

Bus susceptance matrix and phase-shift injections::

    B[f, f] += b_l;  B[t, t] += b_l;  B[f, t] -= b_l;  B[t, f] -= b_l;
    P_inj[f] += -b_l * shift_l_rad
    P_inj[t] += +b_l * shift_l_rad

Power balance at each bus::

    M Pg - B theta = Pd - P_inj

Branch flow with phase shift::

    flow_l = b_l * (theta_f - theta_t - shift_l_rad)
    -rateA_l <= flow_l <= rateA_l   (when rateA_l > 0)

Pre-fix the implementation used ``b_l = 1 / x_l`` and ignored both
tap and shift, so cases with non-unity taps (common in IEEE
benchmarks) built a different LP from the MATPOWER source.

Variable layout: ``x = [Pg_1, ..., Pg_G, theta_1, ..., theta_N]``.

Costs are linearized (MATPOWER polynomial ``model = 2``); quadratic
terms and piecewise-linear (``model = 1``) cost rows are dropped
with ``info["dropped_cost_rows"]`` recording which generators
were affected, so reports do not present the resulting objective as
the original MATPOWER OPF cost when it isn't.
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

    # Build the bus susceptance matrix B (DC) following MATPOWER's
    # ``makeBdc``: per-branch susceptance ``b_l = 1 / (x_l * tau_l)``
    # with phase-shift injections at the from/to buses.
    b_rows: list[int] = []
    b_cols: list[int] = []
    b_data: list[float] = []
    line_susceptance = np.zeros(n_branch, dtype=float)
    line_shift_rad = np.zeros(n_branch, dtype=float)
    line_status = np.ones(n_branch, dtype=bool)
    p_businj = np.zeros(n_bus, dtype=float)
    has_taps = False
    has_phase_shifts = False
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
            line_status[k] = False
            continue
        # Tap ratio (column 8): MATPOWER convention is "0 means 1".
        tap_raw = float(branch[k, 8]) if branch.shape[1] > 8 else 0.0
        tap = tap_raw if tap_raw != 0.0 else 1.0
        if tap_raw not in (0.0, 1.0):
            has_taps = True
        # Phase shift (column 9, in degrees in MATPOWER).
        shift_deg = float(branch[k, 9]) if branch.shape[1] > 9 else 0.0
        shift_rad = float(np.deg2rad(shift_deg))
        if shift_rad != 0.0:
            has_phase_shifts = True
        susceptance = 1.0 / (x * tap)
        line_susceptance[k] = susceptance
        line_shift_rad[k] = shift_rad
        b_rows.extend([f, t, f, t])
        b_cols.extend([t, f, f, t])
        b_data.extend([-susceptance, -susceptance, susceptance, susceptance])
        # Phase-shift injection: -b_l * shift_rad enters the from-bus
        # power balance (treated as additional injection on the LHS),
        # +b_l * shift_rad enters the to-bus.
        p_businj[f] += -susceptance * shift_rad
        p_businj[t] += susceptance * shift_rad
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

    # Power balance per bus, accounting for phase-shift injections::
    #     M Pg - B theta + P_inj = Pd
    #     ⇔  M Pg - B theta = Pd - P_inj
    eq_a = sp.hstack([m_matrix, -b_matrix], format="csc")
    eq_b = pd_pu - p_businj

    # Reference bus angle: theta_ref = 0.
    ref_row = sp.csc_matrix(
        ([1.0], ([0], [n_gen + ref_idx])), shape=(1, n_vars)
    )

    # Line flow limits with tap and phase-shift handling::
    #     flow_l = b_l * (theta_f - theta_t - shift_l_rad)
    #     -rateA <= flow_l <= rateA   (rateA in MW, converted to p.u.)
    # Bring the constant ``b_l * shift_rad`` to the bound side:
    #     -rateA + b_l * shift <= b_l * (theta_f - theta_t) <= rateA + b_l * shift
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
        shift_offset = susceptance * line_shift_rad[k]
        row_data = [susceptance, -susceptance]
        row_cols = [n_gen + f, n_gen + t]
        flow_rows.append(
            sp.csc_matrix(
                (row_data, ([0, 0], row_cols)), shape=(1, n_vars)
            )
        )
        flow_l.append(-rate_pu + shift_offset)
        flow_u.append(rate_pu + shift_offset)

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

    # Linear cost from gencost. MATPOWER polynomial model
    # (``model = 2``): ``cost(P) = c_n P^n + ... + c_1 P + c_0``.
    # We take only the linear term and constant; quadratic and
    # higher-order coefficients are dropped because the QP-form
    # solvers in this benchmark expect a *linear* OPF objective.
    # Piecewise-linear (``model = 1``) rows are also dropped — they
    # need additional epigraph variables that don't fit the LP
    # surface this transform produces.
    #
    # Both drops are recorded on the metadata so reports can flag
    # that the optimal value is the linearized OPF cost, not the
    # original MATPOWER OPF cost.
    q_vec = np.zeros(n_vars, dtype=float)
    r_const = 0.0
    dropped_quadratic_rows: list[int] = []
    dropped_pwl_rows: list[int] = []
    dropped_unknown_model_rows: list[tuple[int, float]] = []
    for g in range(n_gen):
        if g >= gencost.shape[0]:
            continue
        model = float(gencost[g, 0])
        if model == 1.0:
            dropped_pwl_rows.append(g)
            continue
        if model != 2.0:
            dropped_unknown_model_rows.append((g, model))
            continue
        n_coef = int(gencost[g, 3])
        coefs = gencost[g, 4 : 4 + n_coef]
        if n_coef >= 2:
            q_vec[g] += float(coefs[-2])
        if n_coef >= 1:
            r_const += float(coefs[-1])
        # Costs are in $/MWh per MW; the per-unit objective gets a
        # factor of baseMVA.
        q_vec[g] *= base_mva
        # Track any quadratic/cubic terms we dropped so reports can
        # surface the linearization.
        if n_coef >= 3:
            higher_order = [
                float(c) for c in coefs[: n_coef - 2] if float(c) != 0.0
            ]
            if higher_order:
                dropped_quadratic_rows.append(g)

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
        "has_transformer_taps": has_taps,
        "has_phase_shifts": has_phase_shifts,
        # Cost-linearization provenance: callers can detect that
        # the LP objective is the linearized OPF cost rather than
        # the original MATPOWER OPF cost. ``dropped_quadratic_rows``
        # is generators whose quadratic / higher-order term was
        # dropped; ``dropped_pwl_rows`` is generators with a
        # piecewise-linear cost model (``model=1``) that we cannot
        # represent without epigraph variables; ``dropped_unknown_*``
        # is anything else.
        "dropped_cost_rows": {
            "quadratic": dropped_quadratic_rows,
            "piecewise_linear": dropped_pwl_rows,
            "unknown_model": dropped_unknown_model_rows,
        },
    }
    return problem, metadata
