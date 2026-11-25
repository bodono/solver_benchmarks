"""Canonical benchmark statuses."""

OPTIMAL = "optimal"
OPTIMAL_INACCURATE = "optimal_inaccurate"
PRIMAL_INFEASIBLE = "primal_infeasible"
PRIMAL_INFEASIBLE_INACCURATE = "primal_infeasible_inaccurate"
DUAL_INFEASIBLE = "dual_infeasible"
DUAL_INFEASIBLE_INACCURATE = "dual_infeasible_inaccurate"
PRIMAL_OR_DUAL_INFEASIBLE = "primal_or_dual_infeasible"
MAX_ITER_REACHED = "max_iter_reached"
TIME_LIMIT = "time_limit"
SOLVER_ERROR = "solver_error"
WORKER_ERROR = "worker_error"
SKIPPED_UNSUPPORTED = "skipped_unsupported"

SOLUTION_PRESENT = {OPTIMAL}
# These statuses may carry a feasible point; KKT checks determine its accuracy.
ANY_FEASIBLE = {
    OPTIMAL,
    OPTIMAL_INACCURATE,
    MAX_ITER_REACHED,
    TIME_LIMIT,
    SOLVER_ERROR,
}
ANY_INFEASIBLE = {
    PRIMAL_INFEASIBLE,
    PRIMAL_INFEASIBLE_INACCURATE,
    DUAL_INFEASIBLE,
    DUAL_INFEASIBLE_INACCURATE,
    PRIMAL_OR_DUAL_INFEASIBLE,
}

# Known outcome of a problem, recorded by datasets as ``metadata["expected_status"]``.
# Problems without one are assumed to have an optimal solution, so success for
# them is unchanged: the status must be in the caller's success set.
EXPECT_OPTIMAL = "optimal"
# Primal infeasible; whether the dual is feasible is not stated.
EXPECT_INFEASIBLE = "infeasible"
# Primal feasible and dual infeasible: the objective is unbounded below.
EXPECT_UNBOUNDED = "unbounded"
# Primal and dual infeasible at once (NETLIB cplex1).
EXPECT_PRIMAL_AND_DUAL_INFEASIBLE = "primal_and_dual_infeasible"
EXPECTED_OUTCOMES = (EXPECT_OPTIMAL, EXPECT_INFEASIBLE, EXPECT_UNBOUNDED, EXPECT_PRIMAL_AND_DUAL_INFEASIBLE)

# What a known outcome says about each kind of infeasibility certificate, for
# the KKT verification (analysis/verify.py). A certificate of a kind the
# outcome confirms is accepted when its residual (kkt.cert_res) is within the
# row's tolerance. A certificate of a kind the outcome rules out is false and
# never accepted. Any other certificate (every one on a problem without a
# known outcome, and a dual-infeasibility certificate on a problem known only
# to be primal infeasible) is accepted only when it is a proof (kkt.cert_proof).
CONFIRMED_CERTIFICATES = {
    EXPECT_INFEASIBLE: frozenset({PRIMAL_INFEASIBLE}),
    EXPECT_UNBOUNDED: frozenset({DUAL_INFEASIBLE}),
    EXPECT_PRIMAL_AND_DUAL_INFEASIBLE: frozenset({PRIMAL_INFEASIBLE, DUAL_INFEASIBLE}),
}
RULED_OUT_CERTIFICATES = {
    EXPECT_OPTIMAL: frozenset({PRIMAL_INFEASIBLE, DUAL_INFEASIBLE}),
    EXPECT_UNBOUNDED: frozenset({PRIMAL_INFEASIBLE}),
}

# Which certificate statuses solve a problem of each known outcome. Only an
# accurate certificate counts, as only ``optimal`` counts for a solution; the
# ``*_inaccurate`` and ambiguous ``primal_or_dual_infeasible`` statuses carry
# no certificate the harness can trust, unless the KKT verification checked
# it and promoted it. On a primal-infeasible problem a dual-infeasibility
# certificate counts too: verification accepts one there only as a proof,
# and a valid one shows that no optimal pair exists.
CERTIFICATE_SUCCESS = {
    EXPECT_INFEASIBLE: frozenset({PRIMAL_INFEASIBLE, DUAL_INFEASIBLE}),
    EXPECT_UNBOUNDED: frozenset({DUAL_INFEASIBLE}),
    EXPECT_PRIMAL_AND_DUAL_INFEASIBLE: frozenset({PRIMAL_INFEASIBLE, DUAL_INFEASIBLE}),
}
