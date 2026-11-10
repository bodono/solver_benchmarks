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
# Primal infeasible. Either certificate counts: a valid dual-infeasibility
# certificate also proves no optimal pair exists, and some infeasible
# problems (e.g. NETLIB cplex1) are primal and dual infeasible at once.
EXPECT_INFEASIBLE = "infeasible"
# Primal feasible but unbounded: only a dual-infeasibility certificate is
# correct, since no valid primal-infeasibility certificate can exist.
EXPECT_UNBOUNDED = "unbounded"
EXPECTED_OUTCOMES = (EXPECT_OPTIMAL, EXPECT_INFEASIBLE, EXPECT_UNBOUNDED)

# Only an accurate certificate counts, as only ``optimal`` counts for a
# solution; the ``*_inaccurate`` and ambiguous ``primal_or_dual_infeasible``
# statuses carry no certificate the harness can trust.
CERTIFICATE_SUCCESS = {
    EXPECT_INFEASIBLE: frozenset({PRIMAL_INFEASIBLE, DUAL_INFEASIBLE}),
    EXPECT_UNBOUNDED: frozenset({DUAL_INFEASIBLE}),
}
