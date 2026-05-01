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
ANY_INFEASIBLE = {
    PRIMAL_INFEASIBLE,
    PRIMAL_INFEASIBLE_INACCURATE,
    DUAL_INFEASIBLE,
    DUAL_INFEASIBLE_INACCURATE,
    PRIMAL_OR_DUAL_INFEASIBLE,
}
