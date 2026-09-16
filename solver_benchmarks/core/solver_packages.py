"""Distribution names used by per-solve and manifest version metadata.

Keep this mapping free of solver imports so optional native packages can be
probed through importlib.metadata without loading them into the parent process.
"""

SOLVER_PACKAGES = {
    "clarabel": ("clarabel",),
    "cplex": ("cplex",),
    "cvxopt": ("cvxopt",),
    "ecos": ("ecos",),
    "gurobi": ("gurobipy",),
    "highs": ("highspy",),
    "mosek": ("Mosek", "mosek"),
    "osqp": ("osqp",),
    "pdlp": ("ortools",),
    "piqp": ("piqp",),
    "proxqp": ("proxsuite",),
    "qtqp": ("qtqp",),
    "scs": ("scs",),
    "sdpa": ("sdpa-python",),
}
