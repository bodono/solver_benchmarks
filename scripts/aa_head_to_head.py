"""Head-to-head: type-I vs type-II on the union of both 'weak' sets."""
from __future__ import annotations

import csv
import sys
import time

from solver_benchmarks.datasets.maros_meszaros import MarosMeszarosDataset
from solver_benchmarks.datasets.mps import NetlibDataset
from solver_benchmarks.transforms.cones import qp_to_scs_box_cone

csv.field_size_limit(sys.maxsize)

# Type-II's failure set (33 problems where t2 originally failed)
T2_FAIL_SET = [
    ("maros_meszaros", "CONT-200"), ("maros_meszaros", "CVXQP1_L"),
    ("maros_meszaros", "QPILOTNO"), ("maros_meszaros", "QSHIP12L"),
    ("maros_meszaros", "UBH1"), ("netlib_feasible", "bnl2"),
    ("netlib_feasible", "fit2p"), ("netlib_feasible", "pilot87"),
    ("netlib_feasible", "maros"), ("netlib_feasible", "scfxm3"),
    ("maros_meszaros", "QSCFXM3"), ("netlib_feasible", "scfxm2"),
    ("netlib_feasible", "fffff800"), ("maros_meszaros", "QSHIP12S"),
    ("netlib_feasible", "finnis"), ("netlib_feasible", "boeing1"),
    ("netlib_feasible", "scfxm1"), ("maros_meszaros", "PRIMALC8"),
    ("netlib_feasible", "stair"), ("netlib_feasible", "capri"),
    ("maros_meszaros", "PRIMALC2"), ("maros_meszaros", "Q25FV47"),
    ("netlib_feasible", "ship12s"), ("netlib_feasible", "boeing2"),
    ("maros_meszaros", "QSCFXM2"), ("maros_meszaros", "QSCFXM1"),
    ("maros_meszaros", "QSHIP04S"), ("netlib_feasible", "standgub"),
    ("netlib_feasible", "standata"), ("maros_meszaros", "PRIMALC1"),
    ("netlib_feasible", "share2b"), ("maros_meszaros", "QSHARE1B"),
    ("maros_meszaros", "QSHARE2B"),
]
# Type-I's weak set (13 problems where t1 originally struggled)
T1_WEAK_SET = [
    ("netlib_feasible", "afiro"), ("netlib_feasible", "israel"),
    ("maros_meszaros", "QISRAEL"), ("netlib_feasible", "vtp.base"),
    ("netlib_feasible", "lotfi"), ("netlib_feasible", "forplan"),
    ("netlib_feasible", "agg3"), ("netlib_feasible", "agg2"),
    ("netlib_feasible", "tuff"), ("maros_meszaros", "QFFFFF80"),
    ("netlib_feasible", "pilot4"), ("maros_meszaros", "QGFRDXPN"),
    ("netlib_feasible", "perold"),
]

_ds_cache = {}
def get_ds(n):
    if n not in _ds_cache:
        _ds_cache[n] = (MarosMeszarosDataset() if n == "maros_meszaros"
                        else NetlibDataset(dataset_options={"subset": "feasible"}))
    return _ds_cache[n]


def run_one(qp_data, qp_cone, t1):
    import scs
    settings = dict(eps_abs=1e-4, eps_rel=1e-4, max_iters=100000, linear_solver="qdldl",
                    verbose=False, acceleration_lookback=10, acceleration_interval=5,
                    acceleration_regularization=1e-8, acceleration_relaxation=1.2,
                    acceleration_type_1=t1)
    t0 = time.perf_counter()
    res = scs.SCS(qp_data, qp_cone, **settings).solve()
    return time.perf_counter() - t0, res["info"]["status"]


def head_to_head(problems, label):
    print(f"\n=== {label} ({len(problems)} problems) ===")
    print(f"{'problem':18s}  {'t1_time':>8s}  {'t1_status':>30s}  {'t2_time':>8s}  {'t2_status':>30s}")
    t1_solved = t2_solved = 0
    t1_total = t2_total = 0.0
    for ds, name in problems:
        qp = get_ds(ds).load_problem(name)
        d, c, _ = qp_to_scs_box_cone(qp.qp)
        t1_t, t1_s = run_one(d, c, True)
        t2_t, t2_s = run_one(d, c, False)
        if t1_s == "solved":
            t1_solved += 1
        if t2_s == "solved":
            t2_solved += 1
        t1_total += t1_t
        t2_total += t2_t
        print(f"{name:18s}  {t1_t:8.2f}  {t1_s:>30s}  {t2_t:8.2f}  {t2_s:>30s}")
    print(f"  Type-I:  {t1_solved}/{len(problems)} clean solved, total {t1_total:.1f}s")
    print(f"  Type-II: {t2_solved}/{len(problems)} clean solved, total {t2_total:.1f}s")


head_to_head(T1_WEAK_SET, "Type-I's WEAK set")
head_to_head(T2_FAIL_SET, "Type-II's FAIL set")
