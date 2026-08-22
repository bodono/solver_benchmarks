"""Regression test: run type-I and previously-working type-II on a sample."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from solver_benchmarks.datasets.maros_meszaros import MarosMeszarosDataset
from solver_benchmarks.datasets.mps import NetlibDataset
from solver_benchmarks.transforms.cones import qp_to_scs_box_cone

csv.field_size_limit(sys.maxsize)

REPORT = Path("results/scs_anderson_sweep_2026-04-27_08-44-15_UTC/report/problem_solver_comparison.csv")
T1_ID = "scs_aa_lb10_int5_t1True_reg1e-08_relax1.2"
T2_ID = "scs_aa_lb10_int5_t1False_reg1e-08_relax1.2"


def f2(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


_ds_cache = {}
def get_ds(name):
    if name not in _ds_cache:
        if name == "maros_meszaros":
            _ds_cache[name] = MarosMeszarosDataset()
        elif name == "netlib_feasible":
            _ds_cache[name] = NetlibDataset(dataset_options={"subset": "feasible"})
    return _ds_cache[name]


def load_qp(ds_id, name):
    return get_ds(ds_id).load_problem(name)


def time_solve(qp, settings):
    import scs
    data, cone, _ = qp_to_scs_box_cone(qp.qp)
    solver = scs.SCS(data, cone, **settings)
    t0 = time.perf_counter()
    res = solver.solve()
    return time.perf_counter() - t0, res["info"]["status"]


def main():
    # Type-I regression: pick 15 problems where type-I solved < 5s in the original
    t1_sample = []
    t2_sample = []
    with REPORT.open() as f:
        for row in csv.DictReader(f):
            t1_t = f2(row.get(f"{T1_ID}__run_time_seconds"))
            t2_t = f2(row.get(f"{T2_ID}__run_time_seconds"))
            if row.get(f"{T1_ID}__status") == "optimal" and 0.1 < (t1_t or 0) < 5:
                t1_sample.append((row["dataset"], row["problem"], t1_t))
            if row.get(f"{T2_ID}__status") == "optimal" and 0.1 < (t2_t or 0) < 5:
                t2_sample.append((row["dataset"], row["problem"], t2_t))
    t1_sample = sorted(t1_sample)[:15]
    t2_sample = sorted(t2_sample)[:15]

    common = dict(
        eps_abs=1e-4, eps_rel=1e-4, max_iters=100000,
        linear_solver="qdldl", verbose=False,
        acceleration_lookback=10, acceleration_interval=5,
        acceleration_regularization=1e-8, acceleration_relaxation=1.2,
    )

    print("=== TYPE-I regression ===")
    print(f"{'problem':18s} {'orig_t':>8s} {'new_t':>8s} {'status':>30s}")
    regressed_t1 = 0
    for ds, name, orig_t in t1_sample:
        qp = load_qp(ds, name)
        t, status = time_solve(qp, {**common, "acceleration_type_1": True})
        regressed = status != "solved"
        if regressed:
            regressed_t1 += 1
        flag = " REGRESS" if regressed else ""
        print(f"{name:18s} {orig_t:8.2f} {t:8.2f} {status:>30s}{flag}")

    print("\n=== TYPE-II regression (problems originally solving) ===")
    print(f"{'problem':18s} {'orig_t':>8s} {'new_t':>8s} {'status':>30s}")
    regressed_t2 = 0
    for ds, name, orig_t in t2_sample:
        qp = load_qp(ds, name)
        t, status = time_solve(qp, {**common, "acceleration_type_1": False})
        regressed = status != "solved"
        if regressed:
            regressed_t2 += 1
        flag = " REGRESS" if regressed else ""
        print(f"{name:18s} {orig_t:8.2f} {t:8.2f} {status:>30s}{flag}")

    print(f"\nType-I regressions:  {regressed_t1}/{len(t1_sample)}")
    print(f"Type-II regressions: {regressed_t2}/{len(t2_sample)}")


if __name__ == "__main__":
    main()
