"""Quick comparison: type-I vs type-II across the 33 'type-I wins' problems."""
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


def list_gap_problems():
    out = []
    with REPORT.open() as f:
        for row in csv.DictReader(f):
            if row.get(f"{T1_ID}__status") == "optimal" and row.get(f"{T2_ID}__status") != "optimal":
                out.append((row["dataset"], row["problem"]))
    return out


_ds_cache: dict[str, object] = {}
def get_ds(name):
    if name not in _ds_cache:
        if name == "maros_meszaros":
            _ds_cache[name] = MarosMeszarosDataset()
        elif name == "netlib_feasible":
            _ds_cache[name] = NetlibDataset(dataset_options={"subset": "feasible"})
        else:
            raise ValueError(name)
    return _ds_cache[name]


def run(qp, settings):
    import scs
    data, cone, _ = qp_to_scs_box_cone(qp.qp)
    solver = scs.SCS(data, cone, **settings)
    t0 = time.perf_counter()
    res = solver.solve()
    return time.perf_counter() - t0, res["info"]


def main():
    problems = list_gap_problems()
    print(f"Running {len(problems)} 'type-I wins' problems...")
    common = dict(
        eps_abs=1e-4, eps_rel=1e-4, max_iters=100000,
        linear_solver="qdldl", verbose=False,
        acceleration_lookback=10, acceleration_interval=5,
        acceleration_regularization=1e-8, acceleration_relaxation=1.2,
    )
    print(f"{'dataset':18s} {'problem':30s} {'t1_time':>9s} {'t1_status':>20s} {'t2_time':>9s} {'t2_status':>22s}")
    t2_solved = 0
    t1_solved = 0
    for ds_id, name in problems:
        try:
            qp = get_ds(ds_id).load_problem(name)
        except Exception as e:
            print(f"{ds_id:18s} {name:30s} LOAD-FAIL: {e}")
            continue
        t_t1, info_t1 = run(qp, {**common, "acceleration_type_1": True})
        t_t2, info_t2 = run(qp, {**common, "acceleration_type_1": False})
        s1, s2 = info_t1["status"], info_t2["status"]
        if "solved" in s1 and "inaccurate" not in s1:
            t1_solved += 1
        if "solved" in s2 and "inaccurate" not in s2:
            t2_solved += 1
        print(f"{ds_id:18s} {name:30s} {t_t1:9.3f} {s1:>20s} {t_t2:9.3f} {s2:>22s}")
    print(f"\nType-I optimal: {t1_solved}/{len(problems)}  Type-II optimal: {t2_solved}/{len(problems)}")


if __name__ == "__main__":
    main()
