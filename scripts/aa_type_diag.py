"""Compare AA type-I vs type-II on diagnostic problems, dumping aa_stats."""
from __future__ import annotations

import argparse
import time

import numpy as np

from solver_benchmarks.datasets.maros_meszaros import MarosMeszarosDataset
from solver_benchmarks.transforms.cones import qp_to_scs_box_cone


def load_qp(dataset_id: str, name: str):
    ds = MarosMeszarosDataset()
    pd = ds.load_problem(name)
    return pd


def run_one(qp, settings, label):
    import scs
    data, cone, _ = qp_to_scs_box_cone(qp.qp)
    solver = scs.SCS(data, cone, **settings)
    t0 = time.perf_counter()
    res = solver.solve()
    elapsed = time.perf_counter() - t0
    info = res["info"]
    aa = info.get("aa_stats", {})
    print(f"\n--- {label} ---")
    print(f"  status              : {info['status']}")
    print(f"  iter                : {info['iter']}")
    print(f"  wall time           : {elapsed:.3f}s   solve_time(scs): {info['solve_time']/1000:.3f}s")
    print(f"  accel_time(scs)     : {info['accel_time']/1000:.3f}s")
    print(f"  accepted_accel      : {info['accepted_accel_steps']}")
    print(f"  rejected_accel      : {info['rejected_accel_steps']}")
    print(f"  aa_stats:")
    for k, v in aa.items():
        print(f"    {k:24s} {v}")
    return info, aa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="maros_meszaros")
    p.add_argument("--problem", default="PRIMALC1")
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--reg", type=float, default=1e-8)
    p.add_argument("--relax", type=float, default=1.2)
    args = p.parse_args()

    qp = load_qp(args.dataset, args.problem)
    print(f"Loaded {args.dataset}/{args.problem}")

    common = dict(
        eps_abs=1e-4, eps_rel=1e-4, max_iters=100000,
        linear_solver="qdldl", verbose=False,
        acceleration_lookback=args.lookback,
        acceleration_interval=args.interval,
        acceleration_regularization=args.reg,
        acceleration_relaxation=args.relax,
    )

    run_one(qp, {**common, "acceleration_lookback": 0}, "AA OFF (baseline)")
    run_one(qp, {**common, "acceleration_type_1": True},  "Type-I  (t1=True)")
    run_one(qp, {**common, "acceleration_type_1": False}, "Type-II (t1=False)")


if __name__ == "__main__":
    main()
