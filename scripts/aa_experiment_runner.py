"""Run a sweep on the 33 'type-I wins, type-II loses' problems, dump JSON.

Used to compare type-II variants of aa.c. Expects scs-python to already be
rebuilt with the desired aa.c modifications. Captures status, wall time,
iters, and aa_stats per problem so we can compare variants.
"""
from __future__ import annotations

import argparse
import csv
import json
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


SMALL_SET = [
    ("netlib_feasible", "scfxm2"),
    ("netlib_feasible", "fffff800"),
    ("netlib_feasible", "finnis"),
    ("netlib_feasible", "boeing1"),
    ("netlib_feasible", "scfxm1"),
    ("maros_meszaros", "PRIMALC8"),
    ("netlib_feasible", "stair"),
    ("netlib_feasible", "capri"),
    ("maros_meszaros", "PRIMALC2"),
    ("netlib_feasible", "boeing2"),
    ("maros_meszaros", "QSCFXM2"),
    ("maros_meszaros", "QSCFXM1"),
    ("maros_meszaros", "QSHIP04S"),
    ("netlib_feasible", "standgub"),
    ("netlib_feasible", "standata"),
    ("maros_meszaros", "PRIMALC1"),
    ("netlib_feasible", "share2b"),
    ("maros_meszaros", "QSHARE1B"),
    ("maros_meszaros", "QSHARE2B"),
]


def gap_problems(small_only: bool = False):
    if small_only:
        return SMALL_SET
    out = []
    with REPORT.open() as f:
        for row in csv.DictReader(f):
            if row.get(f"{T1_ID}__status") == "optimal" and row.get(f"{T2_ID}__status") != "optimal":
                out.append((row["dataset"], row["problem"]))
    return out


def run_problem(qp, settings):
    import scs
    data, cone, _ = qp_to_scs_box_cone(qp.qp)
    solver = scs.SCS(data, cone, **settings)
    t0 = time.perf_counter()
    res = solver.solve()
    elapsed = time.perf_counter() - t0
    info = res["info"]
    return {
        "status": info["status"],
        "iters": info["iter"],
        "wall_time_s": elapsed,
        "accepted_accel": info["accepted_accel_steps"],
        "rejected_accel": info["rejected_accel_steps"],
        "aa_stats": info.get("aa_stats", {}),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="Experiment label for output JSON")
    p.add_argument("--out-dir", default="experiments/aa_type2")
    p.add_argument("--max-iters", type=int, default=100000)
    p.add_argument("--max-prob-time", type=float, default=60.0,
                   help="Skip per-problem when individual run would exceed this; informational")
    p.add_argument("--small-only", action="store_true",
                   help="Run only the fast subset (19 problems) instead of all 33")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    problems = gap_problems(small_only=args.small_only)

    common = dict(
        eps_abs=1e-4, eps_rel=1e-4, max_iters=args.max_iters,
        linear_solver="qdldl", verbose=False,
        acceleration_lookback=10, acceleration_interval=5,
        acceleration_regularization=1e-8, acceleration_relaxation=1.2,
        acceleration_type_1=False,
    )

    results = []
    for i, (ds_id, name) in enumerate(problems, 1):
        try:
            qp = get_ds(ds_id).load_problem(name)
        except Exception as e:
            print(f"[{i:2d}/{len(problems)}] {ds_id}/{name} LOAD-FAIL: {e}")
            continue
        try:
            r = run_problem(qp, common)
        except Exception as e:
            print(f"[{i:2d}/{len(problems)}] {ds_id}/{name} RUN-FAIL: {e}")
            r = {"status": f"error: {e}", "wall_time_s": -1, "iters": -1}
        r["dataset"] = ds_id
        r["problem"] = name
        results.append(r)
        status_str = r["status"]
        is_clean = status_str == "solved"
        marker = "✓" if is_clean else " "
        print(f"[{i:2d}/{len(problems)}] {marker} {ds_id:18s} {name:18s} {r['wall_time_s']:7.2f}s  {status_str}")

    out = {
        "label": args.label,
        "n_problems": len(results),
        "n_clean_solved": sum(1 for r in results if r["status"] == "solved"),
        "n_max_iters_inaccurate": sum(1 for r in results if "max_iters" in r["status"]),
        "total_wall_time_s": sum(r["wall_time_s"] for r in results if r["wall_time_s"] > 0),
        "results": results,
    }
    out_path = out_dir / f"{args.label}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== {args.label} ===")
    print(f"  clean solved        : {out['n_clean_solved']}/{out['n_problems']}")
    print(f"  hit max_iters       : {out['n_max_iters_inaccurate']}")
    print(f"  total wall time     : {out['total_wall_time_s']:.1f}s")
    print(f"  written to          : {out_path}")


if __name__ == "__main__":
    main()
