"""Run type-I experiments on the type-I weak-spot set."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

csv.field_size_limit(sys.maxsize)

from solver_benchmarks.datasets.maros_meszaros import MarosMeszarosDataset
from solver_benchmarks.datasets.mps import NetlibDataset
from solver_benchmarks.transforms.cones import qp_to_scs_box_cone

# Type-I weak spots — problems where best type-I config did NOT solve cleanly
# but the problem IS solvable (excludes genuine infeasibility detections and timeouts)
TYPE1_WEAK_SET = [
    ("netlib_feasible", "afiro"),       # 91 — AA actively hurts
    ("netlib_feasible", "israel"),      # 458
    ("maros_meszaros", "QISRAEL"),      # 458
    ("netlib_feasible", "vtp.base"),    # 603
    ("netlib_feasible", "lotfi"),       # 769
    ("netlib_feasible", "forplan"),     # 1003
    ("netlib_feasible", "agg3"),        # 1120
    ("netlib_feasible", "agg2"),        # 1120
    ("netlib_feasible", "tuff"),        # 1505
    ("maros_meszaros", "QFFFFF80"),     # 2232
    ("netlib_feasible", "pilot4"),      # 2322
    ("maros_meszaros", "QGFRDXPN"),     # 2800
    ("netlib_feasible", "perold"),      # 3289
]


_ds_cache = {}
def get_ds(name):
    if name not in _ds_cache:
        if name == "maros_meszaros":
            _ds_cache[name] = MarosMeszarosDataset()
        elif name == "netlib_feasible":
            _ds_cache[name] = NetlibDataset(dataset_options={"subset": "feasible"})
    return _ds_cache[name]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--out-dir", default="experiments/aa_type1")
    p.add_argument("--type", choices=["t1", "t2"], default="t1")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    import scs

    common = dict(
        eps_abs=1e-4, eps_rel=1e-4, max_iters=100000,
        linear_solver="qdldl", verbose=False,
        acceleration_lookback=10, acceleration_interval=5,
        acceleration_regularization=1e-8, acceleration_relaxation=1.2,
        acceleration_type_1=(args.type == "t1"),
    )

    results = []
    for i, (ds_id, name) in enumerate(TYPE1_WEAK_SET, 1):
        qp = get_ds(ds_id).load_problem(name)
        data, cone, _ = qp_to_scs_box_cone(qp.qp)
        t0 = time.perf_counter()
        res = scs.SCS(data, cone, **common).solve()
        elapsed = time.perf_counter() - t0
        info = res["info"]
        r = {
            "dataset": ds_id, "problem": name,
            "status": info["status"], "iters": info["iter"],
            "wall_time_s": elapsed,
            "accepted_accel": info["accepted_accel_steps"],
            "rejected_accel": info["rejected_accel_steps"],
            "aa_stats": info.get("aa_stats", {}),
        }
        results.append(r)
        m = "✓" if r["status"] == "solved" else " "
        print(f"[{i:2d}/{len(TYPE1_WEAK_SET)}] {m} {ds_id:18s} {name:18s} {r['wall_time_s']:7.2f}s  {r['status']}")

    out = {
        "label": args.label, "type": args.type,
        "n_problems": len(results),
        "n_clean_solved": sum(1 for r in results if r["status"] == "solved"),
        "n_max_iters_inaccurate": sum(1 for r in results if "max_iters" in r["status"]),
        "total_wall_time_s": sum(r["wall_time_s"] for r in results),
        "results": results,
    }
    out_path = Path(args.out_dir) / f"{args.label}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== {args.label} ({args.type}) ===")
    print(f"  clean solved : {out['n_clean_solved']}/{out['n_problems']}")
    print(f"  max_iters    : {out['n_max_iters_inaccurate']}")
    print(f"  total time   : {out['total_wall_time_s']:.1f}s")


if __name__ == "__main__":
    main()
