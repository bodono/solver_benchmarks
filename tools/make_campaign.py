"""Generate the sharded run configs for the SCS performance campaign.

One shard = one (dataset, solver variant, tolerance) so that every shard can
run in its own container with ``parallelism: 1`` (uncontended timings) and
the shards can fan out across as many machines as there are shards. The
output is a JSON spec consumed by tools/bench_modal.py; the same YAML
configs can be run locally with ``bench run``.

Usage: python tools/make_campaign.py OUT_DIR [--families qp,lp,sdp] [--tols 1e-4,1e-6]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

QP_LP_TIMEOUT = 300.0
SDP_TIMEOUT = 900.0
LPBIG_TIMEOUT = 1800.0

# Mittelmann LP benchmark set (plato.asu.edu/ftp/lptestset), largest first by
# download size, dealt round-robin into three chunks so that no shard can run
# longer than ~13 x (LPBIG_TIMEOUT + 60 s) and every chunk carries a mix of
# large and small instances.
MITTELMANN_LP = [
    "Dual2_5000", "dlr2", "L2CTA3D", "thk_48", "thk_63", "set-cover-model", "a2864", "dlr1",
    "L1_sixm1000obs", "fhnw-binschedule1", "tpl-tub-ws1617", "s82", "scpm1", "Primal2_1000",
    "square41", "supportcase19", "bharat", "L1_sixm250obs", "neos-3025225", "neos-5052403-cygnet",
    "woodlands09", "neos-5251015", "s100", "savsched1", "graph40-40", "datt256_lp", "s250r10",
    "ex10", "physiciansched3-3", "rmine15", "bdry2", "supportcase10", "Linf_520c",
    "chromaticindex1024-7", "irish-electricity", "brazil3", "qap15",
]
MITTELMANN_CHUNKS = [sorted(MITTELMANN_LP[i::3]) for i in range(3)]

# Datasets per family. `id` is the result-table label; options select subsets.
FAMILIES: dict[str, list[dict]] = {
    "qp": [
        {"name": "maros_meszaros", "id": "maros_meszaros"},
        {"name": "qplib", "id": "qplib", "dataset_options": {"subset": "all"}},
    ],
    "lp": [
        {"name": "netlib", "id": "netlib", "dataset_options": {"subset": "feasible"}},
        {"name": "kennington", "id": "kennington"},
        {"name": "miplib", "id": "miplib_relax", "dataset_options": {"max_size_mb": 20}},
        {"name": "mittelmann", "id": "mittelmann"},
    ],
    "sdp": [
        {"name": "sdplib", "id": "sdplib"},
        {"name": "mittelmann_sdp", "id": "mittelmann_sdp", "dataset_options": {"subset": "all"}},
    ],
    # Large LPs, 1e-4 only, 1800 s limit, 64 GB containers (see bench_modal --cpu-kind cpu_big).
    "lpbig": [
        {"name": "mittelmann", "id": f"mittelmann{k}", "include": chunk}
        for k, chunk in enumerate(MITTELMANN_CHUNKS)
    ],
}


def solver_variants(family: str, tol: float, timeout: float) -> list[dict]:
    """Solver settings at a common nominal tolerance ``tol``.

    Each solver's own tolerance semantics differ, so nothing here is treated
    as proof of accuracy: solves are re-verified afterwards with
    ``bench kkt-verify --tol`` on the harness's independent KKT residuals.
    """
    t = float(tol)
    tag = f"{tol:.0e}".replace("e-0", "e-")
    scs_common = {"eps_abs": t, "eps_rel": t, "max_iters": 1_000_000, "time_limit_secs": timeout}
    variants = [
        {"id": f"scs_cpu_{tag}", "solver": "scs", "settings": dict(scs_common), "gpu": False},
        {"id": f"scs_cudss_{tag}", "solver": "scs", "settings": {**scs_common, "linear_solver": "cudss"}, "gpu": True},
        {"id": f"clarabel_{tag}", "solver": "clarabel",
         "settings": {"tol_feas": t, "tol_gap_abs": t, "tol_gap_rel": t, "time_limit": timeout}, "gpu": False},
    ]
    if family == "lpbig":
        pass
    if family in ("qp", "lp"):
        variants += [
            {"id": f"osqp_{tag}", "solver": "osqp",
             "settings": {"eps_abs": t, "eps_rel": t, "max_iter": 1_000_000, "time_limit": timeout}, "gpu": False},
        ]
    if family in ("qp", "lp", "lpbig"):
        variants += [
            {"id": f"piqp_{tag}", "solver": "piqp",
             "settings": {"eps_abs": t, "eps_rel": t, "max_iter": 1_000_000, "time_limit": timeout}, "gpu": False},
            {"id": f"highs_{tag}", "solver": "highs",
             "settings": {"primal_feasibility_tolerance": t, "dual_feasibility_tolerance": t,
                          "time_limit": timeout}, "gpu": False},
        ]
    if family == "qp":
        # ProxQP is a QP method; on pure LPs it declares feasible problems
        # infeasible at these tolerances (netlib afiro, adlittle), so it is
        # compared on the QP family only.
        variants.append(
            {"id": f"proxqp_{tag}", "solver": "proxqp",
             "settings": {"eps_abs": t, "eps_rel": t, "max_iter": 1_000_000, "time_limit": timeout}, "gpu": False}
        )
    if family in ("qp", "lp", "lpbig"):
        # NVIDIA cuOpt: GPU PDLP for LP, GPU barrier for QP; the like-for-like GPU comparator.
        variants.append(
            {"id": f"cuopt_{tag}", "solver": "cuopt",
             "settings": {"eps": t, "time_limit": timeout}, "gpu": True, "runner": "cuopt"}
        )
    if family in ("qp", "lp", "lpbig"):
        # QTQP (pure-Python interior point): MKL Pardiso and cuDSS backends. Kept for
        # the record, not shown on the SCS site. No time-limit knob: the harness
        # worker timeout is the only stop.
        qtqp_common = {"atol": t, "rtol": t, "max_iter": 200, "verbose": False}
        variants += [
            {"id": f"qtqp_mkl_{tag}", "solver": "qtqp", "settings": {**qtqp_common, "linear_solver": "pardiso"},
             "gpu": False, "runner": "qtqp_cpu"},
            {"id": f"qtqp_cudss_{tag}", "solver": "qtqp", "settings": {**qtqp_common, "linear_solver": "cudss"},
             "gpu": True, "runner": "qtqp_gpu"},
        ]
    if family in ("lp", "lpbig"):
        variants.append(
            {"id": f"pdlp_{tag}", "solver": "pdlp",
             "settings": {"eps_abs": t, "eps_rel": t, "solver_time_limit_sec": timeout}, "gpu": False, "runner": "pdlp"}
        )
    if family == "sdp":
        variants += [
            {"id": f"cvxopt_{tag}", "solver": "cvxopt",
             "settings": {"abstol": t, "reltol": t, "feastol": t}, "gpu": False},
            {"id": f"sdpa_{tag}", "solver": "sdpa",
             "settings": {"eps_abs": t, "eps_rel": t, "time_limit": timeout}, "gpu": False},
        ]
    return variants


# Instances a solver cannot attempt at all, counted as failures for it. Clarabel
# forms a dense scaling block per PSD cone, which for a block of side >= 500 is
# hundreds of GiB; the process dies rather than reporting a failure, taking the
# rest of the shard with it.
LARGE_PSD_BLOCKS = [
    "equalG11", "equalG51", "maxG11", "maxG32", "maxG51", "maxG55", "maxG60",
    "thetaG11", "thetaG51", "gpp500-1", "gpp500-2", "gpp500-3", "gpp500-4",
    "mcp500-1", "mcp500-2", "mcp500-3", "mcp500-4", "qpG11", "qpG51",
]
SOLVER_DATASET_EXCLUDE = {
    ("clarabel", "sdplib"): LARGE_PSD_BLOCKS,
    ("clarabel", "mittelmann_sdp"): ["G40mc"],
}


def shard_config(family: str, dataset: dict, variant: dict, timeout: float) -> dict:
    solver = {k: v for k, v in variant.items() if k not in ("gpu", "runner")}
    excluded = SOLVER_DATASET_EXCLUDE.get((variant["solver"], dataset["id"]))
    if excluded:
        dataset = {**dataset, "exclude": list(excluded)}
    return {
        "run": {
            "name": f"{family}_{dataset['id']}_{variant['id']}",
            "output_dir": "results",
            "parallelism": 1,
            "resume": True,
            "timeout_seconds": timeout + 60.0,
            "datasets": [dataset],
        },
        "solvers": [solver],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--families", default="qp,lp,sdp")
    ap.add_argument("--tols", default="1e-4,1e-6")
    ap.add_argument("--only-solver", default=None, help="comma-separated substring filters on solver id")
    ap.add_argument("--only-dataset", default=None, help="substring filter on dataset id")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for family in args.families.split(","):
        timeout = {"sdp": SDP_TIMEOUT, "lpbig": LPBIG_TIMEOUT}.get(family, QP_LP_TIMEOUT)
        for dataset in FAMILIES[family]:
            if args.only_dataset and args.only_dataset not in dataset["id"]:
                continue
            for tol in (float(x) for x in args.tols.split(",")):
                for variant in solver_variants(family, tol, timeout):
                    if args.only_solver and not any(x in variant["id"] for x in args.only_solver.split(",")):
                        continue
                    cfg = shard_config(family, dataset, variant, timeout)
                    name = cfg["run"]["name"]
                    path = args.out_dir / f"{name}.yaml"
                    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    shards.append({"name": name, "family": family, "dataset": dataset["id"],
                                   "solver_id": variant["id"], "tol": tol, "gpu": variant["gpu"],
                                   "runner": variant.get("runner", "gpu" if variant["gpu"] else "cpu"),
                                   "config": str(path)})
    spec = args.out_dir / "campaign.json"
    spec.write_text(json.dumps(shards, indent=2))
    gpu = sum(1 for s in shards if s["gpu"])
    print(f"{len(shards)} shards ({gpu} GPU) written under {args.out_dir}; spec: {spec}")


if __name__ == "__main__":
    main()
