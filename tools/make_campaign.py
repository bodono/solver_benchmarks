"""Generate the sharded run configs for the SCS performance campaign.

One shard = one (dataset, solver variant, tolerance) so that every shard can
run in its own container with ``parallelism: 1`` (uncontended timings) and
the shards can fan out across as many machines as there are shards. The
output is a JSON spec consumed by tools/bench_modal.py; the same YAML
configs can be run locally with ``bench run``.

Usage: python tools/make_campaign.py OUT_DIR [--families qp,lp,sdp] [--tols 1e-4,1e-6]

Families: qp, lp, sdp, lpbig (the Mittelmann LP set) and infeas (the
infeasible NETLIB LPs, with the LP solvers). Solver ids end in the tolerance
tag (``clarabel_1e-4``, ``clarabel_5e-5``), which is how tools/make_figures.py
tells the tolerances apart.
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
    ],
    "sdp": [
        {"name": "sdplib", "id": "sdplib"},
        {"name": "mittelmann_sdp", "id": "mittelmann_sdp", "dataset_options": {"subset": "all"}},
    ],
    # Large LPs, 1800 s limit, 64 GB containers (see bench_modal --cpu-kind cpu_big). The
    # published campaign ran this family at 1e-4 only: generate it with --tols 1e-4.
    "lpbig": [
        {"name": "mittelmann", "id": f"mittelmann{k}", "include": chunk}
        for k, chunk in enumerate(MITTELMANN_CHUNKS)
    ],
    # Infeasible NETLIB LPs: success is a verified infeasibility certificate.
    "infeas": [
        {"name": "netlib", "id": "netlib_infeasible", "dataset_options": {"subset": "infeasible"}},
    ],
}


def tolerance_tag(tol: float) -> str:
    """The suffix of a solver id: ``1e-4`` for 1e-4, ``1.5e-5`` for 1.5e-5."""
    mantissa, exponent = f"{float(tol):e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


# The infeasibility (certificate) tolerances of each campaign solver, which
# solver_variants sets to the variant's tag (approved 2026-11-19).
# Verification holds a row's certificates to the same 10 x tag as its
# solutions (README, "The tolerance"), so a campaign asks every solver for
# certificates at the tag: the request and the check are at the same
# nominal tolerance, though each solver still measures certificates in its
# own units. For Clarabel, cpo3 and QTQP, tol_infeas_rel is the residual
# test and tol_infeas_abs the margin a certificate's objective term must
# clear (b'y < -tol_infeas_abs), stricter at the tag than its 1e-8 default.
# The knobs are set in every family. On a feasible problem a certificate is
# always a failure, so there a knob looser than the solver's default can
# only end solves early: at 1e-4, SCS (eps_infeas default 1e-7, no
# kappa/tau gate) solved 4 fewer of the 93 feasible NETLIB LPs and 7 fewer
# of the 138 Maros-Meszaros QPs (README, "The tolerance").
# An empty tuple is a solver with no such knob the adapter can set:
# - piqp 0.6.4: none; it declares infeasibility from eps_abs / eps_rel (set
#   to the tag) and infeasibility_threshold, a threshold on its proximal-point
#   residuals, not a certificate tolerance.
# - proxqp: ProxQP's eps_primal_inf / eps_dual_inf (default 1e-4) exist, but
#   the one-shot proxqp solve() the adapter calls does not take them.
# - highs: none; its infeasibility verdicts use its feasibility tolerances.
# - cvxopt: none of its own; feastol, set to the tag, is also its
#   certificate test.
# - sdpa: none.
# PDLP's are TerminationCriteria fields, set through parameters_text. cuOpt's
# apply to its PDLP (whose infeasibility detection is off by default).
CERTIFICATE_KNOBS: dict[str, tuple[str, ...]] = {
    "scs": ("eps_infeas",),
    "clarabel": ("tol_infeas_abs", "tol_infeas_rel"),
    "osqp": ("eps_prim_inf", "eps_dual_inf"),
    "cpo3": ("tol_infeas_abs", "tol_infeas_rel"),
    "qtqp": ("tol_infeas_abs", "tol_infeas_rel"),
    "cuopt": ("primal_infeasible_tolerance", "dual_infeasible_tolerance"),
    "pdlp": ("eps_primal_infeasible", "eps_dual_infeasible"),
    "piqp": (),
    "proxqp": (),
    "highs": (),
    "cvxopt": (),
    "sdpa": (),
}


def certificate_settings(solver: str, tol: float) -> dict:
    """The settings that ask ``solver`` for certificates at ``tol``
    (``CERTIFICATE_KNOBS``; a solver missing there is a KeyError, so a new
    campaign solver needs a decision)."""
    knobs = CERTIFICATE_KNOBS[solver]
    if solver == "pdlp":
        fields = " ".join(f"{knob}: {float(tol)!r}" for knob in knobs)
        return {"parameters_text": f"termination_criteria {{ {fields} }}"}
    return dict.fromkeys(knobs, float(tol))


def with_certificate_settings(variant: dict, tol: float) -> dict:
    """``variant`` with its solver's ``certificate_settings`` at ``tol``
    added to its settings. A variant that sets one of those keys itself (a
    PDLP variant's own ``parameters_text``) is refused rather than silently
    overwritten: merge the two by hand."""
    extra = certificate_settings(variant["solver"], tol)
    clash = sorted(set(extra) & set(variant["settings"]))
    if clash:
        raise ValueError(f"campaign variant {variant['id']} sets {', '.join(clash)} itself, which its "
                         "certificate settings would overwrite; merge them by hand")
    variant["settings"].update(extra)
    return variant


def solver_variants(family: str, tol: float, timeout: float) -> list[dict]:
    """Solver settings at a common nominal tolerance ``tol``.

    Each solver's own tolerance semantics differ, so nothing here is treated
    as proof of accuracy: every analysis of the results (``bench report``,
    tools/make_figures.py) verifies each solve against the harness's
    independent KKT residuals at 10 x the tolerance its settings request,
    which for every variant here except HiGHS below 1e-7 (its residual
    tolerances stay at their 1e-7 default) is ``tol``, for its solutions
    and its infeasibility certificates alike. Every variant also asks its
    solver for certificates at ``tol`` (``certificate_settings``, in every
    family; see ``CERTIFICATE_KNOBS`` for what that costs on feasible
    problems), so the certificates it asks for and the ones verification
    accepts are at the same nominal tolerance.
    """
    t = float(tol)
    tag = tolerance_tag(t)
    # The infeasible LPs are run with the LP family's solvers.
    if family == "infeas":
        family = "lp"
    scs_common = {"eps_abs": t, "eps_rel": t, "max_iters": 1_000_000, "time_limit_secs": timeout}
    variants = [
        {"id": f"scs_cpu_{tag}", "solver": "scs", "settings": dict(scs_common), "gpu": False},
        {"id": f"scs_cudss_{tag}", "solver": "scs", "settings": {**scs_common, "linear_solver": "cudss"}, "gpu": True},
        {"id": f"clarabel_{tag}", "solver": "clarabel",
         "settings": {"tol_feas": t, "tol_gap_abs": t, "tol_gap_rel": t, "time_limit": timeout}, "gpu": False},
    ]
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
        # QTQP 0.0.7 uses Clarabel-style termination settings.
        qtqp_common = {"tol_feas": t, "tol_gap_abs": t, "tol_gap_rel": t, "max_iter": 200, "verbose": False}
        variants += [
            {"id": f"qtqp_mkl_{tag}", "solver": "qtqp", "settings": {**qtqp_common, "linear_solver": "pardiso"},
             "gpu": False, "runner": "qtqp_cpu"},
            {"id": f"qtqp_cudss_{tag}", "solver": "qtqp", "settings": {**qtqp_common, "linear_solver": "cudss"},
             "gpu": True, "runner": "qtqp_gpu"},
        ]
    if family in ("qp", "lp", "lpbig"):
        # cpo3 (Rust interior point, Faer backend): Clarabel-style tolerances and
        # a native time limit; runs on its own image (runner cpo3_cpu).
        variants.append(
            {"id": f"cpo3_{tag}", "solver": "cpo3",
             "settings": {"tol_feas": t, "tol_gap_abs": t, "tol_gap_rel": t, "max_iters": 200,
                          "time_limit": timeout, "verbose": False},
             "gpu": False, "runner": "cpo3_cpu"}
        )
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
    return [with_certificate_settings(variant, t) for variant in variants]


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
