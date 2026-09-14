"""Figures for the SCS performance page.

Input: one KKT-verified run directory per family (the output of
``bench merge`` followed by ``bench kkt-verify --promote``). Output: for each
family and tolerance, a Dolan-Moré performance profile over all problems and
over the largest quartile by nonzeros, plus a shifted-geometric-mean bar
chart, with the SCS series drawn in front.

Usage:
  python tools/make_figures.py OUT_DIR qp=results/qp_verified_1e-4 lp=... [--tol-tag 1e-4]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean

FAMILY_TITLES = {"qp": "Quadratic programs", "lp": "Linear programs", "sdp": "Semidefinite programs",
                 "lpbig": "Mittelmann LP benchmark set"}
SOLVER_LABELS = {
    "scs_cpu": "SCS (CPU, MKL Pardiso)",
    "scs_cudss": "SCS (GPU, cuDSS)",
    "cuopt": "cuOpt (GPU)",
    "osqp": "OSQP", "clarabel": "Clarabel", "piqp": "PIQP", "proxqp": "ProxQP",
    "highs": "HiGHS", "pdlp": "PDLP (OR-Tools)", "cvxopt": "CVXOPT", "sdpa": "SDPA",
}
SCS_STYLE = {"scs_cpu": ("#1f77b4", "-", 2.8), "scs_cudss": ("#d62728", "-", 2.8)}
OTHER_COLORS = ["#7f7f7f", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#ff7f0e"]


def base_solver(solver_id: str) -> str:
    return re.sub(r"_1e-\d+$", "", solver_id)


def solver_tol(solver_id: str) -> str | None:
    m = re.search(r"_(1e-\d+)$", solver_id)
    return m.group(1) if m else None


def problem_size(df: pd.DataFrame) -> pd.Series:
    def col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[name], errors="coerce").fillna(0)

    return col("metadata.nnz_a") + col("metadata.nnz_p")


def largest_quartile(df: pd.DataFrame) -> pd.DataFrame:
    sizes = df.assign(__size=problem_size(df)).groupby(["dataset", "problem"])["__size"].max()
    cutoff = sizes.quantile(0.75)
    keep = sizes[sizes >= cutoff].index
    idx = pd.MultiIndex.from_frame(df[["dataset", "problem"]])
    return df[idx.isin(keep)].copy()


def style_for(solver_id: str, i: int):
    base = base_solver(solver_id)
    if base in SCS_STYLE:
        return SCS_STYLE[base]
    return OTHER_COLORS[i % len(OTHER_COLORS)], "--", 1.6


TIME_FLOOR = 0.01
TAU_MAX = 1.0e4


def plot_profile(df: pd.DataFrame, title: str, path: Path) -> None:
    # Failures count as "never solved" (infinite ratio), a 10 ms floor keeps
    # sub-millisecond timings from producing meaningless ratios, and the
    # tau axis is capped at 1e4.
    df = df.assign(run_time_seconds=df["run_time_seconds"].clip(lower=TIME_FLOOR))
    prof = performance_profile(df, metric="run_time_seconds", max_value=float("inf"), tau_max=TAU_MAX)
    if prof.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    solvers = [c for c in prof.columns if c != "tau"]
    others = [s for s in solvers if base_solver(s) not in SCS_STYLE]
    for i, s in enumerate(others + [s for s in solvers if base_solver(s) in SCS_STYLE]):
        color, ls, lw = style_for(s, others.index(s) if s in others else 0)
        ax.plot(prof["tau"], prof[s], color=color, linestyle=ls, linewidth=lw,
                label=SOLVER_LABELS.get(base_solver(s), base_solver(s)), zorder=3 if base_solver(s) in SCS_STYLE else 2)
    ax.set_xscale("log")
    ax.set_xlim(1, prof["tau"].max())
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time ratio to fastest solver, τ")
    ax.set_ylabel("fraction of problems solved within τ")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_geomean(df: pd.DataFrame, title: str, path: Path) -> pd.DataFrame:
    gm = shifted_geomean(df, metric="run_time_seconds")
    if gm.empty:
        return gm
    value_col = "run_time_seconds"  # shifted_geomean names its value column after the metric
    gm = gm.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = [SCS_STYLE[base_solver(s)][0] if base_solver(s) in SCS_STYLE else "#9e9e9e" for s in gm["solver_id"]]
    ax.barh([SOLVER_LABELS.get(base_solver(s), base_solver(s)) for s in gm["solver_id"]], gm[value_col], color=colors)
    ax.set_xscale("log")
    ax.set_xlim(left=float(gm[value_col].min()) / 2.0)
    ax.set_xlabel("shifted geometric mean of solve time (s), failures penalised")
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return gm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("runs", nargs="+", help="family=run_dir pairs")
    ap.add_argument("--tol-tag", default=None, help="only solver ids ending in this tag, e.g. 1e-4")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for pair in args.runs:
        family, run_dir = pair.split("=", 1)
        df = load_results(run_dir)
        if df.empty:
            print(f"{family}: no results in {run_dir}")
            continue
        tags = sorted({solver_tol(s) for s in df["solver_id"].unique() if solver_tol(s)})
        if args.tol_tag:
            tags = [t for t in tags if t == args.tol_tag]
        for tag in tags:
            sub = df[df["solver_id"].str.endswith(f"_{tag}")]
            n_all = sub.groupby(["dataset", "problem"]).ngroups
            title = f"{FAMILY_TITLES.get(family, family)}, tolerance {tag}, {n_all} problems"
            plot_profile(sub, title, args.out_dir / f"{family}_{tag}_profile.png")
            gm = plot_geomean(sub, title, args.out_dir / f"{family}_{tag}_geomean.png")
            big = largest_quartile(sub)
            n_big = big.groupby(["dataset", "problem"]).ngroups
            title_big = f"{FAMILY_TITLES.get(family, family)}, largest quartile ({n_big} problems), tolerance {tag}"
            plot_profile(big, title_big, args.out_dir / f"{family}_{tag}_profile_largest.png")
            gm_big = plot_geomean(big, title_big, args.out_dir / f"{family}_{tag}_geomean_largest.png")
            for label, table in (("all", gm), ("largest", gm_big)):
                if table is not None and not table.empty:
                    t = table.copy(); t.insert(0, "subset", label); t.insert(0, "tol", tag); t.insert(0, "family", family)
                    summary.append(t)
            print(f"{family} {tag}: {n_all} problems, largest quartile {n_big}")
    if summary:
        pd.concat(summary).to_csv(args.out_dir / "geomean_summary.csv", index=False)
        print(args.out_dir / "geomean_summary.csv")


if __name__ == "__main__":
    main()
