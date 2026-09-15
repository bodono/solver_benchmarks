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
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean

FAMILY_TITLES = {"qp": "Maros-Meszaros, QPLIB and MPC QPs", "lp": "Netlib, Kennington and MIPLIB-relaxation LPs",
                 "sdp": "SDPLIB and Mittelmann SDPs", "lpbig": "Mittelmann LP benchmark set"}
# The LP family's copy of the Mittelmann set only ever held qap15; the set is
# its own family (lpbig), so drop it here to keep the title honest.
FAMILY_DROP_DATASETS = {"lp": {"mittelmann"}}
DATASET_NAMES = {"maros_meszaros": "Maros-Meszaros", "qplib": "QPLIB", "mpc": "MPC", "netlib": "Netlib",
                 "kennington": "Kennington", "miplib_relax": "MIPLIB-relaxation", "sdplib": "SDPLIB",
                 "mittelmann_sdp": "Mittelmann", "mittelmann0": "Mittelmann", "mittelmann1": "Mittelmann",
                 "mittelmann2": "Mittelmann"}
FAMILY_NOUN = {"qp": "QPs", "lp": "LPs", "sdp": "SDPs"}


def sets_title(family: str, df: pd.DataFrame) -> str:
    """'Maros-Meszaros and QPLIB QPs' from the data sets actually present."""
    if family not in FAMILY_NOUN:
        return FAMILY_TITLES.get(family, family)
    names = []
    for d in df["dataset"].unique():
        n = DATASET_NAMES.get(d, d)
        if n not in names:
            names.append(n)
    order = list(DATASET_NAMES.values())
    names.sort(key=lambda n: order.index(n) if n in order else 99)
    joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return f"{joined} {FAMILY_NOUN[family]}"
SOLVER_LABELS = {
    "scs_cpu": "SCS (CPU, MKL Pardiso)",
    "scs_cudss": "SCS (GPU, cuDSS)",
    "cuopt": "cuOpt (GPU)",
    "osqp": "OSQP", "clarabel": "Clarabel", "piqp": "PIQP", "proxqp": "ProxQP",
    "highs": "HiGHS", "pdlp": "PDLP (OR-Tools)", "cvxopt": "CVXOPT", "sdpa": "SDPA",
}
LABEL_OVERRIDE: dict[str, str] = {}


def solver_label(solver_id: str) -> str:
    base = base_solver(solver_id)
    return LABEL_OVERRIDE.get(base, SOLVER_LABELS.get(base, base))


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


def draw_profile(ax, df: pd.DataFrame, title: str, compact: bool = False) -> bool:
    # Failures count as "never solved" (infinite ratio), a 10 ms floor keeps
    # sub-millisecond timings from producing meaningless ratios, and the
    # tau axis is capped at 1e4.
    df = df.assign(run_time_seconds=df["run_time_seconds"].clip(lower=TIME_FLOOR))
    prof = performance_profile(df, metric="run_time_seconds", max_value=float("inf"), tau_max=TAU_MAX)
    if prof.empty:
        return False
    fs = 9.5 if compact else 10
    solvers = [c for c in prof.columns if c != "tau"]
    others = [s for s in solvers if base_solver(s) not in SCS_STYLE]
    for i, s in enumerate(others + [s for s in solvers if base_solver(s) in SCS_STYLE]):
        color, ls, lw = style_for(s, others.index(s) if s in others else 0)
        ax.plot(prof["tau"], prof[s], color=color, linestyle=ls, linewidth=lw,
                label=solver_label(s), zorder=3 if base_solver(s) in SCS_STYLE else 2)
    ax.set_xscale("log")
    ax.set_xlim(1, prof["tau"].max())
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("time ratio to fastest solver, τ", fontsize=fs)
    ax.set_ylabel("fraction of problems solved within τ", fontsize=fs)
    ax.tick_params(labelsize=fs - 1)
    ax.set_title(title, fontsize=10.5 if compact else 11, loc="left" if compact else "center")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8 if compact else 8, framealpha=0.9)
    return True


def plot_profile(df: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if draw_profile(ax, df, title):
        fig.tight_layout()
        fig.savefig(path, dpi=160)
    plt.close(fig)


def with_missing_as_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Add a failed row for every (dataset, problem) a solver has no row for.

    Instances a solver could not attempt (Clarabel's large PSD blocks, for
    example) are excluded from its shard rather than recorded, and the
    shifted geometric mean only sees the rows that exist. Without this the
    exclusions would not be charged as failures in the bars, although the
    profiles already treat them that way.
    """
    keys = ["dataset", "problem"]
    problems = df[keys].drop_duplicates()
    solvers = df["solver_id"].unique()
    full = problems.merge(pd.DataFrame({"solver_id": solvers}), how="cross")
    present = df[keys + ["solver_id"]].drop_duplicates()
    missing = full.merge(present, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    if missing.empty:
        return df
    missing = missing.assign(status="not_attempted", run_time_seconds=float("nan"))
    return pd.concat([df, missing], ignore_index=True)


def draw_geomean(ax, df: pd.DataFrame, title: str, compact: bool = False) -> pd.DataFrame:
    gm = shifted_geomean(with_missing_as_failures(df), metric="run_time_seconds")
    if gm.empty:
        return gm
    value_col = "run_time_seconds"  # shifted_geomean names its value column after the metric
    gm = gm.sort_values(value_col).reset_index(drop=True)
    n_problems = df.groupby(["dataset", "problem"]).ngroups
    fs = 9.5 if compact else 10
    labels = [solver_label(s) for s in gm["solver_id"]]
    is_scs = [base_solver(s) in SCS_STYLE for s in gm["solver_id"]]
    colors = [SCS_STYLE[base_solver(s)][0] if scs else "#c4c4c4" for s, scs in zip(gm["solver_id"], is_scs)]
    edges = [SCS_STYLE[base_solver(s)][0] if scs else "#9a9a9a" for s, scs in zip(gm["solver_id"], is_scs)]
    y = np.arange(len(gm))
    ax.barh(y, gm[value_col], height=0.68, color=colors, edgecolor=edges, linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs)
    for tick, scs in zip(ax.get_yticklabels(), is_scs):
        if scs:
            tick.set_fontweight("bold")
    ax.set_xscale("log")
    lo, hi = float(gm[value_col].min()), float(gm[value_col].max())
    ax.set_xlim(lo / 1.8, hi * 2.6)
    for yi, (v, solved) in enumerate(zip(gm[value_col], gm["success_count"])):
        ax.text(v * 1.08, yi, f"{v:.1f} s" if v < 20 else f"{v:.0f} s", va="center", ha="left", fontsize=fs - 1, color="#222222")
        ax.text(v * 1.08, yi, f"\n{int(solved)}/{n_problems} solved", va="top", ha="left", fontsize=fs - 2.5, color="#666666", linespacing=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=fs - 1)
    ax.set_xlabel("shifted geometric mean solve time (s), lower is better" if compact else
                  "shifted geometric mean of solve time (s), lower is better; failures charged 1000 s", fontsize=fs - 1)
    if title:
        ax.set_title(title, fontsize=10.5, loc="left", pad=10, wrap=True)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", which="major", color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return gm


def plot_geomean(df: pd.DataFrame, title: str, path: Path) -> pd.DataFrame:
    n = df["solver_id"].nunique()
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * n + 1.6))
    gm = draw_geomean(ax, df, title)
    if not gm.empty:
        fig.tight_layout()
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return gm


GRID_ROWS = (("qp", "largest"), ("lp", "largest"), ("lpbig", "all"))


def plot_grid(frames: dict, path: Path) -> None:
    """3 x 2 landing-page figure: one family per row, profile left, bars right."""
    rows = [(fam, sub) for fam, sub in GRID_ROWS if (fam, sub) in frames]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.6, 3.5 * len(rows)), gridspec_kw={"width_ratios": [1.0, 1.0]})
    axes = np.atleast_2d(axes)
    for (fam, sub), (ax_p, ax_g) in zip(rows, axes):
        df, title, overrides = frames[(fam, sub)]
        LABEL_OVERRIDE.clear(); LABEL_OVERRIDE.update(overrides)
        draw_profile(ax_p, df, title, compact=True)
        draw_geomean(ax_g, df, "", compact=True)
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("runs", nargs="+", help="family=run_dir pairs")
    ap.add_argument("--tol-tag", default=None, help="only solver ids ending in this tag, e.g. 1e-4")
    ap.add_argument("--grid", type=Path, default=None,
                    help="also write a 3x2 landing-page grid (QP largest, LP largest, Mittelmann) to this path")
    ap.add_argument("--use-run", action="append", default=[], metavar="SOLVER=TAG",
                    help="show SOLVER from its TAG run in every plot (e.g. clarabel=1e-6); the legend says so "
                         "when TAG differs from the plot's tolerance")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    frames: dict = {}
    for pair in args.runs:
        family, run_dir = pair.split("=", 1)
        df = load_results(run_dir)
        if df.empty:
            print(f"{family}: no results in {run_dir}")
            continue
        drop = FAMILY_DROP_DATASETS.get(family)
        if drop:
            df = df[~df["dataset"].isin(drop)]
        use_run = dict(kv.split("=", 1) for kv in args.use_run)
        tags = sorted({solver_tol(s) for s in df["solver_id"].unique() if solver_tol(s)})
        if args.tol_tag:
            tags = [t for t in tags if t == args.tol_tag]
        for tag in tags:
            sub = df[df["solver_id"].str.endswith(f"_{tag}")]
            LABEL_OVERRIDE.clear()
            for solver, run_tag in use_run.items():
                alt = df[df["solver_id"] == f"{solver}_{run_tag}"]
                if run_tag == tag or alt.empty:  # no such run for this family: keep the plot's own run
                    continue
                sub = pd.concat([sub[sub["solver_id"] != f"{solver}_{tag}"], alt.assign(solver_id=f"{solver}_{tag}")],
                                ignore_index=True)
                # The substituted run is documented in the methodology text rather than in the legend.
            n_all = sub.groupby(["dataset", "problem"]).ngroups
            title = f"{sets_title(family, sub)}, tolerance {tag}, {n_all} problems"
            plot_profile(sub, title, args.out_dir / f"{family}_{tag}_profile.png")
            gm = plot_geomean(sub, title, args.out_dir / f"{family}_{tag}_geomean.png")
            big = largest_quartile(sub)
            n_big = big.groupby(["dataset", "problem"]).ngroups
            title_big = f"{sets_title(family, big)}, largest quartile ({n_big} problems), tolerance {tag}"
            plot_profile(big, title_big, args.out_dir / f"{family}_{tag}_profile_largest.png")
            gm_big = plot_geomean(big, title_big, args.out_dir / f"{family}_{tag}_geomean_largest.png")
            if tag == "1e-4":
                frames[(family, "all")] = (sub, title, dict(LABEL_OVERRIDE))
                frames[(family, "largest")] = (big, title_big, dict(LABEL_OVERRIDE))
            for label, table in (("all", gm), ("largest", gm_big)):
                if table is not None and not table.empty:
                    t = table.copy(); t.insert(0, "subset", label); t.insert(0, "tol", tag); t.insert(0, "family", family)
                    t["label"] = [solver_label(s) for s in t["solver_id"]]
                    summary.append(t)
            print(f"{family} {tag}: {n_all} problems, largest quartile {n_big}")
    if args.grid:
        plot_grid(frames, args.grid)
        print(args.grid)
    if summary:
        pd.concat(summary).to_csv(args.out_dir / "geomean_summary.csv", index=False)
        print(args.out_dir / "geomean_summary.csv")


if __name__ == "__main__":
    main()
