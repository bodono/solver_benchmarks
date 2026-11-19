"""Figures for the SCS performance page.

Input: one run directory per family, typically the output of ``bench merge``.
The family's rows (after its dataset, problem and solver filters, and
without solver ids that carry no tolerance tag) are KKT-verified when they
are loaded, as every analysis command verifies them: each claimed solution
and certificate is checked against the harness's own KKT residuals at 10 x
the tolerance its solver was asked for, claims with nothing to check count as
unsolved, and inaccurate or limit stops that pass are promoted (a run written
by ``bench kkt-verify`` is used as it recorded its verification). The 1 %
late rule (``mark_late``) is applied after verification, so a row returned
after the limit counts as a failure whatever verification made of it.
``--unverified``, ``--verify-tol``, ``--reverify``, ``--allow-missing``,
``--requested-tol`` and ``--tol-from-solver-id`` choose the verification as
for ``bench report``. A series ``--use-run`` takes from another tolerance's
run keeps that run's verification (10 x what it requested), which stdout
notes and ``geomean_summary.csv``'s ``verify_tol`` records.
Output: for each family and tolerance, a Dolan-Moré performance profile over
all problems and over the largest quartile by nonzeros, plus a
shifted-geometric-mean bar chart, with the SCS series drawn in front;
``verification_summary.csv`` lists, per family and solver, the tolerances
its rows were checked at and what verification changed, after the late rule
(``late`` counts the claims it kept or promoted that the late rule made
failures; ``kept`` and ``promoted`` no longer include them).

Usage:
  python tools/make_figures.py OUT_DIR qp=results/qp_merged lp=... [--tol-tag 1e-4]

A problem that no result row sized (every solver timed out, crashed or
raised on it) is loaded from its dataset to size it for the largest-quartile
figures. For the lpbig family (the Mittelmann LP set) that can mean parsing
several very large MPS files, which can be slow and need a lot of memory; pass
--no-size-lookup to skip it, in which case such problems are left out of the
largest quartile and named on stderr.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from solver_benchmarks.analysis.load import load_results
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean
from solver_benchmarks.analysis.verify import (
    VerifyInputError,
    VerifyPolicy,
    parse_requested_tol,
    verification_header,
    verification_info,
    verification_table,
)
from solver_benchmarks.core.config import manifest_dataset_entries
from solver_benchmarks.core.problem import QP, cone_dimensions, qp_dimensions
from solver_benchmarks.datasets import get_dataset
from solver_benchmarks.datasets.sdplib import SDPLIB_INFEASIBLE

FAMILY_TITLES = {"qp": "Maros-Meszaros, QPLIB and MPC QPs", "lp": "Netlib, Kennington and MIPLIB-relaxation LPs",
                 "sdp": "SDPLIB and Mittelmann SDPs", "lpbig": "Mittelmann LP benchmark set", "infeas": "Netlib infeasible LPs"}
# The Mittelmann LP set is its own family (lpbig); LP-family runs made when
# make_campaign still gave that family a Mittelmann shard carry its rows, so
# drop them here to keep the title honest.
FAMILY_DROP_DATASETS = {"lp": {"mittelmann"}, "qp": {"mpc"}}
# Instances staged from an older MIPLIB listing that are not in the 240-instance
# MIPLIB 2017 benchmark set; dropped so the LP family is exactly that set.
FAMILY_DROP_PROBLEMS = {"lp": {("miplib_relax", n) for n in
                        ("n9-3", "neos-3754224-navua", "neos-5075914-elvire", "rococoC11-011100", "toll-like")},
                        # SDPLIB's infeasible instances, recorded under the plain "sdplib"
                        # id by runs made before the dataset had subsets (they are the
                        # "infeasible" subset now); the plots measure time to a verified
                        # optimum, and every solver reports these infeasible.
                        "sdp": {("sdplib", n) for n in SDPLIB_INFEASIBLE}}
# Nominal time limit per family (make_campaign gives it to every solver that
# takes one). It applies to run_time_seconds, the wall time of the whole solve
# including setup. The SCS adapter gives SCS's solve the limit minus its setup
# time, since SCS's own limit covers only its iterations; runs made with an SCS
# whose update() cannot change settings, and runs before that change, gave the
# whole limit to the solve. A solve that takes longer than the limit plus
# LATE_GRACE counts as a failure for every solver, whatever status it
# reported: solvers that ignore their time limit (PIQP, ProxQP, SDPA) or have
# none (QTQP, CVXOPT) run on until the harness kills them at the limit plus
# make_campaign's 60 s margin, and must not be scored on time the
# self-limiting solvers were not given.
FAMILY_LIMIT = {"qp": 300.0, "lp": 300.0, "sdp": 900.0, "lpbig": 1800.0, "infeas": 300.0}
# A solve is late when run_time_seconds > limit * (1 + LATE_GRACE): 3 s for the
# 300 s families, 9 s for 900 s, 18 s for 1800 s. A solver that stops itself
# at its limit reports time_limit and fails anyway, so the grace does not
# rescue limit stops as such. What it admits is a row that ends inside the
# window with a solved status: a solve that converges on the iteration that
# crosses the limit (solvers check the clock only between iterations), or a
# limit stop whose point verification promotes, e.g. an SCS stop at the
# limit plus one iteration. The tradeoff: a solver that ignores its
# limit is credited the same way, with up to 1 % more time than the limit;
# the grace is kept that small so this cannot buy a solver much.
LATE_GRACE = 0.01
# Failures are charged this multiple of the family's time limit in the shifted
# geometric mean, so a failure always costs more than any admitted success.
FAILURE_PENALTY_FACTOR = 3.0
DATASET_NAMES = {"maros_meszaros": "Maros-Meszaros", "qplib": "QPLIB", "mpc": "MPC", "netlib": "Netlib",
                 "kennington": "Kennington", "miplib_relax": "MIPLIB-relaxation", "sdplib": "SDPLIB",
                 "mittelmann_sdp": "Mittelmann", "mittelmann0": "Mittelmann", "mittelmann1": "Mittelmann",
                 "mittelmann2": "Mittelmann"}
FAMILY_NOUN = {"qp": "QPs", "lp": "LPs", "sdp": "SDPs"}


def sets_title(family: str, df: pd.DataFrame) -> str:
    """'Maros-Meszaros and QPLIB QPs' from the data sets actually present."""
    if family not in FAMILY_NOUN or df.empty:
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
    "qtqp_mkl": "QTQP (CPU, MKL Pardiso)", "qtqp_cudss": "QTQP (GPU, cuDSS)",
    "scs_cpu": "SCS (CPU, MKL Pardiso)",
    "scs_cudss": "SCS (GPU, cuDSS)",
    "cuopt": "cuOpt (GPU)",
    "osqp": "OSQP", "clarabel": "Clarabel", "piqp": "PIQP", "proxqp": "ProxQP",
    "highs": "HiGHS", "pdlp": "PDLP (OR-Tools)", "cvxopt": "CVXOPT", "sdpa": "SDPA",
    "cpo3": "cpo3",
}
LABEL_OVERRIDE: dict[str, str] = {}


def solver_label(solver_id: str) -> str:
    base = base_solver(solver_id)
    return LABEL_OVERRIDE.get(base, SOLVER_LABELS.get(base, base))


SCS_STYLE = {"scs_cpu": ("#1f77b4", "-", 2.8), "scs_cudss": ("#d62728", "-", 2.8)}
OTHER_COLORS = ["#7f7f7f", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#ff7f0e"]
# One fixed colour per solver, the same in every plot (lines in both modes; bars in
# --neutral mode, where no solver is highlighted).
NEUTRAL = False
SOLVER_COLORS = {"scs_cpu": "#1f77b4", "scs_cudss": "#d62728", "clarabel": "#7f7f7f", "piqp": "#e377c2",
                 "osqp": "#8c564b", "proxqp": "#bcbd22", "highs": "#9467bd", "pdlp": "#17becf",
                 "cuopt": "#2ca02c", "cvxopt": "#8c6d31", "sdpa": "#a55194", "qtqp_mkl": "#393b79",
                 "qtqp_cudss": "#ff7f0e", "cpo3": "#637939"}
NEUTRAL_COLORS = SOLVER_COLORS


# The tolerance tag that ends a campaign solver id (make_campaign's
# tolerance_tag): clarabel_1e-4, clarabel_5e-5, clarabel_1.5e-5.
TOL_TAG = r"\d(?:\.\d+)?e-\d+"


def base_solver(solver_id: str) -> str:
    return re.sub(rf"_{TOL_TAG}$", "", solver_id)


def solver_tol(solver_id: str) -> str | None:
    m = re.search(rf"_({TOL_TAG})$", solver_id)
    return m.group(1) if m else None


PROBLEM_KEYS = ["dataset", "problem"]
SizeLookup = Callable[[str, str], "float | None"]


def problem_sizes(df: pd.DataFrame, size_lookup: SizeLookup | None = None) -> pd.Series:
    """nnz(A) + nnz(P) of every (dataset, problem) in ``df``, NaN where unknown.

    Size is a property of the problem, not of a solve. When the solver
    returns, the worker records the problem's dimensions (``metadata.nnz_a`` /
    ``metadata.nnz_p``) on its row, whatever status the solver reported, so
    any such row sizes the problem. Three kinds of row carry no dimensions:
    the ``worker_error`` row the worker writes when loading the problem or the
    solver raised, and the rows the runner writes itself for a solve killed at
    the harness timeout or a worker that crashed. When every solver's row for
    a problem is one of those, typically on the largest instances,
    ``size_lookup(dataset, problem)`` is asked instead (see
    :func:`registry_size_lookup`).
    """
    def col(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[name], errors="coerce")

    nnz_a = col("metadata.nnz_a")
    # Cone problems have no P; a row is sized exactly when it recorded nnz(A).
    row_size = (nnz_a + col("metadata.nnz_p").fillna(0)).where(nnz_a.notna())
    sizes = df.assign(__size=row_size).groupby(PROBLEM_KEYS)["__size"].max()
    if size_lookup is not None:
        for key in sizes.index[sizes.isna()]:
            size = size_lookup(*key)
            if size is not None:
                sizes.loc[key] = float(size)
    return sizes


def registry_size_lookup(run_dir: str | Path, repo_root: str | Path | None = None) -> SizeLookup:
    """A ``size_lookup`` that loads the problem through the dataset registry.

    The dataset id in a result row is the config entry id; the registry name
    and dataset options behind it come from ``RUN_DIR/manifest.json`` (a
    merged run's manifest lists every shard's entries). Returns None for a
    dataset the manifest does not list or a problem that cannot be loaded.
    """
    path = Path(run_dir) / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    entries = {entry["id"]: entry for entry in manifest_dataset_entries(manifest.get("config") or {})}
    datasets: dict[str, Any] = {}

    def lookup(dataset: str, problem: str) -> float | None:
        entry = entries.get(dataset)
        if entry is None:
            return None
        try:
            if dataset not in datasets:
                # Options the run recorded: read as the run did, unchecked.
                dataset_cls = get_dataset(entry["name"])
                from_recorded = getattr(dataset_cls, "from_recorded_options", None)
                datasets[dataset] = (
                    from_recorded(repo_root=repo_root, options=entry["dataset_options"])
                    if from_recorded is not None
                    else dataset_cls(repo_root=repo_root, **entry["dataset_options"])
                )
            data = datasets[dataset].load_problem(problem)
            dims = qp_dimensions(data.qp) if data.kind == QP else cone_dimensions(data.cone)
        except Exception as exc:  # a size is optional: report it and go on
            print(f"warning: could not load {dataset}/{problem} to size it: {exc}", file=sys.stderr)
            return None
        return float(dims.get("nnz_a", 0) + dims.get("nnz_p", 0))

    return lookup


def largest_quartile(df: pd.DataFrame, sizes: pd.Series | None = None) -> pd.DataFrame:
    """Rows of the problems whose size is in the top quartile of ``df``'s problems.

    ``sizes`` is indexed by (dataset, problem), as from :func:`problem_sizes`
    (computed from ``df`` alone when omitted). A problem of unknown size is
    left out and named on stderr: counting it as size 0 would both drop it and
    pull the cutoff down.
    """
    if sizes is None:
        sizes = problem_sizes(df)
    present = pd.MultiIndex.from_frame(df[PROBLEM_KEYS].drop_duplicates())
    sizes = sizes.reindex(present)
    unknown = sizes.index[sizes.isna()]
    if len(unknown):
        names = ", ".join(f"{d}/{p}" for d, p in unknown)
        print(f"warning: size unknown for {len(unknown)} problems, left out of the largest quartile: {names}",
              file=sys.stderr)
    known = sizes.dropna()
    if known.empty:
        return df.iloc[0:0].copy()
    keep = known[known >= known.quantile(0.75)].index
    idx = pd.MultiIndex.from_frame(df[PROBLEM_KEYS])
    return df[idx.isin(keep)].copy()


def style_for(solver_id: str, i: int):
    base = base_solver(solver_id)
    color = SOLVER_COLORS.get(base, OTHER_COLORS[i % len(OTHER_COLORS)])
    if NEUTRAL:
        return color, "-", 2.0
    if base in SCS_STYLE:
        return SCS_STYLE[base]
    return color, "--", 1.6


TIME_FLOOR = 0.01
TAU_MAX = 1.0e4


def full_denominator_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Dolan-More profile whose denominator is every problem in ``df``.

    This is the harness's ``performance_profile``, so a solve counts exactly
    when ``solver_benchmarks.analysis.outcomes.success_mask`` says it does: an
    optimal status where a solution is expected, and on problems known to be
    infeasible or unbounded only the matching certificate (one the harness did
    not find invalid). Failures and solves a solver has no row for have an
    infinite ratio; problems every solver failed stay in the denominator, so
    the y axis is the fraction of all problems, matching the solved counts
    next to the bars. Times are floored at ``TIME_FLOOR`` and tau runs from 1
    to ``TAU_MAX``.
    """
    return performance_profile(df, metric="run_time_seconds", min_value=TIME_FLOOR, tau_max=TAU_MAX, n_tau=1000)


def draw_profile(ax, df: pd.DataFrame, title: str, compact: bool = False) -> bool:
    # Failures count as "never solved" (infinite ratio), a 10 ms floor keeps
    # sub-millisecond timings from producing meaningless ratios, and the
    # tau axis is capped at 1e4.
    df = df.assign(run_time_seconds=df["run_time_seconds"].clip(lower=TIME_FLOOR))
    prof = full_denominator_profile(df)
    if prof.empty:
        return False
    fs = 9.5 if compact else 10
    solvers = [c for c in prof.columns if c != "tau"]
    others = [s for s in solvers if base_solver(s) not in SCS_STYLE]
    for _i, s in enumerate(others + [s for s in solvers if base_solver(s) in SCS_STYLE]):
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


def late_rows(df: pd.DataFrame, limit: float) -> pd.Series:
    """Rows slower than the limit plus LATE_GRACE."""
    return df["run_time_seconds"] > limit * (1.0 + LATE_GRACE)


def mark_late(df: pd.DataFrame, limit: float) -> pd.DataFrame:
    """A copy of ``df`` with every row slower than the limit plus LATE_GRACE
    recorded as time_limit, whatever status it reported."""
    late = late_rows(df, limit)
    df = df.copy()
    df.loc[late, "status"] = "time_limit"
    return df


def family_rows(df: pd.DataFrame, family: str, exclude_solvers: Iterable[str] = ()) -> pd.DataFrame:
    """The rows a family's figures use: without its dropped datasets and
    problems (FAMILY_DROP_DATASETS, FAMILY_DROP_PROBLEMS) and the excluded
    solvers (base ids)."""
    drop = FAMILY_DROP_DATASETS.get(family)
    if drop:
        df = df[~df["dataset"].isin(drop)]
    exclude = set(exclude_solvers)
    if exclude:
        df = df[~df["solver_id"].astype(str).apply(base_solver).isin(exclude)]
    drop_p = FAMILY_DROP_PROBLEMS.get(family)
    if drop_p:
        df = df[~pd.Series(list(zip(df["dataset"], df["problem"], strict=True)), index=df.index).isin(drop_p)]
    return df


def verification_after_late(df: pd.DataFrame, late: pd.Series) -> pd.DataFrame:
    """``verification_table`` of a family's verified rows, with the claims
    verification kept or promoted that the late rule then made failures
    taken out of ``kept``, ``promoted`` and ``unverified`` and counted as
    ``late``."""
    table = verification_table(df)
    if table.empty:
        return table
    table["late"] = 0
    if late.any():
        counts = verification_table(df[late]).set_index("solver_id")
        index = table["solver_id"]
        for column in ("kept", "promoted", "unverified"):
            taken = index.map(counts[column]).fillna(0).astype(int)
            table[column] = table[column] - taken
            if column != "unverified":
                table["late"] = table["late"] + taken
    return table


def scs_time_limit_warnings(df: pd.DataFrame) -> list[str]:
    """Warnings about how SCS rows were given their time limit (the SCS
    adapter's info.scs_time_limit_mode): rows whose update() failed, and any
    solver id with rows not given the limit minus setup, which would compare
    runs given different amounts of time. Rows without the field (other solvers, older runs, no
    limit) are ignored."""
    mode_col = "info.scs_time_limit_mode"
    if mode_col not in df.columns:
        return []
    warnings = []
    error_col = "info.scs_time_limit_update_error"
    if error_col in df.columns:
        errors = df[df[error_col].notna()]
        for solver_id, rows in errors.groupby("solver_id"):
            warnings.append(f"{solver_id}: SCS update() failed on {len(rows)} row(s), which were "
                            f"rebuilt with the limit at construction (first: {rows[error_col].iloc[0]})")
    # Any SCS row not given limit minus setup: within one solver id that
    # mixes runs given different amounts of time, and across ids (two SCS
    # builds compared) one build may have had its setup on top of the limit.
    for solver_id, modes in df[df[mode_col].notna()].groupby("solver_id")[mode_col]:
        counts = modes.value_counts()
        other = {mode: n for mode, n in counts.items() if mode != "solve_minus_setup"}
        if other:
            mix = ", ".join(f"{mode} {n}" for mode, n in counts.items())
            warnings.append(f"{solver_id}: SCS rows not all given the limit minus setup ({mix})")
    return warnings


def failure_penalty(family: str | None) -> float:
    return FAILURE_PENALTY_FACTOR * FAMILY_LIMIT.get(family or "", 300.0)


def draw_geomean(ax, df: pd.DataFrame, title: str, compact: bool = False, family: str | None = None) -> pd.DataFrame:
    # Instances a solver could not attempt (Clarabel's large PSD blocks, for
    # example) are excluded from its shard rather than recorded;
    # shifted_geomean charges every (problem, solver) pair without a row as a
    # failure, as the profiles do.
    gm = shifted_geomean(df, metric="run_time_seconds", max_value=failure_penalty(family))
    if gm.empty:
        return gm
    value_col = "run_time_seconds"  # shifted_geomean names its value column after the metric
    gm = gm.sort_values(value_col).reset_index(drop=True)
    n_problems = df.groupby(["dataset", "problem"]).ngroups
    fs = 9.5 if compact else 10
    labels = [solver_label(s) for s in gm["solver_id"]]
    is_scs = [base_solver(s) in SCS_STYLE and not NEUTRAL for s in gm["solver_id"]]
    if NEUTRAL:
        colors = [NEUTRAL_COLORS.get(base_solver(s), "#c4c4c4") for s in gm["solver_id"]]
        edges = colors
    else:
        colors = [SCS_STYLE[base_solver(s)][0] if scs else "#c4c4c4" for s, scs in zip(gm["solver_id"], is_scs, strict=True)]
        edges = [SCS_STYLE[base_solver(s)][0] if scs else "#9a9a9a" for s, scs in zip(gm["solver_id"], is_scs, strict=True)]
    y = np.arange(len(gm))
    ax.barh(y, gm[value_col], height=0.68, color=colors, edgecolor=edges, linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs)
    for tick, scs in zip(ax.get_yticklabels(), is_scs, strict=False):
        if scs:
            tick.set_fontweight("bold")
    hi = float(gm[value_col].max())
    ax.set_xlim(0, hi * 1.28)
    pad = hi * 0.012
    for yi, (v, solved) in enumerate(zip(gm[value_col], gm["success_count"], strict=True)):
        ax.text(v + pad, yi, f"{v:.1f} s" if v < 20 else f"{v:.0f} s", va="center", ha="left", fontsize=fs - 1, color="#222222")
        ax.text(v + pad, yi, f"\n{int(solved)}/{n_problems} solved", va="top", ha="left", fontsize=fs - 2.5, color="#666666", linespacing=0.6)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="x", labelsize=fs - 1)
    pen = failure_penalty(family)
    ax.set_xlabel("shifted geometric mean solve time (s), lower is better" if compact else
                  f"shifted geometric mean of solve time (s), lower is better; failures charged {pen:.0f} s", fontsize=fs - 1)
    if title:
        ax.set_title(title, fontsize=10.5, loc="left", pad=10, wrap=True)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", which="major", color="#e0e0e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return gm


def plot_geomean(df: pd.DataFrame, title: str, path: Path, family: str | None = None) -> pd.DataFrame:
    n = df["solver_id"].nunique()
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * n + 1.6))
    gm = draw_geomean(ax, df, title, family=family)
    if not gm.empty:
        fig.tight_layout()
        fig.savefig(path, dpi=220)
    plt.close(fig)
    return gm


GRID_ROWS = (("qp", "largest"), ("lp", "largest"), ("lpbig", "all"))


def plot_pair(df: pd.DataFrame, title: str, path: Path, family: str) -> None:
    """One row of the landing grid on its own: profile left, geomean bars right."""
    fig, (ax_p, ax_g) = plt.subplots(1, 2, figsize=(11.0, 3.6), gridspec_kw={"width_ratios": [1.0, 1.05]})
    if not draw_profile(ax_p, df, title, compact=True):
        plt.close(fig)
        return
    draw_geomean(ax_g, df, "", compact=True, family=family)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_grid(frames: dict, path: Path) -> None:
    """3 x 2 landing-page figure: one family per row, profile left, bars right."""
    rows = [(fam, sub) for fam, sub in GRID_ROWS if (fam, sub) in frames]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.6, 3.5 * len(rows)), gridspec_kw={"width_ratios": [1.0, 1.0]})
    axes = np.atleast_2d(axes)
    for (fam, sub), (ax_p, ax_g) in zip(rows, axes, strict=True):
        df, title, overrides = frames[(fam, sub)]
        LABEL_OVERRIDE.clear()
        LABEL_OVERRIDE.update(overrides)
        draw_profile(ax_p, df, title, compact=True)
        draw_geomean(ax_g, df, "", compact=True, family=fam)
    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def verification_policy(args: argparse.Namespace) -> bool | VerifyPolicy:
    """``load_results``'s ``verify`` for the verification options (as the CLI's)."""
    others = (args.verify_tol is not None or args.reverify or args.allow_missing or args.requested_tol
              or args.tol_from_solver_id)
    if args.unverified and others:
        sys.exit("--unverified cannot be combined with --verify-tol, --reverify, --allow-missing, "
                 "--requested-tol or --tol-from-solver-id")
    if args.verify_tol is not None and args.reverify:
        sys.exit("--verify-tol and --reverify cannot be combined (--verify-tol always re-checks)")
    if args.verify_tol is not None and (args.requested_tol or args.tol_from_solver_id):
        sys.exit("--verify-tol holds every row to one tolerance; --requested-tol and --tol-from-solver-id "
                 "would not be used")
    if args.unverified:
        return False
    try:
        if args.verify_tol is not None:
            return VerifyPolicy(rule="fixed", tol=args.verify_tol, missing_is_failure=not args.allow_missing)
        return VerifyPolicy(rule="requested" if args.reverify else "auto",
                            missing_is_failure=not args.allow_missing,
                            requested_tol=parse_requested_tol(args.requested_tol),
                            tol_from_solver_id=args.tol_from_solver_id)
    except VerifyInputError as exc:
        sys.exit(str(exc))


def verify_rule(df: pd.DataFrame) -> str:
    """How a family's statuses were decided, for geomean_summary.csv."""
    info = verification_info(df)
    mode = info.get("mode")
    if mode == "unverified":
        return "unverified"
    rule = str(info.get("rule") or "requested")
    if rule == "fixed":
        rule = f"fixed {float(info.get('tol')):g}" if info.get("tol") is not None else "fixed"
    else:
        rule = f"{float(info.get('slack') or 10.0):g} x requested"
    return f"recorded ({rule})" if mode == "recorded" else rule


def verify_tols(df: pd.DataFrame) -> dict[str, str]:
    """The solution tolerance(s) each solver id's rows were checked at."""
    if "verify.tol" not in df.columns:
        return {}
    out = {}
    for solver_id, values in df.groupby("solver_id")["verify.tol"]:
        tols = sorted({float(v) for v in values.dropna()})
        out[str(solver_id)] = ", ".join(f"{v:g}" for v in tols)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("runs", nargs="+", help="family=run_dir pairs")
    ap.add_argument("--tol-tag", default=None, help="only solver ids ending in this tag, e.g. 1e-4")
    ap.add_argument("--neutral", action="store_true", help="no highlighted solver; one colour per solver for lines and bars")
    ap.add_argument("--exclude-solver", action="append", default=[], metavar="BASE",
                    help="drop this solver (base id, e.g. qtqp_mkl) from every plot and table")
    ap.add_argument("--grid", type=Path, default=None,
                    help="also write a 3x2 landing-page grid (QP largest, LP largest, Mittelmann) to this path")
    ap.add_argument("--use-run", action="append", default=[], metavar="SOLVER=TAG",
                    help="show SOLVER from its TAG run in every plot (e.g. clarabel=1e-6); the legend and "
                         "bar labels say so (\"Clarabel (1e-6)\") when TAG differs from the plot's tolerance, "
                         "and geomean_summary.csv records the run used in its source_run column")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="repository whose data directories hold the problems, for sizing problems no "
                         "result row recorded dimensions for (default: this checkout)")
    ap.add_argument("--no-size-lookup", action="store_true",
                    help="do not load problems that no result row sized; leave them out of the largest "
                         "quartile instead (loading the largest Mittelmann LPs is slow and memory-hungry)")
    ap.add_argument("--unverified", action="store_true",
                    help="use the statuses the solvers reported instead of the KKT-verified ones")
    ap.add_argument("--verify-tol", type=float, default=None,
                    help="hold every row to this one tolerance instead of 10 x its solver's requested one "
                         "(1e-3 for the 1e-4 figures reproduces the 3.3.2 docs rule for every solver)")
    ap.add_argument("--reverify", action="store_true",
                    help="verify a bench kkt-verify copy again by the default rule instead of as it recorded")
    ap.add_argument("--allow-missing", action="store_true",
                    help="keep claims that recorded no residuals or no certificate to check (default: they "
                         "count as unsolved, since they cannot be verified)")
    ap.add_argument("--requested-tol", action="append", default=[], metavar="SOLVER_ID=TOL",
                    help="the tolerance SOLVER_ID was asked for, ahead of what the run records; repeatable")
    ap.add_argument("--tol-from-solver-id", action="store_true",
                    help="take a solver's requested tolerance from the tag ending its id when the run gives none")
    args = ap.parse_args()
    policy = verification_policy(args)
    global NEUTRAL
    NEUTRAL = bool(args.neutral)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    verification = []
    frames: dict = {}
    failures: list[str] = []
    for pair in args.runs:
        family, run_dir = pair.split("=", 1)
        found: dict[str, list[str]] = {}

        def select(rows: pd.DataFrame, family=family, found=found) -> pd.DataFrame:
            # Only the rows the figures use are verified, so a solver left
            # out never needs a requested tolerance.
            rows = family_rows(rows, family, args.exclude_solver)
            found["solver_ids"] = sorted(rows["solver_id"].astype(str).unique())
            return rows[rows["solver_id"].astype(str).map(solver_tol).notna()]

        try:
            df = load_results(run_dir, verify=policy, select=select)
        except VerifyInputError as exc:
            failures.append(f"{family}: {exc}")
            continue
        if "solver_ids" not in found:
            print(f"{family}: no results in {run_dir}")
            continue
        solver_ids = found["solver_ids"]
        if not solver_ids:
            print(f"{family}: no results left in {run_dir} after the family's dataset, problem and solver "
                  "filters; skipped")
            continue
        untagged = [s for s in solver_ids if solver_tol(s) is None]
        if len(untagged) == len(solver_ids):
            failures.append(f"{family}: no solver id in {run_dir} ends in a tolerance tag such as _1e-4 "
                            f"(solver ids: {', '.join(solver_ids)}); make_figures groups the figures by that tag")
            continue
        if untagged:
            print(f"warning: {family}: left out solver ids without a tolerance tag: {', '.join(untagged)}",
                  file=sys.stderr)
        info = verification_info(df)
        rule = verify_rule(df)
        print(f"{family}: {verification_header(info)}")
        for warning in info.get("warnings") or []:
            print(f"warning: {family}: {warning}", file=sys.stderr)
        # Sized once per family, over every tolerance's rows, so the quartile
        # is the same set of problems in each tolerance's figures.
        lookup = None if args.no_size_lookup else registry_size_lookup(run_dir, repo_root=args.repo_root)
        sizes = problem_sizes(df, size_lookup=lookup)
        # After verification: a late row is a failure whatever it was verified as.
        limit = FAMILY_LIMIT.get(family)
        late = late_rows(df, limit) if limit else pd.Series(False, index=df.index)
        table = verification_after_late(df, late)
        if not table.empty:
            print(table.to_string(index=False))
            verification.append(table.assign(family=family, verify_rule=rule))
        if limit:
            df = mark_late(df, limit)
        for warning in scs_time_limit_warnings(df):
            print(f"warning: {family}: {warning}", file=sys.stderr)
        use_run = dict(kv.split("=", 1) for kv in args.use_run)
        tags = sorted({solver_tol(s) for s in df["solver_id"].unique() if solver_tol(s)})
        if args.tol_tag:
            tags = [t for t in tags if t == args.tol_tag]
        for tag in tags:
            sub = df[df["solver_id"].str.endswith(f"_{tag}")]
            LABEL_OVERRIDE.clear()
            source_run: dict[str, str] = {}
            for key, run_tag in use_run.items():
                # "solver=tag" applies everywhere, "family:solver=tag" to one family only
                if ":" in key:
                    fam_key, solver = key.split(":", 1)
                    if fam_key != family:
                        continue
                else:
                    solver = key
                    if any(k.startswith(f"{family}:{solver}") for k in use_run):
                        continue  # a family-specific entry wins
                # Use the requested run; if this family lacks it, fall back to the
                # tightest run that exists (interior-point solvers overshoot anyway).
                alt = df[df["solver_id"] == f"{solver}_{run_tag}"]
                if alt.empty:
                    for fallback in ("1e-6", "1e-5", "1e-4"):
                        alt = df[df["solver_id"] == f"{solver}_{fallback}"]
                        if not alt.empty:
                            run_tag = fallback
                            break
                if run_tag == tag or alt.empty:
                    continue
                sub = pd.concat([sub[sub["solver_id"] != f"{solver}_{tag}"], alt.assign(solver_id=f"{solver}_{tag}")],
                                ignore_index=True)
                LABEL_OVERRIDE[solver] = f"{SOLVER_LABELS.get(solver, solver)} ({run_tag})"
                source_run[solver] = run_tag
                # Its rows keep their own run's verification; say at what.
                checked = verify_tols(alt).get(f"{solver}_{run_tag}")
                if checked:
                    others = sorted({v for s, v in verify_tols(sub).items()
                                     if base_solver(s) != solver and s.endswith(f"_{tag}") and v})
                    print(f"{family} {tag}: {solver} is shown from its {run_tag} run, verified at {checked} "
                          f"(other series: {'; '.join(others) or 'none verified'})")
            n_all = sub.groupby(["dataset", "problem"]).ngroups
            title = f"{sets_title(family, sub)}, tolerance {tag}, {n_all} problems"
            plot_profile(sub, title, args.out_dir / f"{family}_{tag}_profile.png")
            plot_pair(sub, title, args.out_dir / f"{family}_{tag}_pair.png", family)
            gm = plot_geomean(sub, title, args.out_dir / f"{family}_{tag}_geomean.png", family=family)
            big = largest_quartile(sub, sizes)
            n_big = big.groupby(["dataset", "problem"]).ngroups
            title_big = f"{sets_title(family, big)}, largest quartile ({n_big} problems), tolerance {tag}"
            plot_profile(big, title_big, args.out_dir / f"{family}_{tag}_profile_largest.png")
            plot_pair(big, title_big, args.out_dir / f"{family}_{tag}_pair_largest.png", family)
            gm_big = plot_geomean(big, title_big, args.out_dir / f"{family}_{tag}_geomean_largest.png", family=family)
            if tag == (args.tol_tag or "1e-4"):
                frames[(family, "all")] = (sub, title, dict(LABEL_OVERRIDE))
                frames[(family, "largest")] = (big, title_big, dict(LABEL_OVERRIDE))
            for label, table in (("all", gm), ("largest", gm_big)):
                if table is not None and not table.empty:
                    t = table.copy()
                    t.insert(0, "subset", label)
                    t.insert(0, "tol", tag)
                    t.insert(0, "family", family)
                    t["label"] = [solver_label(s) for s in t["solver_id"]]
                    t["source_run"] = [source_run.get(base_solver(s), tag) for s in t["solver_id"]]
                    # Last, so readers that pick columns by name are unaffected.
                    tols = verify_tols(sub)
                    t["verify_rule"] = rule
                    t["verify_tol"] = [tols.get(str(s), "") for s in t["solver_id"]]
                    summary.append(t)
            print(f"{family} {tag}: {n_all} problems, largest quartile {n_big}")
    if args.grid:
        plot_grid(frames, args.grid)
        print(args.grid)
    if summary:
        pd.concat(summary).to_csv(args.out_dir / "geomean_summary.csv", index=False)
        print(args.out_dir / "geomean_summary.csv")
    if verification:
        table = pd.concat(verification, ignore_index=True)
        table = table[["family", "verify_rule", *[c for c in table.columns if c not in ("family", "verify_rule")]]]
        table.to_csv(args.out_dir / "verification_summary.csv", index=False)
        print(args.out_dir / "verification_summary.csv")
    if failures:
        sys.exit("\n".join(failures))


if __name__ == "__main__":
    main()
