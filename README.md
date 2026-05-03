# Solver Benchmarks

[![Tests](https://github.com/bodono/solver_benchmarks/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/bodono/solver_benchmarks/actions/workflows/tests.yml)

Config-driven benchmark suite for convex optimization solvers.

The maintained entrypoint is the `bench` CLI backed by the `solver_benchmarks`
package.

## Goals

- Configure solver variants from files, including many settings for the same solver.
- Run named datasets or selected individual problems, optionally combining several datasets in a single run.
- Skip unsupported solver/problem combinations with structured warnings.
- Execute each solve in a subprocess for crash isolation, stdout/stderr capture, and timeouts.
- Resume interrupted runs without duplicating completed solves.
- Preserve old runs by writing every run to an immutable run directory.
- Write both human-readable JSONL and analysis-friendly Parquet.
- Keep CVXPY out of the maintained benchmark path.
- Make it straightforward to add new datasets, solvers, analysis tools, and tests.

## Installation

Create and activate your Python environment, then install the package in editable mode:

```bash
pip install -e ".[all]"
```

The `all` extra installs open-source solver dependencies declared by the package,
including QTQP, SCS, OSQP, Clarabel, ECOS, CVXOPT, HiGHS, PIQP, OR-Tools/PDLP,
PyYAML, and pytest. ProxQP and SDPA are included on Python versions where their
wheels are available; currently they are skipped on Python 3.14 by dependency
markers.

Commercial solvers are optional and must be installed separately with valid licenses:

```bash
pip install -e ".[gurobi]"
pip install -e ".[mosek]"
pip install -e ".[cplex]"
```

Install only one optional solver if you prefer:

```bash
pip install -e ".[scs]"
pip install -e ".[osqp]"
pip install -e ".[clarabel]"
pip install -e ".[ecos]"
pip install -e ".[cvxopt]"
pip install -e ".[highs]"
pip install -e ".[qtqp]"
pip install -e ".[pdlp]"
pip install -e ".[piqp]"
pip install -e ".[proxqp]"
pip install -e ".[sdpa]"
```

The `system_info` extra installs `psutil`, which enriches the system snapshot
captured into each run's `manifest.json` (physical core count, CPU frequency,
total RAM). The capture falls back gracefully when `psutil` isn't installed.

```bash
pip install -e ".[system_info]"
```

Check what is available in the current environment:

```bash
bench list solvers
```

Example output:

```text
clarabel  available               cone,qp
cplex     missing optional extra  qp
cvxopt    available               cone,qp
ecos      available               cone,qp
gurobi    missing optional extra  qp
highs     available               qp
mosek     missing optional extra  qp
osqp      available               qp
pdlp      available               cone,qp
piqp      available               qp
proxqp    available               qp
qtqp      available               qp
sdpa      available               cone
scs       available               cone,qp
```

## Supported Datasets

List datasets:

```bash
bench list datasets
```

Current maintained adapters:

| Dataset | ID | Problem type | Notes |
|---|---|---|---|
| Maros-Meszaros | `maros_meszaros` | QP | Loads `.mat` QP files. |
| NETLIB | `netlib` | LP as QP | Use `dataset_options.subset: feasible` or `infeasible`. |
| Kennington | `kennington` | LP as QP | Standard NETLIB Kennington subset (16 instances). Bundled in the repo as decoded MPS — NETLIB hosts these in EMPS (compressed-MPS) format which our `qpsreader` cannot parse, so the prepare path copies from the bundled checkout instead of fetching from the network. |
| Liu-Pataki | `liu_pataki` | Cone/SDP | Bundled infeasible and weakly infeasible SeDuMi `.mat` SDP instances; converted to canonical PSD triangle cones. |
| MIPLIB root LP relaxation | `miplib` or `miplib_lp_relaxation` | LP as QP | Integrality is ignored; root-node LP relaxation only; downloads official MIPLIB 2017 benchmark `.mps.gz` files on request. |
| QPLIB | `qplib` | QP | Uses QPLIB parser without CVXPY; supports category filters such as `subset: ccb`. |
| Mittelmann | `mittelmann` | LP as QP | External ASU lptestset downloads; default prepare downloads `qap15`. |
| Mittelmann SDP | `mittelmann_sdp` | Cone/SDP | Mittelmann's SDP test set in SDPA-S sparse format (G-graph maxcut relaxations + Lovász theta numbers); downloads from `plato.asu.edu`. |
| SDPLIB | `sdplib` | Cone/SDP | Reads converted `.jld2` files; requires `h5py`. |
| TSPLIB MaxCut SDP | `tsplib_sdp` | Cone/SDP | Goemans-Williamson MaxCut SDP relaxations of TSPLIB instances; supports EUC_2D/3D, MAN/MAX, GEO, ATT, and EXPLICIT weight types. |
| DIMACS | `dimacs` | Cone | Reads `.mat` and `.mat.gz`; rotated Lorentz cones are not yet supported. |
| CBLIB | `cblib` | Cone | Downloads CBF files; the parser handles continuous linear (`L=`/`L+`/`L-`/`F`), second-order (`Q`), and exponential (`EXP`/`EXP*`) cone instances. Instances using other cone kinds (PSD, integer, etc.) are *hidden* from `list_problems()` by default; pass `dataset_options.include_unsupported=true` to surface them with `metadata["supported"]=False`. |
| MPC QP Benchmark | `mpc_qpbenchmark` | QP | Downloads structured MPC QPs from `qpsolvers/mpc_qpbenchmark`. |
| LIBSVM-derived QP | `libsvm_qp` | QP | SVM dual or Markowitz portfolio QPs built from LIBSVM datasets; produces realistic ML/finance-shaped QPs absent from classical test sets. |
| DC OPF | `dc_opf` | LP as QP | DC Optimal Power Flow LPs from MATPOWER `.m` case files; sparse power-balance equalities, line-flow inequality limits, generator box bounds. |
| CUTEst QP exports | `cutest_qp` | QP | Local export target only; no automatic CUTEst downloader. |
| Synthetic smoke test | `synthetic_qp` | QP | Tiny deterministic test problem. |
| Synthetic cone smoke test | `synthetic_cone` | Cone | Tiny deterministic conic LP test problem. |

Check local dataset availability:

```bash
bench data status
bench data status netlib
```

Most benchmark data currently used by the suite is bundled under
`problem_classes/`, including DIMACS. External or intentionally-large sources
are prepared explicitly. The convention is:

- `bench run CONFIG` is offline by default. It uses only local data. If
  requested data from an automatically downloadable dataset is missing, the
  command exits with an actionable error that includes the exact
  `bench data prepare ...` command and the equivalent
  `bench run CONFIG --prepare-data` command.
- `bench run CONFIG --prepare-data` or `run.auto_prepare_data: true` prepares
  data implied by the config before solving. If `include` is set, those problem
  names are prepared. If `dataset_options.subset: all` is set, the dataset's
  full remote index is prepared. Otherwise the dataset's small default subset is
  prepared.
- `bench data prepare DATASET` downloads or extracts a small default subset.
- `bench data prepare DATASET --problem NAME` prepares one or more named problems.
- `bench data prepare DATASET --all` opts into the largest known source for that dataset.
- Prepared files are cached under `problem_classes/<dataset>_data` or the
  dataset-specific cache directory and reused by later runs.

Download-backed examples:

```bash
bench data prepare cblib
bench data prepare mpc_qpbenchmark
bench data prepare qplib
bench data prepare miplib
bench data prepare mittelmann
bench data prepare mittelmann_sdp
bench data prepare tsplib_sdp
bench data prepare libsvm_qp
bench data prepare dc_opf
```

(`bench data prepare kennington` is also valid but it copies from the
bundled checkout rather than the network — see the Kennington note
below.)

Running with automatic preparation:

```bash
bench run configs/my_run.yaml --prepare-data
```

Default prepared subsets are intentionally small:

| Dataset | Default prepare set |
|---|---|
| `cblib` | `nql30`, `qssp30`, `sched_50_50_orig`, `nb`, `nb_L2_bessel` |
| `mpc_qpbenchmark` | `LIPMWALK0`, `WHLIPBAL0`, `QUADCMPC1` |
| `qplib` | `8790`, `8515`, `8495` |
| `kennington` | The 16 standard Kennington LP files. |
| `miplib` | `markshare_4_0` |
| `mittelmann` | `qap15` |
| `mittelmann_sdp` | `trto3`, `rose13`, `cnhil8`, `buck3`, `biggs`, `G40mc` |
| `tsplib_sdp` | `burma14`, `ulysses16`, `gr17`, `gr21`, `gr24`, `bayg29` |
| `libsvm_qp` | `heart`, `breast-cancer`, `australian`, `diabetes`, `ionosphere`, `german-numer` |
| `dc_opf` | `case5`, `case6ww`, `case9`, `case14`, `case30`, `case39` |

DIMACS data is bundled in the repository under `problem_classes/dimacs_data`.
`bench data prepare dimacs --problem NAME` can repair a missing local DIMACS
file from that bundled checkout copy and only falls back to the official
Challenge host when the requested file is not bundled. `dimacs --all` follows
the official Challenge index and therefore depends on that external host.

SDPLIB is different from the download-backed datasets above: the maintained
adapter reads converted `.jld2` files, or extracts them from
`problem_classes/sdplib_data/sdplib.tar` when that archive is present. It does
not download and convert the original SDPLIB files automatically. If the
converted archive is absent, restore `sdplib.tar` or place converted `.jld2`
files in `problem_classes/sdplib_data`.

Every prepare command also has a heavily-signposted wrapper script under
`scripts/`:

```bash
python scripts/prepare_cblib.py
python scripts/prepare_mpc_qpbenchmark.py
python scripts/prepare_qplib.py
python scripts/prepare_kennington.py
python scripts/prepare_miplib.py
python scripts/prepare_mittelmann.py
python scripts/prepare_mittelmann_sdp.py
python scripts/prepare_tsplib_sdp.py
python scripts/prepare_libsvm_qp.py
python scripts/prepare_dc_opf.py
python scripts/prepare_sdplib.py
python scripts/prepare_dimacs.py
python scripts/prepare_cutest_qp.py
```

The `--all` flag is deliberately never implicit. For example, CBLIB `--all`
follows the full CBLIB directory index and may download mixed-integer or
currently unsupported CBF files; the `cblib` adapter lists continuous
linear, second-order, and exponential-cone (`EXP`/`EXP*`) instances by
default and hides anything else (PSD, integer, etc.). Set
`dataset_options.include_unsupported=true` to surface those entries
with `metadata["supported"]=False`. Mittelmann `--all` follows the ASU lptestset
index. MIPLIB `--all` follows the official MIPLIB 2017 benchmark v2 instance
list, and `bench data prepare miplib --option max_size_mb=5` downloads only
benchmark instances whose compressed `.mps.gz` file is at most 5 MB. QPLIB
`--all` uses `problem_classes/qplib_data/list_convex_qps.txt`.

Benchmark runs can opt into preparation:

```yaml
run:
  auto_prepare_data: true
  include:
    - qap15
```

or from the CLI:

```bash
bench run configs/my_run.yaml --prepare-data
```

List problems in a dataset:

```bash
bench list problems netlib --option subset=feasible
bench list problems maros_meszaros
bench list problems qplib
bench list problems qplib --option subset=ccb
bench list problems liu_pataki --option classification=weak --option conditioning=messy
bench list problems mpc_qpbenchmark --option subset=default
bench list problems cblib
bench list problems cblib --option subset_kind=expcone
bench list problems libsvm_qp --option kind=svm_dual
bench list problems libsvm_qp --option kind=markowitz
bench list problems mittelmann_sdp
bench list problems tsplib_sdp
bench list problems dc_opf
```

Dataset options are passed as `key=value` on the CLI or as `dataset_options` in
config files.

Useful dataset options:

| Dataset | Option | Meaning |
|---|---|---|
| All file-backed datasets | `max_size_mb=<number>` | Drop problems whose backing file exceeds the threshold (see below). |
| `netlib` | `subset=feasible` or `subset=infeasible` | Select NETLIB feasibility subset. |
| `qplib` | `subset=default`, `ccb`, `ccl`, `dcl`, `all`, or comma-separated IDs | Filter convex QPLIB instances. |
| `liu_pataki` | `classification=infeas` or `classification=weak` | Select infeasible or weakly infeasible SDP families. |
| `liu_pataki` | `conditioning=clean` or `conditioning=messy` | Select clean or numerically messier Liu-Pataki instances. |
| `liu_pataki` | `constraint_count=10` or `constraint_count=20` | Select Liu-Pataki instances by equality count. |
| `liu_pataki` | `block_dim=10` | Select Liu-Pataki instances by PSD block dimension. |
| `mpc_qpbenchmark` | `subset=default`, `all`, or comma-separated names | Filter downloaded MPC QP `.npz` files. |
| `cutest_qp` | `subset=default`, `all`, or comma-separated names | Filter locally exported CUTEst QP `.npz` files. |
| `cblib` | `subset=default`, `all`, or comma-separated names | Filter downloaded CBF files. |
| `cblib` | `subset_kind=expcone`, `socp`, or `lp` | Filter CBF instances by cone shape rather than by name. Combines with `subset` (both filters must pass). |
| `cblib` | `include_unsupported=true` | Show downloaded CBF files the parser would otherwise hide. |
| `mittelmann_sdp` | `subset=all` (no filter) or comma-separated names | Filter to a subset of locally available SDPA-S instances. |
| `tsplib_sdp` | `subset=all` (no filter) or comma-separated names | Filter to a subset of locally available TSPLIB instances. |
| `dc_opf` | `subset=all` (no filter) or comma-separated names | Filter to a subset of locally available MATPOWER cases. |
| `libsvm_qp` | `kind=svm_dual` (default) or `markowitz` | Choose the QP shape derived from the LIBSVM data. |
| `libsvm_qp` | `kernel=linear` (default) or `rbf` | SVM-dual kernel choice. Linear gives a low-rank `Q` (rank ≤ feature dimension). |
| `libsvm_qp` | `gamma=<float>` | RBF kernel bandwidth; defaults to `1 / num_features`. |
| `libsvm_qp` | `C=<float>` | SVM regularization upper bound on dual variables (default `1.0`). |
| `libsvm_qp` | `risk_aversion=<float>` | Markowitz `λ` in the objective `½ w'Σw − λ μ'w` (default `1.0`). |
| `libsvm_qp` | `max_samples=<int>` | Deterministically subsample to this many rows (default `200`). |
| `libsvm_qp` | `subset=default`, `all`, or comma-separated names | Filter to a subset of LIBSVM datasets. |

### Filtering by file size

Many problem libraries (MIPLIB, NETLIB, QPLIB, CUTEst-QP, ...) include a long
tail of large instances that take a long time to solve. Set
`dataset_options.max_size_mb` to drop problems whose backing file exceeds
the threshold (in megabytes, measured on disk):

```yaml
run:
  dataset: miplib
  dataset_options:
    max_size_mb: 1.0   # only run instances whose .mps[.gz] is <= 1 MB
```

Notes on the semantics:

- The filter is applied uniformly across every file-backed dataset. The same
  visible problem set is used by `bench run`, `bench list problems`,
  `bench data status`, and analysis commands such as `bench summary`,
  `bench missing`, and `bench report`.
- During a run, the filter is applied after `include` / `exclude` selection.
  `bench data status` reports the visible count after filters and notes when
  only a subset of locally listed problems is visible.
- By default the threshold is compared against the on-disk size of the
  selected `ProblemSpec.path`.
- Datasets can advertise a more precise comparison size via
  `ProblemSpec.metadata["size_bytes"]`. The runner and listing/status tools
  prefer that value when present. This is used for archive-backed datasets
  such as `sdplib.tar`, where the threshold applies to each archive member,
  and for MPS datasets that have both `.mps` and `.mps.gz` encodings of the
  same problem, where the smallest available local encoding is used for the
  size decision.
- The `synthetic` dataset has no backing file and is unaffected.

## Supported Solvers

Current maintained adapters:

| Solver | ID | Supported types | Notes |
|---|---|---|---|
| QTQP | `qtqp` | QP | Converts QP bounds to nonnegative cone form internally. |
| SCS | `scs` | QP, cone | Supports SCS box-cone form for QPs and native conic data. |
| Clarabel | `clarabel` | QP, cone | Uses nonnegative cone conversion for QPs and native zero/nonnegative/SOC/PSD cones. |
| OSQP | `osqp` | QP | Direct QP adapter. |
| ECOS | `ecos` | QP, cone (LP/SOCP/expcone) | Interior-point conic solver. QPs are reformulated to SOCP via the standard epigraph trick (Cholesky for PD `P`, eigendecomposition for rank-deficient PSD); the SOC dim is `rank(P) + 2`. PSD cones not supported. |
| CVXOPT | `cvxopt` | QP, cone (LP/QP/SOCP/SDP) | Interior-point solver supporting QPs natively plus PSD cones (canonical PSD-vec layout converted to BLAS unpacked-L automatically). Exponential cones not supported. CVXOPT options live on a global dict; the adapter snapshots and restores it per solve so **sequential** solves in the same process don't leak knobs into each other. This is *not* thread-safe — concurrent in-process solves would still race on the global. The benchmark runner uses subprocess-level parallelism, so this per-process sequential contract is sufficient. |
| PDLP | `pdlp` | LP-only QP, simple linear cone | Uses OR-Tools directly, no CVXPY. |
| HiGHS | `highs` | QP | Direct LP/QP adapter through `highspy`; strong LP baseline. |
| ProxQP | `proxqp` | QP | Direct QP adapter through ProxSuite; optional on Python < 3.14 while wheels are available. |
| PIQP | `piqp` | QP | Direct sparse/dense QP adapter. |
| SDPA | `sdpa` | cone | Linear conic adapter through `sdpa-python`; useful for SDP/SOCP subsets, optional on Python < 3.14 while wheels are available. |
| GUROBI | `gurobi` | QP | Optional extra; requires `gurobipy` and license. |
| MOSEK | `mosek` | QP | Optional extra; requires Mosek package and license. |
| CPLEX | `cplex` | QP | Optional extra; requires IBM CPLEX package and license. |

Unsupported solver/problem combinations are skipped by default and recorded as
`skipped_unsupported`. For example, OSQP on a DIMACS conic problem is skipped.

To fail instead of skipping:

```yaml
run:
  fail_on_unsupported: true
```

## Configuration Files

Runs are driven by JSON or YAML. YAML requires `PyYAML`, which is included in the
base dependency list.

Minimal example:

```yaml
run:
  name: netlib_feasible_smoke
  dataset: netlib
  output_dir: results
  dataset_options:
    subset: feasible
  include:
    - afiro
    - sc50a
  exclude: []
  parallelism: 2
  resume: true
  timeout_seconds: 120

solvers:
  - id: osqp_default
    solver: osqp
    settings:
      eps_abs: 1.0e-4
      eps_rel: 1.0e-4
      max_iter: 100000

  - id: osqp_tight
    solver: osqp
    settings:
      eps_abs: 1.0e-6
      eps_rel: 1.0e-6
      max_iter: 200000

  - id: pdlp_default
    solver: pdlp
    settings:
      time_limit_sec: 120
```

Important fields:

| Field | Meaning |
|---|---|
| `run.name` | Optional human-readable run label. If omitted for CLI runs, the config filename stem is used. Run directories use `<run_name>_<YYYY-MM-DD>_<HH-MM-SS>_UTC`. |
| `run.dataset` | Dataset ID from `bench list datasets`. Use `datasets` instead to run several. |
| `run.datasets` | List of dataset entries. Each entry is either a dataset ID string, or a mapping with `name` plus optional `id`, `dataset_options`, `include`, and `exclude`. Mutually exclusive with `dataset`. |
| `run.output_dir` | Root directory for immutable runs. Relative paths are resolved under the repository root, not next to the config file. Defaults to `results`. |
| `run.dataset_options` | Dataset-specific options, such as `subset: feasible`. Treated as defaults for entries in `datasets`. |
| `run.include` | Optional list of problem names to run. Empty means all. Used as a fallback for any dataset whose own `include` is unset. |
| `run.exclude` | Optional list of problem names to skip. Unioned with each dataset entry's `exclude`. |
| `run.parallelism` | Number of concurrent subprocess solves. |
| `run.resume` | If true, completed `(dataset, problem, solver_id)` triples are not rerun. |
| `run.timeout_seconds` | Default subprocess timeout per solve. |
| `run.auto_prepare_data` | If true, run dataset preparation before listing/solving missing requested problems. |
| `solvers[].id` | Unique label for this solver variant. Used in output paths. |
| `solvers[].solver` | Solver adapter ID, e.g. `scs`, `qtqp`, `pdlp`. |
| `solvers[].settings` | Passed directly to the solver adapter. |
| `solvers[].timeout_seconds` | Optional per-solver timeout override. |
| `solvers[].sweep` | Optional mapping of setting name to a non-empty list of values. Expands to the full Cartesian product. |
| `solvers[].id_template` | Optional Python format string used to name expanded sweep variants. |

The same solver may appear many times with different `id` and `settings`.
Solver output is verbose by default and is captured in each solve's
`stdout.log`/`stderr.log`; set `verbose: false` in a solver's settings to run it
quietly. To keep full solver output in the log files without printing it to the
terminal, pass `--no-stream-output` to `bench run`.

### Multi-Dataset Runs

A single run can solve the same set of solver variants across several datasets.
List the datasets under `run.datasets` instead of `run.dataset`:

```yaml
run:
  output_dir: results
  parallelism: 4
  resume: true
  timeout_seconds: 300
  datasets:
    - name: netlib
      dataset_options:
        subset: feasible
      include:
        - afiro
        - sc50a
    - name: maros_meszaros
      exclude:
        - HUES-MOD
    - synthetic_qp

solvers:
  - id: scs_default
    solver: scs
    settings:
      eps_abs: 1.0e-6
      eps_rel: 1.0e-6
  - id: clarabel_default
    solver: clarabel
```

Behavior:

- Each entry can be a bare dataset ID string or a mapping with `name`,
  optional `id`, `dataset_options`, `include`, and `exclude`. Each entry must
  have a unique identity within a config: by default that is `name`, but you
  can set `id` to disambiguate two entries that share an adapter (see below).
- Run-level `dataset_options` are applied as defaults; entries can override or
  extend them per dataset.
- Run-level `include` is used as a fallback only for datasets whose own
  `include` is unset, so per-dataset selections do not bleed across datasets.
- Run-level `exclude` is unioned with each entry's `exclude`.
- Resume keys are `(dataset, problem, solver_id)` where `dataset` is the
  entry's `id`, so two datasets that share a problem name (e.g. `afiro` in
  NETLIB and a hand-crafted dataset) never get conflated.
- Every result row is tagged with its `dataset` (the entry `id`), and the run
  directory groups per-solve artifacts under
  `problems/<dataset>/<problem>/<solver_id>/`.

Reusing the same adapter with different options requires an explicit `id` per
entry. Without an `id`, two entries with the same `name` would collide on the
duplicate-name check. For example, to run NETLIB feasible and infeasible
subsets side-by-side in one run:

```yaml
run:
  output_dir: results
  datasets:
    - name: netlib
      id: netlib_feasible
      dataset_options:
        subset: feasible
    - name: netlib
      id: netlib_infeasible
      dataset_options:
        subset: infeasible
```

The two entries share the `netlib` adapter but produce independent slots in
the result table, separate `problems/<id>/...` artifact directories, and
distinct rows in the per-dataset analysis. `id` defaults to `name` when not
set, so single-entry and ordinary multi-entry configs are unchanged.

Analysis is dataset-aware out of the box:

- `bench summary` and `bench missing` report one row per `(solver, dataset)`
  pair, so an interrupted run is easy to diagnose.
- `bench report` includes the cross-dataset aggregate tables exactly as before
  and adds a `## By Dataset` section with per-dataset solver metrics, failure
  rates, shifted geomean, and KKT summary slices.
- Performance profiles, pairwise speedup tables, objective spreads,
  performance-ratio/KKT heatmaps, the failures-with-successful-alternatives
  table, and the cactus plot denominator all key on `(dataset, problem)`
  when more than one dataset is present, so two datasets sharing a problem
  name (e.g. `afiro`) are never silently collapsed into a single row or
  cross-matched between datasets. Plots and matrices label the combined
  axis as `dataset/problem`.
- Manifests record the full `datasets:` list, so older single-dataset runs and
  newer multi-dataset runs are both handled by the same analysis tools.

The legacy `run.dataset: <name>` shape is still accepted for single-dataset
runs and is treated as a one-entry `datasets` list internally.

### Hyper-Parameter Sweeps

Use `sweep` when you want to run many variants of the same solver without
manually writing one solver block per setting combination. Sweeps are expanded
when the config is loaded; after expansion, every variant is a normal solver
entry with a concrete `solver_id`, concrete settings, its own logs, and its own
rows in `results.jsonl` and `results.parquet`.

Example:

```yaml
solvers:
  - id: osqp
    solver: osqp
    settings:
      verbose: true
      max_iter: 100000
    sweep:
      eps_abs: [1.0e-4, 1.0e-6, 1.0e-8]
      eps_rel: [1.0e-4, 1.0e-6]
    id_template: "osqp_abs{eps_abs:g}_rel{eps_rel:g}"
```

This expands to six solver variants:

```text
osqp_abs0.0001_rel0.0001
osqp_abs0.0001_rel1e-06
osqp_abs1e-06_rel0.0001
osqp_abs1e-06_rel1e-06
osqp_abs1e-08_rel0.0001
osqp_abs1e-08_rel1e-06
```

The expansion is the full cross-product of all sweep lists. Values from `sweep`
override any same-named value in `settings`, so base settings are a convenient
place for shared configuration and `sweep` should contain only the knobs being
varied.

If `id_template` is omitted, IDs are generated from the base ID and swept values:

```yaml
solvers:
  - id: scs
    solver: scs
    settings:
      max_iters: 20000
    sweep:
      eps_abs: [1.0e-4, 1.0e-6]
      normalize: [true, false]
```

Generated IDs:

```text
scs__eps_abs=0.0001__normalize=true
scs__eps_abs=0.0001__normalize=false
scs__eps_abs=1e-06__normalize=true
scs__eps_abs=1e-06__normalize=false
```

`id_template` uses Python `str.format` with fields from the base solver entry and
expanded settings. Available fields include `id`, `solver`, and every setting
name after sweep expansion. Format specifiers work, so `{eps_abs:g}` is useful
for compact floating-point values.

Sweeps can also be used inside `bench env run` environment configs:

```yaml
environments:
  - id: osqp_1_0
    python: .venvs/osqp-1.0/bin/python
    metadata:
      osqp: "1.0.0"
    solvers:
      - id: osqp_1_0
        solver: osqp
        sweep:
          eps_abs: [1.0e-4, 1.0e-6]
          eps_rel: [1.0e-4, 1.0e-6]
        id_template: "osqp_1_0_abs{eps_abs:g}_rel{eps_rel:g}"
```

Sweep IDs must be unique after expansion, including across all environments in
an environment matrix. This is intentional: resume uses `(problem, solver_id)`,
and duplicate IDs would make two distinct configurations look like the same
completed solve.

## Running Benchmarks

Run a config:

```bash
bench run configs/synthetic_smoke.json
bench run configs/netlib_feasible_example.yaml
bench run configs/maros_meszaros_example.yaml
bench run configs/maros_meszaros_qp_solvers.yaml
bench run configs/multi_dataset_example.yaml
```

Bundled benchmark configs:

| Config | Purpose |
|---|---|
| `configs/synthetic_smoke.json` | Tiny one-problem smoke run. |
| `configs/netlib_feasible_example.yaml` | NETLIB feasible subset example. |
| `configs/maros_meszaros_example.yaml` | Maros-Meszaros QP example. |
| `configs/maros_meszaros_qp_solvers.yaml` | Maros-Meszaros QP run comparing OSQP, SCS, Clarabel, and QTQP. |
| `configs/multi_dataset_example.yaml` | Example of one run spanning several datasets. |
| `configs/scs_anderson_sweep.yaml` | SCS Anderson acceleration sweep over NETLIB feasible, MIPLIB instances up to 5 MB, Maros-Meszaros, and SDPLIB instances up to 1 MB. Includes `scs_aa_disabled` with `acceleration_lookback: 0`, plus the cross-product `acceleration_lookback: [5, 10, 20]` and `acceleration_interval: [1, 5, 10]`. |
| `configs/qtqp_refinement_regularization_sweep.yaml` | QTQP sweep over NETLIB feasible, MIPLIB instances up to 5 MB, and Maros-Meszaros. Uses `linear_solver: qdldl` and sweeps `max_iterative_refinement_steps: [1, 3, 5, 10, 20]` by `min_static_regularization: [1.0e-10, 1.0e-8, 1.0e-6]`. |
| `configs/scs_linear_solver_backend_sweep.yaml` | SCS comparison of `linear_solver: qdldl` versus `linear_solver: accelerate` on the same datasets as the Anderson sweep. |
| `configs/qtqp_linear_solver_backend_sweep.yaml` | QTQP comparison of `linear_solver: qdldl` versus `linear_solver: accelerate` on NETLIB feasible, MIPLIB instances up to 5 MB, and Maros-Meszaros. |

The command prints the run directory:

```text
results/netlib_feasible_example_2026-04-26_14-03-27_UTC
```

Resume a run:

```bash
bench run configs/netlib_feasible_example.yaml --run-dir results/<run_id>
```

If `resume: true`, already completed `(dataset, problem, solver_id)` triples in
`results.jsonl` are skipped. New solver variants or newly included problems are
appended.

Append more work to an existing run by editing the config and reusing the same
run directory:

```yaml
solvers:
  - id: qtqp_default
    solver: qtqp

  - id: clarabel_default
    solver: clarabel

  - id: clarabel_tight
    solver: clarabel
    settings:
      tol_gap_abs: 1.0e-8
      tol_feas: 1.0e-8
```

Then run:

```bash
bench run configs/my_run.yaml --run-dir results/<run_id>
```

The resume key is exactly `(dataset, problem, solver_id)`. If you change solver
settings but keep the same `id`, the old rows are treated as complete and will
not be rerun. Use a new `id` whenever settings change, such as `clarabel_tight`
instead of reusing `clarabel_default`.

Run only specific problems by editing `include`, or list problems first:

```bash
bench list problems netlib --option subset=feasible
```

## Comparing Solver Versions

Comparing two versions of the same Python solver should be done with isolated
Python environments. Do not uninstall and reinstall a solver inside a running
benchmark process; Python imports and native libraries are process/global state,
and that approach is brittle. The benchmark supports version comparisons by
recording runtime metadata for every solve and by providing an environment
matrix runner that invokes one Python executable per environment.

Every result row records metadata under `metadata.runtime`:

| Field | Meaning |
|---|---|
| `metadata.runtime.python_executable` | Python executable used by the solve subprocess. |
| `metadata.runtime.python_version` | Python version. |
| `metadata.runtime.platform` | OS/platform string. |
| `metadata.runtime.solver_package_versions` | Installed package versions relevant to the solver adapter, such as `{"osqp": "1.1.1"}`. |
| `metadata.environment_id` | Optional user-supplied environment label. |
| `metadata.environment_metadata` | Optional user-supplied metadata, usually pinned solver versions or environment notes. |

Manual workflow:

```bash
python3.12 -m venv .venvs/osqp-1.0
.venvs/osqp-1.0/bin/python -m pip install -e ".[test]" "osqp==1.0.0"

python3.12 -m venv .venvs/osqp-1.1
.venvs/osqp-1.1/bin/python -m pip install -e ".[test]" "osqp==1.1.1"

.venvs/osqp-1.0/bin/python -m solver_benchmarks.cli run configs/osqp_1_0.yaml \
  --run-dir results/osqp_version_compare \
  --environment-id osqp_1_0 \
  --environment-metadata '{"osqp":"1.0.0"}'

.venvs/osqp-1.1/bin/python -m solver_benchmarks.cli run configs/osqp_1_1.yaml \
  --run-dir results/osqp_version_compare \
  --environment-id osqp_1_1 \
  --environment-metadata '{"osqp":"1.1.1"}'

bench report results/osqp_version_compare
```

The solver IDs should encode the version, because resume uses
`(problem, solver_id)`:

```yaml
solvers:
  - id: osqp_1_0_default
    solver: osqp
    settings:
      eps_abs: 1.0e-6
      eps_rel: 1.0e-6
```

Environment matrix workflow:

```yaml
run:
  dataset: maros_meszaros
  output_dir: results
  include:
    - QAFIRO
    - QPCBLEND
  parallelism: 1
  resume: true
  timeout_seconds: 120

environments:
  - id: osqp_1_0
    python: .venvs/osqp-1.0/bin/python
    install:
      - "{python} -m pip install -e ."
      - "{python} -m pip install osqp==1.0.0"
    metadata:
      osqp: "1.0.0"
    solvers:
      - id: osqp_1_0_default
        solver: osqp
        settings:
          eps_abs: 1.0e-6
          eps_rel: 1.0e-6
          max_iter: 100000

  - id: osqp_1_1
    python: .venvs/osqp-1.1/bin/python
    install:
      - "{python} -m pip install -e ."
      - "{python} -m pip install osqp==1.1.1"
    metadata:
      osqp: "1.1.1"
    solvers:
      - id: osqp_1_1_default
        solver: osqp
        settings:
          eps_abs: 1.0e-6
          eps_rel: 1.0e-6
          max_iter: 100000
```

Run it with:

```bash
bench env run configs/osqp_versions.yaml --run-dir results/osqp_version_compare
```

`bench env run` deliberately does not create virtual environments. It runs the
`install` commands you provide and then invokes each configured `python` with
`python -m solver_benchmarks.cli run ...`. This keeps the benchmark independent
of a specific environment manager. You can use `venv`, `uv`, conda, Docker, or
pre-existing paths.

Install commands are executed directly, not through a shell. The runner expands
`{python}` to the environment's configured Python executable and `{repo_root}` to
the repository root, then tokenizes the command with shell-like quoting. If you
need shell features such as `&&`, redirects, or environment-variable expansion,
wrap them explicitly, for example `bash -lc 'source setup.sh && pip install ...'`.
The final run manifest records the union of all environment solver variants, so
`bench missing`, `bench summary`, and `bench report` understand the full
cross-environment run.

Examples with `uv`:

```yaml
environments:
  - id: osqp_1_0
    python: .venvs/osqp-1.0/bin/python
    install:
      - "uv venv .venvs/osqp-1.0 --python 3.12"
      - "uv pip install --python {python} -e . osqp==1.0.0"
    metadata:
      osqp: "1.0.0"
    solvers:
      - id: osqp_1_0_default
        solver: osqp
```

Examples with conda:

```yaml
environments:
  - id: osqp_1_0
    python: /opt/miniconda3/envs/osqp-1.0/bin/python
    install:
      - "conda run -n osqp-1.0 python -m pip install -e . osqp==1.0.0"
    metadata:
      osqp: "1.0.0"
    solvers:
      - id: osqp_1_0_default
        solver: osqp
```

After a version comparison run, normal analysis commands work unchanged:

```bash
bench summary results/osqp_version_compare
bench plot results/osqp_version_compare
bench report results/osqp_version_compare
```

Because `results.parquet` is flattened, runtime metadata columns can be inspected
directly with pandas:

```python
import pandas as pd

df = pd.read_parquet("results/osqp_version_compare/results.parquet")
print(df[[
    "solver_id",
    "metadata.environment_id",
    "metadata.runtime.python_version",
    "metadata.runtime.solver_package_versions.osqp",
    "status",
    "run_time_seconds",
]])
```

## CLI Reference

`bench` is the installed console script for `solver_benchmarks.cli`. If the
script is not on your `PATH`, use `python -m solver_benchmarks.cli` with the same
arguments.

Every command supports `--help`. Option names use hyphens, not underscores; for
example use `--max-value`, not `--max_value`.

Discovery commands:

| Command | Arguments | Purpose |
|---|---|---|
| `bench list datasets` | none | List registered dataset IDs and descriptions. |
| `bench list solvers` | none | List registered solver IDs, availability, and supported problem types. |
| `bench list problems DATASET` | `--repo-root PATH`, `--option key=value`, `--prepare/--no-prepare` default `--no-prepare` | List visible problems in one dataset. Repeat `--option` for dataset-specific options such as `subset=feasible` or generic filters such as `max_size_mb=5`. |

Data-management commands:

| Command | Arguments | Purpose |
|---|---|---|
| `bench data status [DATASET]` | `--repo-root PATH`, `--option key=value` | Report whether local data is available, how many problems are visible after options and generic filters, the data path, source, exact preparation command when one exists, and status message. Omit `DATASET` to check all datasets. |
| `bench data prepare DATASET` | `--repo-root PATH`, `--option key=value`, `--problem NAME`, `--all` default false | Download or prepare missing data for datasets that support automatic preparation. Repeat `--problem` for selected instances. Use `--all` only when you intentionally want every known remote problem. |

Run and analysis commands:

| Command | Arguments | Purpose |
|---|---|---|
| `bench run CONFIG_PATH` | `--run-dir PATH`, `--repo-root PATH`, `--prepare-data` default false, `--stream-output` / `--no-stream-output`, `--environment-id ID`, `--environment-metadata JSON` | Execute a benchmark config. Without `--run-dir`, creates a new immutable run directory named from `run.name` or the config filename stem plus a readable UTC timestamp, and copies the source config into the run directory. With `--run-dir`, resumes/appends to that run subject to `resume: true`. Without `--prepare-data`, missing external data is reported with exact preparation commands instead of being downloaded implicitly. `--no-stream-output` keeps solver stdout/stderr in per-solve log files but does not tee it to the terminal; benchmark progress still prints. The environment flags are normally used by version-comparison workflows and are recorded in result metadata. |
| `bench env run CONFIG_PATH` | `--run-dir PATH`, `--repo-root PATH`, `--stream-output` / `--no-stream-output` | Execute an environment matrix config. Each environment supplies a Python executable, optional install commands, metadata, and solver variants; all results are written into one run directory, with the source environment config copied into it. `--no-stream-output` is forwarded to each child benchmark run. |
| `bench summary RUN_DIR` | `--repo-root PATH` | Print solver metrics, status counts, and run completion information. |
| `bench failures RUN_DIR` | none | Print success/failure rates by solver. Only `optimal` counts as success by default. |
| `bench missing RUN_DIR` | `--repo-root PATH` | Print missing `(solver, dataset, problem)` results relative to the run manifest and dataset filters. |
| `bench profile RUN_DIR` | `--metric FIELD` default `run_time_seconds` | Write Dolan-More performance profile data to `performance_profile_<metric>.csv`. |
| `bench geomean RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--shift VALUE` default `10.0`, `--max-value VALUE` default `1000.0`, `--success-only` default false | Write shifted geometric means into the run directory. By default failures are penalized with `--max-value`; use `--success-only` for solved-problem-only geomeans. |
| `bench plot RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--output-dir PATH` default run dir | Write PNG plots for Dolan-More profile, cactus plot, pairwise scatter, performance-ratio heatmap, shifted geomean, status heatmap, failure rates, and three KKT-residual plots (per-solver boxplot, problem-by-solver heatmap, and KKT-accuracy profile). |
| `bench report RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--output-dir PATH` default `RUN_DIR/report`, `--repo-root PATH` | Write a complete report directory containing CSV tables, PNG plots, and a generated Markdown report. |

Common metrics for `profile` and `geomean` are `run_time_seconds`,
`solve_time_seconds`, `setup_time_seconds`, and `iterations`, depending on what
each solver reports.

When run from the CLI, `bench run` writes solver stdout/stderr to each solve's
`stdout.log`/`stderr.log`. By default it also streams that solver output to the
terminal; pass `--no-stream-output` to keep the log files without printing
solver output live. It also prints progress lines such as `[bench] starting ...`
and `[bench] finished ...`.
At the start of a run it prints the number of planned solves, already-complete
resume hits, skipped rows, queued rows, and parallelism. After every result is
written, it prints aggregate progress with completed/total counts, percentage,
this-run completion counts, elapsed time, solves/second, remaining-work ETA, and
the last completed `(dataset, problem, solver)` tuple with its run time. On a
resumed run, the main `progress X/Y` count includes rows completed by earlier
invocations, while `this_run X/Y` and `eta_remaining` only describe the work
queued for the current invocation. The same aggregate snapshots are recorded as
structured `benchmark_plan`, `benchmark_progress`, and `benchmark_complete`
events in `events.jsonl`.
With `parallelism > 1`, live solver output from different workers can interleave;
the per-solve log files remain separated and are the authoritative logs.

## Run Directory Layout

Every run is immutable by default:

```text
results/netlib_feasible_example_2026-04-26_14-03-27_UTC/
  manifest.json
  run_config.yaml
  events.jsonl
  results.jsonl
  results.parquet
  problems/
    synthetic_qp/
      one_variable_eq/
        scs_default/
          payload.json
          stdout.log
          stderr.log
          worker_result.json
          result.json
          trace.jsonl
```

Run directory names come from `run.name` or, for CLI runs without an explicit
name, the config filename stem. The content hash is still recorded in
`manifest.json`, but is not part of the directory name. Per-solve artifacts
always live under `problems/<dataset>/<problem>/<solver_id>/`, so two datasets
that share a problem name remain isolated on disk.

Files:

| File | Purpose |
|---|---|
| `manifest.json` | Run metadata, normalized config, config hash, creation time. |
| `run_config.yaml` / `run_config.json` | Exact source config file passed to `bench run`, copied into the run directory before solving. |
| `events.jsonl` | Structured warnings/events, including unavailable solvers and unsupported combinations. |
| `results.jsonl` | One JSON record per completed or skipped solve. Human-readable and append-friendly. |
| `results.parquet` | Flattened results table for pandas/Arrow analysis. Rewritten after each result. |
| `payload.json` | Exact payload passed to the subprocess worker. |
| `stdout.log` | Everything emitted to subprocess stdout during a solve. |
| `stderr.log` | Everything emitted to subprocess stderr during a solve. |
| `worker_result.json` | Raw worker output before aggregation. |
| `result.json` | Per-solve result record matching `results.jsonl`. |
| `trace.jsonl` | Iteration trace when the solver adapter exposes one. |

## Logging and Provenance

Each solve runs in a subprocess. This gives:

- Native solver crashes do not kill the main runner.
- Timeouts are enforced per solve.
- Solver stdout/stderr are streamed live by the CLI and captured per problem and solver variant.
- The exact worker payload is retained.
- Completed solves are discoverable for resume.

Warnings are structured JSON records. Example `events.jsonl` line:

```json
{
  "timestamp_utc": "2026-04-24T10:00:00+00:00",
  "level": "warning",
  "message": "Solver 'osqp' does not support 'cone' problems; skipping 'qssp30'",
  "solver_id": "osqp_skip",
  "problem": "qssp30",
  "problem_kind": "cone"
}
```

Solver-specific traces:

- SCS can write a CSV trace if `log_csv_filename: true` is set. The adapter maps
  this to a per-solve artifact path.
- QTQP writes `trace.jsonl` when its collected stats are available.
- PDLP writes `pdlp_solve_log.textproto` and `pdlp_response.textproto` when
  OR-Tools is installed and a solve is attempted.

System info: each run's `manifest.json` carries a `system` block captured once
at run start, recording the CPU model / logical and physical core count / max
frequency, total and available memory, OS / kernel version, Python version,
and the installed numpy / scipy / pandas / pyarrow versions. The block is
preserved across manifest rewrites so a re-run on the same run directory
cannot silently overwrite the original provenance. Install the optional
`system_info` extra (`pip install -e ".[system_info]"`) for the richer fields
(physical core count, CPU frequency, swap); the capture falls back gracefully
to stdlib + `/proc` (Linux) / `sysctl` (macOS) without it.

The generated markdown report renders the block as a `### System` section at
the top of the Provenance block:

```text
### System

| Field | Value |
| --- | --- |
| CPU model | Apple M4 |
| CPU cores | 10 logical |
| Total RAM | 16.0 GiB |
| OS | macOS-26.3.1-arm64-arm-64bit |
| Python | 3.12.8 |
| Libraries | numpy 2.4.2, pandas 3.0.0, pyarrow 23.0.1, scipy 1.16.3 |
```

Per-row results also carry `metadata.runtime.cpu_model` so heterogeneous runs
(multiple environments / hosts in one matrix) can be told apart row-by-row.

## Analysis

Summary by solver/status:

```bash
bench summary results/<run_id>
bench failures results/<run_id>
bench missing results/<run_id>
```

`bench summary` includes solver-level runtime and iteration aggregates, status
counts, and completion counts. The completion table is the fastest way to detect
an interrupted run: every `(solver, dataset)` row should have `missing = 0` and
`complete = True`. For multi-dataset runs, `bench summary` and `bench missing`
emit one row per `(solver, dataset)` pair so an interrupted dataset is easy to
spot.

Generate a Dolan-More performance-profile CSV:

```bash
bench profile results/<run_id>
bench profile results/<run_id> --metric iterations
```

The Dolan-More profile uses `r[p, s] = metric[p, s] / min_s metric[p, s]` and
plots the fraction of problems with `r[p, s] <= tau`. Failed or inaccurate solves
are assigned the failure penalty before ratios are computed.

Generate shifted geometric means:

```bash
bench geomean results/<run_id>
bench geomean results/<run_id> --metric iterations
bench geomean results/<run_id> --success-only
```

The default `bench geomean` output is a penalized shifted geometric mean for
benchmark comparison, not a raw average runtime. Non-`optimal` statuses,
including `optimal_inaccurate`, are assigned `--max-value` before the geometric
mean is computed. Use `bench summary` for raw totals/means/medians, or
`bench geomean --success-only` for a geomean over successful solves only. The
default failure penalty is `1000` seconds and can be changed with `--max-value`.

Generate PNG plots:

```bash
bench plot results/<run_id>
bench plot results/<run_id> --metric iterations
```

Generate a complete report directory:

```bash
bench report results/<run_id>
bench report results/<run_id> --metric iterations --output-dir reports/my_run
```

`bench report` writes:

- A generated Markdown report at `index.md`, mirrored to `README.md`, that presents run metadata, completion, solver metrics, status counts, plots, diagnostics, provenance (including the captured `### System` block — CPU model, cores, RAM, OS, library versions), and links to the full CSV artifacts in a single document.
- Solver-level metrics, status counts, failure rates, completion, and missing results.
- Dolan-More performance-profile CSVs and plots.
- Cactus plots showing the fraction of problems solved under each time/iteration budget.
- Pairwise scatter plots for problems solved by both solvers.
- Pairwise speedup tables with wins/losses/ties and largest wins.
- A per-problem cross-solver table, `problem_solver_comparison.csv`, with columns such as `qtqp_default__status`, `qtqp_default__run_time_seconds`, and `qtqp_default__iterations`.
- Per-solver problem tables under `solver_problem_tables/`, with rows as problems and columns for status, runtime, iterations, objective, artifact path, and problem dimensions when available.
- Objective-spread and slowest-solve tables.
- Failures-with-successful-alternatives tables, listing problems where one solver failed but at least one other solver succeeded.
- Status and performance-ratio heatmaps.
- KKT-residual summaries (`kkt_summary.csv`, `kkt_certificate_summary.csv`) plus three KKT plots: per-solver boxplot of `primal_res_rel` / `dual_res_rel` / `comp_slack` / `duality_gap_rel`, a problem-by-solver heatmap of those residuals, and a KKT-accuracy profile (fraction of problems whose worst residual is below tau).
- For multi-dataset runs, a `## By Dataset` section that re-emits the headline tables (solver metrics, failure rates, shifted geomean, KKT summary) sliced per dataset. The cross-dataset aggregates above the section remain the primary view.
- An artifact index listing every CSV and plot written by the report command.

Programmatic use:

```python
from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean

df = load_results("results/<run_id>")
summary = solver_summary("results/<run_id>")
profile = performance_profile(df, metric="run_time_seconds")
geomean = shifted_geomean(df, metric="run_time_seconds")
```

Status handling:

- By default, only `optimal` counts as successful for performance profiles and
  shifted geometric means. `optimal_inaccurate` and other inaccurate statuses
  are penalized because the solver did not hit the requested target.
- Failed, skipped, timeout, and solver-error statuses receive a large penalty.
- You can pass custom `success_statuses` and `max_value` in Python.

## Adding a New Dataset

Dataset adapters live under `solver_benchmarks/datasets/`.

Implement a class derived from `Dataset`:

```python
from pathlib import Path

from solver_benchmarks.core.problem import QP, ProblemData, ProblemSpec
from solver_benchmarks.datasets.base import Dataset


class MyDataset(Dataset):
    dataset_id = "my_dataset"
    description = "My custom benchmark dataset."

    def list_problems(self) -> list[ProblemSpec]:
        return [
            ProblemSpec(
                dataset_id=self.dataset_id,
                name="problem_1",
                kind=QP,
                path=Path("..."),
                metadata={"source": "..."},
            )
        ]

    def load_problem(self, name: str) -> ProblemData:
        qp = {
            "P": P,       # scipy sparse n x n
            "q": q,       # numpy vector n
            "r": 0.0,     # objective constant
            "A": A,       # scipy sparse m x n
            "l": l,       # numpy vector m
            "u": u,       # numpy vector m
            "n": n,
            "m": m,
            "obj_type": "min",
        }
        return ProblemData(self.dataset_id, name, QP, qp)
```

Register it in `solver_benchmarks/datasets/registry.py`:

```python
DATASETS["my_dataset"] = MyDataset
```

QP convention:

```text
minimize    0.5 x' P x + q' x + r
subject to  l <= A x <= u
```

Cone convention:

```python
problem = {
    "P": None,
    "A": A,
    "b": b,
    "q": c,
    "r": 0.0,
    "cone": {"z": 10, "l": 50, "q": [20], "s": [30]},
    "n": A.shape[1],
    "m": A.shape[0],
    "obj_type": "min",
}
```

Current conic support depends on the solver adapter. SCS supports general conic
data accepted by SCS. PDLP supports only linear zero/nonnegative cones; zero
cones may be keyed as either `z` or `f`.

Add tests:

- Dataset is registered.
- `list_problems()` returns stable, deduplicated problem names.
- Generic visibility through `visible_problems()` respects shared filters such
  as `max_size_mb`.
- A small fixture problem loads with expected dimensions.
- Unsupported solver combinations skip with a warning.

## Adding a New Solver

Solver adapters live under `solver_benchmarks/solvers/`.
Use one module per solver, named `<solver>_adapter.py`; for example, the
commercial optional adapters live in `gurobi_adapter.py`, `mosek_adapter.py`,
and `cplex_adapter.py`. Optional adapters should still be importable without the
optional solver package installed, so keep dependency imports inside
`is_available()` or `solve()`.

Implement `SolverAdapter`:

```python
from pathlib import Path

from solver_benchmarks.core.problem import QP, ProblemData
from solver_benchmarks.core.result import SolverResult
from solver_benchmarks.solvers.base import SolverAdapter, SolverUnavailable


class MySolverAdapter(SolverAdapter):
    solver_name = "my_solver"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        try:
            import my_solver
        except ModuleNotFoundError:
            return False
        return True

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import my_solver
        except ModuleNotFoundError as exc:
            raise SolverUnavailable("Install my_solver to use this adapter") from exc

        from solver_benchmarks.analysis import kkt
        from solver_benchmarks.core import status

        qp = problem.qp
        result = my_solver.solve(qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], **self.settings)

        mapped = status.OPTIMAL  # map raw solver status to a canonical value
        kkt_dict = None
        if mapped in {status.OPTIMAL, status.OPTIMAL_INACCURATE}:
            kkt_dict = kkt.qp_residuals(
                qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], result.x, result.y
            )

        return SolverResult(
            status=mapped,
            objective_value=float(result.objective),
            iterations=int(result.iterations),
            run_time_seconds=float(result.runtime),
            info={"raw_status": result.status},
            trace=result.trace,
            kkt=kkt_dict,
        )
```

Register it in `solver_benchmarks/solvers/registry.py`:

```python
SOLVERS["my_solver"] = MySolverAdapter
```

Adapter expectations:

- Do not import optional solver packages at module import time if they may be missing.
- Use `is_available()` for optional dependency checks.
- Keep solver adapter modules solver-specific rather than grouping unrelated adapters together.
- Return canonical statuses from `solver_benchmarks.core.status`. Map every raw solver status the underlying library can emit (including any `*_inaccurate` variants and infeasibility/unboundedness certificates), and fall back to `SOLVER_ERROR` only for genuinely unrecognized states.
- Populate `SolverResult.kkt` whenever the solver returned a primal/dual pair: use `solver_benchmarks.analysis.kkt.qp_residuals` (or `cone_residuals` for cone solvers) on `OPTIMAL` / `OPTIMAL_INACCURATE`, and the matching `*_infeasibility_cert` helpers when the solver reports a Farkas-style certificate. The KKT plots, summaries, and accuracy profile all read this field.
- Put solver-specific scalar metadata in `info`.
- Put large or per-iteration data in `trace` or explicit artifact files.
- Write extra files under `artifacts_dir`, never global paths.
- Return `skipped_unsupported` when a problem is known to be unsupported.

Add tests:

- Solver appears in `bench list solvers`.
- Missing dependency reports `missing optional extra`.
- Unsupported problem kind skips cleanly.
- A fake or tiny solve writes expected result fields.
- Trace/log artifacts are serialized if the solver exposes them.

## PDLP Notes

PDLP uses OR-Tools directly. No CVXPY code or CVXPY reductions are used.
By default the adapter keeps OR-Tools on the pure PDLP path: Glop presolve,
feasibility polishing, and the diagonal-QP trust-region helper are disabled
unless a run config explicitly opts into them. A default PDLP run therefore
requires only the OR-Tools dependency installed by the `pdlp` extra.

Install:

```bash
pip install -e ".[pdlp]"
```

Supported inputs:

- LP datasets represented as QPs with `P.nnz == 0`, such as NETLIB and MIPLIB
  root LP relaxations.
- Simple linear conic problems with only zero and nonnegative cones.
  Zero cones may use either `z` or `f`.

Unsupported inputs:

- QPs with nonzero `P`.
- SDP, SOC, rotated SOC, and other non-LP cones.

Settings:

```yaml
settings:
  time_limit_sec: 120
  use_glop: false  # default; set true only to opt into OR-Tools Glop presolve
  parameters_text: |
    termination_criteria {
      eps_optimal_absolute: 1e-6
      eps_optimal_relative: 1e-6
    }
```

PDLP artifacts:

- `pdlp_solve_log.textproto`
- `pdlp_response.textproto`

## Testing

CI runs the same checks on pull requests and pushes to `master`.
The scheduled run executes every Monday at 06:00 UTC. GitHub only evaluates
scheduled workflows from the repository default branch, so keep the workflow on
the default branch if weekly runs are required.

Run the test suite:

```bash
pip install -e ".[test,all]"
pytest
```

Current tests cover:

- Config parsing and config hash changes.
- Dataset and solver registration.
- Missing external-data errors and `bench data status` output include exact
  `bench data prepare ...` / `bench run ... --prepare-data` commands.
- Synthetic QP loading.
- Generic runner result writing and resume behavior.
- Generic `max_size_mb` filtering across dataset visibility, CLI problem
  listing, data status, runner selection, and analysis completion/missing
  calculations.
- Real external dataset download smoke tests for the `bench data prepare`
  command are marked `network` and run in the scheduled/manual GitHub Actions
  workflow. The normal PR matrix excludes `network` tests so routine CI remains
  deterministic.
- Structured warning events for unsupported combinations.
- Subprocess stdout/stderr capture.
- Worker trace serialization.
- Generated Markdown reports from `bench report`.
- Real solver smoke coverage on a small QP and LP for every open-source adapter
  (OSQP, SCS, Clarabel, QTQP, ECOS, CVXOPT, HiGHS, ProxQP, PIQP, plus PDLP on
  the LP), with KKT residuals checked against tight tolerances.
- SDPA cone smoke coverage on a small linear cone LP with KKT residuals.
- Multi-solver agreement tests on a small SDP with PSD cones (SCS, Clarabel,
  SDPA, CVXOPT) — guarding against PSD-vec layout drift.
- Primal- and dual-infeasibility certificate checks for SCS, Clarabel, OSQP,
  and ECOS against deliberately infeasible / unbounded LPs.
- Adapter status-mapping unit tests (e.g. `test_scs_status_val_mapping`,
  `test_sdpa_phase_mapping`) that pin the raw → canonical status translation
  for every documented solver code.
- Loading JSONL/Parquet-backed result tables.
- Solver summary output.
- Performance profile and shifted geometric mean calculations.
- CLI `summary`, `profile`, and `geomean` commands.
- PDLP registration and clean skip behavior when unavailable or given non-LP data.
- PDLP linear-cone handling for both `z` and `f` zero-cone keys.

GitHub Actions runs additional matrix workflows that install each open-source
solver in isolation and exercise its adapter via `scripts/ci_solver_smoke.py`,
so a packaging or import regression in one extra cannot hide behind another
solver's tests. A separate `commercial-adapters` job confirms that the CPLEX,
Gurobi, and Mosek adapters register and report `is_available() == False`
without their (proprietary) dependencies installed.

Recommended tests for new contributions:

- Unit tests for parser/conversion logic.
- A tiny deterministic problem fixture.
- One end-to-end run using `synthetic_qp` or another small dataset.
- A resume test if result-writing behavior changes.
- CLI tests for any new user-facing command.
- Optional-dependency tests for MOSEK/GUROBI must not require the dependency or a
  license to be installed.

## Development Notes

Run output under `results/` is intentionally not part of source code. Keep large
generated outputs out of commits unless there is a deliberate benchmark-results
artifact policy. New analysis should use run directories and
`solver_benchmarks.analysis`.

The maintained package should avoid CVXPY imports. If a legacy parser still has
a CVXPY helper method, ensure the import is lazy and not used by the maintained
path.

## Roadmap

High-value next steps:

- More direct SDPLIB/DIMACS validation tests with small fixtures.
- Richer plotting commands that write PDFs/PNGs, not only CSV profile data.
- Cross-run comparison commands.
- Solver environment metadata capture: versions, git SHA, platform, CPU info.
- Optional exact-solution/reference-objective checking where datasets provide it.
- Parallel result-write batching for very large runs.
