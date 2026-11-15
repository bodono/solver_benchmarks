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
including QTQP, cpo3, SCS, OSQP, Clarabel, ECOS, CVXOPT, HiGHS, PIQP, OR-Tools/PDLP,
PyYAML, and pytest. ProxQP and SDPA are included on Python versions where their
wheels are available; currently they are skipped on Python 3.14 by dependency
markers. cpo3 publishes wheels for Linux x86-64, Windows x86-64 and macOS; on
other platforms (e.g. Linux aarch64) pip builds it from source, which needs a
Rust toolchain.

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
pip install -e ".[cpo3]"
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
cpo3      available               cone,qp
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
| Kennington | `kennington` | LP as QP | Standard NETLIB Kennington subset (16 instances). Bundled in the repo as decoded MPS — NETLIB hosts these in EMPS (compressed-MPS) format, which HiGHS cannot read, so the prepare path copies from the bundled checkout instead of fetching from the network (the EMPS decoder in `transforms/emps.py`, used for Mittelmann's EMPS files, would also decode the NETLIB originals). |
| Liu-Pataki | `liu_pataki` | Cone/SDP | Bundled infeasible and weakly infeasible SeDuMi `.mat` SDP instances; converted to canonical PSD triangle cones. |
| MIPLIB root LP relaxation | `miplib` or `miplib_lp_relaxation` | LP as QP | Integrality is ignored; root-node LP relaxation only; downloads official MIPLIB 2017 benchmark `.mps.gz` files on request. |
| QPLIB | `qplib` | QP | Uses QPLIB parser without CVXPY; supports category filters such as `subset: ccb`. A download is parsed before it is cached, and only a transport failure (not an HTTP error such as a 404) is retried over plain HTTP. |
| Mittelmann | `mittelmann` | LP as QP | External ASU lptestset downloads; default prepare downloads `qap15`. |
| Mittelmann SDP | `mittelmann_sdp` | Cone/SDP | Mittelmann's SDP test set in SDPA-S sparse format; downloads from `plato.asu.edu`. The default subset is truss topology design (`trto3`, `buck3`), `rose13`, `cnhil8`, `biggs` and one maxcut relaxation (`G40mc`, a PSD block of order 2000); some take a minute or more. Each file's problem `min c'x` s.t. `sum_k x_k F_k - F_0` PSD is posed, as for SDPLIB. A plain `.dat-s` is preferred to a `.dat-s.gz` of the same name. |
| SDPLIB | `sdplib` | Cone/SDP | Reads converted `.jld2` files (or original `.dat-s` files placed beside the archive); requires `h5py`. Problems are posed with the block structure the original SDPLIB files declare (several semidefinite blocks, and diagonal blocks as nonnegative rows); the converted files store each constraint as one matrix and record the structure apart, as `blockvec`. `dataset_options.subset: feasible` (default), `infeasible` (the four instances `infd1`, `infd2`, `infp1`, `infp2`) or `all`. `dataset_options.blocks: merged` poses each problem read from the archive as one semidefinite block of the full order instead, as results from before the loader gave the block-diagonal problems their declared blocks were produced (an original `.dat-s` file has no merged form). |
| TSPLIB MaxCut SDP | `tsplib_sdp` | Cone/SDP | Goemans-Williamson MaxCut SDP relaxations of TSPLIB instances; supports EUC_2D/3D, MAN/MAX, GEO, ATT, and EXPLICIT weight types. |
| DIMACS | `dimacs` | Cone | Reads SeDuMi-format `.mat` and `.mat.gz` and poses the file's dual, `maximize b'y` subject to `c - A'y in K*`, as `minimize -b'y`: rows in SeDuMi's order (free, linear, second-order, PSD), each PSD block as the scaled lower triangle of its symmetric part. The recorded objective is minus the published optimum. `filtinf1`, the library's infeasible variant of `filter48_socp`, is recorded as expected infeasible. Rotated Lorentz cones are not yet supported. |
| CBLIB | `cblib` | Cone | Downloads CBF files; the parser handles continuous linear (`L=`/`L+`/`L-`/`F`), second-order (`Q`), and exponential (`EXP`/`EXP*`) cone instances. Instances using other cone kinds (PSD, integer, etc.) are *hidden* from `list_problems()` by default; pass `dataset_options.include_unsupported=true` to surface them with `metadata["supported"]=False`. Listing reads only each file's header (the sections before the first coordinate section); a file that cannot be read is skipped with a warning, and a plain `.cbf` is preferred to a `.cbf.gz` of the same name. A maximisation is posed as the minimisation of minus its objective, with `original_obj_type: max` in the metadata. |
| MPC QP Benchmark | `mpc_qpbenchmark` | QP | Downloads structured MPC QPs from `qpsolvers/mpc_qpbenchmark`, at a pinned commit (`MPC_QPBENCHMARK_REVISION`). `.npz` files are read without unpickling: an object array is accepted only as a `None` placeholder for a missing array. `P` may be the whole symmetric Hessian or one triangle of it, which is mirrored; a `P` that is neither is rejected. |
| LIBSVM-derived QP | `libsvm_qp` | QP | SVM dual or Markowitz portfolio QPs built from LIBSVM datasets; produces realistic ML/finance-shaped QPs absent from classical test sets. |
| DC OPF | `dc_opf` | QP | DC Optimal Power Flow problems from evaluated MATPOWER cases (`.mat` files downloaded from a pinned snapshot into `dc_opf_data/<revision>/`; the `.m` files bundled in `dc_opf_data/` are MATPOWER's sources of the default cases, kept for reference and not read). An LP when every generator cost is linear or piecewise linear, a QP when costs are quadratic. Sparse power-balance equalities, line-flow inequality limits, generator box bounds. What a case marks as out of service is left out, as MATPOWER does: an isolated bus (type 4) with its demand and the branches and generators attached to it, and a generator that is off, which stays as a variable fixed at zero and adds nothing to the cost. |
| CUTEst QP exports | `cutest_qp` | QP | Local export target only; no automatic CUTEst downloader. Read with the MPC `.npz` schema: `P` dense, the whole symmetric Hessian or one triangle. Preparing checks that the requested exports exist; when some are missing it names them and writes `EXPORT_README.md`, with the schema, into `cutest_qp_data/`. |
| Synthetic smoke test | `synthetic_qp` | QP | Tiny deterministic test problem. |
| Synthetic cone smoke test | `synthetic_cone` | Cone | Tiny deterministic conic LP test problem. |

Every problem is posed as a minimisation. An MPS file that HiGHS reads as a
maximisation (NETLIB, Kennington, MIPLIB and Mittelmann are read from MPS,
through HiGHS) is posed as the minimisation of minus its objective, as CBLIB
and QPLIB maximisations are, and the objective recorded for it is that of the
minimisation. One limit: HiGHS reads the sense from an `OBJSENSE` section
(`MAX` or `MAXIMIZE` on the next line) or from `OBJSENSE MAX` on one line
before `ROWS`, and takes other spellings (`OBJSENSE MAXIMIZE` on one line, an
`OBJSENSEMPS` section) for a minimisation without a warning. A constant term
in the file's objective is part of the recorded objective (of the bundled
files only NETLIB `e226` has one, 7.113: its recorded optimum is -11.6389,
where the Netlib table lists -18.7519 without the constant). The sense the
file states is recorded in the problem's metadata as `original_obj_type`.

An MPS file read by these datasets may also carry a quadratic objective
(`QUADOBJ`, `QMATRIX` or `QSECTION`, in a file named `.mps`); HiGHS reads it
and it becomes the problem's `P` (none of the bundled files has one). A column
whose only bound is an `UP` bound below zero is taken as free below, as CPLEX,
Gurobi and lp_solve read it, with a warning; HiGHS alone keeps the default
lower bound 0, which makes the problem infeasible. A lower bound the file
states is kept. Compressed files are decompressed in pieces to a temporary
file for HiGHS, never whole in memory.

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
  full remote index is prepared. If the subset names problems, those are
  prepared: a single name, a list or a comma-separated string, for the datasets
  whose `subset` takes problem names (`qplib`, where a category such as `ccb`
  also works, `mpc_qpbenchmark`, `cutest_qp`, `cblib`, `mittelmann_sdp`,
  `tsplib_sdp`, `dc_opf` and `libsvm_qp`). Otherwise the dataset's small
  default subset is prepared. Requested problems that are already present are
  not requested again, and a failed preparation stops the run with an error
  naming the dataset.
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
bundled checkout rather than the network — see the Kennington note in
the dataset table above.)

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

Mittelmann preparation accepts both bzip2-compressed MPS and NETLIB EMPS
files. EMPS files are decoded in Python, with checksum validation, and the
resulting MPS is checked with HiGHS before it is cached. Running preparation
again also repairs EMPS files saved with an `.mps` suffix by older versions.
Already cached standard MPS files are reused without a full validation parse.
An empty remote index is an error for `mittelmann --all`; the default
preparation command intentionally downloads only `qap15`.

SDPLIB is different from the download-backed datasets above: the maintained
adapter reads converted `.jld2` files, or extracts them from
`problem_classes/sdplib_data/sdplib.tar` when that archive is present. It does
not download and convert the original SDPLIB files automatically. If the
converted archive is absent, restore `sdplib.tar` or place converted `.jld2`
files in `problem_classes/sdplib_data`. A problem read from the archive is
extracted once into `sdplib_data/.cache/`, through a temporary file renamed
into place when complete, so concurrent workers never read a partial file; a
cached file whose size differs from the archive member is extracted again.
The list of archive members is kept in `.cache/sdplib.tar.index.json` and
rebuilt when the archive's size or modification time changes, since listing
the compressed archive means decompressing all of it. Extracted files and the
index get the mode a plain write would give them (0o666 less the umask), so on
a shared data directory other users can read what one user extracted.

Three members of the archive are damaged and cannot be read: `maxG55`,
`maxG60` and `thetaG51` (stretches of each file are zeros). Their original
SDPLIB files are bundled beside the archive as `.dat-s.gz`, and an original
file takes precedence over the archive member of the same name, so these three
are read from the originals. If you keep the archive somewhere else
(`data_root`), take the three `.dat-s.gz` files with it.

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
| All file-backed datasets | `data_root=<path>` | Read the dataset's files under `<path>` instead of `<repo_root>/problem_classes`. In a config file, a relative path is resolved under the repository root, like `run.output_dir`; `bench run --data-root` resolves it against the current directory. |
| `netlib` | `subset=feasible` or `subset=infeasible` | Select NETLIB feasibility subset. |
| `sdplib` | `subset=feasible`, `subset=infeasible` or `subset=all` | Select the SDPLIB feasibility subset (default `feasible`). |
| `sdplib` | `blocks=declared` or `blocks=merged` | Pose each problem with SDPLIB's declared blocks (default), or as the single merged block the converted archive stores. 39 of the 92 problems differ: `arch*`, `control*`, `hinf*`, `truss*` and `ss30`. Results for those 39 are not comparable between the two forms (statuses, iterations and times all change), so a run directory started before the loader gave these problems their declared blocks resumes its SDPLIB rows only under `blocks=merged`. Under the default they are run again and the merged-form rows move to `results.superseded.jsonl`, so use a new run directory to keep both forms. `bench merge` warns when the runs it combines posed a problem with different numbers of rows. |
| `qplib` | `subset=default`, `all`, or a category (`ccb`, `ccl`, `dcl`, ...), an ID, or a list or comma-separated string of them | Filter convex QPLIB instances. |
| `liu_pataki` | `classification=infeas` or `classification=weak` | Select infeasible or weakly infeasible SDP families. |
| `liu_pataki` | `conditioning=clean` or `conditioning=messy` | Select clean or numerically messier Liu-Pataki instances. |
| `liu_pataki` | `constraint_count=10` or `constraint_count=20` | Select Liu-Pataki instances by equality count. |
| `liu_pataki` | `block_dim=10` | Select Liu-Pataki instances by PSD block dimension. |
| `mpc_qpbenchmark` | `subset=default`, `all`, or comma-separated names | Filter downloaded MPC QP `.npz` files. |
| `cutest_qp` | `subset=default`, `all`, or a list or comma-separated names | Filter locally exported CUTEst QP `.npz` files. |
| `cblib` | `subset=default`, `all`, or comma-separated names | Filter downloaded CBF files. |
| `cblib` | `subset_kind=expcone`, `socp`, or `lp` | Filter CBF instances by cone shape rather than by name. Combines with `subset` (both filters must pass). |
| `cblib` | `include_unsupported=true` | Show downloaded CBF files the parser would otherwise hide. |
| `mittelmann_sdp` | `subset=all` (no filter, the default), `default`, or comma-separated names | Filter to a subset of locally available SDPA-S instances. |
| `tsplib_sdp` | `subset=all` (no filter, the default), `default`, or comma-separated names | Filter to a subset of locally available TSPLIB instances; a plain `.tsp` is preferred to a `.tsp.gz` of the same name. |
| `dc_opf` | `subset=all` (no filter), `feasible`, `infeasible`, or a list or comma-separated names | Filter to a subset of locally available MATPOWER cases; `feasible`/`infeasible` use the snapshot's feasibility labels. |
| `libsvm_qp` | `kind=svm_dual` (default) or `markowitz` | Choose the QP shape derived from the LIBSVM data. |
| `libsvm_qp` | `kernel=linear` (default) or `rbf` | SVM-dual kernel choice. Linear gives a low-rank `Q` (rank ≤ feature dimension). |
| `libsvm_qp` | `gamma=<float>` | RBF kernel bandwidth; defaults to `1 / num_features`. |
| `libsvm_qp` | `C=<float>` | SVM regularization upper bound on dual variables (default `1.0`). |
| `libsvm_qp` | `risk_aversion=<float>` | Markowitz `λ` in the objective `½ w'Σw − λ μ'w` (default `1.0`). |
| `libsvm_qp` | `max_samples=<int>` | Deterministically subsample to this many rows (default `200`): rows `0, k, 2k, ...` with `k = n // max_samples`, the first `max_samples` of them, so a file with fewer than twice that many rows contributes its first `max_samples` rows. |
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
  such as `sdplib.tar`, where the threshold applies to each archive member
  (an original SDPLIB file that replaces a member is filtered by the member's
  size, not its own),
  and for MPS datasets that have both `.mps` and `.mps.gz` encodings of the
  same problem, where the smallest available local encoding is used for the
  size decision.
- The `synthetic` dataset has no backing file and is unaffected.

## Supported Solvers

Current maintained adapters:

| Solver | ID | Supported types | Notes |
|---|---|---|---|
| QTQP | `qtqp` | QP | Converts QP bounds to nonnegative cone form internally. Needs qtqp >= 0.0.7; tolerances are `tol_feas`, `tol_gap_abs`, `tol_gap_rel` (the harness also accepts `eps`/`eps_abs`/`eps_rel` and the pre-0.0.7 `atol`/`rtol`, translated to whatever the installed version takes). |
| cpo3 | `cpo3` | QP, cone (zero / nonnegative only) | Sparse interior-point solver for LPs and QPs, written in Rust ([cpo3-org/cpo3](https://github.com/cpo3-org/cpo3)); install with the `cpo3` extra (`pip install cpo3`, wheels for Linux x86-64, Windows x86-64 and macOS). QPs are converted to zero / nonnegative cone form like QTQP; cone problems pass through when they use only zero and nonnegative cones, and are skipped otherwise (cpo3 0.0.1 does not solve second-order or PSD cones). Tolerances are Clarabel-style `tol_feas`, `tol_gap_abs`, `tol_gap_rel`, `tol_infeas_abs`, `tol_infeas_rel`; `eps` / `eps_abs` / `eps_rel` map onto them and `max_iter` onto `max_iters`. `linear_solver` is `faer` (default, always available), `accelerate`, `cholmod` or `pardiso`; a backend the installed build or host cannot load stops the solve with a `SolverUnavailable` worker error instead of being scored as a solver failure. `threads` and `time_limit` are forwarded. |
| SCS | `scs` | QP, cone | Supports SCS box-cone form for QPs and native conic data. |
| Clarabel | `clarabel` | QP, cone | Uses nonnegative cone conversion for QPs and native zero/nonnegative/SOC/PSD cones. |
| OSQP | `osqp` | QP | Direct QP adapter. |
| ECOS | `ecos` | QP, cone (LP/SOCP/expcone) | Interior-point conic solver. QPs are reformulated to SOCP via the standard epigraph trick (Cholesky for PD `P`, eigendecomposition for rank-deficient PSD); the SOC dim is `rank(P) + 2`. PSD cones not supported. |
| CVXOPT | `cvxopt` | QP, cone (LP/QP/SOCP/SDP) | Interior-point solver supporting QPs natively plus PSD cones (canonical PSD-vec layout converted to BLAS unpacked-L automatically). Exponential cones not supported. CVXOPT options live on a global dict; the adapter snapshots and restores it per solve so **sequential** solves in the same process don't leak knobs into each other. This is *not* thread-safe — concurrent in-process solves would still race on the global. The benchmark runner uses subprocess-level parallelism, so this per-process sequential contract is sufficient. |
| PDLP | `pdlp` | LP-only QP, simple linear cone | Uses OR-Tools directly, no CVXPY. |
| HiGHS | `highs` | QP | Direct LP/QP adapter through `highspy`; strong LP baseline. |
| ProxQP | `proxqp` | QP | Direct QP adapter through ProxSuite; optional on Python < 3.14 while wheels are available. ProxSuite's default stopping test checks only the primal and dual residuals and can call a point with a large duality gap solved, so the adapter turns on its duality-gap test (`check_duality_gap: true`, with `eps_duality_gap_abs` / `eps_duality_gap_rel` taken from `eps_abs` / `eps_rel` unless given); the values used are on `info`. With `eps_rel` unset or 0 (ProxQP's default), `eps_abs` is the relative gap tolerance too: an absolute-only gap test is out of reach of double precision when the objective is large (at `eps_abs: 1.0e-9`, Maros-Meszaros QSCAGR25, objective 2e8, ran to `max_iter`, and QPCBOEI1 ended in a false infeasibility). Set `check_duality_gap: false` for ProxSuite's own default, which ProxQP results recorded before this change used. |
| PIQP | `piqp` | QP | Direct sparse/dense QP adapter. |
| SDPA | `sdpa` | cone | Linear conic adapter through `sdpa-python`; useful for SDP subsets, optional on Python < 3.14 while wheels are available. Problems with second-order cones are skipped: `sdpa-python` ends the process on them. |
| GUROBI | `gurobi` | QP | Optional extra; requires `gurobipy` and license. When presolve can only say "infeasible or unbounded", the adapter re-solves with `DualReductions=0` (Gurobi's documented remedy) so the row gets a definite status; `info.inf_or_unbd_resolved` marks it. |
| MOSEK | `mosek` | QP | Optional extra; requires Mosek package and license. |
| CPLEX | `cplex` | QP | Optional extra; requires IBM CPLEX package and license. After an "infeasible or unbounded" status the adapter re-solves with `preprocessing.reduce: 0` (`info.inf_or_unbd_resolved`). `iterations` is the barrier count when barrier ran (CPLEX's default for QPs), else the simplex count; both are on `info`. |

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

Important fields (a key not listed here is rejected with a suggestion, as are
unknown solver and dataset names; an empty section such as `run:` with nothing
under it counts as empty). Numbers such as `1e-6` and `1.0e6` are read as floats,
as YAML 1.2 and JSON read them:

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
| `run.resume` | If true, completed `(dataset, problem, solver_id)` triples whose solve settings match the config are not rerun (see [Running Benchmarks](#running-benchmarks)). If false, `bench run --run-dir` refuses a directory that already has results for the config's datasets and solvers. |
| `run.max_worker_error_retries` | Maximum retries of a compatible `worker_error` result across resumes; default 2, 0 disables retries. Increase it to retry again after fixing a persistent failure. |
| `run.timeout_seconds` | Default time limit per solve, in seconds; `0` or unset means none. The runner enforces it; it is not passed to the solver (set the solver's own time-limit setting for that). A worker still running after the limit plus a grace for start-up and problem loading (equal to the limit, at least 5 s and at most 30 s) is killed, and the solve is recorded as `time_limit` with `run_time_seconds` equal to the limit. A solve that returns an optimal, inaccurate or infeasibility status after the limit is recorded as `time_limit` too, with the solver's status kept in `info.status_before_timeout`. |
| `run.auto_prepare_data` | If true, prepare the missing requested data before listing and solving: the problems `include` names, else those `dataset_options.subset` names (for the datasets whose subset takes problem names), else the dataset's small default subset. Problems already present are not requested again, and a failed preparation stops the run with an error naming the dataset. The details are under [Supported Datasets](#supported-datasets). |
| `solvers[].id` | Unique label for this solver variant. Used in output paths, so two ids may not differ only in letter case. |
| `solvers[].solver` | Solver adapter ID, e.g. `scs`, `qtqp`, `pdlp`. |
| `solvers[].settings` | Passed directly to the solver adapter. |
| `solvers[].timeout_seconds` | Optional per-solver override of `run.timeout_seconds`. |
| `solvers[].sweep` | Optional mapping of setting name to a non-empty list of values. Expands to the full Cartesian product. |
| `solvers[].id_template` | Optional Python format string used to name expanded sweep variants. |

The same solver may appear many times with different `id` and `settings`.
Solver output is verbose by default and is captured in each solve's
`stdout.log`/`stderr.log`; set `verbose: false` in a solver's settings to run it
quietly. To keep full solver output in the log files without printing it to the
terminal, pass `--no-stream-output` to `bench run`.

### Shared Solver Settings

A few setting names mean the same thing for every adapter:

- `time_limit` (or `time_limit_sec` / `time_limit_secs`): the solver's own
  time limit in seconds. It must be a finite number greater than zero; `0` is
  refused because solvers disagree on what it means (no limit for SCS, stop
  before the first iteration for Clarabel, cpo3, HiGHS and PDLP, an invalid
  setting for OSQP). Leave it out to run without a limit. Adapters whose solver
  has no time limit record `time_limit_ignored` on the result's `info`.
- `threads` (or `num_threads`): the solver's thread count, forwarded where the
  solver has one (e.g. PIQP and PDLP `num_threads`, HiGHS, Gurobi, CPLEX,
  cuOpt, cpo3) and recorded as `threads_ignored` otherwise (e.g. OSQP, ProxQP,
  QTQP).
- `max_iter`: the iteration cap, under the solver's own name where it differs
  (HiGHS: every algorithm's `*_iteration_limit`; Gurobi: `BarIterLimit` and
  `IterationLimit`; CPLEX: `barrier.limits.iteration` and
  `simplex.limits.iterations`; cuOpt: `iteration_limit`; cpo3: `max_iters`). A
  native name given alongside wins.
- `eps_abs` / `eps_rel` (and `eps`, which sets both): the absolute and relative
  tolerances, native for OSQP, PIQP, ProxQP and PDLP and translated for QTQP,
  cpo3 and cuOpt. HiGHS, Gurobi and CPLEX have no such pair and refuse these
  names; set their own tolerances instead.

The QP adapters (OSQP, PIQP, ProxQP, QTQP, HiGHS, PDLP, cpo3, Gurobi, CPLEX,
cuOpt) refuse a setting they do not know, or a value their solver rejects, with
a `ValueError` naming it (recorded as a `worker_error`), so a misspelled option
cannot silently leave a solver on its defaults and a bad value is not scored
as a solver failure. PIQP and PDLP report a rejected value as a status
(`PIQP_INVALID_SETTINGS`, `TERMINATION_REASON_INVALID_PARAMETER`), which the
adapter turns into that error. QTQP's `solve()` raises the same exception for a
bad value as for problem data it cannot take, so the adapter checks the ranges
it enforces first. ProxSuite checks no values, so the ProxQP adapter itself
refuses a negative or non-finite `eps_*` tolerance, a `rho` / `mu_eq` / `mu_in`
that is not positive, and a `max_iter` below 1. HiGHS's on/off options
(`presolve`, `run_crossover`, `parallel`, ...) also accept a YAML boolean,
since YAML reads an unquoted `off` as `false`. Translated aliases are listed on
the result's `info.settings_translated`.

In the QP form `l <= A x <= u`, every adapter named above (MOSEK keeps its own
`1e20` cutoff) treats a bound with magnitude `>= 1e19` on the open side as infinite (the datasets write `+-1e20`, sometimes
with a few ULPs of error; the transform layer and the Maros-Meszaros and QPS
readers use the same cutoff), and passes true infinities to its solver. Rows
with no finite bound constrain nothing and are not passed to OSQP, PIQP,
ProxQP, HiGHS, PDLP, Gurobi or CPLEX; their dual is reported as zero.

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
an environment matrix. This is intentional: resume uses
`(dataset, problem, solver_id)`, and duplicate IDs would make two distinct
configurations look like the same completed solve.

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
| `configs/maros_meszaros_qp_solvers.yaml` | Maros-Meszaros QP run comparing OSQP, SCS, Clarabel, QTQP, and cpo3. |
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
`results.jsonl` are skipped when their solve settings match the current config:
the solver, its settings, the effective timeout and the dataset options, except
`data_root` and `max_size_mb`, which only say where the files are and which
problems are listed. Rows written by earlier releases, which also hashed those
two options, still count when everything else matches, also after either option
changes: the first resume in such a directory works out, from the manifest the
earlier release left, which current signature those rows' signatures stand for,
and records it in the manifest for later resumes. A row solved under other settings is superseded: the
triple is solved again, and at the end of the run the old rows whose
replacements were written move from `results.jsonl` to
`results.superseded.jsonl` in one rewrite, so reports only see results for the
current settings. Until then, and after a crash that skips that step, both rows
are in `results.jsonl`; the next resume moves the old one out. The superseded
attempt's logs stay where its row points; the new attempt writes into a
`rerun-N` subdirectory. Rows with `worker_error` are retried automatically
when their solve settings match the current config, up to
`run.max_worker_error_retries` times (default 2). This bounds both crashed workers
and deterministic exceptions reported as `worker_error`. The latest error stays
visible when the limit is reached; increase the limit after fixing the cause.
Resuming on a host without the solver leaves pending errors and retry budgets
unchanged. Problems skipped because the solver was not installed (their rows
carry `metadata.skip_reason: solver_unavailable`) are solved on the first resume
where it is; skips the adapter reports for an unsupported problem stay final. A
retry atomically replaces
the compatible error rows, so reports count one result for that solve. Previous
logs and solver artifacts remain intact; new artifacts go into a `retry-N`
subdirectory containing a `previous_results.jsonl` snapshot of the replaced
errors. New solver variants or newly included problems are appended.

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
settings but keep the same `id`, the old rows are superseded: every problem is
solved again with the new settings and the old rows move to
`results.superseded.jsonl`. To compare the two settings, give the new variant a
new `id`, such as `clarabel_tight` instead of reusing `clarabel_default`.

With `resume: false`, `bench run --run-dir` refuses a directory that already has
results for the config's datasets and solvers instead of appending a second set
of rows; start a new run directory instead.

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
| `metadata.runtime.library_versions` | Installed numpy and scipy versions, which ship the BLAS/LAPACK most solvers use. |
| `metadata.runtime.usable_cpu_count` | CPUs the solve subprocess may run on (its CPU affinity). |
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
`(dataset, problem, solver_id)`:

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
| `bench run CONFIG_PATH` | `--run-dir PATH`, `--repo-root PATH` (default: the current directory if it has `problem_classes`, else the checkout the package was installed from, as for the other commands), `--data-root PATH`, `--prepare-data` default false, `--stream-output` / `--no-stream-output`, `--environment-id ID`, `--environment-metadata JSON` | Execute a benchmark config. Without `--run-dir`, creates a new immutable run directory named from `run.name` or the config filename stem plus a readable UTC timestamp, and copies the source config into the run directory. With `--run-dir`, resumes/appends to that run subject to `resume: true`. `--data-root` sets `data_root` for every dataset in the config. Without `--prepare-data`, missing external data is reported with exact preparation commands instead of being downloaded implicitly. `--no-stream-output` keeps solver stdout/stderr in per-solve log files but does not tee it to the terminal; benchmark progress still prints. The environment flags are normally used by version-comparison workflows and are recorded in result metadata. |
| `bench env run CONFIG_PATH` | `--run-dir PATH`, `--repo-root PATH`, `--stream-output` / `--no-stream-output` | Execute an environment matrix config. Each environment supplies a Python executable, optional install commands, metadata, and solver variants; all results are written into one run directory, with the source environment config copied into it. `--no-stream-output` is forwarded to each child benchmark run. |
| `bench summary RUN_DIR` | `--repo-root PATH` | Print solver metrics, status counts, and run completion information. Solver metrics count each `(dataset, problem, solver)` once (a duplicated solve counts as its successful attempt); run time and iteration totals cover every solve, means, medians and maxima the successful ones. |
| `bench failures RUN_DIR` | `--repo-root PATH` | Print success/failure rates by solver, one count per solve. Planned solves with no recorded row count as `not_attempted` failures, as in the report (`bench summary` counts them the same way). Only `optimal` counts as success by default; on problems known to be infeasible or unbounded only the matching certificate does (see [Known outcomes](#known-outcomes)). |
| `bench missing RUN_DIR` | `--repo-root PATH` | Print missing `(solver, dataset, problem)` results relative to the run manifest and dataset filters. |
| `bench profile RUN_DIR` | `--metric FIELD` default `run_time_seconds` | Write Dolan-More performance profile data to `performance_profile_<metric>.csv`. |
| `bench geomean RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--shift VALUE` default metric-dependent, `--max-value VALUE` optional, `--success-only` default false, `--repo-root PATH` | Write shifted geometric means into the run directory. Time-metric failures cost three times the manifest's time limit unless `--max-value` overrides it; use `--success-only` for solved-problem-only geomeans. |
| `bench plot RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--output-dir PATH` default run dir, `--max-value VALUE` optional, `--repo-root PATH` | Write PNG plots for Dolan-More profile, cactus plot, pairwise scatter, performance-ratio heatmap, shifted geomean, status heatmap, failure rates, and three KKT-residual plots (per-solver boxplot, problem-by-solver heatmap, and KKT-accuracy profile). |
| `bench report RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--output-dir PATH` default `RUN_DIR/report`, `--max-value VALUE` optional, `--repo-root PATH` | Write a complete report directory containing CSV tables, PNG plots, and a generated Markdown report. |
| `bench merge OUT_DIR RUN_DIR...` | `--overwrite` | Concatenate sharded runs into a new run directory; duplicated `(dataset, problem, solver)` rows are reported, not resolved. |
| `bench kkt-verify RUN_DIR` | `--tol VALUE` required, `--output-dir PATH`, `--allow-missing`, `--promote`, `--overwrite` | Write a copy of the run in which every claimed optimum whose relative KKT residuals (cone distances and absent-bound multipliers included) exceed `--tol` becomes `optimal_inaccurate`, and every `primal_infeasible` / `dual_infeasible` whose scaled certificate residual exceeds it becomes the matching `*_inaccurate` status. Claimed optima with residuals missing are demoted unless `--allow-missing`. Rows whose only unknown residuals are distances to cones the harness cannot project (complex PSD), and infeasibility claims whose certificate cannot be checked (the adapter records none, as HiGHS, PIQP, ProxQP, SDPA and the commercial adapters do not, or a cone cannot be projected), are kept whatever `--allow-missing` says, with the reason in `metadata.kkt_verify` and a warning that counts them by solver. At a strict `--tol` this favours adapters that record no certificate in infeasibility-heavy comparisons: a recorded certificate can fail the check (Clarabel's genuine NETLIB certificates reach 6e-5), an unrecorded one cannot. `--promote` turns inaccurate, iteration- or time-limited solves whose point passes into `optimal`. Unparseable lines (a torn last line) are skipped by both commands, and the output directory is replaced only once complete. |

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
| `manifest.json` | Run metadata, normalized config, config hash, creation time, and a map from the resume signatures that rows from earlier releases carry to the current ones (`legacy_resume_signatures`). |
| `run_config.yaml` / `run_config.json` | Exact source config file passed to `bench run`, copied into the run directory before solving. |
| `events.jsonl` | Structured warnings/events, including unavailable solvers and unsupported combinations. |
| `results.jsonl` | One JSON record per completed or skipped solve. Human-readable and append-friendly. |
| `results.superseded.jsonl` | Rows a resume replaced because they were solved under other settings (see [Running Benchmarks](#running-benchmarks)). Only present after such a resume; analysis ignores it. |
| `results.parquet` | Flattened results table for pandas/Arrow analysis, written when a run finishes; its presence marks a completed run. Columns whose values mix types across adapters (e.g. `info.raw_status`, an integer for Gurobi and CPLEX, a string elsewhere) are stored as text. If it cannot be written, the error is logged to `events.jsonl` and `results.jsonl` remains the complete record. |
| `payload.json` | Exact payload passed to the subprocess worker. |
| `stdout.log` | Everything emitted to subprocess stdout during a solve. |
| `stderr.log` | Everything emitted to subprocess stderr during a solve. |
| `worker_result.json` | Raw worker output before aggregation. |
| `result.json` | Per-solve result record matching `results.jsonl`. |
| `trace.jsonl` | Iteration trace when the solver adapter exposes one. |

## Logging and Provenance

Each solve runs in a subprocess. This gives:

- Native solver crashes do not kill the main runner.
- Timeouts are enforced per solve: the worker's whole process group is killed, so helper processes a solver started do not outlive it. Ctrl+C, or a SIGTERM to `bench run` / `bench env run` (Slurm's `scancel`, `kill`, a CI job timeout), kills the running workers too, and a parallel run keeps the results that had already finished; a SIGTERM exits with status 143.
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
- QTQP writes `trace.jsonl` when its settings include `collect_stats: true`.
  qtqp computes these per-iteration statistics inside the timed solve, so they
  are not collected by default.
- PDLP writes `pdlp_solve_log.textproto` and `pdlp_response.textproto` when
  OR-Tools is installed and a solve is attempted.

System info: each run's `manifest.json` carries a `system` block captured once
at run start, recording the CPU model / logical and physical core count / the
number of CPUs the process may use (`cpu.usable_count`, smaller than the logical
count under `taskset`, Slurm or a container cpuset) / max frequency, total and available memory, OS / kernel version, Python version,
and the installed numpy / scipy / pandas / pyarrow and registered solver-package
versions under `system.library_versions` (uninstalled optional packages are
recorded as `null`). This describes the parent environment at run start;
`metadata.runtime.solver_package_versions` and `metadata.runtime.library_versions`
(numpy and scipy) on each result remain the source for versions actually used
by workers, including isolated `bench env` runs.
The block is
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
have infinite ratios by default and never enter the solved fraction. Problems
where every solver fails remain in the denominator. The per-problem best is
computed only from successful, finite metrics.

### Known outcomes

Datasets that know a problem's outcome record it as
`metadata.expected_status`: `optimal`, `infeasible` or `unbounded`. Every
success count, failure table, profile, geomean and plot scores rows against
it (runs recorded before this field existed are scored the same way, with the
outcome inferred from the dataset id, problem name and source path):

- `infeasible` (NETLIB `subset: infeasible`, all of Liu-Pataki, SDPLIB
  `infd1`/`infd2`/`infp1`/`infp2`, DIMACS `filtinf1`, DC OPF cases labelled
  infeasible): solved
  only by `primal_infeasible` or `dual_infeasible`. Either certificate proves
  no optimal pair exists, and some instances (NETLIB `cplex1`) are primal and
  dual infeasible at once. Weakly infeasible Liu-Pataki instances are held to
  the same rule.
- `unbounded`: solved only by `dual_infeasible`.
- `optimal`, or no recorded outcome: solved only by a success status
  (`optimal` by default).

So a claimed optimum on an infeasible problem is a failure, and so is a
certificate on a solvable one. Inaccurate certificates and the ambiguous
`primal_or_dual_infeasible` never count, as `optimal_inaccurate` never counts,
and a certificate the harness checked and found invalid (`kkt.valid` false) is
a failure. The check scales the certificate so that its objective term (`b'y`,
the support `u'y⁺ − l'y⁻` of the bounds, or `q'x`) is −1 and then requires
its residuals to be at most `1e-3`: `A'y` with the distance of `y` to the dual
cone (or the weight `y` puts on absent bounds), or `Px` with the distance of
`−Ax` to the cone. `kkt.cert_res` records the largest. That default only
rejects what is clearly not a certificate; `bench kkt-verify --tol` holds
certificates to its own tolerance. Records written before `kkt.cert_res`
existed keep, in reports, the sign-only `kkt.valid` they were stored with
(the objective term is negative); only `bench kkt-verify` rebuilds their
scaled residuals from the recorded absolute ones and checks them.

Generate shifted geometric means:

```bash
bench geomean results/<run_id>
bench geomean results/<run_id> --metric iterations
bench geomean results/<run_id> --success-only
```

The default `bench geomean` output is a penalized shifted geometric mean for
benchmark comparison, not a raw average runtime. Non-`optimal` statuses,
including `optimal_inaccurate`, are assigned the failure penalty before the geometric
mean is computed. Use `bench summary` for totals and success-only means/medians, or
`bench geomean --success-only` for a geomean over successful solves only. The
default failure penalty for time metrics is three times the manifest's
`timeout_seconds`. A 300-second limit gives a 900-second penalty, a 900-second
limit gives 2700 seconds, and an 1800-second limit gives 5400 seconds. When
solvers set their own `timeout_seconds`, the limit is the largest any solver of
the run had (a solver allowed an hour can succeed after more than three times a
60-second run limit, and failing must never cost less than succeeding); a run
whose solvers all set one needs no run-level limit. The penalty
does not depend on which solvers are compared or how slowly successful solves
finished; the geomean warns when an explicit penalty is at or below a
successful value. `--max-value` on `geomean`, `plot`, and `report` supplies an explicit
penalty in the selected metric's units, preserved exactly as given.

Merged runs use each dataset's limit from the embedded source manifests. A
per-dataset table reports its fixed penalty in `max_value`. If a pooled table
combines different limits, `max_value` is empty and `max_value_by_dataset` records
the dataset-to-penalty mapping. Missing, unlimited, conflicting, or lost time
limits require an explicit `--max-value`; `bench geomean` raises an error instead
of inferring a penalty from observed results, and `bench plot` and `bench report`
leave out only the penalized time geomean (the report says why) and write
everything else. Archived derived runs therefore need their source limits
preserved in the manifest, or an explicit override.

Iteration and KKT metrics retain fixed defaults in their own units: iterations
use penalty `1e6` and shift `100`, while the supported KKT residuals
(`kkt.primal_res_rel`, `kkt.dual_res_rel`, `kkt.duality_gap_rel`,
`kkt.comp_slack_rel`, `kkt.comp_slack`) use penalty `1` and shift `1e-12`, so a
residual that is exactly zero counts as `1e-12` instead of dragging the mean of
logs towards the smallest float. Time metrics use shift `10`. `--success-only` requires
no failure penalty or time limit. These rules select the cost of failures;
campaign-specific status filtering and timeout grace rules remain separate.

Missing solves are charged too: run-directory commands use the manifest's
configured solver and problem sets, intersecting includes with the dataset
listing when available. `--repo-root` on `geomean`, `profile`, `plot`, and `report`
selects the repository containing the dataset files. If a dataset directory is
absent, explicit includes without a size filter supply the expected names;
unrestricted selections still require local data. Merged-run selections retain
each source's filters and solver associations when determining planned jobs.
Comparisons then use the combined observed/expected problem population for all
retained solvers, whereas completion reports count only planned jobs.
No-manifest analyses use the union of observed problems crossed with observed solvers. Python callers can
pass `expected=` as a DataFrame of `dataset`, `problem`, and `solver_id` identities
to include completely absent solvers or problems. Duplicate attempts contribute
once per identity, using the best successful result, as in performance profiles;
every per-solve table and plot of a report (solver metrics, failure rates, status
matrix, cross-solver comparison, cactus, KKT tables) resolves them the same way.
A solver that reports no value of the metric on any of its successful solves
(several adapters never set `solve_time_seconds` or `setup_time_seconds`) is left
out of profiles and geomeans on that metric with a warning, instead of having
its solved problems scored as failures; a solver that reports it elsewhere is
still charged for a success without a usable value.

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
- KKT-residual summaries (`kkt_summary.csv` over successful optimality claims, `kkt_certificate_summary.csv` over infeasibility claims, counting the recorded certificates as valid, invalid or unchecked (`cert_unknown`), `claimed_optimal_kkt_thresholds.csv` over the residuals `bench kkt-verify` checks) plus three KKT plots: per-solver boxplot of `primal_res_rel` / `dual_res_rel` / `comp_slack_rel` (`comp_slack` for records written before it existed) / `duality_gap_rel`, a problem-by-solver heatmap of those residuals, and a KKT-accuracy profile with one panel per residual showing, for each solver, the fraction of its successful solves whose residual is at most tau. Exact zeros are drawn at `1e-16`.
- For multi-dataset runs, a `## By Dataset` section that re-emits the headline tables (solver metrics, failure rates, shifted geomean, KKT summary) sliced per dataset. The cross-dataset aggregates above the section remain the primary view.
- An artifact index listing every CSV and plot written by the report command.

Programmatic use:

```python
from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean

df = load_results("results/<run_id>")
summary = solver_summary("results/<run_id>")
profile = performance_profile(df, metric="run_time_seconds")
geomean = shifted_geomean(df, metric="run_time_seconds", timeout_seconds=300)
```

Python callers can pass `timeout_seconds={"qp": 300, "sdp": 900}` for mixed
dataset limits, or `max_value=5400` for an explicit common penalty. A time-metric
penalized geomean requires one of these; `penalize_failures=False` does not.

Status handling:

- By default, only `optimal` counts as successful for performance profiles and
  shifted geometric means. `optimal_inaccurate` and other inaccurate statuses
  are penalized because the solver did not hit the requested target.
- Failed, skipped, timeout, and solver-error statuses have infinite performance
  ratios; shifted geometric means use a finite failure penalty instead.
- An explicit finite profile `max_value` opts into finite failure ratios,
  and must be strictly greater than every retained successful value after the
  metric floor is applied. Otherwise the function raises `ValueError`; choose
  a larger penalty in the metric's units or omit it to keep failures at infinity.
  Problems with no successes still have infinite ratios throughout.
- Profiles floor time metrics at 0.01 seconds and iterations at 1 before taking
  ratios, so a zero best value does not hide positive successful values. Other
  metrics retain exact ratios. The performance-ratio heatmap, pairwise speedups
  and pairwise scatter apply the same floors. Python callers can set `min_value` in metric
  units, including 0 to disable the floor, and customize `success_statuses`
  and `max_value`.

## Published Results

Results of the campaigns run with this harness are published in
[bodono/solver_benchmarks_results](https://github.com/bodono/solver_benchmarks_results): one directory
per campaign with the merged `results.jsonl` and `manifest.json` per family (readable directly by
`bench kkt-verify` and `bench report`), the campaign configs, figures, tables, and a manifest of solver
versions and hardware. The complete archive of each campaign is attached to its GitHub release there.

| Campaign | Content |
|---|---|
| [`scs33_2026-09`](https://github.com/bodono/solver_benchmarks_results/tree/main/scs33_2026-09) | SCS 3.3.1 against Clarabel, PIQP, OSQP, ProxQP, HiGHS, PDLP, cuOpt, CVXOPT, SDPA and QTQP on QP, LP, the Mittelmann LP set, SDP and infeasible problems (the [SCS benchmarks page](https://www.cvxgrp.org/scs/benchmarks/)) |

The SDPLIB rows of `scs33_2026-09` were measured on the merged form: every problem posed as one
semidefinite block, before the loader gave the 39 block-diagonal problems (`arch*`, `control*`,
`hinf*`, `truss*`, `ss30`) their declared blocks. Its `sdp_sdplib_*` configs reproduce those rows only
with `dataset_options.blocks: merged`.

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

The constraint is `b - A x in K`. The rows of `A` and `b` are stacked cone by
cone in one fixed order, the one SCS uses: zero (`z`), nonnegative (`l`), box
(`bl`/`bu`), second-order (`q`), PSD (`s`), complex PSD (`cs`), exponential
(`ep`), dual exponential (`ed`), power (`p`). List the keys of `cone` in that
order. The key order does not set the layout: solvers and the KKT check read
the rows in the fixed order. `ProblemData` refuses a cone dictionary whose keys
are listed in another order, and one whose sizes do not add up to the rows of
`A` (checked when it can size every key), so a loader that lists its cones in
the order it stacked them fails at load when that is not the fixed order. A PSD
block of order `d` takes `d (d + 1) / 2` rows: the lower triangle column by
column, off-diagonal entries times `sqrt(2)`.

The check cannot see rows stacked in another order under a correctly listed
dictionary, or a wrong PSD scaling: either still gives a well-posed problem,
and only the optimum shows it. When a source publishes optimal values, add a
test that solves a few small instances and compares against them.

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
from solver_benchmarks.solvers.base import (
    SolverAdapter,
    SolverUnavailable,
    finite_point,
    reported_objective,
)


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
        if mapped in status.ANY_FEASIBLE and finite_point(result.x, result.y):
            kkt_dict = kkt.qp_residuals(
                qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], result.x, result.y
            )

        return SolverResult(
            status=mapped,
            objective_value=reported_objective(mapped, result.objective),
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
- Return canonical statuses from `solver_benchmarks.core.status`. Map every raw solver status the underlying library can emit (including any `*_inaccurate` variants and infeasibility/unboundedness certificates), and fall back to `SOLVER_ERROR` only for genuinely unrecognized states. A solve that stopped on its iteration or time limit is `MAX_ITER_REACHED` / `TIME_LIMIT` even when the solver labels its last iterate "solved inaccurate" or "almost solved"; keep the solver's own status in `info["raw_status"]`.
- Populate `SolverResult.kkt` whenever the solver returned a finite primal/dual pair: use `solver_benchmarks.analysis.kkt.qp_residuals` (or `cone_residuals` for cone solvers) for every status in `status.ANY_FEASIBLE` (optimal, inaccurate, iteration or time limit, solver error), and the matching `*_infeasibility_cert` helpers when the solver reports a Farkas-style certificate (also an inaccurate one that came with a limit status, as OSQP's `*_infeasible_inaccurate` do). `bench kkt-verify --promote` can then judge every solver's limit-hit point by the same check. The KKT plots, summaries, and accuracy profile all read this field. `cone_residuals` projects onto every cone but complex PSD (zero, nonnegative, box, second-order, PSD, exponential and its dual, power and its dual), and its `primal_res` is the larger of `Ax + s − b` and the distance of `s` to the cone, so an adapter may pass `s = b − Ax`.
- Report `objective_value` only for `OPTIMAL` and `OPTIMAL_INACCURATE` (`reported_objective` does this); it is `None` for limits, certificates and errors, whose point's objective is still on `kkt.primal_obj`.
- Return `SOLVER_ERROR` with the message on `info["error"]` (and the elapsed time) when the solver itself fails on a problem by raising, e.g. on a non-convex `P` or crossed bounds; an exception escaping `solve` is recorded as a harness `worker_error` and re-run on resume. Keep raising for configuration errors (unknown settings, out-of-range values, a missing backend), including a bad value the solver reports as a status or raises from the same call as its failures on the problem.
- Time what other adapters time: the solver's setup and solve, not harness-only work such as statistics collection.
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

The MPS reader uses HiGHS in the same worker as OR-Tools. The highspy 1.15.x
wheels can clash with OR-Tools 9.15.6755's native HiGHS symbols when highspy
loads first, causing an import error mentioning `HighsLogOptions` or
`setLocalOptionValue`. This also affects macOS, not only Linux. Dependencies
in the `pdlp` and `all` extras therefore exclude highspy 1.15.x; highspy 1.14.0
has wheels for Python 3.10–3.14 and works with OR-Tools 9.15.6755. Reinstall the `pdlp` extra
if an existing environment has the incompatible pair. CI imports the native
libraries in both orders in fresh processes, then solves a real MPS file
through the reader and PDLP adapter.

Supported inputs:

- LP datasets represented as QPs with `P.nnz == 0`, such as NETLIB and MIPLIB
  root LP relaxations.
- Simple linear conic problems with only zero and nonnegative cones.
  Zero cones may use either `z` or `f`.

Unsupported inputs:

- QPs with nonzero `P`.
- SDP, SOC, rotated SOC, and other non-LP cones.
- Models whose serialized OR-Tools request reaches 2 GiB, the protobuf size
  limit. A conservative bound from the retained sparse entries, variables, and
  constraints skips obviously oversized models before constructing the protobuf,
  with `protobuf_model_bytes_lower_bound` and `protobuf_limit_bytes` in `info`.
  Remaining models get an exact, envelope-aware check before the request-model
  copy; oversized requests report model/request sizes and the byte limit.
  Both paths return `skipped_unsupported`. The exact `ByteSize()` check may
  allocate a serialization buffer internally on the upb protobuf backend.

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
  (OSQP, SCS, Clarabel, QTQP, cpo3, ECOS, CVXOPT, HiGHS, ProxQP, PIQP, plus PDLP
  on the LP), with KKT residuals checked against tight tolerances.
- SDPA and cpo3 cone smoke coverage on a small linear cone LP with KKT residuals.
- Multi-solver agreement tests on a small SDP with PSD cones (SCS, Clarabel,
  SDPA, CVXOPT) — guarding against PSD-vec layout drift.
- Primal- and dual-infeasibility certificate checks for SCS, Clarabel, OSQP,
  ECOS, and cpo3 against deliberately infeasible / unbounded LPs.
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
