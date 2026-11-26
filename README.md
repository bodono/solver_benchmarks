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

The bundled datasets live in the repository, not in the package, so install
from a clone in editable mode. A non-editable install (`pip install .`, or from
a git URL) has no data and takes the current directory as the repository root;
a command that then looks for a dataset's files under a root without
`problem_classes/` stops with an error, unless given `--repo-root` naming a
checkout or `--data-root` (the `data_root` option) naming a data directory.
Datasets without files (`synthetic_qp`, `synthetic_cone`) and the analysis of
run directories whose datasets need no bundled files work without either.

The `all` extra installs open-source solver dependencies declared by the package,
including QTQP, cpo3, SCS, OSQP, Clarabel, ECOS, CVXOPT, HiGHS, PIQP, OR-Tools/PDLP,
PyYAML, and pytest. On Python 3.14 it needs proxsuite >= 0.8.0 (3.14 wheels
exist from 0.7.3) and sdpa-python >= 0.2.3 (the first release with them);
ProxQP is left out on Windows with Python 3.14, where proxsuite has no wheel. cpo3 publishes wheels for Linux x86-64, Windows x86-64 and macOS; on
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
cuopt     missing optional extra  qp
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
scs       available               cone,qp
sdpa      available               cone
```

A solver package that is installed but fails to import (a native library that
does not load, an ABI mismatch) is listed as `unavailable: <the import error>`
instead of `missing optional extra`, and `bench run` prints the same warning.
Either way the solver counts as not installed: its solves are recorded as
skipped with `metadata.skip_reason: solver_unavailable` and the other solvers
run.

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
| QPLIB | `qplib` | QP | Uses QPLIB parser without CVXPY; supports category filters such as `subset: ccb` (categories come from `list_convex_qps.txt` in the data folder, else from the one bundled in the checkout, so they also work with another `data_root`). A download is parsed before it is cached, and only a transport failure (not an HTTP error such as a 404) is retried over plain HTTP. |
| Mittelmann | `mittelmann` | LP as QP | External ASU lptestset downloads; default prepare downloads `qap15`. |
| Mittelmann SDP | `mittelmann_sdp` | Cone/SDP | Mittelmann's SDP test set in SDPA-S sparse format; downloads from `plato.asu.edu`. The default subset is truss topology design (`trto3`, `buck3`), `rose13`, `cnhil8`, `biggs` and one maxcut relaxation (`G40mc`, a PSD block of order 2000); some take a minute or more. Each file's problem `min c'x` s.t. `sum_k x_k F_k - F_0` PSD is posed, as for SDPLIB. A plain `.dat-s` is preferred to a `.dat-s.gz` of the same name. |
| SDPLIB | `sdplib` | Cone/SDP | Reads converted `.jld2` files (or original `.dat-s` files placed beside the archive); requires `h5py`. Problems are posed with the block structure the original SDPLIB files declare (several semidefinite blocks, and diagonal blocks as nonnegative rows); the converted files store each constraint as one matrix and record the structure apart, as `blockvec`. `dataset_options.subset: feasible` (default), `infeasible` (the four instances `infd1`, `infd2`, `infp1`, `infp2`) or `all`. `dataset_options.blocks: merged` poses each problem read from the archive as one semidefinite block of the full order instead, as results from before the loader gave the block-diagonal problems their declared blocks were produced (an original `.dat-s` file has no merged form). |
| TSPLIB MaxCut SDP | `tsplib_sdp` | Cone/SDP | Goemans-Williamson MaxCut SDP relaxations of TSPLIB instances; supports EUC_2D/3D, MAN/MAX, GEO, ATT, and EXPLICIT weight types. A download is parsed before it is cached. |
| DIMACS | `dimacs` | Cone | Reads SeDuMi-format `.mat` and `.mat.gz` and poses the file's dual, `maximize b'y` subject to `c - A'y in K*`, as `minimize -b'y`: rows in SeDuMi's order (free, linear, second-order, PSD), each PSD block as the scaled lower triangle of its symmetric part. The recorded objective is `-max b'y` of the SeDuMi pair: minus the challenge's published value for the instances published as `min c'x` (`nb`, `nql*`, `qssp*`, `truss*`, `sched*`, ...), and equal to it for `hamming_*` and `torus*`, which are published as the theta number or max-cut bound, the value `-max b'y`. `filtinf1`, the library's infeasible variant of `filter48_socp`, is recorded as expected infeasible. Rotated Lorentz cones are not yet supported. |
| CBLIB | `cblib` | Cone | Downloads CBF files; the parser handles continuous linear (`L=`/`L+`/`L-`/`F`), second-order (`Q`), and exponential (`EXP`/`EXP*`) cone instances. Instances using other cone kinds (PSD, integer, etc.) are *hidden* from `list_problems()` by default; pass `dataset_options.include_unsupported=true` to surface them with `metadata["supported"]=False`. Listing reads only each file's header (the sections before the first coordinate section); a file that cannot be read is skipped with a warning, and a plain `.cbf` is preferred to a `.cbf.gz` of the same name. A maximisation is posed as the minimisation of minus its objective, with `original_obj_type: max` in the metadata. |
| MPC QP Benchmark | `mpc_qpbenchmark` | QP | Downloads structured MPC QPs from `qpsolvers/mpc_qpbenchmark`, at a pinned commit (`MPC_QPBENCHMARK_REVISION`). `.npz` files are read without unpickling: an object array is accepted only as a `None` placeholder for a missing array. `P` may be the whole symmetric Hessian or one triangle of it, which is mirrored; a `P` that is neither is rejected, and so is a file whose arrays do not fit together (`P` not `n x n` for `n = len(q)`, `G` or `A` without `n` columns, `G` without `h` or `A` without `b`, a right-hand side or `lb`/`ub` of the wrong length), with an error naming the file and the array. |
| LIBSVM-derived QP | `libsvm_qp` | QP | SVM dual or Markowitz portfolio QPs built from LIBSVM datasets; produces realistic ML/finance-shaped QPs absent from classical test sets. Problems are named `<kind>_<data set>` (`svm_dual_heart`). Without `subset` the dataset lists its six default data sets; another one is selected by naming its data set in `subset` (`subset: a1a`), and `include: [svm_dual_a1a]` alone is an error that says so. Preparing a problem named that way (from `include`, once the subset selects it) downloads its data set. |
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
  `bench run CONFIG --prepare-data` command. That holds for each problem asked
  for by name, through `include` or a `subset` that names problems: if some of
  them are on disk and others are not, the run stops naming the missing ones
  instead of solving only the rest. For a dataset that cannot download its data
  (`cutest_qp`), the problems present are solved and the missing ones get a
  warning naming the dataset's prepare command. Subset problems that
  `include` / `exclude` leave out are neither checked nor prepared. A name the
  dataset cannot have (a bundled dataset lists all it has, and a dataset with a
  name subset lists only those) fails the run with a suggestion, and says how
  to add it to the subset when it is outside it.
- `bench run CONFIG --prepare-data` or `run.auto_prepare_data: true` prepares
  data implied by the config before solving. If `include` is set, those problem
  names are prepared. If `dataset_options.subset: all` is set, the dataset's
  full remote index is prepared. If the subset names problems, those are
  prepared: a single name (a number such as `subset: 8790` included), a list
  or a comma-separated string, for the datasets whose `subset` takes problem
  names (`qplib`, where a category such as `ccb`
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
A worker killed during the copy leaves its temporary file
(`.<name>.jld2.<random>.tmp`) in `.cache/`; a later load of that problem from
the archive removes it once it is more than an hour old, an age a copy that is
still running never reaches. Preparation extracts into `sdplib_data/` and
removes there the leftovers of killed preparations in the same way.
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

Like `bench`, each wrapper takes `--repo-root` (default: the current directory
if it has `problem_classes/`, else the checkout the package was installed from)
and `--data-root` (the directory used in place of `problem_classes/`).

`scripts/research/` holds one-off scripts and SCS patches from the Anderson
acceleration experiments. They need a patched SCS build and local results that
are not in the repository, and they are not maintained; see its `README.md`.

The `--all` flag is deliberately never implicit. For example, CBLIB `--all`
follows the full CBLIB directory index and may download mixed-integer or
currently unsupported CBF files; the `cblib` adapter lists continuous
linear, second-order, and exponential-cone (`EXP`/`EXP*`) instances by
default and hides anything else (PSD, integer, etc.). Set
`dataset_options.include_unsupported=true` to surface those entries
with `metadata["supported"]=False`. Mittelmann `--all` follows the ASU lptestset
index. MIPLIB `--all` follows the official MIPLIB 2017 benchmark v2 instance
list, and `bench data prepare miplib --option max_size_mb=5` downloads only
benchmark instances whose compressed `.mps.gz` file is at most 5 MB: an
instance already on disk is sized there, and only the missing ones are asked
for by HEAD request. When the benchmark index cannot be fetched but instances
are on disk, those are used, with a warning. QPLIB `--all` uses
`list_convex_qps.txt` from the data folder, else the one bundled in
`problem_classes/qplib_data`.

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
config files. Each dataset accepts only the options listed below for it (and the
two generic ones): a misspelt or unknown option is an error that suggests the
closest name, raised when the config is parsed, so `kernal: rbf` does not
silently run the default linear kernel. NETLIB's `subset` must be `feasible` or
`infeasible`. A run directory recorded before options were checked is still
read by the analysis commands, with the options its manifest holds.

Useful dataset options:

| Dataset | Option | Meaning |
|---|---|---|
| All file-backed datasets | `max_size_mb=<number>` | Drop problems whose backing file exceeds the threshold (see below). |
| All file-backed datasets | `data_root=<path>` | Read the dataset's files under `<path>` instead of `<repo_root>/problem_classes`. In a config file, a relative path is resolved under the repository root, like `run.output_dir`; `bench run --data-root` resolves it against the current directory. |
| `netlib` | `subset=feasible` or `subset=infeasible` | Select NETLIB feasibility subset. |
| `netlib` | `infeasible=true` | The infeasible subset, when `subset` is not given. |
| `netlib`, `kennington`, `miplib`, `mittelmann` | `add_quadratic=true` | Add the identity to `P`, turning each LP into a QP with the same constraints. |
| `kennington`, `miplib`, `mittelmann` | `folder=<name>` | Read the problems from `<data_root>/<name>` instead of the dataset's own folder. |
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
  selected `ProblemSpec.path`. A problem stored both plain and gzipped
  (`p.mat` and `p.mat.gz`, `p.cbf` and `p.cbf.gz`, ...) is read from the plain
  file but filtered by the size of the smaller copy, in the DIMACS, CBLIB,
  Mittelmann SDP, TSPLIB and MPS (`netlib`, `kennington`, `miplib`,
  `mittelmann`) datasets.
- Datasets can advertise a more precise comparison size via
  `ProblemSpec.metadata["size_bytes"]`. The runner and listing/status tools
  prefer that value when present. This is used for archive-backed datasets
  such as `sdplib.tar`, where the threshold applies to each archive member
  (an original SDPLIB file that replaces a member is filtered by the member's
  size, not its own),
  and for the plain and gzipped copies of one problem described above.
- The synthetic datasets (`synthetic_qp`, `synthetic_cone`) have no backing file and are unaffected.

## Supported Solvers

Current maintained adapters:

| Solver | ID | Supported types | Notes |
|---|---|---|---|
| QTQP | `qtqp` | QP | Converts QP bounds to nonnegative cone form internally. Needs qtqp >= 0.0.7; tolerances are `tol_feas`, `tol_gap_abs`, `tol_gap_rel` (the harness also accepts `eps`/`eps_abs`/`eps_rel` and the pre-0.0.7 `atol`/`rtol`, translated to whatever the installed version takes). `linear_solver` and the other enum settings take a member name of the installed build (an unknown name is a configuration error); a backend whose Python module is missing (e.g. `qdldl`) stops the solve with a `SolverUnavailable` worker error. |
| cpo3 | `cpo3` | QP, cone (zero / nonnegative only) | Sparse interior-point solver for LPs and QPs, written in Rust ([cpo3-org/cpo3](https://github.com/cpo3-org/cpo3)); install with the `cpo3` extra (`pip install cpo3`, wheels for Linux x86-64, Windows x86-64 and macOS). QPs are converted to zero / nonnegative cone form like QTQP; cone problems pass through when they use only zero and nonnegative cones, and are skipped otherwise (cpo3 0.0.1 does not solve second-order or PSD cones). Tolerances are Clarabel-style `tol_feas`, `tol_gap_abs`, `tol_gap_rel`, `tol_infeas_abs`, `tol_infeas_rel`; `eps` / `eps_abs` / `eps_rel` map onto them and `max_iter` onto `max_iters`. `linear_solver` is `faer` (default, always available), `accelerate`, `cholmod` or `pardiso`; a backend the installed build or host cannot load stops the solve with a `SolverUnavailable` worker error instead of being scored as a solver failure. `time_limit` is forwarded; `threads` is one unless set (Accelerate, which takes only 0, keeps 0). Problem data `cpo3.Problem` refuses (an asymmetric `P`, non-finite entries) is `solver_error`. |
| SCS | `scs` | QP, cone | Supports SCS box-cone form for QPs and native conic data. `max_iter` maps to `max_iters` and `eps` to `eps_abs` / `eps_rel`. A run stopped by `max_iters` or `time_limit_secs` is recorded as `max_iter_reached` / `time_limit` (SCS's own text, e.g. `solved (inaccurate - reached max_iters)`, is kept in `info.raw_status`), not as `optimal_inaccurate`. A `time_limit` covers setup as well: SCS's own `time_limit_secs` covers only its iterations, so the adapter constructs the solver, times that, and gives the solve the rest of the limit through `SCS.update` (scs-python with bodono/scs-python#267); if setup used the whole limit the solve is skipped and the row is `time_limit` with `info.solve_skipped`. An SCS whose `update` cannot change settings gets the whole limit at construction, as before. `info.scs_time_limit_mode` is `solve_minus_setup`, `construction`, or `construction_after_update_error` (an `update` that raised; not expected, and `make_figures` warns about it), with `info.scs_time_limit_secs_given` (and `info.scs_setup_seconds` when measured). Problem data SCS refuses (a non-convex `P`, crossed bounds, non-finite entries, no constraint rows) is `solver_error`; an unknown setting or a value SCS refuses is a configuration error. |
| Clarabel | `clarabel` | QP, cone | Uses nonnegative cone conversion for QPs and native zero/nonnegative/SOC/PSD/exponential (`ep`) and power (`p`, 0 < alpha < 1) cones; exponential and power cone rows pass through unchanged (Clarabel uses the same orientation as SCS), dual exponential (`ed`) and dual power (negative alpha) cones are skipped. `eps` / `eps_abs` / `eps_rel` map onto `tol_feas`, `tol_gap_abs`, `tol_gap_rel`; a negative or non-finite tolerance is refused, and so are `tol_feas: 0` and `tol_gap_abs` and `tol_gap_rel` both 0, which can never be met; one zero gap or infeasibility tolerance (e.g. `eps_rel: 0`) is a valid absolute-only or relative-only test. `threads` sets `max_threads`, one unless set. A `direct_solve_method` Clarabel knows but cannot load on this host (`mkl`, `panua`) stops the solve as a `worker_error` (`SolverUnavailable`); an unknown one is a rejected setting. Clarabel relabels a stop at `max_iter` or `time_limit` as `AlmostSolved` (or `Almost*Infeasible`) when the last iterate meets its reduced tolerances; such a stop is `max_iter_reached` / `time_limit`, with Clarabel's status in `info.raw_status`, while an `Almost*` status for any other stop stays `*_inaccurate`. On some SDPs Clarabel's chordal decomposition (on by default) reports `Solved` at a point that misses the original constraints by far more than its tolerance (SDPLIB `control1`: 18.05 against the optimum 17.7846, harness `dual_res_rel` about 2e-2); the KKT record shows it, and `chordal_decomposition_enable: false` solves that instance. |
| OSQP | `osqp` | QP | Direct QP adapter. |
| ECOS | `ecos` | QP, cone (LP/SOCP/expcone) | Interior-point conic solver. QPs are reformulated to SOCP via the standard epigraph trick with `P = R'R`; the SOC dim is `rank(P) + 2`. A row/column of `P` whose only entry is its diagonal gives an `R` row `sqrt(p_ii)` (so a diagonal `P` works at any size); only the rows/columns coupled by off-diagonal entries are factored densely, one connected block at a time (Cholesky for PD, eigendecomposition for rank-deficient PSD, keeping every eigenvalue above the block's round-off `k * eps * max|eigenvalue|`, so `R'R` reproduces `P` even when its spectrum spans many orders of magnitude; an eigenvalue below `-1e-12` times the scale of `P` makes `P` indefinite and the problem `skipped_unsupported`). A QP whose coupled blocks would need more than 2000² dense entries together (for example one coupled block larger than 2000 × 2000, as in CVXQP*_L) is skipped as `skipped_unsupported` instead of densified; mostly diagonal problems such as MOSARQP1 (10 coupled rows of 2500) and STCQP1/2 (1534 of 4097) are solved. The reformulation is included in `run_time_seconds` (and recorded as `info.socp_reformulation_seconds`). `max_iter` maps to `max_iters`, `eps` / `eps_abs` to `feastol` and `abstol`, `eps_rel` to `reltol`; an unknown setting or a value ECOS refuses is a configuration error. ECOS answers "close to optimal" (exit flags 10-12) when it stops at `max_iters` with an iterate within its reduced tolerances; that stop is `max_iter_reached` (ECOS's text in `info.raw_status`), and the same flags for an earlier stop stay `*_inaccurate`. PSD cones not supported. |
| CVXOPT | `cvxopt` | QP, cone (LP/QP/SOCP/SDP) | Interior-point solver supporting QPs natively plus PSD cones (canonical PSD-vec layout converted to BLAS unpacked-L automatically). Problems with a linear objective (LP/SOCP/SDP) go through `cvxopt.solvers.conelp`, which certifies primal and dual infeasibility; problems with a quadratic objective go through `coneqp`, which cannot. `conelp`'s infeasibility test divides a residual by minus the objective, so at its least-squares starting point it only checks whether that objective exceeds `1/feastol` in magnitude; a claim at iteration 0 (NETLIB agg, agg2, agg3, grow7) is discarded and the problem solved with `coneqp` (`info.conelp_discarded_claim`, time of both calls reported). A later claim can be false as well on a badly scaled feasible problem (NETLIB scagr25, gfrd-pnc, nesm), with a scaled residual that passes the certificate check, so the adapter also solves such a problem with `coneqp`: the claim stands unless `coneqp` finds a solution, which is then reported (`info.coneqp_cross_check`; a genuinely infeasible problem costs two solves). NETLIB shell remains a false claim (`coneqp` fails on it). With second-order cones the adapter passes `conelp` the sparse `chol` KKT solver `coneqp` used, not CVXOPT's dense `qr` default, which turned DIMACS sched_50_50_scaled from `optimal` in 80 s into an error after 490 s. PSD-only problems keep `qr`: `conelp` with `chol` fails on SDPLIB control1 and hinf9 and is inaccurate on control2 and truss6/7, which `qr` solves; `qr` is slower (qap6/7 about 8x) and stops at `maxiters` on hinf1/6/14 and qap6/7. `kktsolver` overrides either choice. A problem without cone rows (equalities only) is solved by `coneqp`'s single KKT solve of `[[P, A'], [A, 0]]`, which always says `optimal`; a CVXOPT residual above `feastol` makes it `optimal_inaccurate` (below 1e-3) or `solver_error`. The default `chol2` KKT solver returns such a wrong point for singular `P` (HS52, GENHS28), so when `n + p` is at most 2000 the adapter uses the `ldl` KKT solver; `ldl` factors the KKT matrix as a dense array (7.3 GB for AUG2DC, `n + p` = 30200), so larger systems keep `chol2`. CVXOPT's `unknown` status is `max_iter_reached` only when `maxiters` was reached; an earlier stop is `optimal_inaccurate` when gap and residuals are below 1e-3 and `solver_error` otherwise. Settings are CVXOPT's options (`maxiters`, `abstol`, `reltol`, `feastol`, `refinement`, `kktreg`, ...) plus `kktsolver`; `max_iter` maps to `maxiters`, `eps` / `eps_abs` to `abstol` and `feastol`, `eps_rel` to `reltol`, and any other name raises (CVXOPT would ignore it). CVXOPT checks option values and `kktsolver` in the call that solves, raising the same exception as for numerical trouble, so the adapter checks them first (`maxiters` >= 1, `feastol` > 0, `refinement` >= 0, one of `abstol` / `reltol` positive, `kktsolver` one CVXOPT accepts for the entry point used): a bad value is a configuration error, not `solver_error` on every problem. A numerical failure is `solver_error` with the message on `info.error`. Exponential cones not supported. CVXOPT options live on a global dict; the adapter snapshots and restores it per solve so **sequential** solves in the same process don't leak knobs into each other. This is *not* thread-safe — concurrent in-process solves would still race on the global. The benchmark runner uses subprocess-level parallelism, so this per-process sequential contract is sufficient. |
| PDLP | `pdlp` | LP-only QP, simple linear cone | Uses OR-Tools directly, no CVXPY. QPs and cone problems with a nonzero `P` are skipped; cone entries without rows (`q: []`, `ep: 0`) are ignored. `max_iter` maps to `iteration_limit`; the legacy `solver_time_limit_sec` spelling is validated like `time_limit` (which wins when both are given). PDLP runs on one thread by default. See [PDLP Notes](#pdlp-notes). |
| HiGHS | `highs` | QP | Direct LP/QP adapter through `highspy`; strong LP baseline. Model data HiGHS refuses when it is added (a coefficient at or above `large_matrix_value`, an infinite matrix entry, a wrong-signed infinite bound) is `solver_error`, not a solve of the truncated model. An infeasibility claim carries HiGHS's ray as its certificate (`getDualRay` for `primal_infeasible`, `getPrimalRay` for `dual_infeasible`), checked like every other adapter's. When HiGHS proved the claim on the presolved LP, it finds the ray by solving the original LP again without presolve, which can take far longer than the claim (NETLIB gosh: about 0.01 s to the claim, 3 to 7 s to the ray). That time is recorded as `info.certificate_seconds`, is covered by the solver's `time_limit` setting, and counts in `run_time_seconds` when a ray is recorded. `run.timeout_seconds` does not cover it, since the runner does not pass it to the solver: with `timeout_seconds` alone the search for the ray has no limit, and if it outlasts `timeout_seconds` plus the grace the worker is killed and the claim, however quickly it was made, is recorded as a `time_limit` row with no status or info (campaigns set both). An attempt that gives no ray is left out of `run_time_seconds`, since the claim is unverified whatever it cost and counting it would turn a claim made in time into a `time_limit` row (gosh with a 1 s limit). HiGHS gives no ray for an unbounded QP or on NETLIB cplex2, klein1 and qual (its re-solve ends with status Unknown), and highspy 1.15.1 none for a claim made by interior point or PDLP (`solver: ipm`, `pdlp`; 1.16.0 gives one); such a claim, and one whose ray HiGHS fails to produce (an error from the re-solve) or that cannot be used (wrong size, not finite), has no certificate (`info.certificate_unavailable` says why) and counts as unsolved unless `--allow-missing`. `iterations` is the count of the algorithm that finished the solve (QP, else interior point, else PDLP, else simplex); every algorithm's count is on `info`. |
| ProxQP | `proxqp` | QP | Direct QP adapter through ProxSuite. The adapter builds ProxSuite's QP object (`proxqp.sparse.QP`, or `proxqp.dense.QP` with the dense backend), sets its settings and calls `init` and `solve`, the steps of ProxSuite's one-shot `solve` function (which the adapter called before) in the same order and with the same defaults, so the settings that function takes give the same solves and `run_time_seconds` covers the same work. Settings are that function's keywords (`rho`, `mu_eq`, `mu_in`, `compute_preconditioner` and `default_H_eigenvalue_estimate` go to `init`) and any other field of `proxqp.Settings`, such as its certificate tolerances `eps_primal_inf` and `eps_dual_inf`, which the one-shot function does not take; `sparse_backend` only with the sparse backend; a name ProxQP does not have, or a value of the wrong type, raises `SolverConfigError` naming it, and a setting given as `null` is left at ProxQP's default. ProxSuite's default stopping test checks only the primal and dual residuals and can call a point with a large duality gap solved, so the adapter turns on its duality-gap test (`check_duality_gap: true`, with `eps_duality_gap_abs` / `eps_duality_gap_rel` taken from `eps_abs` / `eps_rel` unless given); the values used are on `info`. With `eps_rel` unset or 0 (ProxQP's default), `eps_abs` is the relative gap tolerance too: an absolute-only gap test is out of reach of double precision when the objective is large (at `eps_abs: 1.0e-9`, Maros-Meszaros QSCAGR25, objective 2e8, ran to `max_iter`, and QPCBOEI1 ended in a false infeasibility). Set `check_duality_gap: false` for ProxSuite's own default, which ProxQP results recorded before this change used. Infeasibility claims carry the harness certificate check: on feasible NETLIB LPs ProxQP often reports `primal_infeasible` (17 of the 30 smallest) and those claims fail the check, and so do most of its claims on infeasible problems: ProxSuite returns its dual iterate, which often is not a certificate within the tolerance. `backend` must be `sparse` or `dense` and `dense` a boolean. |
| PIQP | `piqp` | QP | Direct sparse/dense QP adapter. PIQP checks the duality gap against its own `eps_duality_gap_abs` / `eps_duality_gap_rel` (1e-8 / 1e-9); when the config sets `eps_abs` / `eps_rel` (or `eps`) and not those, they follow `eps_abs` / `eps_rel` (`eps_abs` when `eps_rel` is 0), as in the ProxQP adapter, and are listed on `info.settings_translated`; with no tolerance configured PIQP's defaults stay. Infeasibility claims carry the harness certificate check. `threads` sets `num_threads`, one unless set; `backend` must be `sparse` or `dense` and `dense` a boolean. |
| SDPA | `sdpa` | cone | Linear conic adapter through `sdpa-python`; useful for SDP subsets. Problems with second-order cones are skipped: `sdpa-python` ends the process on them. `lowerBound` / `upperBound` default to -1e30 / 1e30 (with sdpa-python's -1e5 / 1e5, SDPA stops as soon as an objective crosses them, so feasible problems with an optimum beyond 1e5 in magnitude came back infeasible or unbounded). `threads` maps to `numThreads`, which defaults to 1 (sdpa-python otherwise uses every core; `threads: 0` leaves it to sdpa-python); `max_iter` maps to `maxIteration`, `eps_abs` / `eps_rel` (or `eps`) to `epsilonStar` / `epsilonDash`; other names must be sdpa-python parameters, and `frvMethod`, `convMethod`, `domainMethod` and `rangeMethod` one of the values sdpa-python takes (`split` / `elimination`, `LMI` / `EQ`, `none` / `clique` / `basis`, `none` / `clique` / `decomp`), which it would otherwise replace by its default with only a printed note. `pdOPT` / `pdFEAS` are `optimal` when sdpa-python's primal / dual errors and gap are within `optimality_tolerance` (default `max(1e-6, epsilonStar, epsilonDash)`); an error sdpa-python could not compute (`None`) is left out. Infeasibility claims carry the harness certificate check. A problem without constraint rows, which sdpa-python refuses ("Inconsistent Size"), is `solver_error`. |
| GUROBI | `gurobi` | QP | Optional extra; requires `gurobipy` and license. A license failure raised by the solve (the size-limited license of the PyPI package refuses larger models) stops it as a `worker_error` (`SolverUnavailable`) instead of being scored as Gurobi failing the problem; another `GurobiError` from the solve is `solver_error`. When presolve can only say "infeasible or unbounded", the adapter re-solves with `DualReductions=0` (Gurobi's documented remedy) so the row gets a definite status; `info.inf_or_unbd_resolved` marks it. The re-solve gets what is left of the time limit (`time_limit` or Gurobi's own `TimeLimit`), since Gurobi's limit bounds each `optimize()`; with no time left it is skipped and the row stays `primal_or_dual_infeasible`. `info.raw_status` is the status name (`OPTIMAL`, ...), `info.raw_status_code` Gurobi's integer. An infeasibility claim on an LP carries Gurobi's `FarkasDual` (for `primal_infeasible`) or `UnbdRay` (for `dual_infeasible`) as its certificate: the adapter sets `InfUnbdInfo=1` unless it is configured (`info.inf_unbd_info` is the value the solve ran with), which Gurobi documents as extra work done only once an LP is found infeasible or unbounded. That work happens inside `optimize()`, so it counts in `run_time_seconds` and cannot be timed apart, and it can include solving the unpresolved LP again, after which a numerically marginal LP can end with another status than it would without it; `InfUnbdInfo: 0` turns it off, and the claims then have no certificate. Gurobi computes neither for a QP or after barrier without crossover; such a claim has no certificate (`info.certificate_unavailable`). The conversion follows Gurobi's reference manual and is tested against a fake Gurobi only. |
| MOSEK | `mosek` | QP | Optional extra; requires Mosek package and license. A license failure raised by the solve (missing, expired or exhausted license) stops it as a `worker_error` (`SolverUnavailable`) instead of being scored as MOSEK failing the problem. Settings are MOSEK parameter names (`MSK_IPAR_*`, `MSK_DPAR_*`, `MSK_SPAR_*`); `max_iter` sets `MSK_IPAR_INTPNT_MAX_ITERATIONS` and `MSK_IPAR_SIM_MAX_ITERATIONS`, and the `eps` aliases are refused. An infeasibility claim carries MOSEK's certificate (its dual solution for `primal_infeasible`, its primal solution for `dual_infeasible`), for LPs and QPs, with no setting needed; one that is not finite, or that MOSEK fails to return, is not recorded (`info.certificate_unavailable` says why). The conversion follows MOSEK's documentation and is tested against a fake MOSEK only. |
| CPLEX | `cplex` | QP | Optional extra; requires IBM CPLEX package and license. A license failure raised by the solve (Community Edition size limits, error 1016, or no license) stops it as a `worker_error` (`SolverUnavailable`); another CPLEX error from the solve is `solver_error`. `info.raw_status` is CPLEX's status string, `info.raw_status_code` its integer. After an "infeasible or unbounded" status the adapter re-solves with `preprocessing.reduce: 0` (`info.inf_or_unbd_resolved`). `iterations` is the barrier count when barrier ran (CPLEX's default for QPs), else the simplex count; both are on `info`. An infeasibility claim on an LP carries CPLEX's dual Farkas proof (for `primal_infeasible`) or unbounded ray (for `dual_infeasible`) as its certificate. CPLEX leaves one only after a simplex solve proved the claim; when presolve or another method proved it (or the vector CPLEX gives is clearly not a certificate, failing `kkt.CERT_TOL`), the adapter solves the LP again with presolve off, by the dual simplex for a Farkas proof or the primal simplex for a ray (`info.certificate_resolved`). That time is recorded as `info.certificate_seconds` and counts in `run_time_seconds` when a certificate is recorded, as for HiGHS. Each further solve of a run (the re-solve after "infeasible or unbounded", the one for the certificate) gets what is left of the time limit, set by `time_limit` or by CPLEX's own `timelimit`, since CPLEX's limit bounds each solve; without either those solves have no limit, as HiGHS's search for a ray has none (`run.timeout_seconds` is not passed to the solver). CPLEX has no certificate for a QP; such a claim has none, as has one whose certificate cannot be had or used (`info.certificate_unavailable` says why). The conversion follows the CPLEX reference manual (CPXdualfarkas, CPXgetray) and is tested against a fake CPLEX only. |
| cuOpt | `cuopt` | QP | NVIDIA's GPU solver (PDLP for LPs, barrier for QPs). Not a package extra: install `cuopt-cu12` from NVIDIA's pip index on a CUDA machine. The QP maps onto cuOpt's objective with `Q = P / 2`. `eps` sets all of cuOpt's optimality tolerances and `eps_abs` / `eps_rel` its absolute and relative primal, dual and gap tolerances; `max_iter` sets `iteration_limit`, `time_limit` and `threads` are forwarded, and any other setting goes to `set_parameter` (a value cuOpt rejects is a `ValueError` naming it). The adapter turns on PDLP's `infeasibility_detection` unless it is configured (`info.infeasibility_detection` is the value the solve ran with): off, cuOpt's default, PDLP never stops on an infeasible problem, which suits only a problem known in advance to be feasible, and a user cannot know that; cuOpt's concurrent dual simplex and barrier detect infeasibility either way. cuOpt rows recorded before this ran with it off. cuOpt reports duals with the opposite sign to the harness, so the KKT residuals use `dual_sign: -1` by default. These conventions were checked by hand on a GPU with `probe_cuopt` in `tools/bench_modal.py`, and CI tests them only against a fake cuOpt; rerun the probe after upgrading. cuOpt's API exposes no infeasibility certificate (on an infeasibility stop it returns its last iterate, not the ray its test is made on), so its infeasibility claims have nothing to check and count as unsolved unless `--allow-missing`. |

Comparing with run directories written before the conic adapters adopted the
shared result policy: SCS stops on `max_iters` / `time_limit_secs` are now
`max_iter_reached` / `time_limit` (they were `optimal_inaccurate`, so configs
that count `optimal_inaccurate` as success lose those rows); CVXOPT `unknown`
stops before `maxiters` are now `solver_error` or `optimal_inaccurate` (they
were `max_iter_reached` or `*_infeasible_inaccurate`), and stops at `maxiters`
are `max_iter_reached` even with small residuals (they were
`optimal_inaccurate`: NETLIB grow15 / grow22, SDPLIB hinf / qap, DIMACS
minphase, several Liu-Pataki problems); CVXOPT solves LPs,
SOCPs and SDPs with `conelp`; ECOS `run_time_seconds` now includes the QP to
SOCP reformulation; SDPA runs single-threaded unless `threads` is set (it used
every core, so its timings are slower than in old runs); and limit stops and
infeasibility certificates carry no `objective_value` in any conic adapter.

Comparing with runs recorded before the thread rule and the remaining
adapters adopted that policy: Clarabel and ECOS stops at `max_iter` (or
Clarabel's `time_limit`) that the solver labelled "almost solved" / "close to
optimal" are `max_iter_reached` / `time_limit` with no `objective_value` (they
were `optimal_inaccurate`); cpo3, Clarabel, HiGHS, PIQP, Gurobi, CPLEX and
MOSEK run on one thread unless `threads` is set (they used the solver's
default, usually every core, so timings of medium and large problems change,
and for Clarabel, whose iterates depend on `max_threads`, occasionally the
iteration count or an `optimal` / `optimal_inaccurate` verdict);
PIQP and ProxQP infeasibility claims carry certificates, so failed ones are no
longer successes on infeasible problems and `kkt-verify` checks them; PIQP
with `eps_abs` / `eps_rel` set closes the duality gap only to those tolerances
(fewer iterations at loose tolerances); HiGHS `iterations` is the finishing
algorithm's count (a QP no longer reports its phase-1 simplex count) and its
infeasibility rows have no KKT record; PDLP skips cone problems with a nonzero
`P` (it solved the LP without it); and data SCS, cpo3 or SDPA refuses is
`solver_error` instead of a `worker_error` retried on resume.

Comparing with runs recorded before the runner capped each worker's BLAS /
OpenMP threads (see `threads` under
[Shared Solver Settings](#shared-solver-settings)): at default settings,
timings change for every solve path that goes through NumPy / SciPy's
OpenBLAS, whatever the solver's own wheel bundles: Clarabel on PSD cones,
QTQP, and ECOS on QPs with an off-diagonal `P` (the adapter's dense
factorization of `P`, part of `run_time_seconds`); and for CVXOPT, SDPA and
SCS builds linked to a threaded BLAS (the PyPI wheels of those three bundle a
serial OpenBLAS). With `run.parallelism` above one, solves that returned in
time but whose post-solve KKT check (PSD cones) ran past the kill deadline
were recorded as `time_limit` with no KKT record; they now keep their own
status.

Comparing with runs recorded before the HiGHS, Gurobi, CPLEX and MOSEK
adapters recorded infeasibility certificates: their infeasibility rows have
no KKT record, so the verification counts every such claim as unsolved
(unless `--allow-missing`); re-run them to have the claims checked. In the
new rows, HiGHS's and CPLEX's `run_time_seconds` of an infeasibility claim
include the time taken to produce its certificate when one is recorded
(`info.certificate_seconds`; CPLEX's second solve,
`info.certificate_resolved`), Gurobi runs with `InfUnbdInfo=1`
(`info.inf_unbd_info`), and CPLEX's re-solve after "infeasible or unbounded"
gets what is left of the time limit instead of the whole limit again.

A `solver_error` row carries the reason the adapter gives (`info.error`) in its
`error` field, the column report tables show; rows recorded before that have it
only under `info`.

Unsupported solver/problem combinations are skipped by default and recorded as
`skipped_unsupported`. For example, OSQP on a DIMACS conic problem is skipped,
and so is a problem whose cone types an adapter cannot take (ECOS on a PSD cone),
which the adapter reports from the solve. A solver that is not installed has
every one of its solves skipped the same way.

To fail instead of skipping (a solver that is not installed, or installed but
not loadable, then fails the run before anything is solved; a skip the adapter reports is recorded, then the run
stops):

```yaml
run:
  fail_on_unsupported: true
```

### Solve Times

Every result has `run_time_seconds`, the wall-clock time the adapter measures
around the solver's calls (problem loading excluded); for an infeasibility
claim that includes the calls that produce the certificate it is checked on,
which for HiGHS and CPLEX can be a second solve (an attempt that gives no
certificate is not counted; both record the attempt as
`info.certificate_seconds`, see [Supported Solvers](#supported-solvers)). `solve_time_seconds` is
the solve time the solver itself reports; where a solver reports its setup
(scaling, factorization set-up) separately, that part is in
`setup_time_seconds` and not in `solve_time_seconds`, and where it reports one
total, `solve_time_seconds` includes the setup and `setup_time_seconds` is
empty. Compare `solve_time_seconds` across solvers with
that in mind, or use `run_time_seconds`. What each adapter records:

| Solver | `solve_time_seconds` | `setup_time_seconds` |
|---|---|---|
| `clarabel` | `solve_time` (setup included) | not reported |
| `cplex` | not reported | not reported |
| `cpo3` | `solve_time` (presolve, equilibration and factorizations included) | not reported |
| `cuopt` | `get_solve_time()` | not reported |
| `cvxopt` | not reported | not reported |
| `ecos` | `timing.tsolve` | `timing.tsetup` |
| `gurobi` | not reported | not reported |
| `highs` | not reported | not reported |
| `mosek` | not reported | not reported |
| `osqp` | `solve_time` | `setup_time` |
| `pdlp` | not reported | not reported |
| `piqp` | `solve_time` | `setup_time` |
| `proxqp` | `solve_time` | `setup_time` |
| `qtqp` | not reported | not reported |
| `scs` | `solve_time` | `setup_time` |
| `sdpa` | sdpa-python's `sdpa` time | not reported |

### GPU warm-up

A solve whose settings select a GPU (SCS with `linear_solver: cudss` or
`gpu_indirect`, cuOpt, QTQP with `linear_solver: cudss` or `cupy_dense`) has
the GPU warmed up before its timed region. Every solve runs in a worker
process of its own, so without it the `run_time_seconds` of a GPU row would
include costs that a process pays once, whatever it then solves: loading the
solver's GPU module and the CUDA libraries it links (scs-python imports its
cuDSS and GPU modules inside `scs.SCS`), creating the CUDA context (0.245 s on
an NVIDIA GB10 with CUDA 13), and cuBLAS loading its matrix-product kernels at
the first `gemm` or `gemv` of the process (cuBLAS 13 runs both through
cuBLASLt: 0.128 s on the GB10, whatever the problem size). A Python session
that solves many problems pays these once, and the warm-up times every GPU
solve as such a session's later solves see it.

Before the adapter's `solve`, the worker imports the modules the adapter
names (`SolverAdapter.gpu_modules`), creates the primary CUDA context of
device 0 through the driver API (`libcuda.so.1`), and, for each cuBLAS
library the solver's modules loaded (read from `/proc/self/maps`), runs one
`gemm` and one `gemv` of each kind on 10 x 10 operands, on a handle and a
buffer of its own that it then frees. It loads no cuBLAS of its own, so it
never takes the place of a copy a solver bundles (cuOpt bundles its CUDA
libraries).

Each GPU row records the warm-up in `metadata.gpu_warmup` (the columns
`metadata.gpu_warmup.*` of the results frame): `seconds`, the whole warm-up,
which is not in `run_time_seconds`; `modules` and `import_seconds`;
`cuda_context_seconds`; `cublas_libraries` (the paths warmed) and
`cublas_seconds`; and `notes`, one line for each part that did not run and
why (no CUDA driver or device, a module that does not import, no cuBLAS
loaded, a CUDA error). The warm-up never fails a run: a part that cannot run
is skipped and the solve runs as it would have without it. CPU rows have no
such record.

What stays in the timed region: a library that a solver loads only during
its solve (a cuBLAS loaded then is not warmed, and the notes say that none
was found), a cuBLAS linked statically into a solver, kernels that other
libraries load at their first use (cuDSS's own; cuSPARSE's first transpose,
21.7 ms on the GB10), cuBLAS kernels of other shapes and types, and the
solver's own handles and memory. The context and the loaded kernels hold
some GPU memory during the solve that a solver making no `gemm` or `gemv`
call would not have held.

Comparing with runs recorded before the warm-up: the `run_time_seconds` of
their GPU rows (and SCS's `setup_time_seconds`, which covers its first CUDA
calls) include the module loading, the CUDA context and, for a solver whose
solve calls `gemm` or `gemv`, cuBLAS's kernel loading. A GPU row recorded
with the warm-up has `metadata.gpu_warmup`.

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
      # Equal to timeout_seconds: PDLP's own time-limit stops return slightly
      # after 120 s, so the KKT verification never promotes them. Set it
      # below timeout_seconds (say 110) if those points should be eligible.
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
| `run.dataset_options` | Dataset options, such as `subset: feasible`. With `datasets`, defaults for the listed datasets that read each option (an entry's own `dataset_options` override them); an option no listed dataset reads is an error. |
| `run.include` | Optional list of problem names to run. Empty means all. Used as a fallback for any dataset whose own `include` is unset. Names are case-sensitive. A name that matches no problem fails the run with a suggestion, unless the dataset downloads or prepares its data (it may just not be local yet): then a dataset that downloads automatically stops the run with the command that prepares it (or prepares it first under `--prepare-data`), and one prepared by hand gets a warning; a name removed by `max_size_mb` gets a warning. |
| `run.exclude` | Optional list of problem names to skip. Unioned with each dataset entry's `exclude`. A name that matches no problem gets a warning. |
| `run.parallelism` | Number of concurrent subprocess solves. |
| `run.resume` | If true, completed `(dataset, problem, solver_id)` triples whose solve settings match the config are not rerun (see [Running Benchmarks](#running-benchmarks)). If false, `bench run --run-dir` refuses a directory that already has results for the config's datasets and solvers. |
| `run.max_worker_error_retries` | Maximum retries of a compatible `worker_error` result across resumes; default 2, 0 disables retries. Increase it to retry again after fixing a persistent failure. |
| `run.timeout_seconds` | Default time limit per solve, in seconds; `0` or unset means none. The runner enforces it; it is not passed to the solver (set the solver's own time-limit setting for that). A worker still running after the limit plus a grace for start-up and problem loading (equal to the limit, at least 5 s and at most 30 s) is killed, and the solve is recorded as `time_limit` with `run_time_seconds` equal to the limit. A solve that returns an optimal, inaccurate or infeasibility status after the limit is recorded as `time_limit` too, with the solver's status kept in `info.status_before_timeout` and its objective moved to `info.objective_before_timeout`. Every result returned after the limit, whatever its status, carries `info.exceeded_time_limit`, and the KKT verification never promotes it. A solver stopped by its own time limit returns slightly after that limit, so when its limit equals `timeout_seconds` its time-limit stops are over the limit and are never promoted; set the solver's limit below `timeout_seconds` if those points should be eligible. |
| `run.auto_prepare_data` | If true, prepare the missing requested data before listing and solving: the problems `include` names, else those `dataset_options.subset` names (for the datasets whose subset takes problem names), else the dataset's small default subset. Problems already present are not requested again, and a failed preparation stops the run with an error naming the dataset. The details are under [Supported Datasets](#supported-datasets). |
| `run.fail_on_unsupported` | If true, an unsupported solver/problem combination fails the run instead of being recorded as `skipped_unsupported`: a problem kind the solver does not take, a problem the adapter reports it cannot take (the row is written, then the run stops), and a solver that is not installed (the run stops before solving anything). See [Supported Solvers](#supported-solvers). Default false. |
| `run.worker_memory_limit_mb` | Optional cap, in MiB, on the data segment of each solve's worker process (`RLIMIT_DATA`, set in the worker before it loads the problem). On Linux this covers heap and anonymous `mmap` allocations, and the libraries a solver loads count against it too. macOS largely ignores the limit, and on Windows, which has no `resource` module, every solve fails when it is set. A solve that exceeds the limit is normally recorded as `worker_error` (a solver that catches the failed allocation itself may report its own error status instead), so set the limit well above the largest problem's needs. Integer >= 1; unset or null means no limit. |
| `solvers[].id` | Unique label for this solver variant. Used in output paths, so two ids may not differ only in letter case. |
| `solvers[].solver` | Solver adapter ID, e.g. `scs`, `qtqp`, `pdlp`. |
| `solvers[].settings` | Passed directly to the solver adapter. |
| `solvers[].timeout_seconds` | Optional per-solver override of `run.timeout_seconds`. |
| `solvers[].sweep` | Optional mapping of setting name to a non-empty list of values. Expands to the full Cartesian product. |
| `solvers[].id_template` | Optional Python format string used to name expanded sweep variants. |

The same solver may appear many times with different `id` and `settings`.
Solvers run quietly by default (`verbose: false`), so each solve's
`stdout.log`/`stderr.log` is normally empty; set `verbose: true` in a solver's
settings to have its log captured there and streamed to the terminal. To keep
the log files without printing solver output to the terminal, pass
`--no-stream-output` to `bench run`.

### Shared Solver Settings

A few setting names mean the same thing for every adapter:

- `time_limit` (or `time_limit_sec` / `time_limit_secs`): the solver's own
  time limit in seconds. It must be a finite number greater than zero; `0` is
  refused because solvers disagree on what it means (no limit for SCS, stop
  before the first iteration for Clarabel, cpo3, HiGHS and PDLP, an invalid
  setting for OSQP). Leave it out to run without a limit. Adapters whose solver
  has no time limit record `time_limit_ignored` on the result's `info`.
- `threads` (or `num_threads`): the solver's thread count. Every adapter whose
  solver has a thread setting (cpo3, Clarabel `max_threads`, HiGHS, PIQP and
  PDLP `num_threads`, SDPA `numThreads`, Gurobi, CPLEX, MOSEK) runs it on one
  thread unless `threads` or the solver's own name for it is given, and
  `threads: 0` leaves the choice to the solver (usually every core). One thread
  keeps timings comparable with the single-threaded solvers, and the runner may
  run several workers in parallel (`run.parallelism`), which a solver using
  every core would oversubscribe. The count the solver was given is on
  `info.threads` (0: the solver's choice); it is `None` where the adapter cannot
  set it (OSQP, ProxQP, QTQP, ECOS, CVXOPT, SCS without OpenMP), and a
  `threads` request there is recorded as `threads_ignored`. cuOpt, a GPU
  solver, forwards `threads` to `num_cpu_threads` only when it is set.
  The runner caps the worker's BLAS / OpenMP pools to the same number: each
  worker starts with `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` set to `threads` (one unless
  set; `threads: 0` sets none of them). Export any of these four yourself
  (with a value that is not blank; an empty or whitespace-only one counts
  as unset) and the runner
  leaves all of them alone, whatever `threads` says (the solver's own thread
  option still follows `threads`); `tools/bench_modal.py` does this with
  `OMP_NUM_THREADS`. This is what bounds the harness's own KKT check (an
  eigenvalue decomposition per PSD block) and the BLAS inside solvers that
  take no thread setting of their own; without it every one of
  `run.parallelism` workers would start a BLAS thread per core. The cap
  follows `threads` (or `num_threads`) only: a solver's native thread option
  (Clarabel `max_threads`, SDPA `numThreads`) sizes the solver but not the
  worker's BLAS / OpenMP pools, so use the shared setting to give BLAS more
  threads. The values a solve ran with are on `metadata.runtime.thread_env`
  (`null` for a variable that was not set).
- `max_iter`: the iteration cap: an integer (an integer-valued number such as
  YAML's `1e3` is accepted; `true`, `1.5` or a string is refused), `null` for
  the solver's default. It goes under the solver's own name where it differs
  (HiGHS: every algorithm's `*_iteration_limit`; Gurobi: `BarIterLimit` and
  `IterationLimit`; CPLEX: `barrier.limits.iteration` and
  `simplex.limits.iterations`; MOSEK: `MSK_IPAR_INTPNT_MAX_ITERATIONS` and
  `MSK_IPAR_SIM_MAX_ITERATIONS`; cuOpt and PDLP: `iteration_limit`; cpo3, SCS
  and ECOS: `max_iters`; CVXOPT: `maxiters`; SDPA: `maxIteration`). A native
  name given alongside wins. HiGHS's own `*_iteration_limit` settings and
  ECOS's `max_iters` and `nitref` (refinement steps) follow the same integer
  rule.
- `eps_abs` / `eps_rel` (and `eps`, which sets both): the absolute and relative
  tolerances, native for OSQP, PIQP, ProxQP, PDLP and SCS and translated for
  QTQP, cpo3, Clarabel, ECOS, CVXOPT, SDPA and cuOpt. An alias set to `null`
  counts as unset (`eps` then still applies), and a boolean is refused (a
  solver that takes any number would read `eps_abs: true` as 1.0). HiGHS,
  Gurobi, CPLEX and MOSEK have no such pair and refuse these names; set their
  own tolerances instead.

Every adapter refuses a setting it does not know, or a value it or its solver
rejects, with a `SolverConfigError` (a `ValueError`) naming it, so a misspelled
option cannot silently leave a solver on its defaults and a bad value is not
scored as a solver failure. A rejected setting stops that solver id at the
first solve that hits it: its remaining solves are not run and none of its
solves is recorded (a `[bench] error:` line and an `error` event in
`events.jsonl` say so), while the other solvers' solves run to completion.
The run then exits with an error naming every rejected solver and writes no
`results.parquet`; the error names the run directory to resume with
`--run-dir`. `bench missing` lists the rejected solver's solves (rows it
already had from other settings or an earlier release stay until a corrected
config replaces them), a re-run with the same settings fails the same way
after one solve per rejected solver in a serial run (a few, up to about
`run.parallelism`, in a parallel run), and a corrected config runs them. PIQP and PDLP report a rejected value as a status
(`PIQP_INVALID_SETTINGS`, `TERMINATION_REASON_INVALID_PARAMETER`), which the
adapter turns into that error. QTQP's `solve()` raises the same exception for a
bad value as for problem data it cannot take, so the adapter checks the ranges
it enforces first. ProxSuite checks no values, so the ProxQP adapter itself
refuses a negative or non-finite `eps_*` tolerance, a `rho` / `mu_eq` / `mu_in`
that is not positive, and a `max_iter` below 1. SCS, ECOS, CVXOPT and
Clarabel get the same treatment: the adapter checks the values their solver
would refuse from the solving call (or, for SCS and ECOS, turns the solver's
own refusal of a setting into that error). HiGHS's on/off options
(`presolve`, `run_crossover`, `parallel`, ...) also accept a YAML boolean,
since YAML reads an unquoted `off` as `false`. Translated aliases are listed on
the result's `info.settings_translated`.

In the QP form `l <= A x <= u`, every QP adapter (MOSEK included) treats a bound
with magnitude `>= 1e19` on the open side as infinite (the datasets write `+-1e20`, sometimes
with a few ULPs of error; the transform layer and the Maros-Meszaros and QPS
readers use the same cutoff), and passes true infinities to its solver. Rows
with no finite bound constrain nothing and are not passed to SCS, OSQP, PIQP,
ProxQP, HiGHS, PDLP, Gurobi or CPLEX (unless every row is free, for SCS); their
dual is reported as zero. The nonnegative-cone form used by Clarabel, ECOS,
CVXOPT, QTQP and cpo3 has no row for them.

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
- Run-level `dataset_options` are applied as defaults to the listed datasets
  that read each option, and entries can override or extend them per dataset:
  with `datasets: [netlib, maros_meszaros]`, a run-level `subset: feasible`
  selects the feasible NETLIB problems and leaves Maros-Meszaros, which has no
  `subset`, whole. An option that no listed dataset reads is an error.
- Run-level `include` is used as a fallback only for datasets whose own
  `include` is unset, so per-dataset selections do not bleed across datasets.
  Each of its names must match a problem in at least one of those datasets;
  an entry's own `include` names must match in that dataset.
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
| `configs/synthetic_smoke.json` | Tiny smoke run: the two synthetic QPs with SCS and OSQP. |
| `configs/netlib_feasible_example.yaml` | NETLIB feasible subset example. |
| `configs/maros_meszaros_example.yaml` | Maros-Meszaros QP example: QTQP with its default backend and with `linear_solver: scipy`, SCS, and Clarabel. |
| `configs/maros_meszaros_qp_solvers.yaml` | Maros-Meszaros QP run comparing OSQP, SCS, Clarabel, QTQP, and cpo3. |
| `configs/multi_dataset_example.yaml` | Example of one run spanning several datasets. |
| `configs/scs_anderson_sweep.yaml` | SCS Anderson acceleration sweep over NETLIB feasible, MIPLIB instances up to 5 MB, Maros-Meszaros, and SDPLIB instances up to 1 MB. Includes `scs_aa_disabled` with `acceleration_lookback: 0`, plus the cross-product `acceleration_lookback: [5, 10, 20]` and `acceleration_interval: [1, 5, 10]`. |
| `configs/qtqp_refinement_regularization_sweep.yaml` | QTQP sweep over NETLIB feasible, MIPLIB instances up to 5 MB, and Maros-Meszaros. Uses QTQP's default backend (MKL Pardiso on x86-64 Linux and Windows, Apple Accelerate on arm64 macOS) and sweeps `max_iterative_refinement_steps: [1, 3, 5, 10, 20]` by `min_static_regularization: [1.0e-10, 1.0e-8, 1.0e-6]`. |
| `configs/scs_linear_solver_backend_sweep.yaml` | SCS comparison of `linear_solver: qdldl` versus `linear_solver: cpu_indirect` (both bundled in every SCS wheel) on the same datasets as the Anderson sweep. `mkl` (x86-64 Linux and Windows wheels) and `accelerate` (macOS wheels) can be added as arms where the installed wheel has them. |
| `configs/qtqp_linear_solver_backend_sweep.yaml` | QTQP comparison of its default backend (MKL Pardiso on x86-64 Linux and Windows, Apple Accelerate on arm64 macOS) versus `linear_solver: scipy` on NETLIB feasible, MIPLIB instances up to 5 MB, and Maros-Meszaros. Other backends such as `qdldl` need their own package (`pip install qdldl`); the `qtqp` extra does not install them. |

These run with the released solvers that `pip install -e ".[all]"` installs. Most
other files in `configs/` are research sweeps that need development builds of SCS
or QTQP. Each of those starts with an `# EXPERIMENTAL:` header that names what it
needs. With the released packages, arms whose settings the solver rejects are
not run (the run finishes the other arms and exits with an error naming them),
and arms that need a missing backend module fail as `worker_error`. See
[`configs/README.md`](configs/README.md).

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
problems are listed, and the options that only choose which of a dataset's
problems are listed: `subset` of `qplib`, `cblib`, `mpc_qpbenchmark`,
`cutest_qp`, `mittelmann_sdp`, `tsplib_sdp`, `dc_opf`, `libsvm_qp` and `sdplib`,
`cblib`'s `subset_kind` and `include_unsupported`, and the four `liu_pataki`
filters. Widening a subset therefore solves only the problems it adds. NETLIB's
`subset` stays in the signature (its two folders share problem names for
different problems), as do the options that change the problem posed, such as
`libsvm_qp`'s `kind` or `sdplib`'s `blocks`. Rows written by earlier releases,
which also hashed the location options or the selection options, still count
when everything else matches, also after those options change: the first resume
in such a directory works out, from the manifest the earlier release left,
which current signature those rows' signatures stand for, and records it in the
manifest for later resumes. A row solved under other settings is superseded: the
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
retry, or the solve of such a skip, appends its row, and the error or skip rows
it replaced are dropped from `results.jsonl` in the same end-of-run rewrite;
after a crash that skips that step, the next resume drops them. Until then
the readers (`load_results`, so `bench summary`, `failures`, `report` and the
other analysis commands, and `bench merge` / `kkt-verify`) leave out a
`worker_error` or `solver_unavailable` skip row that a later row of the same
run id, solve and resume signature follows, so they count one result for that
solve while a resume runs and after it was killed (in a `bench merge` output,
rows of runs with different run ids are never paired this way). Previous
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

A run directory keeps the run id it was created with (recorded in its
`manifest.json`), also when it is renamed or copied: a resume into it writes
rows under that run id, not under the directory's new name. Without a readable
manifest the directory's name is the run id.

A run directory has one writer at a time: a run holds a lock on
`<run_dir>/.bench.lock` until it ends, and a second `bench run` on the same
directory fails at once, naming the process that holds it.

A run that fails or is interrupted while planning its solves (a dataset
without local data, a solver `run.fail_on_unsupported` does not let it skip,
Ctrl+C) removes the run directory it created, timestamped or named by
`--run-dir`, and the empty directories it created to hold it; a directory that
existed before the run is never removed.

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
| `metadata.runtime.python_executable` | Name (basename) of the Python executable used by the solve subprocess, e.g. `python3.12`. |
| `metadata.runtime.python_version` | Python version. |
| `metadata.runtime.platform` | OS/platform string. |
| `metadata.runtime.solver_package_versions` | Installed package versions relevant to the solver adapter, such as `{"osqp": "1.1.1"}`. |
| `metadata.runtime.library_versions` | Installed numpy and scipy versions, which ship the BLAS/LAPACK most solvers use. |
| `metadata.runtime.usable_cpu_count` | CPUs the solve subprocess may run on (its CPU affinity). |
| `metadata.runtime.thread_env` | The solve subprocess's `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` (`null` when not set); see `threads` under [Shared Solver Settings](#shared-solver-settings). |
| `metadata.environment_id` | Optional user-supplied environment label. |
| `metadata.environment_metadata` | Optional user-supplied metadata, usually pinned solver versions or environment notes. |
| `metadata.gpu_warmup` | GPU rows only: what the warm-up before the timed region did and how long it took (`seconds`, `cuda_context_seconds`, `cublas_seconds`, `notes`, ...); see [GPU warm-up](#gpu-warm-up). |

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

Each `bench run` into the directory rewrites `manifest.json` and
`run_config.yaml` from its own config, so after the manual workflow they
describe only the last config run (here `osqp_1_1.yaml`). The results of both
environments are kept and analysed, but `bench missing` and the completion
table in `bench summary` / `bench report` check only the last config's solver
variants: an interrupted first-environment run is not reported as incomplete.
When the full plan matters, use the environment matrix below, whose manifest
records every environment's variants, or run each environment into its own
directory and combine them with `bench merge`.

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
`python -P -m solver_benchmarks.cli run ...`, so the interpreters must be
Python 3.11 or newer (the `-P` flag). When `bench env run` itself imports
`solver_benchmarks` and `problem_classes` from a checkout or a `PYTHONPATH`
directory (an editable install, `python -m` from a clone), those directories
come first on each environment's `PYTHONPATH`: every environment runs this
harness's code with its own interpreter and installed packages (numpy, scipy,
the solvers, and the harness's dependencies, which the `pip install -e .` in
the examples installs). When `bench` is imported from site-packages nothing is
handed over (the run prints a line saying so): each environment must have
`solver_benchmarks` installed and runs that copy. An inherited `PYTHONPATH`
follows on each environment's `PYTHONPATH`, with relative entries taken
relative to the directory the command is run from; empty entries are
dropped, so a package reachable only through one must be listed explicitly
(`.` or an absolute path).
This keeps the benchmark independent of a specific environment manager. You can use `venv`, `uv`, conda, Docker, or
pre-existing paths.

Install commands are executed directly, not through a shell. The runner
tokenizes the command with shell-like quoting, then expands `{python}` to the
environment's configured Python executable and `{repo_root}` to the repository
root in each token, so a path containing spaces stays one argument. If you
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

After a version comparison run, normal analysis commands work unchanged (with
the manual workflow's completion caveat above):

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
| `bench data status [DATASET]` | `--repo-root PATH`, `--option key=value` | Report whether local data is available, how many problems are visible after options and generic filters, the data path, source, exact preparation command when one exists, and status message. Omit `DATASET` to check all datasets: each then gets only the `--option`s it reads, a dataset that rejects the value given is skipped with a note, and an option no dataset reads is an error. |
| `bench data prepare DATASET` | `--repo-root PATH`, `--option key=value`, `--problem NAME`, `--all` default false | Download or prepare missing data for datasets that support automatic preparation. Repeat `--problem` for selected instances. Use `--all` only when you intentionally want every known remote problem. |

Run and analysis commands:

| Command | Arguments | Purpose |
|---|---|---|
| `bench run CONFIG_PATH` | `--run-dir PATH`, `--repo-root PATH` (default: the current directory if it has `problem_classes`, else the checkout the package was installed from, as for the other commands), `--data-root PATH`, `--prepare-data` default false, `--stream-output` / `--no-stream-output`, `--environment-id ID`, `--environment-metadata JSON` | Execute a benchmark config. Without `--run-dir`, creates a new immutable run directory named from `run.name` or the config filename stem plus a readable UTC timestamp, and copies the source config into the run directory. With `--run-dir`, resumes/appends to that run subject to `resume: true`. `--data-root` sets the `data_root` dataset option of every dataset in the config: the directory used in place of `problem_classes/`, holding each dataset's `*_data` folder. Without `--prepare-data`, missing external data is reported with exact preparation commands instead of being downloaded implicitly. `--no-stream-output` keeps solver stdout/stderr in per-solve log files but does not tee it to the terminal; benchmark progress still prints. The environment flags are normally used by version-comparison workflows and are recorded in result metadata. |
| `bench env run CONFIG_PATH` | `--run-dir PATH`, `--repo-root PATH`, `--stream-output` / `--no-stream-output` | Execute an environment matrix config. Each environment supplies a Python executable, optional install commands, metadata, and solver variants; all results are written into one run directory, with the source environment config copied into it. `--no-stream-output` is forwarded to each child benchmark run. |
| `bench summary RUN_DIR` | `--repo-root PATH`, `--data-root PATH`, `--unverified`, `--verify-tol VALUE`, `--reverify`, `--allow-missing`, `--requested-tol SOLVER_ID=TOL` (repeatable), `--tol-from-solver-id` | Print how the statuses were verified, solver metrics, status counts, a verification table (per solver: the tolerance its settings requested and where that came from, the tolerances its rows were checked at, and the claims kept, demoted, promoted, returned after the time limit and kept unverified), and run completion information. Statuses are KKT-verified by default (see [Independent KKT verification](#independent-kkt-verification)); the verification options choose another rule. Solver metrics count each `(dataset, problem, solver)` once (a duplicated solve counts as its successful attempt); `total` is the number of solves the success and failure counts and rates are over (planned solves never recorded included), `completed` the number with a recorded result; run time and iteration totals cover every solve, means, medians and maxima the successful ones. A dataset whose expected problems are unknown (no problems of it found here) has empty `expected`, `missing` and `unexpected` counts, `complete` False, and a line saying so. |
| `bench failures RUN_DIR` | `--repo-root PATH`, `--data-root PATH`, `--unverified`, `--verify-tol VALUE`, `--reverify`, `--allow-missing`, `--requested-tol SOLVER_ID=TOL` (repeatable), `--tol-from-solver-id` | Print success/failure rates by solver over KKT-verified statuses, one count per solve. Planned solves with no recorded row count as `not_attempted` failures, as in the report (`bench summary` counts them the same way). Only `optimal` counts as success by default; on problems known to be infeasible or unbounded only the matching certificate does (see [Known outcomes](#known-outcomes)). |
| `bench missing RUN_DIR` | `--repo-root PATH`, `--data-root PATH` | Print missing `(solver, dataset, problem)` results relative to the run manifest and dataset filters (which rows exist, so no verification is needed). A dataset whose expected problems cannot be listed (its directory is not found here or holds none of its problems) is named with that directory, instead of being reported complete. |
| `bench profile RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--repo-root PATH`, `--data-root PATH`, `--unverified`, `--verify-tol VALUE`, `--reverify`, `--allow-missing`, `--requested-tol SOLVER_ID=TOL` (repeatable), `--tol-from-solver-id` | Write Dolan-More performance profile data, over KKT-verified statuses, to `performance_profile_<metric>.csv` (how they were verified is printed on stderr). Solves the manifest planned but the run did not record count as failures; `--repo-root` and `--data-root` locate the dataset directories used to list them (see below). |
| `bench geomean RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--shift VALUE` default metric-dependent, `--max-value VALUE` optional, `--success-only` default false, `--repo-root PATH`, `--data-root PATH`, `--unverified`, `--verify-tol VALUE`, `--reverify`, `--allow-missing`, `--requested-tol SOLVER_ID=TOL` (repeatable), `--tol-from-solver-id` | Write shifted geometric means, over KKT-verified statuses, into the run directory. Time-metric failures cost three times the manifest's time limit unless `--max-value` overrides it; use `--success-only` for solved-problem-only geomeans. |
| `bench plot RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--output-dir PATH` default run dir, `--max-value VALUE` optional, `--repo-root PATH`, `--data-root PATH`, `--unverified`, `--verify-tol VALUE`, `--reverify`, `--allow-missing`, `--requested-tol SOLVER_ID=TOL` (repeatable), `--tol-from-solver-id` | Write PNG plots, over KKT-verified statuses, for Dolan-More profile, cactus plot, pairwise scatter, performance-ratio heatmap, shifted geomean, status heatmap, failure rates, three KKT-residual plots (per-solver boxplot, problem-by-solver heatmap, and KKT-accuracy profile), a setup-vs-solve time breakdown (`setup_solve_breakdown.png`), and a difficulty-scaling plot of the metric by problem size (`difficulty_scaling_<metric>.png`). |
| `bench report RUN_DIR` | `--metric FIELD` default `run_time_seconds`, `--output-dir PATH` default `RUN_DIR/report`, `--max-value VALUE` optional, `--repo-root PATH`, `--data-root PATH`, `--unverified`, `--verify-tol VALUE`, `--reverify`, `--allow-missing`, `--requested-tol SOLVER_ID=TOL` (repeatable), `--tol-from-solver-id` | Write a complete report directory containing CSV tables, PNG plots, and a generated Markdown report, all over KKT-verified statuses; the report opens with how they were verified (`verification.csv`, `verification_changes.csv`). |
| `bench merge OUT_DIR RUN_DIRS...` | `--overwrite` default false | Concatenate the results of run directories sharded across machines (for example one solver or dataset per run) into a new run directory `OUT_DIR` for analysis. Rows are kept as they are, except a `worker_error` or `solver_unavailable` skip row that a later row of the same run id, solve and resume signature replaced (an input whose resume is running or was killed). Duplicate `(dataset, problem, solver_id)` keys across the inputs are counted and reported, not resolved, and a warning lists problems that the inputs posed with different numbers of rows. The merged manifest unions the inputs' datasets and solvers and keeps each input's selection and manifest (each row's settings are those of the input whose run id it carries), so `summary`, `missing` and `report` treat the merged directory like one run. A warning says when the inputs were verified differently (some by `bench kkt-verify`, or at different tolerances); the merged run is verified again from the rows' reported statuses when it is analysed. `OUT_DIR` must not exist unless `--overwrite` is given, and must not overlap an input. `--overwrite` replaces only an empty directory or a run written by `bench merge` / `bench kkt-verify` (its manifest has `derived`); a base run directory, a file or a directory of other files is refused. |
| `bench kkt-verify RUN_DIR` | `--tol VALUE` optional, `--slack VALUE` default `10`, `--output-dir PATH` default `RUN_DIR_verified` (`RUN_DIR_kkt<tol>` with `--tol`) next to `RUN_DIR`, `--allow-missing` default false, `--promote` / `--no-promote` default `--promote`, `--requested-tol SOLVER_ID=TOL` (repeatable), `--tol-from-solver-id`, `--overwrite` default false | Write a copy of the run with KKT-verified statuses, by the rule every analysis command applies when it reads a run (see [Independent KKT verification](#independent-kkt-verification)): every claimed optimum whose relative KKT residuals (cone distances and absent-bound multipliers included) exceed the row's tolerance becomes `optimal_inaccurate`, and every `primal_infeasible` / `dual_infeasible` whose certificate is not accepted (neither a proof nor of a kind the problem's known outcome confirms with its scaled certificate residual within the tolerance, see [Certificates](#certificates)) the matching `*_inaccurate` status. The tolerance is `--slack` times the tolerance the solver's settings requested, for solutions and certificates alike; `--tol` holds every row to one value instead (and refuses `--slack`, `--requested-tol` and `--tol-from-solver-id`, which it would not use). `--requested-tol` and `--tol-from-solver-id` supply requested tolerances the run does not record; without one where it is needed nothing is written and the command says which solvers lack one. Claimed optima with residuals missing, and infeasibility claims whose adapter recorded no certificate (cuOpt never records one, its API exposing none; HiGHS, Gurobi and CPLEX do not when the solver produces none, as for an unbounded QP, see [Supported Solvers](#supported-solvers); PIQP, ProxQP, SDPA and MOSEK do not when the solver returns no finite certificate), cannot be verified and are demoted unless `--allow-missing`. With `--allow-missing`, infeasibility-heavy comparisons favour adapters that record no certificate: a recorded certificate can fail the check (Clarabel's genuine NETLIB certificates reach 6e-5), an unrecorded one cannot. Claims whose only unknown residuals are distances to cones the harness cannot project (complex PSD) are kept whatever `--allow-missing` says, with a warning that counts them by solver. `--promote` (the default; `--no-promote` turns it off) turns inaccurate, iteration- or time-limited solves whose point passes into `optimal`, and `primal_infeasible_inaccurate` / `dual_infeasible_inaccurate` rows whose certificate is fully checked and passes into `primal_infeasible` / `dual_infeasible`, as it does an iteration- or time-limited row whose record holds such a certificate instead of a point (OSQP's inaccurate certificates) (never when the certificate is missing or only partly checked), except a result returned after the run's time limit, which keeps its status and is counted as `over_time_limit`. The runner marks every such result with `info.exceeded_time_limit`; in runs recorded before that marker, only a conclusion recorded as `time_limit` (with `info.status_before_timeout`) is recognised, and an iteration- or time-limited stop returned after the limit is still promoted. A solver stopped by its own time limit returns slightly after it, so when that limit equals `timeout_seconds` its time-limit stops are never promoted; set the solver's limit below `timeout_seconds` if those points should be eligible. A row promoted to `optimal` without an `objective_value` (adapters report one only for a solution) takes the objective of its returned point, which the worker records as `kkt.primal_obj_reported` (constant term and maximisation applied, as for `objective_value`); `metadata.kkt_verify.objective_from` says so. Every checked row gets `metadata.kkt_verify` (its verdict, rule, tolerance, slack, promotion and `--allow-missing` setting, worst residual, reason and `original_status`), and every row is judged from the status it was reported with, so verifying a verified run under the same rule writes the same rows (a copy of a copy written before this version recorded the status of the step before as `original_status`, and starts from it). Unparseable lines (a torn last line) are skipped by both commands, and the output directory is replaced only once complete. The command prints kept, demoted, promoted and no-residual counts per solver, the over-time-limit count where there is one, and the requested and checked tolerances (`requested=1e-04`, `check=1e-03`). Both commands refuse an existing output directory unless `--overwrite`, which replaces only an empty directory or an earlier derived run, never a base run or other files; a source directory with neither `results.jsonl` nor `manifest.json`; and a `manifest.json` that is not valid JSON. |

Common metrics for `profile` and `geomean` are `run_time_seconds`,
`solve_time_seconds`, `setup_time_seconds`, and `iterations`, depending on what
each solver reports (see [Solve Times](#solve-times) for which adapters record
the two solver-reported times and what they cover).

When run from the CLI, `bench run` writes solver stdout/stderr to each solve's
`stdout.log`/`stderr.log` (solvers print nothing there unless their settings
say `verbose: true`). By default it also streams that solver output to the
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
    netlib/
      afiro/
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
| `results.parquet` | Flattened results table for pandas/Arrow analysis, written when a run finishes; its presence marks a completed run. Columns whose values mix types across adapters (e.g. `info.raw_status`, an integer in Gurobi and CPLEX rows recorded by older versions, a string elsewhere) are stored as text. If it cannot be written, the error is logged to `events.jsonl` and `results.jsonl` remains the complete record. |
| `payload.json` | Exact payload passed to the subprocess worker. |
| `stdout.log` | Everything emitted to subprocess stdout during a solve. |
| `stderr.log` | Everything emitted to subprocess stderr during a solve. |
| `worker_result.json` | Raw worker output before aggregation. |
| `result.json` | Per-solve result record matching `results.jsonl`. |
| `trace.jsonl` | Iteration trace when the solver adapter exposes one. |

## Logging and Provenance

Each solve runs in a subprocess. This gives:

- Native solver crashes do not kill the main runner. The `worker_error` row of a worker that died names the signal (SIGKILL usually means out of memory) and quotes the last lines of its stderr.
- Timeouts are enforced per solve: the worker's whole process group is killed, so helper processes a solver started do not outlive it. Ctrl+C, or a SIGTERM or SIGHUP to `bench run` / `bench env run` (Slurm's `scancel`, `kill`, a CI job timeout, a closed terminal or dropped ssh session), kills the running workers too, and a parallel run keeps the results that had already finished; a SIGTERM exits with status 143, a SIGHUP with 129. Under `nohup` the SIGHUP stays ignored. A runner killed outright (SIGKILL, the out-of-memory killer) takes its workers with it on Linux; on other POSIX systems they exit within about a second, or, while the solver holds Python's interpreter lock in a long native call, as soon as it releases it. On Windows workers are not tied to the runner. A `bench env run` killed outright stops the current environment's `bench run` as a SIGTERM would (on POSIX systems; at once on Linux, within about a second elsewhere), and that run kills its workers.
- Workers run the harness code `bench` itself runs: they start with `python -P` in the repository root (`--repo-root`, the data's location), so that directory is not on `sys.path`. When `bench` imports `solver_benchmarks` and `problem_classes` from a checkout or a `PYTHONPATH` directory (an editable install, `python -m` from a clone), those directories come first on the workers' `PYTHONPATH`; when it imports them from site-packages nothing is handed over, and the worker's interpreter finds the same site-packages. An inherited `PYTHONPATH` follows, without empty entries (which would mean the working directory) and with relative entries made absolute against the runner's working directory. A `solver_benchmarks` copy or a file named like a standard module (`json.py`) in the data root is not what runs.
- Each worker has a private temporary directory (`TMPDIR`, `TEMP` and `TMP`) under the runner's own, removed with its contents once the worker's process group is gone, after a normal exit, a timeout or an interrupted run. A worker killed while it holds the decompressed copy of a `.mps.gz` file or a solver's scratch files leaves nothing behind. A runner killed outright (SIGKILL, the out-of-memory killer) cannot clean up: each live worker's `bench-worker-*` directory stays in the temporary directory and can be deleted by hand.
- Solver stdout/stderr (from solvers set to `verbose: true`) are streamed live by the CLI and captured per problem and solver variant.
- The exact worker payload is retained.
- Completed solves are discoverable for resume.

Warnings are structured JSON records. Example `events.jsonl` line:

```json
{
  "timestamp_utc": "2026-04-24T10:00:00+00:00",
  "level": "warning",
  "message": "Solver 'osqp' does not support 'cone' problems; skipping 'qssp30'",
  "solver_id": "osqp_skip",
  "dataset": "dimacs",
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
the top of the Provenance block (after the `### Derived Run` block of a run
written by `bench kkt-verify` or `bench merge`):

```text
### System

| Field | Value |
| --- | --- |
| CPU model | Apple M4 |
| CPU cores | 10 logical, 10 usable by this run |
| Total RAM | 16.0 GiB |
| OS | macOS-26.3.1-arm64-arm-64bit |
| Python | 3.12.8 |
| Libraries | numpy 2.4.2, pandas 3.0.0, pyarrow 23.0.1, scipy 1.16.3 |
```

When the manifest records `cpu.usable_count`, the `CPU cores` row ends with the
number of CPUs the run could use, e.g. `64 physical / 128 logical, 4 usable by this run`.

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

### Independent KKT verification

Every analysis reads KKT-verified statuses: `bench summary`, `failures`,
`profile`, `geomean`, `plot` and `report`, `tools/make_figures.py`, and
`load_results` in Python. A solver's status is a claim; the harness checks it
against KKT residuals it computes itself from the returned point or
certificate and the problem data (the `kkt.*` fields every solve records, see
`solver_benchmarks/analysis/kkt.py`), and the status every table, plot and
count uses is the verified one:

- A claimed solution (`optimal`) stays `optimal` only if the worst of its
  relative primal residual, dual residual and duality gap (`kkt.primal_res_rel`,
  `kkt.dual_res_rel`, `kkt.duality_gap_rel`, plus the absent-bound multipliers
  `kkt.dual_bound_res_rel` of QP-form records or the relative cone distances
  `kkt.primal_cone_res_rel` / `kkt.dual_cone_res_rel` of cone-form ones) is at
  or below the row's tolerance; otherwise it becomes `optimal_inaccurate`. A
  claimed solution without complete residuals is demoted too.
- A claimed certificate (`primal_infeasible`, `dual_infeasible`) stays only if
  it is a proof (`kkt.cert_proof`, see [Certificates](#certificates)), or if
  the problem's known outcome confirms its kind and its scaled certificate
  residual `kkt.cert_res` (the certificate scaled so that its objective term
  is −1) is at or below the row's tolerance, the same as for its solutions.
  A certificate the known outcome rules out (any certificate on a problem
  with an optimal solution, a primal one on an unbounded problem) never
  stays. Otherwise it becomes the matching `*_inaccurate` status.
- An inaccurate, iteration-limited or time-limited stop whose point passes
  becomes `optimal` (taking the point's objective, `kkt.primal_obj_reported`,
  when it has none), and an inaccurate certificate, or a limit stop that
  recorded a certificate, that passes becomes the accurate certificate status.
  A result returned after the run's time limit (`info.exceeded_time_limit`) is
  never promoted, and `tools/make_figures.py` applies its 1 % late rule after
  verification.
- A claim with nothing to check counts as unsolved, since it cannot be
  verified: a claimed solution without complete residuals, a claimed
  certificate the adapter did not record (cuOpt never records one, its API
  exposing none; HiGHS, Gurobi and CPLEX do not when the solver produces
  none, as for an unbounded QP, see [Supported Solvers](#supported-solvers);
  PIQP, ProxQP, SDPA and MOSEK do not when the solver returns no finite
  certificate), and a certificate that needs a proof in a record written
  before proofs were recorded. `--allow-missing`
  (`VerifyPolicy(missing_is_failure=False)`) keeps them, counted as
  unverified; in infeasibility-heavy comparisons that favours adapters that
  record no certificate, since a recorded certificate can fail the check and
  an unrecorded one cannot. A record without a proof whose scaled residual
  `kkt.cert_res` exceeds the row's tolerance is demoted either way: the
  unknown proof does not hide a residual known to fail.
- Claims whose only unknown residuals are distances to cones the harness
  cannot project (complex PSD) are kept when everything else passes, and
  counted as unverified. The verification header every command prints says
  how many claims were kept without a full check; `bench summary` and the
  report count them per solver.

**The tolerance.** A row is held to `10` (`VERIFY_SLACK` in
`analysis/verify.py`) times the tolerance its configuration asked the solver
for. That requested tolerance is the loosest of the feasibility and gap
tolerances, absolute and relative, the solver is asked to meet, with every
knob the settings leave unset at the library's default: a solver that stops
on `r <= abs + rel * N` has `r / (1 + N) <= max(abs, rel)`, which is how the
harness measures its residuals, and the factor of ten absorbs the different
norms and scalings solvers measure in (it is the factor of the published
3.3.2 figures, which were verified at `--tol 10*eps`). Each adapter computes
it with its own alias translation (`SolverAdapter.requested_tolerance`), so
`eps`, `eps_abs` and `eps_rel` mean what they mean in a solve. Certificates
are held to the same tolerance as the row's solutions. The solvers' own
infeasibility tolerances are not used, because they measure certificates in
units other than `kkt.cert_res`: OSQP (and, by its documentation, ProxQP)
divides the residual by the norm of the ray, Clarabel by the larger of that
norm and one, cpo3 by the larger of that norm and the objective term, and
QTQP adds a term in the ray's norm; only SCS's and PDLP's tests divide by the
objective term alone. Held to those knobs at their defaults, the certificates
in one figure would be checked at tolerances up to a thousand times apart,
and Clarabel's genuine NETLIB certificates (`cert_res` up to `6e-5`) would
fail at its `1e-8`. So the infeasibility knobs (SCS's `eps_infeas`, OSQP's
`eps_prim_inf` and `eps_dual_inf`, Clarabel's `tol_infeas_abs` and
`tol_infeas_rel`, and the like) are deliberately not read when a row is
verified, whatever a run sets them to. Campaigns instead ask each solver for
its certificates at the same tag as its solutions: every variant
`tools/make_campaign.py` writes requests exactly its tolerance tag in its
solution knobs (HiGHS below `1e-7` excepted: its residual tolerances stay at
their `1e-7` default), and sets every infeasibility knob its solver has to the
tag as well (`CERTIFICATE_KNOBS` there; the last column of the table below),
except SCS's `eps_infeas`, which is never looser than SCS's default (below);
and a campaign's solutions and certificates are both verified at ten times
the tag, the rule of the published 3.3.2 figures. That puts the request and
the check at the same nominal tolerance (for SCS at tags above `1e-7`, a
tighter request), not the same test: each solver still
measures its certificates in its own units (above), and Clarabel's, cpo3's
and QTQP's `tol_infeas_abs` is not a residual tolerance but the margin a
certificate's objective term must clear (`b'y < −tol_infeas_abs`), so at the
tag it is stricter than its `1e-8` default, while `tol_infeas_rel`, their
residual test, is looser. PIQP, HiGHS, CVXOPT and SDPA have no such knob
the adapter can set, and the table says why.

The knobs are set in every family, the feasible ones included, where a
certificate is always a failure (see [Known outcomes](#known-outcomes)).
There a knob looser than the solver's default can only end solves early, on
feasible problems close enough to infeasible to admit an approximate
certificate. A solver whose certificate test is not gated on `κ/τ`, as
Clarabel's, cpo3's and QTQP's are, is exposed. SCS's is not gated, and a
campaign never sets its `eps_infeas` looser than SCS's default: it is the
tighter of the tag and SCS's default `1e-7` (maintainer decision,
2026-11-21: "it should be the tighter of the tag and 1e-7"), so
`1e-7` at the `1e-4` to `1e-6` tags and the tag at `1e-8`, in every family
(`SCS_EPS_INFEAS_DEFAULT` in `tools/make_campaign.py`). SCS's default does
not change with `eps_abs` and `eps_rel`, and a certificate threshold looser
than the default is not how users run SCS. SCS still looks for a
certificate on every problem, and its certificates are verified at ten
times the tag like every solver's. The tag would also have cost SCS
feasible solves: on the 93 feasible NETLIB LPs at `1e-4` (SCS 3.3.1, a 60 s
limit, verified statuses) SCS solved 79 with `eps_infeas` at the tag against
83 at its default, because `agg3`, `fffff800`, `fit2d` and `israel` ended in
certificates instead of solutions, and at `1e-6` it solved 76 against 77
(`agg3`). On the 138 Maros-Meszaros QPs at `1e-4` it solved 124 against
131: `AUG2DCQP`, `AUG2DQP`, `HUES-MOD`, `HUESTIS`, `LISWET5`, `QGFRDXPN` and
`STADAT1` ended in certificates. On the 29 infeasible NETLIB LPs at `1e-4`
the tag gained SCS one certificate (`bgindy`, 24 against 23). On the
feasible NETLIB LPs at `1e-4` Clarabel solved 90 either way, and OSQP's
knobs at `1e-4` are its defaults. PDLP (default `1e-8`), cuOpt, the SDP
family and the large LPs are not measured. Nor are the knobs the tag sets tighter than
their defaults, which on an infeasible problem can only lose or delay a
certificate: OSQP's `eps_prim_inf` / `eps_dual_inf` and ProxQP's
`eps_primal_inf` / `eps_dual_inf` (default `1e-4`) at tags below `1e-4`,
SCS's `eps_infeas` at tags below `1e-7`, and the `tol_infeas_abs` margin of
Clarabel, cpo3 and QTQP at every tag above `1e-8`. Nor is turning on cuOpt's
PDLP infeasibility detection (see [Published Results](#published-results)),
which needs a GPU.

Archived campaigns keep the settings they recorded and are verified by the
same uniform rule: the `scs332_2026-11` infeasible-LP runs gave their `1e-8`
SCS, OSQP and Clarabel variants infeasibility tolerances of `1e-4`, and those
variants' certificates are still checked at `1e-7` (they are not re-run).

Verification uses the residuals the worker recorded at solve time; it does
not recompute them. Runs recorded before #75 (`f942f1f`, 2026-08-29) carry a
known-wrong `duality_gap_rel` of about `1` on Maros-Meszaros, so most of their
claimed optima there are demoted; read them with `--unverified`, or re-run
them. The defaults below are those of the versions named; a row of an older
version that leaves a knob unset is checked at today's default, except SCS
before 3.0 and qtqp before 0.0.7, whose rows must set their tolerances (the
tolerance is otherwise unknown).

| Solver | Knobs (the loosest is requested) | Default request | Defaults | Infeasibility knobs `make_campaign` sets to the tag |
|---|---|---|---|---|
| `clarabel` | `tol_feas`, `tol_gap_abs`, `tol_gap_rel` (`eps_abs` / `eps` set the first two, `eps_rel` / `eps` the last) | `1e-8` | checked on clarabel 0.11.1 | `tol_infeas_abs`, `tol_infeas_rel` (default `1e-8`; its `reduced_tol_infeas_*`, like its other reduced tolerances, stay at their defaults) |
| `cplex` | `simplex.tolerances.feasibility`, `simplex.tolerances.optimality`, `barrier.convergetol` (no `eps` aliases) | `1e-6` | documented, not checked | not in campaigns |
| `cpo3` | `tol_feas`, `tol_gap_abs`, `tol_gap_rel` (as Clarabel) | `1e-8` | cpo3 0.0.1 source (`src/settings.rs`), not checked against an installed cpo3 | `tol_infeas_abs`, `tol_infeas_rel` (as Clarabel) |
| `cuopt` | `absolute_primal_tolerance`, `absolute_dual_tolerance`, `absolute_gap_tolerance`, `relative_primal_tolerance`, `relative_dual_tolerance`, `relative_gap_tolerance` (`eps` and `optimality_tolerance` set all six, `eps_abs` / `eps_rel` three each) | `1e-4` | documented, not checked | `primal_infeasible_tolerance`, `dual_infeasible_tolerance` (its PDLP's; default `1e-10` in the cuOpt 26.08 and 26.10 source, not checked on a GPU; they act only with PDLP's `infeasibility_detection`, off by cuOpt's default and on in every campaign variant and by the adapter's default; by cuOpt's documentation its dual simplex detects infeasibility either way) |
| `cvxopt` | `abstol`, `reltol`, `feastol` (`eps_abs` / `eps` set `abstol` and `feastol`, `eps_rel` / `eps` `reltol`) | `1e-6` | checked on cvxopt 1.3.3 | none of its own: its certificate test is `feastol`, already the tag |
| `ecos` | `feastol`, `abstol`, `reltol` (as CVXOPT) | `1e-8` | documented, not checked | not in campaigns |
| `gurobi` | `FeasibilityTol`, `OptimalityTol`, `BarConvTol` (any case; no `eps` aliases) | `1e-6` | documented, not checked | not in campaigns |
| `highs` | `primal_feasibility_tolerance`, `dual_feasibility_tolerance`, `primal_residual_tolerance`, `dual_residual_tolerance`, `optimality_tolerance`, and `ipm_optimality_tolerance` (`solver` `choose`, `ipm`, `ipx`, `hipo`) or `pdlp_optimality_tolerance` (`pdlp`, `hipdlp`); `kkt_tolerance` changed from its `1e-7` default replaces them all (no `eps` aliases) | `1e-7` | checked on highspy 1.15.1 | none: its infeasibility verdicts use its feasibility tolerances |
| `mosek` | `MSK_DPAR_INTPNT_TOL_PFEAS`, `MSK_DPAR_INTPNT_TOL_DFEAS`, `MSK_DPAR_INTPNT_TOL_REL_GAP` for LPs, `MSK_DPAR_INTPNT_QO_TOL_PFEAS`, `MSK_DPAR_INTPNT_QO_TOL_DFEAS`, `MSK_DPAR_INTPNT_QO_TOL_REL_GAP` for QPs (both when the row does not say which; no `eps` aliases) | `1e-8` | documented, not checked | not in campaigns |
| `osqp` | `eps_abs`, `eps_rel` | `1e-3` | checked on osqp 1.1.3 | `eps_prim_inf`, `eps_dual_inf` (default `1e-4`) |
| `pdlp` | `eps_optimal_absolute`, `eps_optimal_relative` (set by `eps_abs` / `eps_rel`; a `parameters_text` that sets the detailed criteria is read with OR-Tools) | `1e-6` | checked on OR-Tools 9.15 | `eps_primal_infeasible`, `eps_dual_infeasible` (termination criteria, default `1e-8`), through `parameters_text` |
| `piqp` | `eps_abs`, `eps_rel`, and while `check_duality_gap` is on `eps_duality_gap_abs`, `eps_duality_gap_rel` (filled from `eps_abs` / `eps_rel`) | `1e-8` | checked on piqp 0.6.4 | none in piqp 0.6.4, which declares infeasibility from `eps_abs`, `eps_rel` (the tag) and `infeasibility_threshold`, a threshold on its proximal-point residuals, not a certificate tolerance |
| `proxqp` | `eps_abs`, `eps_rel`, and while `check_duality_gap` is on (the adapter's default) `eps_duality_gap_abs`, `eps_duality_gap_rel` (filled from `eps_abs` / `eps_rel`) | `1e-5` | checked on proxsuite 0.7.3 | `eps_primal_inf`, `eps_dual_inf` (default `1e-4`; fields of the settings of the QP object the adapter builds) |
| `qtqp` | `tol_feas`, `tol_gap_abs`, `tol_gap_rel` (as Clarabel; `atol` / `rtol` supply them; before qtqp 0.0.7 `atol` and `rtol` themselves, which must then both be set) | `1e-8` | checked on qtqp 0.0.7 | `tol_infeas_abs`, `tol_infeas_rel` (default `1e-8`) |
| `scs` | `eps_abs`, `eps_rel` (`eps` sets both) | `1e-4` | SCS 3.x documentation | `eps_infeas` (default `1e-7`; set to the tighter of the tag and `1e-7`, never looser than the default) |
| `sdpa` | `epsilonStar`, `epsilonDash` (`eps_abs` / `eps_rel`; the adapter's `optimality_tolerance` is a status rule the verification supersedes) | `1e-7` | checked on sdpa-python 0.2.3 | none |

The requested tolerance of a row comes from, first match wins:
`--requested-tol SOLVER_ID=TOL` (Python: `VerifyPolicy(requested_tol=...)`);
the row's `metadata.requested_tolerance`, which the worker records with every
solve; the settings the run's manifest gives the solver id (in a merge, the
manifest of the input whose `run_id` the row carries, else every input's,
which must then agree); and, only with `--tol-from-solver-id`, a tolerance tag
ending the solver id (`clarabel_1e-4`; nominal only). A row that needs one and has none stops the command with a message
naming the solvers and these remedies. Rows that compare nothing (no
residuals, errors, skips) never need one. Runs recorded before the worker
recorded it resolve from their manifests; the `scs33_2026-09` archive's merged
manifests list one solver's settings, so it needs `--tol-from-solver-id`.

**Columns.** `status` is the verified status and `status_reported` the one the
solver reported. `verify.verdict` is one of `kept`, `demoted`, `promoted`,
`over_time_limit`, `missing_residuals`, `unverified_cones`,
`unverified_certificate` and `not_checked`; `verify.rule`, `verify.tol`,
`verify.requested_tol`, `verify.requested_tol_source`, `verify.worst`,
`verify.worst_field`, `verify.reason` and `verify.objective_from` say why.
Scoring a certificate follows its verdict (made at the row's tolerance), not
the solve-time `kkt.valid`.

**Other rules and the opt-out.** On the analysis commands (and
`tools/make_figures.py`): `--unverified` uses the statuses the solvers reported
(Python: `load_results(run, verify=False)`, which also undoes a `bench
kkt-verify` copy); `--verify-tol T` holds every row, solutions and
certificates, to `T` (`VerifyPolicy(rule="fixed", tol=T)`); `--reverify`
re-checks a `bench kkt-verify` copy by the default rule instead of using what
it recorded (`VerifyPolicy(rule="requested")`); `--allow-missing` keeps claims
with nothing to check. Options that would not be used are refused:
`--verify-tol` with `--reverify`, `--requested-tol` or `--tol-from-solver-id`,
and `--unverified` with any other.

**Figures.** `tools/make_figures.py` verifies only the rows its figures use
(after the family's dataset, problem and `--exclude-solver` filters, and
without solver ids that carry no tolerance tag), so a solver it leaves out
never needs a requested tolerance. Its 1 % late rule applies after
verification, and `verification_summary.csv` is counted after it: `late`
counts the claims verification kept or promoted that the late rule then made
failures, and `kept`, `promoted` and `unverified` leave them out. A figure
compares solvers at one accuracy, so a series `--use-run` takes from another
tolerance's run (an interior-point solver at its `1e-6` run in the `1e-4`
figures) is verified at ten times the figure's tolerance tag, as the
published 3.3.2 figures were, not at ten times what its own run requested:
Clarabel from its `1e-6` run is checked at `1e-3` in the `1e-4` figures, like
the other series there, and at `1e-5` in its own `1e-6` figures. Its rows are
verified again from their reported statuses, with the tag standing in for
their requested tolerance and the family's slack, promotion and
`--allow-missing` setting (a `bench kkt-verify` copy's as recorded), and the
1 % late rule applies after. stdout says so, `geomean_summary.csv`'s
`verify_rule` (`10 x the figure's tolerance`) and `verify_tol` record it, and
`verification_summary.csv` counts that verification in a row of its own,
whose `figure` column names the figure's tag (it is empty on the per-solver
rows). Under `--verify-tol`, or a `bench kkt-verify --tol` copy, every
series is held to that one tolerance already.

**Derived runs.** A run written by `bench kkt-verify` is read as it recorded
its verification (its statuses, and its rule: `--tol`, or the per-row rule).
Every other run, a merge of verified copies included, is verified from its
rows' reported statuses (`metadata.kkt_verify.original_status` keeps the status
an earlier verification started from), with a warning when its rows carry an
earlier verification under another rule (tolerance, slack, promotion or
`--allow-missing`), naming the `--verify-tol` that reproduces a fixed one. A
`bench kkt-verify` copy read as it recorded does not use `--requested-tol`,
`--tol-from-solver-id` or `--allow-missing`, and warns when they are given;
`--reverify` applies them. Verification depends only on the reported status,
the KKT record, the time-limit markers and the rule, so verifying a verified
run again under the same rule changes nothing.

**`kkt.CERT_TOL`.** The `1e-3` in `kkt.CERT_TOL` is the solve-time check that
sets `kkt.valid`; it only rejects what is clearly not a certificate, and
scoring uses it only for statuses read with `--unverified`.

### Known outcomes

Datasets that know a problem's outcome record it as
`metadata.expected_status`: `optimal`, `infeasible` (primal infeasible),
`unbounded` (primal feasible, dual infeasible) or
`primal_and_dual_infeasible`. Every success count, failure table, profile,
geomean and plot scores rows against it, and the [independent KKT
verification](#independent-kkt-verification) uses it to judge certificates
(see [Certificates](#certificates)). Runs recorded before this field existed
are scored the same way, with the outcome inferred from the dataset id,
problem name and source path; rows that record `infeasible` for a problem
whose kind is now stated more precisely (NETLIB `cplex1`, SDPLIB `infd1` and
`infd2`, DIMACS `filtinf1`) take the precise one.

- `infeasible` (NETLIB `subset: infeasible` but `cplex1`, all of Liu-Pataki,
  SDPLIB `infp1`/`infp2`, DC OPF cases labelled infeasible): solved only by
  `primal_infeasible`, or by `dual_infeasible` verified as a proof (which
  shows that no optimal pair exists). Weakly infeasible Liu-Pataki instances
  are held to the same rule.
- `unbounded` (SDPLIB `infd1`/`infd2` and DIMACS `filtinf1` as posed: each
  has a feasible point, and the collection states that it has no optimal
  pair): solved only by `dual_infeasible`.
- `primal_and_dual_infeasible` (NETLIB `cplex1`): solved by either
  certificate.
- `optimal`, or no recorded outcome: solved only by a success status
  (`optimal` by default).

So a claimed optimum on an infeasible problem is a failure, and so is a
certificate on a solvable one. Inaccurate certificates (unless verification
accepted them and promoted them) and the ambiguous
`primal_or_dual_infeasible` never count, as `optimal_inaccurate` never counts.
Statuses read with `--unverified` fall back on the check recorded at solve
time: there a certificate with `kkt.valid` false (a residual above
`kkt.CERT_TOL`, `1e-3`, which only rejects what is clearly not a certificate)
is a failure.

### Certificates

A certificate's scaled residual `kkt.cert_res` is the measure SCS and
Clarabel stop on: the certificate is scaled so that its objective term
(`b'y`, the support `u'y⁺ − l'y⁻` of the bounds, or `q'x`) is −1, and
`cert_res` is the largest remaining residual (`A'y` with the distance of `y`
to the dual cone or the weight `y` puts on absent bounds, or `Px` with the
distance of `−Ax` to the cone). A small `cert_res` does not prove
infeasibility. A primal certificate implies `r'x ≤ −1` (with `r = A'y`) for
every feasible `x`, which only excludes feasible points with
`‖x‖₁ < 1/‖r‖∞`; by LP duality the smallest `cert_res` any certificate
reaches on a feasible problem is `1 / min ‖x‖₁` over its feasible points. So
a feasible problem whose feasible points are all large has certificates that
pass any tolerance a campaign uses, and the verdict depends on the units of
the variables. Measured on stored results: false certificates (on problems
with an optimum) reach `cert_res` 1.6e-11 (QPLIB 9002), true ones go up to
8.8e-4, and on MIPLIB mas76 single-precision SCS returns certificates at
2.57e-5, the best a certificate can do there since every feasible point has
`x₁₅₀ > 38,893`. A dual (unboundedness) certificate is the same statement
about the size of the dual's feasible points.

`kkt.cert_proof` says whether the certificate is a proof: whether the
inequality it implies contradicts the problem's own data, read as the exact
numbers its doubles are. For a primal certificate the residual `r` must be
paid for by finite bounds on `x` (the bounds the data states in its
single-entry rows), however large they are, and the contradiction must
exceed the rounding error of evaluating it (the safe bound of Neumaier and
Shcherbina, Math. Program. 99, 2004); a residual entry on a variable without
the bound its sign needs must be exactly zero, or of the sign the bound it
has allows. A dual certificate has nothing to pay with, so `Px` must vanish
and `Ax` lie in the recession cone exactly, and `q'x` must be negative beyond
its rounding error. Where rounding cannot decide the sign of a residual
entry (it is within its rounding error of zero), the sign is computed in
exact arithmetic (each double is an integer times a power of two, summed as
Python integers). So the check accepts no certificate on a feasible problem,
nor a dual one on a problem whose dual is feasible. It has no tolerance, does
not depend on units or on the scale of the certificate, and has no solver or
iteration in it (two sparse products and a few passes over the data in the
worker, plus the exact sums of the undecided entries). `kkt.cert_proof_open`
counts the residual entries that are exactly nonzero where nothing pays for
them, and `kkt.cert_proof_gap` is the margin of a primal contradiction
relative to the support (MIPLIB mas76's false certificates: −2.6e7, its
`1e12` bound being far too loose to pay for their residual).

Taking a residual within its rounding error as zero instead (the rule this
replaced) is not sound. A certificate whose entries span more than `2⁵³`
leaves exactly nonzero residuals below that error wherever rows imply an
equality (an equality written as two inequalities, dependent equality rows),
on data as plain as integers:
- `a + b ≤ 0`, `−a − b ≤ 0`, `a − b ≤ −1` is feasible (`a = −0.5`,
  `b = 0.5`), and `y = (1e20, 1e20, 1)` has support −1 and `A'y = (1, −1)`
  exactly, against a rounding error of about 1e4 on each free column;
- `min −x₃` subject to `x₁ − x₂ + x₃ ≤ 0`, `x₁ − x₂ = 0` is bounded
  (`x₃ ≤ 0`), and `x = (1e20, 1e20, 1)` has `q'x = −1` and `Ax = (1, 0)`
  exactly;
- `x − y ≥ 1` and `−(1 + 2⁻⁵²)x + y ≥ 1` are feasible (only for
  `x ≤ −2⁵³`), and `y = (−1, −1)` leaves one unit in the last place on `x`.

Each of these was a proof up to rounding, and none is one exactly. A
certificate whose residuals cancel exactly is still a proof however wide the
range of its entries (the first problem with `a − b ≥ 0` added, and the same
`y` with weight 1 on that row).

A bound of `1e18` or more, which the residuals read as absent
(`kkt._INF_BOUND`), counts as a bound for the dual proof (a direction must
respect it: a solver may hold the row to it) and not for the primal one (no
weight on it, not in the box), so each proof holds whichever way such a bound
is meant.

Its limits:
- It covers linear constraints only (QP bounds, zero and nonnegative cones).
  A certificate on any other cone has `kkt.cert_proof` null and its cones
  listed in `kkt.cert_proof_cones`, since a floating-point projection onto a
  second-order, semidefinite, exponential or power cone (or the box cone) is
  not exactly in the cone.
- Few certificates of non-simplex solvers are proofs: they leave residuals
  of about 1e-10 on free variables, and even HiGHS's simplex rays often
  leave exactly nonzero residuals below their rounding error. Regenerated
  locally on the 29 NETLIB infeasible LPs, 48 of 282 certificates were
  proofs (25 of HiGHS's 49; 80 and 41 with residuals within their rounding
  error taken as zero), and none of the dual certificates of any solver.
  Those problems are labelled, so all of them are still accepted when their
  `cert_res` is within the tolerance.

So the verification accepts a certificate when it is a proof, or when the
problem's known outcome (`status.CONFIRMED_CERTIFICATES`) confirms its kind
and `cert_res` is within the row's tolerance (a solver's usable answer to a
question whose answer is known); it never accepts one the known outcome rules
out (`status.RULED_OUT_CERTIFICATES`), and says so in `verify.reason`, noting
when the record holds a proof, which means the label or the problem's loader
is wrong. On a problem without a known outcome only proofs are accepted, and
records written before proofs were recorded (archived runs) have no proof to
check, so their certificates on such problems count as unsolved unless
`--allow-missing` (and whatever it says when their `cert_res` fails the
tolerance). The known outcome is the judge's knowledge, not the
solver's: every solver runs the same settings on every problem, with
infeasibility detection on. The labels are only as good as the datasets and
loaders: NETLIB `cplex2` is labelled infeasible but has points violating
its rows by about 1e-8. Code that checks certificates of a labelled problem
outside a run passes the outcome to `verify.classify` as the view's
`expected_status`. Records written before `kkt.cert_res` existed have their
scaled residuals rebuilt from the recorded absolute ones by the
verification.

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
use penalty `1e6` and shift `100`, while every relative KKT residual (any
`kkt.*_rel` field, such as `kkt.primal_res_rel`, `kkt.dual_res_rel`,
`kkt.duality_gap_rel`, `kkt.comp_slack_rel`, `kkt.dual_bound_res_rel`,
`kkt.primal_eq_res_rel`, `kkt.primal_cone_res_rel` and `kkt.dual_cone_res_rel`),
the scaled certificate residual `kkt.cert_res` and `kkt.comp_slack` use penalty
`1` and shift `1e-12`, so a residual that is exactly zero counts as `1e-12`
instead of dragging the mean of logs towards the smallest float. Absolute
residuals and other ad-hoc columns keep the run-time-style defaults (penalty
`1000`, shift `10`) unless `--shift` / `--max-value` override them. Time
metrics use shift `10`. `--success-only` requires no failure penalty or time
limit. These rules select the cost of failures; campaign-specific status
filtering and time-limit rules remain separate.

Missing solves are charged too: run-directory commands use the manifest's
configured solver and problem sets, intersecting includes with the dataset
listing when available. `--repo-root` on `summary`, `failures`, `missing`,
`geomean`, `profile`, `plot`, and `report` selects the repository whose
`problem_classes/` holds the dataset files. A `data_root` the run recorded (an
absolute path of the machine it ran on) is used when the dataset's folder is
under it here; when it is not, it is dropped with a warning naming the path and
the dataset is looked for under the repository root. `--data-root` on the same commands replaces the recorded
`data_root` of every dataset, as `bench run --data-root` sets it. If a dataset
directory is absent, explicit includes without a size filter supply the expected
names; otherwise, and also when the directory holds none of the dataset's
problems (the folders of the downloaded datasets are in every checkout, empty),
the dataset's expected problems are unknown: completion leaves its counts empty
and marks it incomplete, `bench missing` and `bench summary` say so with the
directory that was not found or held no problems, the report names the dataset,
and solves on its problems that no solver recorded are not charged. Merged-run selections retain
each source's filters and solver associations when determining planned jobs.
Comparisons then use the combined observed/expected problem population for all
retained solvers, whereas completion reports count only planned jobs.
No-manifest analyses use the union of observed problems crossed with observed solvers. Python callers can
pass `expected=` as a DataFrame of `dataset`, `problem`, and `solver_id` identities
to include completely absent solvers or problems. Duplicate attempts contribute
once per identity, using the best successful result, as in performance profiles:
a success beats a failure, then a success with a usable value of the metric beats
one without (a retried success still counts when the solver does not record the
metric), then the best value of the metric, then file order.
A report resolves each duplicated solve once, by its `--metric`, and every
per-solve table and plot (solver metrics, failure rates, status matrix,
cross-solver comparison, objective spreads, difficulty scaling, cactus, KKT and
certificate tables) describes that one attempt; status counts, completion and the
slowest-solves table list the rows as recorded. Python callers of the table
functions get the same resolution through their `metric=` argument
(`run_time_seconds` by default).
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

- A `## Verification` section near the top (and one sentence in the executive summary) saying how the statuses of every table and plot were verified, with the per-solver table of `bench summary` in `verification.csv` and, in `verification_changes.csv`, the rows of each solver by reported and verified status.
- A generated Markdown report at `index.md`, mirrored to `README.md`, that presents run metadata, completion, solver metrics, status counts, plots, diagnostics, provenance (including the captured `### System` block — CPU model, cores, RAM, OS, library versions), and links to the full CSV artifacts in a single document.
- Solver-level metrics, status counts, failure rates, completion, and missing results.
- Dolan-More performance-profile CSVs and plots. In the profile, cactus, KKT-accuracy and difficulty-scaling plots each solver has one colour and line style (and marker), the same in every plot of the run: ten colours, then the same colours dashed, dash-dotted and dotted, so up to 40 solvers look different.
- Cactus plots showing the fraction of problems solved under each time/iteration budget, over the same problems as the performance profile (planned solves never recorded included) and with values floored at the profile's resolution (0.01 s, one iteration).
- A difficulty-scaling table and plot: problems in equal-population bins of `metadata.n`, each sized from any of its rows that recorded it (a solve killed at the time limit or lost to a worker crash records no dimensions), with the solves of problems no row sized in a row with an empty `size_bin`.
- Pairwise scatter plots for problems solved by both solvers.
- Pairwise speedup tables with wins/losses/ties and largest wins.
- A per-problem cross-solver table, `problem_solver_comparison.csv`, with columns such as `qtqp_default__status`, `qtqp_default__run_time_seconds`, and `qtqp_default__iterations`.
- Per-solver problem tables under `solver_problem_tables/`, with rows as problems and columns for status, runtime, iterations, objective, the KKT residuals `kkt.primal_res_rel`, `kkt.dual_res_rel`, `kkt.comp_slack`, `kkt.comp_slack_rel` and `kkt.duality_gap_rel`, artifact path, and problem dimensions when available. Complementary slackness is shown absolute (`kkt.comp_slack`, in objective units) and relative (`kkt.comp_slack_rel`, scaled like the duality gap) for inspection only: verification does not check it, so it never changes a status. A residual a row did not record (a failed solve, an infeasibility certificate) is blank. Per-solver files and per-dataset `by_dataset/` folders are named like the run's artifact directories (`solver_problem_tables/scs__scale=1e+06.csv`); ids that would still share a name (compared case-insensitively) each get `~` and the first 8 hex digits of the id's SHA-256 appended instead, and the artifact index names the id of each such file.
- Objective-spread and slowest-solve tables.
- Failures-with-successful-alternatives tables, listing problems where one solver failed but at least one other solver succeeded.
- Status and performance-ratio heatmaps.
- A setup-vs-solve time breakdown per solver over successful solves (`setup_solve_breakdown.csv` / `.png`, for adapters that report the split) and a difficulty-scaling table and plot of the median metric per solver across problem-size bins (`difficulty_scaling_<metric>.csv` / `.png`).
- KKT-residual summaries (`kkt_summary.csv` over successful optimality claims, `kkt_certificate_summary.csv` over infeasibility claims, counting the recorded certificates as valid, invalid or unchecked (`cert_unknown`) by the verification's verdicts, `claimed_optimal_kkt_thresholds.csv` over the residuals verification checks; both claim tables select the statuses the solvers reported, `status_reported`, so claims verification demoted are counted) plus three KKT plots: per-solver boxplot of `primal_res_rel` / `dual_res_rel` / `comp_slack_rel` (`comp_slack` for records written before it existed) / `duality_gap_rel`, a problem-by-solver heatmap of those residuals, and a KKT-accuracy profile with one panel per residual showing, for each solver, the fraction of its successful solves whose residual is at most tau. Exact zeros are drawn at `1e-16`.
- For multi-dataset runs, a `## By Dataset` section that re-emits the headline tables (solver metrics, failure rates, shifted geomean, KKT summary) sliced per dataset. The cross-dataset aggregates above the section remain the primary view.
- For a run written by `bench kkt-verify` or `bench merge`, a `### Derived Run` block in Provenance (and one sentence in the executive summary): the rule and tolerance, `--promote`, `--allow-missing` (`missing_is_failure`) and every per-solver tally `kkt-verify` recorded, and for a merge the merged run directories with their row counts. The copied source config of a merge is labelled as the first merged run's.
- An artifact index listing every CSV and plot written by the report command.

The report owns the file names it writes: re-running `bench report` with the same
`--metric` into the same directory removes a table, plot or `by_dataset/` file of
an earlier report that the new one no longer writes (for example
`missing_results.csv` once the run is complete), the `solver_problem_tables/`
file of a configured solver that now has no rows, and the per-solver and
per-dataset files an earlier report wrote under the older name rule (`+`, `=` and
other characters outside `[A-Za-z0-9_.-]` replaced by `-`), together with such a
`by_dataset/` folder once it is empty. Files of another metric, and files of
datasets or solvers no longer in the manifest, are not tracked and stay (unless
the file's name is the older-rule name of a current id); other files in the
directory are left alone. `bench plot` does the same for its
plot names of the metric.

Programmatic use:

```python
from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean

df = load_results("results/<run_id>")  # KKT-verified statuses; verify=False for the reported ones
summary = solver_summary("results/<run_id>")
profile = performance_profile(df, metric="run_time_seconds")
geomean = shifted_geomean(df, metric="run_time_seconds", timeout_seconds=300)
```

Python callers can pass `timeout_seconds={"qp": 300, "sdp": 900}` for mixed
dataset limits, or `max_value=5400` for an explicit common penalty. A time-metric
penalized geomean requires one of these; `penalize_failures=False` does not.

Status handling:

- By default, only `optimal` counts as successful for performance profiles and
  shifted geometric means, and statuses are KKT-verified first (see
  [Independent KKT verification](#independent-kkt-verification)): a claim the
  harness's residuals do not support is not `optimal`, and an inaccurate or
  limit stop whose point passes is. `optimal_inaccurate` and other inaccurate
  statuses are penalized because the solver did not hit the requested target.
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
per campaign with the merged `results.jsonl` and `manifest.json` per family (readable by `bench
kkt-verify` and `bench report`; the `scs33_2026-09` merges record only one solver's settings in their
manifests, so they need `--tol-from-solver-id` to be verified), the campaign configs, figures, tables,
and a manifest of solver versions and hardware. The complete archive of each campaign is attached to its GitHub release there.

Campaign configs are written by `tools/make_campaign.py` under one rule:
every solver runs the same general settings on every problem, and none is
told whether a problem is feasible. A user cannot know that in advance, and
a solver that knew its problems were feasible could be faster by ignoring
infeasibility. So a solver variant's settings are the same in every family
it runs in, apart from its time limit, which is the harness's budget for the
family (300 s for QPs, LPs and the infeasible LPs, 900 s for SDPs, 1800 s for
the large LPs), and the same at every tolerance tag apart from the
tolerances the tag sets (its solution tolerances and its infeasibility knobs,
SCS's never looser than its default, see
[Independent KKT verification](#independent-kkt-verification)). Every
solver keeps its infeasibility detection on everywhere. cuOpt is the one
campaign solver whose detection is off by its default (its PDLP's
`infeasibility_detection`): every cuOpt variant turns it on, in every family,
as the adapter now does unless configured; earlier campaigns, `scs33_2026-09`
among them, ran cuOpt with it off. A family picks which solvers run by
problem class (ProxQP on the QP family, CVXOPT and SDPA on the SDP family),
never by feasibility: the infeasible LPs run the LP family's solvers with the
LP family's settings. `tests/test_tools.py` checks all of this on the
generated settings.

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

Register it in `solver_benchmarks/datasets/registry.py` by module and class name
(say it is in `solver_benchmarks/datasets/my_dataset.py`), as an entry of the
`DATASETS` table there; the module is imported when the dataset is first looked
up, so a worker imports only the loader it reads from:

```python
DATASETS: LazyRegistry[type[Dataset]] = LazyRegistry(
    {
        ...
        "my_dataset": _loader("my_dataset", "MyDataset"),
    }
)
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
    IMPORT_ERRORS,
    SolverAdapter,
    finite_point,
    module_available,
    reported_objective,
    solver_import_failed,
)


class MySolverAdapter(SolverAdapter):
    solver_name = "my_solver"
    supported_problem_kinds = {QP}

    @classmethod
    def is_available(cls) -> bool:
        return module_available("my_solver")

    def solve(self, problem: ProblemData, artifacts_dir: Path) -> SolverResult:
        try:
            import my_solver
        except IMPORT_ERRORS as exc:
            raise solver_import_failed(
                "my_solver", "Install my_solver to use this adapter", exc
            ) from exc

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
            info={"raw_status": str(result.status), "threads": None},
            trace=result.trace,
            kkt=kkt_dict,
        )
```

Register it in `solver_benchmarks/solvers/registry.py` by module and class name
(say it is in `solver_benchmarks/solvers/my_solver_adapter.py`), as an entry of
the `SOLVERS` table there; the module is imported when the solver is first looked
up, so a worker imports only the adapter it runs (do not import the adapter in
`registry.py`):

```python
SOLVERS: LazyRegistry[type[SolverAdapter]] = LazyRegistry(
    {
        ...
        "my_solver": _adapter("my_solver_adapter", "MySolverAdapter"),
    }
)
```

Tests and scripts can still register a class directly with
`SOLVERS["my_solver"] = MySolverAdapter` (or `monkeypatch.setitem`).

Adapter expectations:

- Do not import optional solver packages at module import time if they may be missing.
- Use `is_available()` for optional dependency checks, through
  `module_available()`: an installed package that fails to import (a native
  library that does not load raises `ImportError` or `OSError`) then makes the
  solver unavailable with a warning instead of stopping the run, and
  `solver_import_failed()` turns the same failure in `solve()` into
  `SolverUnavailable`.
- Keep solver adapter modules solver-specific rather than grouping unrelated adapters together.
- Return canonical statuses from `solver_benchmarks.core.status`. Map every raw solver status the underlying library can emit (including any `*_inaccurate` variants and infeasibility/unboundedness certificates), and fall back to `SOLVER_ERROR` only for genuinely unrecognized states. A solve that stopped on its iteration or time limit is `MAX_ITER_REACHED` / `TIME_LIMIT` even when the solver labels its last iterate "solved inaccurate" or "almost solved"; keep the solver's own status in `info["raw_status"]`.
- Populate `SolverResult.kkt` whenever the solver returned a finite primal/dual pair: use `solver_benchmarks.analysis.kkt.qp_residuals` (or `cone_residuals` for cone solvers) for every status in `status.ANY_FEASIBLE` (optimal, inaccurate, iteration or time limit, solver error), and the matching `*_infeasibility_cert` helpers when the solver reports a Farkas-style certificate (also an inaccurate one that came with a limit status, as OSQP's `*_infeasible_inaccurate` do). A QP adapter can hand the certificate to `qp_split.qp_certificate_kkt` in the harness's sign (a Farkas vector positive on the rows whose upper bound it uses, as the solution duals are; the ray as it is); set whatever option makes the solver produce one, and count the time it takes in `run_time_seconds`. A claim without a certificate cannot be verified and counts as unsolved. The verification can then judge every solver's limit-hit point by the same check. The KKT plots, summaries, and accuracy profile all read this field. `cone_residuals` projects onto every cone but complex PSD (zero, nonnegative, box, second-order, PSD, exponential and its dual, power and its dual), and its `primal_res` is the larger of `Ax + s − b` and the distance of `s` to the cone, so an adapter may pass `s = b − Ax`.
- Report `objective_value` only for `OPTIMAL` and `OPTIMAL_INACCURATE` (`reported_objective` does this); it is `None` for limits, certificates and errors, whose point's objective is still on `kkt.primal_obj`. The worker adds `kkt.primal_obj_reported`, that objective with the problem's constant term and sense applied, which the verification uses for a row it promotes to `optimal`.
- Keep the solver's own status as a string in `info["raw_status"]` (a name, not an integer code; put a code under another key).
- Return `SOLVER_ERROR` with the message on `info["error"]` (and the elapsed time) when the solver itself fails on a problem by raising, e.g. on a non-convex `P`, crossed bounds or data it refuses; an exception escaping `solve` is recorded as a harness `worker_error` and re-run on resume, so catch only the solver's own exception types. Keep raising for configuration errors: `SolverConfigError` (from `solver_benchmarks.solvers.base`; the runner stops that solver id on it, for the whole run) for unknown settings and out-of-range or wrong-typed values, including a bad value the solver reports as a status or raises from the same call as its failures on the problem (check those values before the solve), and `SolverUnavailable` for a missing backend or a license failure.
- Pass a NaN-free point to the KKT helpers: `finite_point` returns False for a missing or non-finite vector, and the record is then `None`.
- Time what other adapters time: the solver's setup and solve, not harness-only work such as statistics collection.
- A solver that runs on a GPU (with some or all settings) defines `gpu_modules()`, returning for its settings the modules its GPU backend imports (empty on the CPU; it must not raise). The worker then warms the GPU up before `solve` (see [GPU warm-up](#gpu-warm-up)), so its `run_time_seconds`, like the other GPU solvers', leaves out the per-process loading.
- Reject unknown setting names with a `SolverConfigError` naming them, and translate the shared aliases (`eps` / `eps_abs` / `eps_rel` with `translate_tolerance_aliases`, `max_iter` with `pop_max_iter`, `time_limit*` with `pop_time_limit`, `threads` with `pop_threads`) where the solver has an equivalent, recording what was inserted in `info["settings_translated"]`, or mark them ignored. A solver with a thread setting runs on `threads_or_default(threads)` (one thread unless configured) and records it on `info["threads"]`; set `info["threads"] = None` where the adapter cannot set it.
- Define the classmethod `requested_tolerance(settings, *, problem_kind, is_lp, versions)`: run the adapter's own alias translation on a copy of the settings and return `loosest_tolerance(settings, {knob: library default, ...})` (see `solver_benchmarks/solvers/base.py`), listing the solver's feasibility and gap tolerances (not its infeasibility tolerances, which no adapter uses). The [independent KKT verification](#independent-kkt-verification) holds every row of the solver, solutions and certificates alike, to ten times it; the base class raises, so an adapter without it leaves its rows unverifiable (and a test fails). Add its row to the adapter table there. If the solver joins `tools/make_campaign.py`, add its infeasibility knobs to `CERTIFICATE_KNOBS` (an empty tuple, with the reason, when it has none the adapter can set) and to the last column of that row; `solver_variants` refuses a solver without an entry.
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
wheels ship a separate HiGHS library that clashes with OR-Tools 9.15.6755's
native HiGHS symbols, causing an import error mentioning `HighsLogOptions`,
`setLocalOptionValue` or `releaseMemory`. This also affects macOS, not only
Linux. Dependencies in the `pdlp` and `all` extras therefore exclude highspy
1.15.x (`highspy>=1.0,!=1.15.*`); highspy 1.14.0 and 1.16.0 link HiGHS into
their own module and work with OR-Tools 9.15.6755. Reinstall the `pdlp` extra
if an existing environment has the incompatible pair. CI imports the native
libraries in both orders in fresh processes, then solves a real MPS file
through the reader and PDLP adapter.

Supported inputs:

- LP datasets represented as QPs with `P.nnz == 0`, such as NETLIB and MIPLIB
  root LP relaxations.
- Simple linear conic problems with only zero and nonnegative cones.
  Zero cones may use either `z` or `f`.

Unsupported inputs:

- QPs and cone problems with nonzero `P`.
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

- `pdlp_solve_log.textproto`: PDLP's solve log (termination, iteration
  statistics).
- `pdlp_response.textproto`: OR-Tools' response without the solution vectors
  (`variable_value`, `dual_value`, `reduced_cost`), which as text take seconds
  and tens of MB per solve on large LPs. Set `save_response: true` in the
  solver's settings to write the complete response; the result row is the
  same either way.

The model sent to PDLP gives its variables and constraints no names.

## Testing

CI runs the same checks on pull requests and pushes to `master`.
The scheduled run executes every Monday at 06:00 UTC. GitHub only evaluates
scheduled workflows from the repository default branch, so keep the workflow on
the default branch if weekly runs are required.

Run the test suite:

```bash
pip install -e ".[test,all]"
pytest -m "not network"
```

`-m "not network"` leaves out the tests marked `network`. They download from
external dataset hosts and fail offline or behind a proxy, so CI runs them only
on the weekly schedule (`pytest -m network tests/test_dataset_downloads.py`).
The tests marked `slow` take from about ten seconds to minutes: two read every
member of the SDPLIB archive, and one has Clarabel look for an improving
direction of two DIMACS problems. For a quick run, use
`pytest -m "not network and not slow"`, as the CI matrix jobs do; the coverage
job runs the slow tests on every pull request.

Most tests that need a solver skip when it is not installed; some runner,
worker and PDLP tests need OSQP, SCS or OR-Tools installed and fail without
them. A solver that is installed but fails to import fails the tests that use
it.
Set `BENCH_REQUIRE_ALL_SOLVERS=1` (any value but empty, `0`, `false`, `no` or
`off`) to make those skips fail instead, across the whole suite: a skip whose
reason is an open-source solver that is not installed (`pytest.importorskip`,
"... not installed"), or an OSQP build that never stops at a tested status, fails and names the solver. Only the solvers the `all`
extra installs on the running Python and platform count; skips for the
commercial solvers (Gurobi, MOSEK, CPLEX, cuOpt), for absent data and for the
platform stay skips. CI sets it in the jobs that install every extra, because
there a missing solver means a packaging or import regression.

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
- cuOpt adapter conventions (objective mapping, dual sign, status names, option
  forwarding) against a fake cuOpt, since CI has no GPU.
- The campaign tools `tools/make_figures.py` and `tools/make_campaign.py`
  against the analysis API they use and against each other (families, time
  limits, tolerance tags), the infeasibility knobs every campaign solver is
  given, that every solver's settings are the same in every family apart
  from the tolerance tag and the time limit and keep infeasibility detection
  on (cuOpt's included), every CPU solver variant of the campaign against the installed
  solvers, and `tools/bench_modal.py` against a stand-in for Modal.
- Every solver variant of the maintained configs in `configs/` against the
  installed solvers, `configs/README.md`'s list of maintained configs against
  the configs without the `# EXPERIMENTAL:` header, and the README's CLI,
  config-field and solver tables against the code.

GitHub Actions runs additional matrix workflows that install each open-source
solver in isolation and exercise its adapter via `scripts/ci_solver_smoke.py`,
so a packaging or import regression in one extra cannot hide behind another
solver's tests. A separate `commercial-adapters` job confirms that the CPLEX,
Gurobi, and Mosek adapters register and import without their (proprietary)
dependencies installed.

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
- Cross-run comparison commands.
- Optional exact-solution/reference-objective checking where datasets provide it.
- Parallel result-write batching for very large runs.
