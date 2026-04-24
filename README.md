# Solver Benchmarks

[![Tests](https://github.com/bodono/solver_benchmarks/actions/workflows/tests.yml/badge.svg?branch=qtqp)](https://github.com/bodono/solver_benchmarks/actions/workflows/tests.yml)

Config-driven benchmark suite for convex optimization solvers.

The maintained path is the `solver_benchmarks` package and the `bench` CLI. Older
top-level scripts are legacy entrypoints and should not be extended for new work.

## Goals

- Configure solver variants from files, including many settings for the same solver.
- Run named datasets or selected individual problems.
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
including QTQP, SCS, OSQP, Clarabel, OR-Tools/PDLP, PyYAML, and pytest.

Commercial solvers are optional and must be installed separately with valid licenses:

```bash
pip install -e ".[gurobi]"
pip install -e ".[mosek]"
```

Install only one optional solver if you prefer:

```bash
pip install -e ".[scs]"
pip install -e ".[osqp]"
pip install -e ".[clarabel]"
pip install -e ".[qtqp]"
pip install -e ".[pdlp]"
```

Check what is available in the current environment:

```bash
bench list solvers
```

Example output:

```text
clarabel  available               qp
gurobi    missing optional extra  qp
mosek     missing optional extra  qp
osqp      available               qp
pdlp      available               cone,qp
qtqp      available               qp
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
| MIPLIB root LP relaxation | `miplib` or `miplib_lp_relaxation` | LP as QP | Integrality is ignored; root-node LP relaxation only. |
| QPLIB | `qplib` | QP | Uses QPLIB parser without CVXPY. |
| Mittelmann | `mittelmann` | LP as QP | Adapter exists; data may not be present locally. |
| SDPLIB | `sdplib` | Cone/SDP | Reads `.jld2`; requires `h5py`. |
| DIMACS | `dimacs` | Cone | Reads `.mat` and `.mat.gz`; rotated Lorentz cones are not yet supported. |
| Synthetic smoke test | `synthetic_qp` | QP | Tiny deterministic test problem. |

List problems in a dataset:

```bash
bench list problems netlib --option subset=feasible
bench list problems maros_meszaros
bench list problems qplib
```

Dataset options are passed as `key=value` on the CLI or as `dataset_options` in
config files.

## Supported Solvers

Current maintained adapters:

| Solver | ID | Supported types | Notes |
|---|---|---|---|
| QTQP | `qtqp` | QP | Converts QP bounds to nonnegative cone form internally. |
| SCS | `scs` | QP, cone | Supports SCS box-cone form for QPs and native conic data. |
| Clarabel | `clarabel` | QP | Uses nonnegative cone conversion for `l <= Ax <= u`. |
| OSQP | `osqp` | QP | Direct QP adapter. |
| PDLP | `pdlp` | LP-only QP, simple linear cone | Uses OR-Tools directly, no CVXPY. |
| GUROBI | `gurobi` | QP | Optional extra; requires `gurobipy` and license. |
| MOSEK | `mosek` | QP | Optional extra; requires Mosek package and license. |

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
  dataset: netlib
  output_dir: runs
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
      verbose: false
      eps_abs: 1.0e-4
      eps_rel: 1.0e-4
      max_iter: 100000

  - id: osqp_tight
    solver: osqp
    settings:
      verbose: false
      eps_abs: 1.0e-6
      eps_rel: 1.0e-6
      max_iter: 200000

  - id: pdlp_default
    solver: pdlp
    settings:
      verbose: false
      time_limit_sec: 120
```

Important fields:

| Field | Meaning |
|---|---|
| `run.dataset` | Dataset ID from `bench list datasets`. |
| `run.output_dir` | Root directory for immutable runs. |
| `run.dataset_options` | Dataset-specific options, such as `subset: feasible`. |
| `run.include` | Optional list of problem names to run. Empty means all. |
| `run.exclude` | Optional list of problem names to skip. |
| `run.parallelism` | Number of concurrent subprocess solves. |
| `run.resume` | If true, completed `(problem, solver_id)` pairs are not rerun. |
| `run.timeout_seconds` | Default subprocess timeout per solve. |
| `solvers[].id` | Unique label for this solver variant. Used in output paths. |
| `solvers[].solver` | Solver adapter ID, e.g. `scs`, `qtqp`, `pdlp`. |
| `solvers[].settings` | Passed directly to the solver adapter. |
| `solvers[].timeout_seconds` | Optional per-solver timeout override. |

The same solver may appear many times with different `id` and `settings`.

## Running Benchmarks

Run a config:

```bash
bench run configs/synthetic_smoke.json
bench run configs/netlib_feasible_example.yaml
bench run configs/maros_meszaros_example.yaml
```

The command prints the run directory:

```text
runs/20260424T101304Z_synthetic_qp_d5939d8c1f2d
```

Resume a run:

```bash
bench run configs/netlib_feasible_example.yaml --run-dir runs/<run_id>
```

If `resume: true`, already completed `(problem, solver_id)` pairs in
`results.jsonl` are skipped. New solver variants or newly included problems are
appended.

Run only specific problems by editing `include`, or list problems first:

```bash
bench list problems netlib --option subset=feasible
```

## Run Directory Layout

Every run is immutable by default:

```text
runs/20260424T101304Z_synthetic_qp_d5939d8c1f2d/
  manifest.json
  events.jsonl
  results.jsonl
  results.parquet
  problems/
    one_variable_eq/
      scs_default/
        payload.json
        stdout.log
        stderr.log
        worker_result.json
        result.json
        trace.jsonl
```

Files:

| File | Purpose |
|---|---|
| `manifest.json` | Run metadata, normalized config, config hash, creation time. |
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
- Solver stdout/stderr are captured per problem and solver variant.
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

## Analysis

Summary by solver/status:

```bash
bench summary runs/<run_id>
```

Generate a performance-profile CSV:

```bash
bench profile runs/<run_id>
bench profile runs/<run_id> --metric iterations
```

Generate shifted geometric means:

```bash
bench geomean runs/<run_id>
bench geomean runs/<run_id> --metric iterations
```

Programmatic use:

```python
from solver_benchmarks.analysis.load import load_results, solver_summary
from solver_benchmarks.analysis.profiles import performance_profile, shifted_geomean

df = load_results("runs/<run_id>")
summary = solver_summary("runs/<run_id>")
profile = performance_profile(df, metric="run_time_seconds")
geomean = shifted_geomean(df, metric="run_time_seconds")
```

Status handling:

- By default, `optimal` and `optimal_inaccurate` count as successful for
  performance profiles and shifted geometric means.
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
data accepted by SCS. PDLP supports only linear zero/nonnegative cones.

Add tests:

- Dataset is registered.
- `list_problems()` returns stable, deduplicated problem names.
- A small fixture problem loads with expected dimensions.
- Unsupported solver combinations skip with a warning.

## Adding a New Solver

Solver adapters live under `solver_benchmarks/solvers/`.

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

        qp = problem.qp
        result = my_solver.solve(qp["P"], qp["q"], qp["A"], qp["l"], qp["u"], **self.settings)

        return SolverResult(
            status="optimal",
            objective_value=float(result.objective),
            iterations=int(result.iterations),
            run_time_seconds=float(result.runtime),
            info={"raw_status": result.status},
            trace=result.trace,
        )
```

Register it in `solver_benchmarks/solvers/registry.py`:

```python
SOLVERS["my_solver"] = MySolverAdapter
```

Adapter expectations:

- Do not import optional solver packages at module import time if they may be missing.
- Use `is_available()` for optional dependency checks.
- Return canonical statuses from `solver_benchmarks.core.status`.
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

Install:

```bash
pip install -e ".[pdlp]"
```

Supported inputs:

- LP datasets represented as QPs with `P.nnz == 0`, such as NETLIB and MIPLIB
  root LP relaxations.
- Simple linear conic problems with only zero and nonnegative cones.

Unsupported inputs:

- QPs with nonzero `P`.
- SDP, SOC, rotated SOC, and other non-LP cones.

Settings:

```yaml
settings:
  verbose: false
  time_limit_sec: 120
  use_glop: true
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

CI runs the same checks on pull requests and pushes to `master` or `qtqp`.
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
- Synthetic QP loading.
- Generic runner result writing and resume behavior.
- Structured warning events for unsupported combinations.
- Subprocess stdout/stderr capture.
- Worker trace serialization.
- Real solver smoke coverage for QTQP, SCS, Clarabel, OSQP, and PDLP on a tiny LP.
- Loading JSONL/Parquet-backed result tables.
- Solver summary output.
- Performance profile and shifted geometric mean calculations.
- CLI `summary`, `profile`, and `geomean` commands.
- PDLP registration and clean skip behavior when unavailable or given non-LP data.

Recommended tests for new contributions:

- Unit tests for parser/conversion logic.
- A tiny deterministic problem fixture.
- One end-to-end run using `synthetic_qp` or another small dataset.
- A resume test if result-writing behavior changes.
- CLI tests for any new user-facing command.
- Optional-dependency tests for MOSEK/GUROBI must not require the dependency or a
  license to be installed.

## Development Notes

Run output under `runs/` is intentionally not part of source code. Keep large
generated outputs out of commits unless there is a deliberate benchmark-results
artifact policy.

The old `results/` tree was removed during the refactor. New analysis should use
run directories and `solver_benchmarks.analysis`.

The maintained package should avoid CVXPY imports. If a legacy parser still has
a CVXPY helper method, ensure the import is lazy and not used by the maintained
path.

## Roadmap

High-value next steps:

- More direct SDPLIB/DIMACS validation tests with small fixtures.
- Richer plotting commands that write PDFs/PNGs, not only CSV profile data.
- Cross-run comparison commands.
- Solver environment metadata capture: versions, git SHA, platform, CPU info.
- More comprehensive status normalization across solvers.
- Optional exact-solution/reference-objective checking where datasets provide it.
- Parallel result-write batching for very large runs.
