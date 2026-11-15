# Research scripts

These are one-off scripts from the 2026 SCS Anderson-acceleration (AA)
experiments. They are kept so that work can be reproduced. They are not
maintained, no test runs them, and they are not part of the benchmark
workflow. The maintained scripts are the `prepare_*.py` data wrappers and
`ci_solver_smoke.py` in the parent directory.

Most of them call SCS directly on problems loaded through the dataset
classes. They expect a scs-python build that carries the AA changes under
study, so the stock `scs` wheel will not reproduce their results and may not
accept their settings.

| File | What it is |
|---|---|
| `aa_experiment_runner.py` | Runs a type-II AA variant on the 33 "type-I wins, type-II loses" problems and writes JSON to `experiments/aa_type2/`. |
| `aa_type1_runner.py` | Runs type-I experiments on the type-I weak-spot set and writes JSON to `experiments/aa_type1/`. |
| `aa_head_to_head.py` | Compares type-I and type-II on the union of both weak sets. |
| `aa_regression_check.py` | Re-runs type-I and previously working type-II settings on a sample of problems. |
| `aa_type_sweep.py` | Compares type-I and type-II across the 33 "type-I wins" problems. |
| `aa_type_diag.py` | Prints AA statistics for type-I and type-II on one diagnostic problem. |
| `aa_adaptive_r.patch`, `aa_combined_best.patch`, `aa_type2_best.patch`, `aa_type2_regularization.patch` | Patches to SCS's `aa.c` for the variants above, made against a local scs-python checkout. |
| `apply_pr54_to_scs.py` | Applies the SCS PR #54 AA trust-region changes to an scs-python checkout: `python scripts/research/apply_pr54_to_scs.py /path/to/scs-python/scs_source`. |

`aa_experiment_runner.py`, `aa_regression_check.py` and `aa_type_sweep.py`
choose their problems from
`results/scs_anderson_sweep_2026-04-27_08-44-15_UTC/report/problem_solver_comparison.csv`.
That is the report of the AA sweep from 2026-04-27, which is not in the
repository. Without that file they stop with `FileNotFoundError`, except
`aa_experiment_runner.py --small-only`, which uses a built-in problem list.
Run the scripts from the repository root.
