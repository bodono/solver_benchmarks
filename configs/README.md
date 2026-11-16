# Benchmark configs

The configs here fall into two groups.

**Maintained configs** run with the released solvers that the package extras
install (`pip install -e ".[all]"`) on every platform those extras support.
They are exactly these files:

- Examples (listed in the main README under *Running Benchmarks*):
  `synthetic_smoke.json`, `netlib_feasible_example.yaml`,
  `maros_meszaros_example.yaml`, `maros_meszaros_qp_solvers.yaml` and
  `multi_dataset_example.yaml`.
- Sweeps (also listed there): `scs_anderson_sweep.yaml`,
  `scs_linear_solver_backend_sweep.yaml`,
  `qtqp_refinement_regularization_sweep.yaml` and
  `qtqp_linear_solver_backend_sweep.yaml`.
- SCS baselines: `scs_miplib_baseline.yaml`, `scs_mm_baseline.yaml`,
  `scs_oos_baseline.yaml`, `scs_sdp_baseline.yaml`,
  `scs_anderson_refined.yaml`, `scs_pr54_validation.yaml` and the four
  configs in `regress331/` (`kennington.yaml`, `kennington_e6.yaml`,
  `miplib.yaml`, `netlib.yaml`).

**Experimental configs** start with an `# EXPERIMENTAL:` header that names
what they need. They are all the other configs here; names alone do not
decide the group (`scs_indirect_baseline.yaml` and `scs_aa_sweep_v2.yaml`,
for example, are experimental). They are research sweeps that were run
against development builds of SCS or QTQP. They are kept so those results
can be reproduced, but they will not run with the released packages:

- Most `exp_*.yaml` and `scs_*.yaml` sweeps pass SCS settings that only exist
  on development branches: `den_floor_frac`, `num_l2_passes`,
  `num_ruiz_passes`, `diag_rho_x_floor`, `diag_scale_damp`, `restart`,
  `adaptive_alpha`, `acceleration_trust_factor`, `metric_freeze_factor`,
  `rescaling_min_iters`, `scale_mode`, `soc_metric`, `tau_factor`, and
  `adaptive_diag_scale` values above 1 (released SCS accepts only 0 and 1).
  Released SCS rejects them, so the arms that use them are not run: the
  first solve of such an arm stops that arm, the run finishes the other
  arms (their results are written as usual) and then exits with an error
  naming every rejected arm. Remove those arms to get a clean exit; a
  re-run with them fails the same way and solves nothing new for them.
- `qtqp_*.yaml` (apart from the two maintained sweeps above) pass
  QTQP's `init_strategy`, which released QTQP 0.0.7 rejects (those arms are
  not run, as above), or select `linear_solver: qdldl`. That backend needs
  the `qdldl` Python package (`pip install qdldl`), which the `qtqp` extra
  does not install; without it each solve of such an arm fails as
  `worker_error`, retried on resume up to `run.max_worker_error_retries`
  times.

Each header says exactly what its config needs. Build or install that, then
run the config as usual with `bench run`.

`tests/test_bundled_configs.py` runs every solver variant in the maintained
configs on a one-variable LP with the installed solvers. It fails when a
config without the header uses a setting or a linear-solver backend that the
released packages do not provide, and when the list of maintained configs
above does not match the configs without the header. When you add a config
that needs an unreleased build, give it the header; when you add one that
runs with the released packages, add it to the list.
