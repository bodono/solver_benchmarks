# Benchmark configs

The configs here fall into two groups.

**Maintained configs** run with the released solvers that the package extras
install (`pip install -e ".[all]"`) on every platform those extras support.
The main README lists them under *Running Benchmarks*. They include
`synthetic_smoke.json`, the `*_example.yaml` files,
`maros_meszaros_qp_solvers.yaml`, the Anderson and linear-solver sweeps, and
the SCS baseline configs (`scs_*_baseline.yaml`, `scs_anderson_refined.yaml`,
`scs_pr54_validation.yaml`, `regress331/`).

**Experimental configs** start with an `# EXPERIMENTAL:` header that names
what they need. They are research sweeps that were run against development
builds of SCS or QTQP. They are kept so those results can be reproduced, but
they will not run with the released packages:

- Most `exp_*.yaml` and `scs_*.yaml` sweeps pass SCS settings that only exist
  on development branches: `den_floor_frac`, `num_l2_passes`,
  `num_ruiz_passes`, `diag_rho_x_floor`, `diag_scale_damp`, `restart`,
  `adaptive_alpha`, `acceleration_trust_factor`, `metric_freeze_factor`,
  `rescaling_min_iters`, `scale_mode`, `soc_metric`, `tau_factor`, and
  `adaptive_diag_scale` values above 1 (released SCS accepts only 0 and 1).
  Released SCS rejects them, so every solve that uses them fails as
  `worker_error`. Each failure is retried on resume, up to
  `run.max_worker_error_retries` times.
- `qtqp_*.yaml` (apart from the two sweeps listed in the main README) pass
  QTQP's `init_strategy`, which released QTQP 0.0.7 rejects, or select
  `linear_solver: qdldl`. That backend needs the `qdldl` Python package
  (`pip install qdldl`), which the `qtqp` extra does not install.

Each header says exactly what its config needs. Build or install that, then
run the config as usual with `bench run`.

`tests/test_bundled_configs.py` runs every solver variant in the maintained
configs on a one-variable LP with the installed solvers. It fails when a
config without the header uses a setting or a linear-solver backend that the
released packages do not provide. When you add a config that needs an
unreleased build, give it the header.
