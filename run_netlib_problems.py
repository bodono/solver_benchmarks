'''
Run netlib problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from netlib_problems.netlib_problem import NETLIBRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
from utils.make_table import make_latex_table
import os
import argparse
import shutil
import qtqp


parser = argparse.ArgumentParser(description='NETLIB Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=False,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--infeasible', help='Run of infeasible', default=False,
                    action='store_true')
parser.add_argument('--add_quadratic', help='Add a quadratic term,', default=False,
                    action='store_true')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
infeasible = args.infeasible
add_quadratic = args.add_quadratic

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

solvers = [s.QTQP, s.SCS] # , s.OSQP, s.SCS_AA1, s.SCS_AA2] #, s.COSMO] #, s.SCS_ALT #s.ECOS, s.QPALM, s.COSMO]
if infeasible:
  OUTPUT_FOLDER = 'netlib_infeasible'
else:
  OUTPUT_FOLDER = 'netlib_feasible'

if add_quadratic:
  OUTPUT_FOLDER = OUTPUT_FOLDER + '_quadratic'

if high_accuracy:
    solvers = [solver + s.HIGH for solver in solvers]
    OUTPUT_FOLDER += s.HIGH

settings = s.get_settings(infeasible)

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = verbose

settings = {}
solvers = []
for corrector_new_variables in [True, False]:
  for corrector_vars_scale_by_sigma in [True, False]:
    for linf_neighborhood_scale in [0., 0.001, 0.01, 0.1]:
      solvers.append(f"QTQP_corr_new_{corrector_new_variables}_corr_sigma_{corrector_vars_scale_by_sigma}_linfn_{str(linf_neighborhood_scale).replace('.', '_')}")
      settings[solvers[-1]] = dict(
          corrector_new_variables=corrector_new_variables,
          corrector_vars_scale_by_sigma=corrector_vars_scale_by_sigma,
          linf_neighborhood_scale=linf_neighborhood_scale,
          verbose=verbose,
          solver=s.QTQPSolver
        )

settings = {}
solvers = []

solvers.append("QTQP_0_0_3_pardiso")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.PARDISO)
solvers.append("QTQP_0_0_3_qdldl")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL)
solvers.append("QTQP_clarabel_term_conds_qdldl")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e8_static_mu_times_10_no_normalization")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e8_static_mu_times_100_no_normalization")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e8_static_mu_times_4_no_normalization")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e8_static_orig_anchor")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e8_static_orig_terms_equilibrate_before_init")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e8_static_orig_terms_equilibrate_after_init")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_qdldl_1e8_static_orig_terms_equilibrate_after_init")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_qdldl_1e8_static_init_with_10mu_newton_step")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_qdldl_1e8_static_init_with_10mu_newton_step_10_times")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_qdldl_1e8_static_init_with_100mu_newton_step_20_times")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_qdldl_1e8_static_init_with_10mu_newton_step_s_times_mu")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_mcc")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_equilibration_fix_xyt_anchor_1e8_min_static_reg")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_equilibration_fix_mu_fix_tatc_fix_1e8_min_static_reg")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_equilibration_fix_1e8_min_static_reg")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e8_static")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-8)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e9_static")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-9)
solvers.append("QTQP_clarabel_term_conds_qdldl_1e10_static")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL, min_static_regularization=1e-10)
solvers.append("Clarabel")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.ClarabelSolver)
solvers.append("Clarabel_no_dyn_reg")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.ClarabelSolver, dynamic_regularization_enable=False, presolve_enable=False)

# Run all examples
netlib_runner = NETLIBRunner(solvers,
                             settings,
                             OUTPUT_FOLDER,
                             infeasible,
                             add_quadratic)

# debug
# netlib_runner.problems = ["fffff800"]
#netlib_runner.problems = \
#  netlib_runner.problems[netlib_runner.problems.index("pilot4i"):]
#
#probs = netlib_runner.problems
#for prob in probs:
#  netlib_runner = NETLIBRunner(solvers,
#                             settings,
#                             OUTPUT_FOLDER,
#                             infeasible)
#  netlib_runner.problems = [prob]
#  netlib_runner.solve(parallel=parallel, cores=12)
#  if infeasible:
#    shutil.rmtree("./results/netlib_infeasible")
#  else:
#    shutil.rmtree("./results/netlib_feasible")


netlib_runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy,
                   infeasible_test=infeasible)

make_latex_table(solvers, OUTPUT_FOLDER, infeasible)
