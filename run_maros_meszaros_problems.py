'''
Run Maros-Meszaros problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from maros_meszaros_problems.maros_meszaros_problem import MarosMeszarosRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
from utils.make_table import make_latex_table
import os
import argparse
import qtqp

import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)


parser = argparse.ArgumentParser(description='Maros Meszaros Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy',
                    default=False, action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=False,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)


OUTPUT_FOLDER = 'maros_meszaros_problems'

solvers = [s.QTQP] #, s.SCS_AA2, s.SCS_AA1] #, s.ECOS, s.qpOASES, s.QPALM

if high_accuracy:
    solvers = [solver + s.HIGH for solver in solvers]
    OUTPUT_FOLDER += s.HIGH

settings = s.get_settings()

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = verbose

settings = {}
solvers = []

solvers = []
settings = {}


for solver in [qtqp.LinearSolver.PARDISO]: #, qtqp.LinearSolver.EIGEN, qtqp.LinearSolver.QDLDL]:
  solvers.append(f"QTQP_{solver}_0_0_3_clarabel_terms")
  settings[solvers[-1]] = dict(
                              solver=s.QTQPSolver,
                              linear_solver=solver,
                              )

for solver in [qtqp.LinearSolver.PARDISO]: #, qtqp.LinearSolver.EIGEN, qtqp.LinearSolver.QDLDL]:
  solvers.append(f"QTQP_{solver}_0_0_3_always_1_it_refinement")
  settings[solvers[-1]] = dict(
                              solver=s.QTQPSolver,
                              linear_solver=solver,
                              )
#
#for solver in [qtqp.LinearSolver.QDLDL, qtqp.LinearSolver.PARDISO]:
#  for atol in [1e-7]:
#    for rtol in [1e-8]:
#      for it in [1, 5, 10, 20, 50]:
#        for streg in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
#          for equilibrate in [True, False]:
#            solvers.append(f"QTQP_{solver}_atol{atol}_rtol{rtol}_it{it}_streg{streg}_equilibrate{equilibrate}")
#            settings[solvers[-1]] = dict(verbose=verbose, 
#                                        solver=s.QTQPSolver, 
#                                        linear_solver=solver,
#                                        atol=atol,
#                                        rtol=rtol,
#                                        max_iterative_refinement_steps=it,
#                                        min_static_regularization=streg,
#                                        equilibrate=equilibrate)
#
#solvers.append(f"QTQP_qdldl_5_stalling_ratio")
#settings[solvers[-1]] = dict(verbose=verbose, 
#                            solver=s.QTQPSolver, 
#                            linear_solver=qtqp.LinearSolver.QDLDL)
#
#solvers.append(f"QTQP_qdldl_1_stalling_ratio")
#settings[solvers[-1]] = dict(verbose=verbose, 
#                            solver=s.QTQPSolver, 
#                            linear_solver=qtqp.LinearSolver.QDLDL)

#
#for solver in [qtqp.LinearSolver.QDLDL, qtqp.LinearSolver.PARDISO]:
#  for atol in [1e-7]:
#    for rtol in [1e-8]:
#      for it in [1, 5, 10, 20, 50]:
#        for streg in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:
#          for equilibrate in [True, False]:
#            solvers.append(f"QTQP_{solver}_atol{atol}_rtol{rtol}_it{it}_streg{streg}_equilibrate{equilibrate}")
#            settings[solvers[-1]] = dict(verbose=verbose, 
#                                        solver=s.QTQPSolver, 
#                                        linear_solver=solver,
#                                        atol=atol,
#                                        rtol=rtol,
#                                        max_iterative_refinement_steps=it,
#                                        min_static_regularization=streg,
#                                        equilibrate=equilibrate)
#
#solvers.append(f"QTQP_qdldl_5_stalling_ratio")
#settings[solvers[-1]] = dict(verbose=verbose, 
#                            solver=s.QTQPSolver, 
#                            linear_solver=qtqp.LinearSolver.QDLDL)
#
#solvers.append(f"QTQP_qdldl_1_stalling_ratio")
#settings[solvers[-1]] = dict(verbose=verbose, 
#                            solver=s.QTQPSolver, 
#                            linear_solver=qtqp.LinearSolver.QDLDL)

solvers.append("Clarabel_all_default_settings_no_m_z")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.ClarabelSolver)
#solvers.append("Clarabel_equil_on_presolve_off_dyn_reg_off")
#settings[solvers[-1]] = dict(verbose=verbose, solver=s.ClarabelSolver)


# Run all examples
maros_meszaros_runner = MarosMeszarosRunner(solvers,
                                            settings,
                                            OUTPUT_FOLDER)

# maros_meszaros_runner.problems = ["YAO"]
maros_meszaros_runner.solve(parallel=parallel, cores=12)

# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)

make_latex_table(solvers, OUTPUT_FOLDER, False)
