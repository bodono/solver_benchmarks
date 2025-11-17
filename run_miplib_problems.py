'''
Run miplib problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from miplib_problems.miplib_problem import MIPLIBRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
from utils.make_table import make_latex_table
import numpy as np
import os
import argparse
import shutil
import qtqp
import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

MAX_PROB_SIZE_MB = 5

parser = argparse.ArgumentParser(description='miplib Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=False,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--quick', help='Run quick probs only', default=False,
                    action='store_true')
parser.add_argument('--bisco', help='Run bisco probs only', default=False,
                    action='store_true')
parser.add_argument('--preprocessed', help='Run preprocessed probs only',
                    default=False, action='store_true')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
quick = args.quick
bisco = args.bisco
preprocessed = args.preprocessed

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

settings = s.get_settings()

OUTPUT_FOLDER = 'miplib_problems'
solvers=[s.QTQP, s.OSQP] # , s.SCS_AA1, s.SCS_AA2, s.OSQP]

if high_accuracy:
    solvers = [solver + s.HIGH for solver in solvers]
    OUTPUT_FOLDER += s.HIGH

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = verbose

solvers = []
solvers.append("Clarabel_w_equil")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.ClarabelSolver)
#for solver in [qtqp.LinearSolver.QDLDL]: #, qtqp.LinearSolver.PARDISO]:
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
for solver in [qtqp.LinearSolver.PARDISO]: #, qtqp.LinearSolver.SCIPY, qtqp.LinearSolver.EIGEN, qtqp.LinearSolver.QDLDL]:
  solvers.append(f"QTQP_{solver}_0_0_3_always_1_it_refinement")
  settings[solvers[-1]] = dict(
                              solver=s.QTQPSolver,
                              linear_solver=solver,
                              )


miplib_runner = MIPLIBRunner(solvers,
                             settings,
                             OUTPUT_FOLDER,
                             MAX_PROB_SIZE_MB)


# Run all examples
if preprocessed:
  OUTPUT_FOLDER += '_preprocessed'
  miplib_runner = MIPLIBRunner(solvers,
                             settings,
                             OUTPUT_FOLDER,
                             MAX_PROB_SIZE_MB,
                             "miplib2017-firstorderlp-paper-papilo")
elif quick:
  OUTPUT_FOLDER += '_quick'
  probs = []
  print("QUICK test set")
  with open('miplib_problems/quick_test.txt', 'r') as f:
    probs = f.read().splitlines()
  miplib_runner = MIPLIBRunner(solvers,
                           settings,
                           OUTPUT_FOLDER,
                           MAX_PROB_SIZE_MB)
  miplib_runner.problems = sorted(list(set(probs) &
                                  set(miplib_runner.problems)))
elif bisco:
  OUTPUT_FOLDER += '_bisco'
  probs = []
  print("BISCO test set")
  with open('miplib_problems/bisco_probs.txt', 'r') as f:
    probs = f.read().splitlines()
  miplib_runner = MIPLIBRunner(solvers,
                           settings,
                           OUTPUT_FOLDER,
                           MAX_PROB_SIZE_MB)
  miplib_runner.problems = sorted(list(set(probs) &
                                  set(miplib_runner.problems)))

print("Final problem set:")
print(miplib_runner.problems)
# debug
# miplib_runner.problems = ["cryptanalysiskb128n5obj16"]
#miplib_runner.problems = \
#  miplib_runner.problems[miplib_runner.problems.index("lotsize"):]
#
#probs = miplib_runner.problems
#for prob in probs:
#  miplib_runner = miplibRunner(solvers,
#                             settings,
#                             OUTPUT_FOLDER,
#                             infeasible)
#  miplib_runner.problems = [prob]
#  miplib_runner.solve(parallel=parallel, cores=12)
#  if infeasible:
#    shutil.rmtree("./results/miplib_infeasible")
#  else:
#    shutil.rmtree("./results/miplib_feasible")

miplib_runner.solve(parallel=parallel, cores=24)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)

make_latex_table(solvers, OUTPUT_FOLDER, False)
