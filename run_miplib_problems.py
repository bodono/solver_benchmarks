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

MAX_PROB_SIZE_MB = 5

parser = argparse.ArgumentParser(description='miplib Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=True,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--quick', help='Run quick probs only', default=False,
                    action='store_true')
parser.add_argument('--bisco', help='Run bisco probs only', default=False,
                    action='store_true')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
quick = args.quick
bisco = args.bisco

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

solvers=[s.SCS, s.OSQP, s.COSMO]

if high_accuracy:
    solvers = [solver + s.HIGH for solver in solvers]

settings = s.get_settings()

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = verbose

OUTPUT_FOLDER = 'miplib_problems'

# Run all examples
miplib_runner = MIPLIBRunner(solvers,
                             settings,
                             OUTPUT_FOLDER,
                             MAX_PROB_SIZE_MB)

if quick:
  OUTPUT_FOLDER += '_quick'
  probs = []
  print("QUICK test set")
  with open('miplib_problems/quick_test.txt', 'r') as f:
    probs = f.read().splitlines()
  #miplib_runner.problems = sorted(probs)
  miplib_runner.problems = sorted(list(set(probs) &
                                  set(miplib_runner.problems)))
elif bisco:
  OUTPUT_FOLDER += '_bisco'
  probs = []
  print("BISCO test set")
  with open('miplib_problems/bisco_probs.txt', 'r') as f:
    probs = f.read().splitlines()
  #miplib_runner.problems = sorted(probs)
  miplib_runner.problems = sorted(list(set(probs) &
                                  set(miplib_runner.problems)))

print("Final problem set:")
print(miplib_runner.problems)
# debug
miplib_runner.problems = ["neos-5049753-cuanza"]
#miplib_runner.problems = \
#  miplib_runner.problems[miplib_runner.problems.index("lotsize"):]
#
#probs = miplib_runner.problems
#for prob in probs:
#  miplib_runner = miplibRunner(solvers,
#                             s.settings,
#                             OUTPUT_FOLDER,
#                             infeasible)
#  miplib_runner.problems = [prob]
#  miplib_runner.solve(parallel=parallel, cores=12)
#  if infeasible:
#    shutil.rmtree("./results/miplib_infeasible")
#  else:
#    shutil.rmtree("./results/miplib_feasible")

miplib_runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)

make_latex_table(solvers, OUTPUT_FOLDER, False)
