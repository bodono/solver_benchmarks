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
import os
import argparse
import shutil


parser = argparse.ArgumentParser(description='miplib Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=True,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--quick', help='Run quick probs only', default=False,
                    action='store_true')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
quick = args.quick


print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

solvers=[s.SCS, s.OSQP, s.SCS_AA, s.qpOASES, s.ECOS, s.COSMO]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

OUTPUT_FOLDER = 'miplib_problems'

# Run all examples
miplib_runner = MIPLIBRunner(solvers,
                             s.settings,
                             OUTPUT_FOLDER)

if quick:
  probs = []
  with open('problem_classes/miplib_data/quick_test.txt', 'r') as f:
    probs = f.read().splitlines()
  miplib_runner.problems = sorted(probs)
  #miplib_runner.problems = sorted(list(set(probs) &
  #                                set(miplib_runner.problems)))

print(miplib_runner.problems)
# debug
#miplib_runner.problems = ["fhnw-binpack4-48"]
#miplib_runner.problems = \
#  miplib_runner.problems[miplib_runner.problems.index("pilot4i"):]
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
