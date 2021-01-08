'''
Run dimacs problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from dimacs_problems.dimacs_problem import DIMACSRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
import os
import argparse
import shutil


parser = argparse.ArgumentParser(description='DIMACS Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=True,
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

solvers=[s.SCS, s.COSMO]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

OUTPUT_FOLDER = 'dimacs_problems'


# Run all examples
dimacs_runner = DIMACSRunner(solvers,
                             s.settings,
                             OUTPUT_FOLDER)

# debug
#dimacs_runner.problems = ["filtinf1"]
#dimacs_runner.problems = \
#  dimacs_runner.problems[dimacs_runner.problems.index("hamming_7_5_6"):]
#
#probs = dimacs_runner.problems
#for prob in probs:
#  dimacs_runner = dimacsRunner(solvers,
#                             s.settings,
#                             OUTPUT_FOLDER)
#  dimacs_runner.problems = [prob]
#  dimacs_runner.solve(parallel=parallel, cores=12)
#  shutil.rmtree("./results/dimacs_feasible")

dimacs_runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)
