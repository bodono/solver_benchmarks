'''
Run random problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from random_problems.random_problem import RandomProbRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
import os
import argparse
import shutil


parser = argparse.ArgumentParser(description='random Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=True,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--infeasible', help='Run of infeasible', default=False,
                    action='store_true')
parser.add_argument('--unbounded', help='Add a quadratic term,', default=False,
                    action='store_true')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
infeasible = args.infeasible
unbounded = args.unbounded

if infeasible and unbounded:
  raise ValueError('cannot run both infeasible and unbounded')

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

solvers=[s.SCS, s.OSQP, s.QPALM] #, s.COSMO, s.SCS_AA, s.ECOS, s.qpOASES, s.QPALM]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

if infeasible:
  OUTPUT_FOLDER = 'random_infeasible'
elif unbounded:
  OUTPUT_FOLDER = 'random_unbounded'
else:
  OUTPUT_FOLDER = 'random_feasible'

m=1500
n=1000

SEED = 1234
NUM_PROBS = 1000

# Run all examples
runner = RandomProbRunner(m, n,
                          SEED,
                          NUM_PROBS,
                          solvers,
                          s.settings,
                          OUTPUT_FOLDER,
                          infeasible,
                          unbounded)


runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy,
                   infeasible_test=infeasible or unbounded)
