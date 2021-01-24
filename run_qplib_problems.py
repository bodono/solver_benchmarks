'''
Run QPLIB problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from qplib_problems.qplib_problem import QPLIBRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
import os
import argparse


parser = argparse.ArgumentParser(description='QPLIB Runner')
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

OUTPUT_FOLDER = 'qplib_problems'
solvers=[s.OSQP, s.SCS, s.COSMO, s.SCS_AA, s.ECOS, s.qpOASES]
,
# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

# Run all examples
qplib_runner = QPLIBRunner(solvers,
                           s.settings,
                           OUTPUT_FOLDER)

# DEBUG only: Choose only 2 problems
#qplib_runner.problems = ["9002"]
qplib_runner.solve(parallel=parallel, cores=12)

# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)
