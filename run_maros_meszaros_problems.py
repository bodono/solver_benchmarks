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
import os
import argparse


parser = argparse.ArgumentParser(description='Maros Meszaros Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy',
                    default=False, action='store_true')
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


OUTPUT_FOLDER = 'maros_meszaros_problems_tmp'

solvers = [s.SCS, s.OSQP]#, s.ECOS, s.COSMO, s.qpOASES]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

# Run all examples
maros_meszaros_runner = MarosMeszarosRunner(solvers,
                                            s.settings,
                                            OUTPUT_FOLDER)

#maros_meszaros_runner.problems = ["BOYD1"]
maros_meszaros_runner.solve(parallel=parallel, cores=12)

# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)
