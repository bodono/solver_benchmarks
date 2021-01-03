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
import os
import argparse


parser = argparse.ArgumentParser(description='NETLIB Runner')
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

solvers=[s.SCS, s.OSQP]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

OUTPUT_FOLDER = 'netlib_problems'

# Run all examples
netlib_runner = NETLIBRunner(solvers,
                             s.settings,
                             OUTPUT_FOLDER)

# debug
netlib_runner.problems = [
"woodinfe",
"bgdbg1",
"bgetam",
"bgprtr",
"box1"]

netlib_runner.solve(parallel=parallel, cores=12)

# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)
