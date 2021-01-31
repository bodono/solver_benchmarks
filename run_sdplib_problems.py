'''
Run sdplib problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from sdplib_problems.sdplib_problem import SDPLIBRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
from utils.make_table import make_latex_table
import os
import argparse
import shutil


parser = argparse.ArgumentParser(description='SDPLIB Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=True,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--infeasible', help='Run of infeasible', default=False,
                    action='store_true')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
infeasible = args.infeasible

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

solvers=[s.SCS, s.COSMO]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

if infeasible:
  OUTPUT_FOLDER = 'sdplib_infeasible'
else:
  OUTPUT_FOLDER = 'sdplib_feasible'


# Run all examples
sdplib_runner = SDPLIBRunner(solvers,
                             s.settings,
                             OUTPUT_FOLDER)

INFEASIBLE_PROBLEMS = ["infd1", "infd2", "infp1", "infp2"]

if infeasible:
  sdplib_runner.problems = [p for p in sdplib_runner.problems if p in
                              INFEASIBLE_PROBLEMS]
else:
  sdplib_runner.problems = [p for p in sdplib_runner.problems if p not in
                              INFEASIBLE_PROBLEMS]

# debug
#sdplib_runner.problems = ["qap5"]
#sdplib_runner.problems = \
#  sdplib_runner.problems[sdplib_runner.problems.index("maxG55"):]
#
#probs = sdplib_runner.problems
#for prob in probs:
#  sdplib_runner = sdplibRunner(solvers,
#                             s.settings,
#                             OUTPUT_FOLDER)
#  sdplib_runner.problems = [prob]
#  sdplib_runner.solve(parallel=parallel, cores=12)
#  shutil.rmtree("./results/sdplib_feasible")

sdplib_runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy,
                   infeasible_test=infeasible)

make_latex_table(solvers, OUTPUT_FOLDER, infeasible)
