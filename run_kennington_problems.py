'''
Run kennington problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from kennington_problems.kennington_problem import KENNINGTONRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
from utils.make_table import make_latex_table
import os
import argparse
import shutil


parser = argparse.ArgumentParser(description='kennington Runner')
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

solvers=[s.SCS, s.OSQP, s.COSMO]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

OUTPUT_FOLDER = 'kennington_problems'

# Run all examples
kennington_runner = KENNINGTONRunner(solvers,
                             s.settings,
                             OUTPUT_FOLDER)

print(kennington_runner.problems)
# debug
#kennington_runner.problems = ["fhnw-binpack4-48"]
#kennington_runner.problems = \
#  kennington_runner.problems[kennington_runner.problems.index("lotsize"):]
#
#probs = kennington_runner.problems
#for prob in probs:
#  kennington_runner = kenningtonRunner(solvers,
#                             s.settings,
#                             OUTPUT_FOLDER,
#                             infeasible)
#  kennington_runner.problems = [prob]
#  kennington_runner.solve(parallel=parallel, cores=12)
#  if infeasible:
#    shutil.rmtree("./results/kennington_infeasible")
#  else:
#    shutil.rmtree("./results/kennington_feasible")

kennington_runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)

make_latex_table(solvers, OUTPUT_FOLDER, False)
