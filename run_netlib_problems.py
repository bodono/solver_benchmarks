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
from utils.make_table import make_latex_table
import os
import argparse
import shutil


parser = argparse.ArgumentParser(description='NETLIB Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=True,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--infeasible', help='Run of infeasible', default=False,
                    action='store_true')
parser.add_argument('--add_quadratic', help='Add a quadratic term,', default=False,
                    action='store_true')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel
infeasible = args.infeasible
add_quadratic = args.add_quadratic

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

solvers=[s.SCS, s.OSQP, s.COSMO] #s.ECOS, s.QPALM, s.COSMO]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

if infeasible:
  OUTPUT_FOLDER = 'netlib_infeasible'
else:
  OUTPUT_FOLDER = 'netlib_feasible'

if add_quadratic:
  OUTPUT_FOLDER = OUTPUT_FOLDER + '_quadratic'

# Run all examples
netlib_runner = NETLIBRunner(solvers,
                             s.settings,
                             OUTPUT_FOLDER,
                             infeasible,
                             add_quadratic)

# debug
#netlib_runner.problems = ["mondou2"]
#netlib_runner.problems = \
#  netlib_runner.problems[netlib_runner.problems.index("pilot4i"):]
#
#probs = netlib_runner.problems
#for prob in probs:
#  netlib_runner = NETLIBRunner(solvers,
#                             s.settings,
#                             OUTPUT_FOLDER,
#                             infeasible)
#  netlib_runner.problems = [prob]
#  netlib_runner.solve(parallel=parallel, cores=12)
#  if infeasible:
#    shutil.rmtree("./results/netlib_infeasible")
#  else:
#    shutil.rmtree("./results/netlib_feasible")


netlib_runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy,
                   infeasible_test=infeasible)

make_latex_table(solvers, OUTPUT_FOLDER, infeasible)
