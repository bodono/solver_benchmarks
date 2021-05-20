'''
Run mittelmann problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK

'''
from mittelmann_problems.mittelmann_problem import MITTELMANNRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
from utils.make_table import make_latex_table
import os
import argparse
import shutil


parser = argparse.ArgumentParser(description='mittelmann Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=True,
                    action='store_true')
parser.add_argument('--parallel', help='Parallel solution', default=False,
                    action='store_true')
parser.add_argument('--preprocessed', help='Run preprocessed probs only',
                    default=False, action='store_true')

args = parser.parse_args()
high_accuracy = args.high_accuracy
verbose = args.verbose
parallel = args.parallel

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


# Run all examples
if preprocessed:
  OUTPUT_FOLDER += '_preprocessed'
  mittelmann_runner = MITTELMANNRunner(solvers,
                             s.settings,
                             OUTPUT_FOLDER)
                             "mittelmann-papilo-preprocessed")
else:
  # Run all examples
  mittelmann_runner = MITTELMANNRunner(solvers,
                                       s.settings,
                                       OUTPUT_FOLDER)


print(mittelmann_runner.problems)
# debug
#mittelmann_runner.problems = ["Linf_520c"]
#mittelmann_runner.problems = \
#  mittelmann_runner.problems[mittelmann_runner.problems.index("lotsize"):]
#
#probs = mittelmann_runner.problems
#for prob in probs:
#  mittelmann_runner = mittelmannRunner(solvers,
#                             s.settings,
#                             OUTPUT_FOLDER,
#                             infeasible)
#  mittelmann_runner.problems = [prob]
#  mittelmann_runner.solve(parallel=parallel, cores=12)
#  if infeasible:
#    shutil.rmtree("./results/mittelmann_infeasible")
#  else:
#    shutil.rmtree("./results/mittelmann_feasible")

mittelmann_runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)

make_latex_table(solvers, OUTPUT_FOLDER, False)
