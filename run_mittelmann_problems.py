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
import qtqp
import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

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
preprocessed = args.preprocessed

print('high_accuracy', high_accuracy)
print('verbose', verbose)
print('parallel', parallel)

solvers=[s.QTQP] #, s.OSQP]
OUTPUT_FOLDER = "mittelmann_problems"

if high_accuracy:
    solvers = [solver + s.HIGH for solver in solvers]
    OUTPUT_FOLDER += s.HIGH

settings = s.get_settings()

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = verbose

solvers = []
settings = {}
#solvers = ["QTQP_new_params"]
#settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.PARDISO)
#
#solvers += ["QTQP_new_params_w_equil"]
#settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.PARDISO)
#
for solver in [qtqp.LinearSolver.PARDISO]: #, qtqp.LinearSolver.EIGEN, qtqp.LinearSolver.QDLDL]:

  solvers.append(f"QTQP_{solver}_0_0_3_always_1_it_refinement")
  settings[solvers[-1]] = dict(
                              solver=s.QTQPSolver,
                              linear_solver=solver,
                              )


solvers.append("Clarabel")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.ClarabelSolver)




# Run all examples
if preprocessed:
  OUTPUT_FOLDER += '_preprocessed'
  mittelmann_runner = MITTELMANNRunner(solvers,
                             settings,
                             OUTPUT_FOLDER,
                             "mittelmann-papilo-preprocessed")
else:
  # Run all examples
  mittelmann_runner = MITTELMANNRunner(solvers,
                                       settings,
                                       OUTPUT_FOLDER)


print(mittelmann_runner.problems)
# debug
# mittelmann_runner.problems = ["heat-source-instance-easy"]
#mittelmann_runner.problems = ["synthetic-design-match"]
#mittelmann_runner.problems = \
#  mittelmann_runner.problems[mittelmann_runner.problems.index("lotsize"):]
#
#probs = mittelmann_runner.problems
#for prob in probs:
#  mittelmann_runner = mittelmannRunner(solvers,
#                             settings,
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
