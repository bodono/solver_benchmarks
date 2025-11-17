'''Run Maros-Meszaros problems'''
from maros_meszaros_problems.maros_meszaros_problem import MarosMeszarosRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
from utils.make_table import make_latex_table
import os
import argparse
import qtqp

import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)


parser = argparse.ArgumentParser(description='Maros Meszaros Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy',
                    default=False, action='store_true')
parser.add_argument('--verbose', help='Verbose solvers', default=False,
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


OUTPUT_FOLDER = 'maros_meszaros_problems'

if high_accuracy:
    solvers = [solver + s.HIGH for solver in solvers]
    OUTPUT_FOLDER += s.HIGH

settings = {}
solvers = []
solvers.append("Clarabel")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.ClarabelSolver)
solvers.append("QTQP_qdldl")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.QDLDL)

# Run all examples
maros_meszaros_runner = MarosMeszarosRunner(solvers,
                                            settings,
                                            OUTPUT_FOLDER)

# To test a single problem:
# maros_meszaros_runner.problems = ["YAO"]
maros_meszaros_runner.solve(parallel=parallel, cores=12)

# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)

make_latex_table(solvers, OUTPUT_FOLDER, False)
