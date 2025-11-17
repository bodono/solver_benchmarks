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
from utils.make_table import make_latex_table
import os
import argparse
import qtqp

parser = argparse.ArgumentParser(description='QPLIB Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy', default=False,
                    action='store_true')
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

OUTPUT_FOLDER = 'qplib_problems'
solvers=[s.QTQP] #, s.COSMO, s.SCS_ALT]#, s.SCS_AA1, s.SCS_AA2]#, s.ECOS, s.qpOASES]nnn

if high_accuracy:
    solvers = [solver + s.HIGH for solver in solvers]
    OUTPUT_FOLDER += s.HIGH

settings = s.get_settings()

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = verbose

settings = {}
solvers = []
#solvers.append("QTQP_new")
#settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.PARDISO)
solvers.append("QTQP_0_0_3")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.QTQPSolver, linear_solver=qtqp.LinearSolver.PARDISO)
solvers.append("Clarabel")
settings[solvers[-1]] = dict(verbose=verbose, solver=s.ClarabelSolver)


# Run all examples
qplib_runner = QPLIBRunner(solvers,
                           settings,
                           OUTPUT_FOLDER)

# DEBUG only: Choose only 2 problems
#qplib_runner.problems.remove("9008")
qplib_runner.solve(parallel=parallel, cores=12)

# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy)

make_latex_table(solvers, OUTPUT_FOLDER, False)
