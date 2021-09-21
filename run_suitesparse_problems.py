'''
Run Suitesparse problems for the OSQP paper

This code tests the solvers:
    - OSQP
    - GUROBI
    - MOSEK
    - ECOS

'''
from suitesparse_problems.suitesparse_problem import SuitesparseRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
import os
import argparse


MAX_PROB_SIZE_MB = 5

parser = argparse.ArgumentParser(description='Suitesparse Lasso Runner')
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

settings = s.get_settings()
solvers=[s.SCS, s.OSQP] #, s.COSMO]
OUTPUT_FOLDER = 'suitesparse_problems_NEW'

if high_accuracy:
    solvers = [solver + s.HIGH for solver in solvers]
    OUTPUT_FOLDER += s.HIGH

# Shut up solvers
for key in settings:
    settings[key]['verbose'] = verbose


problems = [
            'Lasso',
            'Huber',
            ]

# Run all examples
for problem in problems:
    suitesparse_runner = SuitesparseRunner(problem,
                                           solvers,
                                           settings,
                                           OUTPUT_FOLDER,
                                           MAX_PROB_SIZE_MB)
    # DEBUG: To test
    # suitesparse_runner.problems = ['Bai_qh768']
    suitesparse_runner.solve(parallel=parallel, cores=12)

#  suitesparse_lasso_runner = SuitesparseLassoRunner(solvers,
                                                  #  settings,
                                                  #  OUTPUT_FOLDER)

# DEBUG Only two problems
#  suitesparse_lasso_runner.problems = ['Springer_ESOC']  # ['HB_abb313', 'HB_ash331']
#  suitesparse_lasso_runner.problems = ['Rucci_Rucci1']  # Problematic
#  suitesparse_lasso_runner.problems.remove('Rucci_Rucci1')  # Problematic, makes Gurobi crash the whole system
#  suitesparse_lasso_runner.solve(parallel=parallel, cores=12)

compute_stats_info(solvers, OUTPUT_FOLDER,
                   problems=problems,
                   high_accuracy=high_accuracy)

make_latex_table(solvers, OUTPUT_FOLDER, False)
