from liu_pataki_problems.liu_pataki_problem import LIU_PATAKIRunner
import solvers.solvers as s
from utils.benchmark import compute_stats_info
from utils.make_table import make_latex_table
import os
import argparse
import shutil


parser = argparse.ArgumentParser(description='LIU PATAKI Runner')
parser.add_argument('--high_accuracy', help='Test with high accuracy',
                    default=False,
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

solvers=[s.SCS, s.COSMO]

# Shut up solvers
if verbose:
    for key in s.settings:
        s.settings[key]['verbose'] = True

OUTPUT_FOLDER = 'liu_pataki_problems'


# Run all examples
runner = LIU_PATAKIRunner(solvers,
                          s.settings,
                          OUTPUT_FOLDER)
#runner.problems = ["infeas_clean_10_10_99"]

runner.solve(parallel=parallel, cores=12)
# Compute results statistics
compute_stats_info(solvers, OUTPUT_FOLDER,
                   high_accuracy=high_accuracy, infeasible_test=True)

make_latex_table(solvers, OUTPUT_FOLDER, True)
