import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

from solvers.solvers import SOLVER_MAP
from problem_classes.qplib import QPLIB
from utils.general import make_sure_path_exists

import numpy as np

PROBLEMS_FOLDER = "qplib_data"

# from http://qplib.zib.de/qplib.solu
OPT_VALS = {
    10034:      -0.0660101760500000,
    10038:       0.0000000000000000,
    8495:    42857.4963900000002468,
    8500:        np.nan,  # not on website
    8515:      319.9999887000000172,
    8547:       -0.0073984697840000,
    8559: 74223239.8299999982118607,
    8567: 78965987.9500000029802322,
    8602:    27353.3872000000010303,
    8616:      245.0685977999999920,
    8785:     7867.4911490000004051,
    8790:       -0.0001562421091000,
    8792:     3593.5183550000001560,
    8845: 10907992.4900000002235174,
    8906:  2699111.5129999998025596,
    8938:      -35.7794529499999996,
    8991:       -0.0016678673780000,
    9002:        np.nan,  # not on website
    9008:        0.0000000000000000,
}

class QPLIBRunner(object):
    '''
    Examples runner
    '''
    def __init__(self,
                 solvers,
                 settings,
                 output_folder):
        self.solvers = solvers
        self.settings = settings
        self.output_folder = output_folder

        # Get maros problems list
        problems_dir = os.path.join(".", "problem_classes", PROBLEMS_FOLDER)
        # List of problems in .mat format
        lst_probs = sorted([f for f in os.listdir(problems_dir) if \
            f.endswith('.qplib')])
        self.problems = [f[6:-6] for f in lst_probs]   # List of problem names
        print(self.problems)

    def solve(self, parallel=True, cores=32):
        '''
        Solve problems of type example

        The results are stored as

            ./results/{self.output_folder}/{solver}/results.csv

        using a pandas table with fields
            - 'name': QPLIB problem name
            - 'solver': solver name
            - 'status': solver status
            - 'run_time': execution time
            - 'iter': number of iterations
            - 'obj_val': objective value from solver
            - 'obj_opt': optimal objective value
            - 'n': leading dimension
            - 'N': nnz dimension (nnz(P) + nnz(A))
        '''

        print("Solving Convex QPLIB problems")
        print("-------------------------------")

        if parallel:
            pool = Pool(processes=min(cores, cpu_count()))

        # Iterate over all solvers
        for solver in self.solvers:
            settings = self.settings[solver]

            #  # Initialize solver results
            #  results_solver = []

            # Solution directory
            path = os.path.join('.', 'results', self.output_folder,
                                solver)

            # Create directory for the results
            make_sure_path_exists(path)

            # Get solver file name
            results_file_name = os.path.join(path, 'results.csv')

            # Check if file name already exists
            if not os.path.isfile(results_file_name):
                # Solve Maros Meszaros problems
                if parallel:
                    results = pool.starmap(self.solve_single_example,
                                           zip(self.problems,
                                               repeat(solver),
                                               repeat(settings)))
                else:
                    results = []
                    for problem in self.problems:
                        results.append(self.solve_single_example(problem,
                                                                 solver,
                                                                 settings))
                # Create dataframe
                df = pd.concat(results)

                # Store results
                df.to_csv(results_file_name, index=False)

            #  else:
            #      # Load from file
            #      df = pd.read_csv(results_file_name)
            #
            #      # Combine list of dataframes
            #      results_solver.append(df)

        if parallel:
            pool.close()  # Not accepting any more jobs on this pool
            pool.join()   # Wait for all processes to finish

    def solve_single_example(self,
                             problem,
                             solver, settings):
        '''
        Solve QPLIB 'problem' with 'solver'

        Args:
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''
        print(" - Solving %s with solver %s" % (problem, solver))
        print(f" - OPT : {OPT_VALS[int(problem)]}")

        # Create example instance
        full_name = os.path.join(".", "problem_classes",
                                 PROBLEMS_FOLDER, "QPLIB_%s.qplib" % problem)
        instance = QPLIB(full_name)


        # Solve problem
        s = SOLVER_MAP[solver](settings)
        results = s.solve(instance)

        # Create solution as pandas table
        P = instance.qp_problem['P']
        A = instance.qp_problem['A']
        N = P.nnz + A.nnz

        # Add constant part to objective value
        obj = results.obj_val
        if results.obj_val is not None:
            obj += instance.qp_problem["r"]

        # Change sign of objective if maximization problem
        if instance.obj_type == 'max':
            obj *= -1

        # Optimal cost distance from Maros Meszaros results
        # (For DEBUG)
        # ( obj - opt_obj )/(|opt_obj|)
        #  if obj is not None:
        #      obj_dist = abs(obj - OPT_COST_MAP[problem])
        #      if abs(OPT_COST_MAP[problem]) > 1e-20:
        #          # Normalize cost distance
        #          obj_dist /= abs(OPT_COST_MAP[problem])
        #  else:
        #      obj_dist = np.inf

        solution_dict = {'name': [problem],
                         'solver': [solver],
                         'status': [results.status],
                         'run_time': [results.run_time],
                         'iter': [results.niter],
                         'obj_val': [obj],
                         'n': [instance.qp_problem["n"]],
                         'm': [instance.qp_problem["m"]],
                         'N': [N]}

        # Add status polish if OSQP
        if 'OSQP' in solver:
            solution_dict['status_polish'] = results.status_polish
            solution_dict['setup_time'] = results.setup_time
            solution_dict['solve_time'] = results.solve_time
            solution_dict['update_time'] = results.update_time
            solution_dict['rho_updates'] = results.rho_updates
            solution_dict['rho_estimate'] = results.rho_estimate

        if 'SCS' in solver:
            solution_dict['setup_time'] = results.setup_time
            solution_dict['solve_time'] = results.solve_time
            solution_dict['scale'] = results.scale
            solution_dict['scale_updates'] = results.scale_updates

        print(" - Solved %s with solver %s" % (problem, solver))

        # Return solution
        return pd.DataFrame(solution_dict)
