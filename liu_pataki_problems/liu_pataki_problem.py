import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

from solvers.solvers import SOLVER_MAP
from problem_classes.liu_pataki import LIU_PATAKI
from utils.general import make_sure_path_exists

import numpy as np

BASE_PROBLEMS_FOLDER = "liu_pataki_data"

class LIU_PATAKIRunner(object):
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

        # Get problems list
        self.problems_dir = os.path.join(".", "problem_classes", BASE_PROBLEMS_FOLDER)

        # List of problems in .mat format
        lst_probs = sorted([f for f in os.listdir(self.problems_dir) if
                            f.endswith('.mat')])
        self.problems = [f[:-4] for f in lst_probs]   # List of problem names
        # cannot handle rotated lorentz cones:
        print(self.problems)

    def solve(self, parallel=True, cores=32):
        '''
        Solve problems of type example

        The results are stored as

            ./results/{self.output_folder}/{solver}/results.csv

        using a pandas table with fields
            - 'name': Maros problem name
            - 'solver': solver name
            - 'status': solver status
            - 'run_time': execution time
            - 'iter': number of iterations
            - 'obj_val': objective value from solver
            - 'obj_opt': optimal objective value
            - 'n': leading dimension
            - 'N': nnz dimension (nnz(P) + nnz(A))
        '''

        print("Solving Liu Pataki instances")
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

            # If results file already exists read in solved problems
            if os.path.isfile(results_file_name):
                df = pd.read_csv(results_file_name)
                # filter down to unsolved only
                solver_problems = [p for p in self.problems if p not in df['name'].values]
            else:
                solver_problems = self.problems.copy()


            for problem in solver_problems:
                df = pd.DataFrame(self.solve_single_example(problem, solver, settings))
                if os.path.isfile(results_file_name):
                    # append to existing csv
                    df.to_csv(results_file_name, mode='a', header=False, index=False)
                else:
                    # csv is new, write with header
                    df.to_csv(results_file_name, mode='w', header=True, index=False)

        if parallel:
            pool.close()  # Not accepting any more jobs on this pool
            pool.join()   # Wait for all processes to finish

    def solve_single_example(self,
                             problem,
                             solver, settings):
        '''
        Solve 'problem' with 'solver'

        Args:
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''
        print(" - Solving %s with solver %s" % (problem, solver))

        # Create example instance
        full_name = os.path.join(".", "problem_classes",
                                 BASE_PROBLEMS_FOLDER, "%s.mat" % problem)
        instance = LIU_PATAKI(full_name, problem)


        # Solve problem
        s = SOLVER_MAP[solver](settings)
        results = s.solve(instance)

        # Create solution as pandas table
        P = instance.sdp_problem['P']
        A = instance.sdp_problem['A']
        N = A.nnz + (P.nnz if P is not None else 0)

        # Add constant part to objective value
        obj = results.obj_val
        if results.obj_val is not None:
            obj += instance.sdp_problem["r"]

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

        # Add status polish if OSQP
        if 'OSQP' in solver:
            solution_dict['status_polish'] = results.status_polish
            solution_dict['setup_time'] = results.setup_time
            solution_dict['solve_time'] = results.solve_time
            solution_dict['update_time'] = results.update_time
            solution_dict['rho_updates'] = results.rho_updates
            solution_dict['rho_estimate'] = results.rho_estimate

        if 'SCS' in solver:
            for k, v in results.info.items():
                if k not in solution_dict:  # don't overwrite existing
                    solution_dict[k] = v

        print(" - Solved %s with solver %s" % (problem, solver))

        # Return solution
        return pd.DataFrame(solution_dict)
