"""
Solve Lasso problem as parametric QP by updating iteratively lambda
"""
import numpy as np
import pandas as pd
import os
from solvers.solvers import SOLVER_MAP  # AVOID CIRCULAR DEPENDENCY
from problem_classes.lasso import LassoExample
from utils.general import make_sure_path_exists
import scipy.sparse
import scs


class LassoParametric(object):
    def __init__(self,
                 settings,
                 dimension,
                 minimum_lambda_over_max=0.01,
                 n_problems=100):
        """
        Generate Parametric Lasso object

        Args:
            settings: solver settings
            dimension: leading dimension for the problem
            minimum_lambda_over_max: min ratio between lambda and lambda_max
            n_problem: number of lasso problems to solve
        """
        self.settings = settings
        self.dimension = dimension
        self.minimum_lambda_over_max = minimum_lambda_over_max
        self.n_problems = n_problems

    def solve(self):
        """
        Solve Lasso problem
        """

        print("Solve Lasso problem for dimension %i" % self.dimension)

        # Create example instance
        instance = LassoExample(self.dimension)
        qp = instance.qp_problem

        # Create lambda array
        lambda_array = np.logspace(np.log10(self.minimum_lambda_over_max *
                                            instance.lambda_max),
                                   np.log10(instance.lambda_max),
                                   self.n_problems)[::-1]   # From max to min

        '''
        Solve problem without warm start
        '''
        #  print("Solving without warm start")
        # Solution directory
        no_ws_path = os.path.join('.', 'results', 'parametric_problems',
                                  'no warmstart',
                                  'Lasso',
                                  )

        # Create directory for the results
        make_sure_path_exists(no_ws_path)

        # Check if solution already exists
        n_file_name = os.path.join(no_ws_path, 'n%i.csv' % self.dimension)

        if not os.path.isfile(n_file_name):

            res_list_no_ws = []  # Initialize results
            for lambda_val in lambda_array:
                # Update lambda
                instance.update_lambda(lambda_val)

                A = qp['A']
                (m, n) = A.shape
                # Hack out the equality constraints
                idxs = (qp['u'] - qp['l'] < 1e-6)
                idxs &= (qp['u'] < 1e20)
                idxs &= (qp['l'] > -1e20)
                o_idxs = np.array(range(A.shape[0])) # no need +1 for box cone
                o_idxs = np.hstack((o_idxs[idxs], o_idxs[~idxs]))
                inv_perm = np.argsort(o_idxs)

                A_scs = scipy.sparse.vstack((A[idxs, :], np.zeros((1, n)), -A[~idxs, :]))
                b_scs = np.hstack((qp['u'][idxs], 1, np.zeros(m - np.sum(idxs))))

                data = dict(P=scipy.sparse.csc_matrix(qp['P']), c=qp['q'],
                            A=scipy.sparse.csc_matrix(A_scs), b=b_scs)
                cone = dict(z=np.int(np.sum(idxs)), bl=qp['l'][~idxs].tolist(),
                              bu=qp['u'][~idxs].tolist())

                results = scs.solve(data, cone, **self.settings)


                # DEBUG
                #  print("Lambda = %.4e,\t niter = %d" % (lambda_val, r.info.iter))

                if results['info']['status'] != "solved":
                    print("SCS no warmstart did not solve the problem")

                run_time = (results['info']['setup_time'] +
                            results['info']['solve_time']) /1000.
                solution_dict = {'status': [results['info']['status']],
                                 'run_time': [run_time],
                                 'iter': [results['info']['iter']]}

                res_list_no_ws.append(pd.DataFrame(solution_dict))

            # Get full warm-start
            res_no_ws = pd.concat(res_list_no_ws)

            # Store file
            res_no_ws.to_csv(n_file_name, index=False)

