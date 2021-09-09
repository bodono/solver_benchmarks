"""
Solve Lasso problem as parametric QP by updating iteratively lambda
"""
import numpy as np
import pandas as pd
import os
from problem_classes.control import ControlExample
from utils.general import make_sure_path_exists
import scipy.sparse
import scs


class MPCParametric(object):
    def __init__(self,
                 settings,
                 dimension,
                 n_simulation=100):
        """
        Generate MPC problem as parametric QP

        Args:
            settings: solver settings
            dimension: leading dimension for the problem
            minimum_lambda_over_max: min ratio between lambda and lambda_max
            n_simulation: number of MPC problems to solve
        """
        self.settings = settings
        self.dimension = dimension
        self.n_simulation = n_simulation

    def solve(self):
        """
        Solve MPC problem
        """

        print("Solve MPC problem for dimension %i" % self.dimension)

        # Create example instance
        instance = ControlExample(self.dimension)
        qp = instance.qp_problem
        x0 = np.copy(instance.x0)

        '''
        Solve problem without warm start
        '''
        # Solution directory
        no_ws_path = os.path.join('.', 'results', 'parametric_problems',
                                  'no warmstart',
                                  'MPC',
                                  )

        # Create directory for the results
        make_sure_path_exists(no_ws_path)

        # Check if solution already exists
        n_file_name = os.path.join(no_ws_path, 'n%i.csv' % self.dimension)

        if not os.path.isfile(n_file_name):
            # Initialize states and inputs for the whole simulation
            X_no_ws = np.zeros((instance.nx, self.n_simulation + 1))
            U_no_ws = np.zeros((instance.nu, self.n_simulation))
            X_no_ws[:, 0] = x0

            res_list_no_ws = []  # Initialize results
            for i in range(self.n_simulation):

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

                run_time = (results['info']['setup_time'] +
                            results['info']['solve_time']) /1000.
                solution_dict = {'status': [results['info']['status']],
                                 'run_time': [run_time],
                                 'iter': [results['info']['iter']]}

                if results['info']['status'] != "solved":
                    print("SCS no warmstart did not solve the problem")


                res_list_no_ws.append(pd.DataFrame(solution_dict))

                # Get input
                U_no_ws[:, i] = results['x'][instance.nx * (instance.T + 1):
                                    instance.nx * (instance.T + 1)+instance.nu]

                # Propagate state
                X_no_ws[:, i + 1] = instance.A.dot(X_no_ws[:, i]) + \
                    instance.B.dot(U_no_ws[:, i])

                # Update initial state
                instance.update_x0(X_no_ws[:, i + 1])

            # Get full warm-start
            res_no_ws = pd.concat(res_list_no_ws)

            # Store file
            res_no_ws.to_csv(n_file_name, index=False)

            # Plot results
            # import matplotlib.pylab as plt
            # plt.figure(1)
            # plt.plot(X_no_ws.T)
            # plt.title("No Warm Start")
            # plt.show(block=False)

