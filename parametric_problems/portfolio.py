"""
Solve Portoflio problem for one year simulation
"""
import os
import numpy as np
from utils.general import make_sure_path_exists
import pandas as pd
from problem_classes.portfolio import PortfolioExample
import scipy.sparse
import scs

class PortfolioParametric(object):
    def __init__(self,
                 settings,
                 n_factors=100,
                 n_assets=3000,
                 n_months_per_risk_model_update=3,
                 n_years=4):
        """
        Generate Portfolio problem as parametric QP

        Args:
            settings: solver settings
            n_factors: number of factors in risk model
            n_assets: number of assets to be optimized
            n_months_per_risk_model_update: number of months for every risk
                                            model update
            n_years: number of years to run the simulation
        """
        self.settings = settings
        self.n_factors = n_factors
        self.n_assets = n_assets
        self.n_qp_per_month = 20  # Number of trading days
        self.n_qp_per_update = self.n_qp_per_month * \
            n_months_per_risk_model_update
        self.n_problems = n_years * 240
        self.alpha = 0.1  # Relaxation parameter between new data nad old ones

    def solve(self):
        """
        Solve Portfolio problem
        """

        print("Solve Portfolio problem for dimension %i" % self.n_factors)

        # Create example instance
        instance = PortfolioExample(self.n_factors, n=self.n_assets)

        # Store number of nonzeros in F and D for updates
        nnzF = instance.F.nnz

        # Store alpha
        alpha = self.alpha

        '''
        Solve problem without warm start
        '''
        #  print("Solving without warm start")

        # Solution directory
        no_ws_path = os.path.join('.', 'results', 'parametric_problems',
                                  'no warmstart',
                                  'Portfolio',
                                  )

        # Create directory for the results
        make_sure_path_exists(no_ws_path)

        # Check if solution already exists
        n_file_name = os.path.join(no_ws_path, 'n%i.csv' % self.n_factors)

        if not os.path.isfile(n_file_name):

            res_list_no_ws = []  # Initialize results
            for i in range(self.n_problems):
                qp = instance.qp_problem

                # Solve problem
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
                #  print("niter = %d" % r.info.iter)

                run_time = (results['info']['setup_time'] +
                            results['info']['solve_time']) /1000.
                solution_dict = {'status': [results['info']['status']],
                                 'run_time': [run_time],
                                 'iter': [results['info']['iter']],
                                 'obj_val': [results['info']['pobj']]}


                if results['info']['status'] != "solved":
                    print("SCS no warmstart did not solve the problem")


                res_list_no_ws.append(pd.DataFrame(solution_dict))

                # Update model
                current_mu = instance.mu
                current_F_data = instance.F.data
                current_D_data = instance.D.data

                if i % self.n_qp_per_update == 0:
                    #  print("Update everything: mu, F, D")
                    # Update everything
                    new_mu = alpha * np.random.randn(instance.n) + (1 - alpha) * current_mu
                    new_F = instance.F.copy()
                    new_F.data = alpha * np.random.randn(nnzF) + (1 - alpha) * current_F_data
                    new_D = instance.D.copy()
                    new_D.data = alpha * np.random.rand(instance.n) * \
                        np.sqrt(instance.k) + (1 - alpha) * current_D_data
                    instance.update_parameters(new_mu, new_F, new_D)
                else:
                    #  print("Update only mu")
                    # Update only mu
                    new_mu = alpha * np.random.randn(instance.n) + (1 - alpha) * current_mu
                    instance.update_parameters(new_mu)

            # Get full warm-start
            res_no_ws = pd.concat(res_list_no_ws)

            # Store file
            res_no_ws.to_csv(n_file_name, index=False)

            # Plot results
            # import matplotlib.pylab as plt
            # plt.figure(0)
            # plt.plot(X_no_ws.T)
            # plt.title("No Warm Start")
            # plt.show(block=False)

