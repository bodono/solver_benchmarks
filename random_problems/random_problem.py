import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
import pandas as pd

from solvers.solvers import SOLVER_MAP
from problem_classes.netlib import NETLIB
from utils.general import make_sure_path_exists

import numpy as np
import scipy.sparse.linalg
import cvxpy

# This code is used to randomly generate problems of the form:
#
# min. (1/2) x'Px + c'x
# s.t.  Ax + s == b
#       s >= 0 (elementwise)
#
# over variable x \in R^n, s \in R^m
# P is symmetric positive semi-definite
#
# each fn returns: P \in R^{n x n}, A \in R^{m x n}, b \in R^m, c \in R^n
#
# We can generate feasible, infeasible, or unbounded problems
# The resulting problems typically have nice numerical conditioning and can
# generally be solved very quickly.


# Project onto positive orthant
def pos(x):
    return (x + np.abs(x)) / 2.

# Generate a FEASIBLE problem, n variables, m constraints
def gen_feasible(m, n):
    # randomly generate data and point (x, s, y) that satisfies KKT conditions
    # Ax + s = 0
    # Px + A'y == c
    # s >= 0, y >= 0, s'y = 0

    z = np.random.randn(m)
    y = pos(z)
    s = y - z # s >= 0, s'y = 0 via Moreau

    P = np.random.randn(n,n)
    P = 0.1 * P.T @ P
    eigs, V = np.linalg.eig(P)
    # rank 5
    eigs[:-5] = 0 # set some eigs to 0 to be slightly more challenging
    P = (V * eigs) @ V.T
    P = 0.5 * (P + P.T) # symmetrize just to be sure

    # Make problem slightly more numerically challenging:
    A = np.random.randn(m, n)
    U, S, V = np.linalg.svd(A, full_matrices=False)
    S = S**4
    S /= np.max(S)
    A = (U * S) @ V

    x = np.random.randn(n)
    c = -A.T @ y - P @ x
    b = A.dot(x) + s
    b /= np.linalg.norm(b)

    return (P, A, b, c)

# Generate an INFEASIBLE problem, n variables, m constraints
def gen_infeasible(m, n):
    # infeasible cert: b'y < 0, A'y == 0
    # also make sure dual feasible:
    # P _x + A'_y + c = 0, _y \in K^*

    z = np.random.randn(m)
    y = pos(z)  # certificate

    A = np.random.randn(m, n)

    # A := A - y(A'y)' / y'y ==> A'y = 0
    A = A - np.outer(y, np.transpose(A).dot(y)) / np.linalg.norm(y)**2

    b = np.random.randn(m)
    b = -b / np.dot(b, y)

    P = np.random.randn(n,n)
    P = 0.1 * P.T @ P
    eigs, V = np.linalg.eig(P)
    # rank 5 
    eigs[:-5] = 0 # set some eigs to 0 to be slightly more challenging
    P = (V * eigs) @ V.T
    P = 0.5 * (P + P.T) # symmetrize just to be sure

    # make a random point dual feasible
    _y = pos(np.random.randn(m))
    _x = np.random.randn(n)
    c = -A.T @ _y - P @ _x

    return (P, A, b, c)


# Generate an UNBOUNDED problem, n variables, m constraints
def gen_unbounded(m, n):
    # unbounded cert: c'x < 0, Ax + s = 0, Px = 0
    # also make sure primal feasible:
    # A _x + _s = b, _s \in K

    z = np.random.randn(m)
    y = pos(z)
    s = y - z

    P = 0.1 * np.random.randn(n,n)
    P = P.T @ P
    eigs, V = np.linalg.eig(P)

    i = np.argmin(eigs)
    # rank n / 2
    eigs[:n//2] = 0
    P = (V * eigs) @ V.T
    P = 0.5 * (P + P.T) # symmetrize just to be sure
    # Px = 0
    x = V[:,0] # certificate

    A = np.random.randn(m, n)
    # A := A - (s + Ax)x' / x'x ===> Ax + s == 0
    A = A - np.outer(s + A.dot(x), x) / np.linalg.norm(x)**2
    c = np.random.randn(n)
    c = -c / np.dot(c, x)

    # make sure a point is feasible
    _x = np.random.randn(n)
    _s = pos(np.random.randn(m))
    # A _x + _s == b
    b = A.dot(_x) + _s

    return (P, A, b, c)


class _Problem(dict):
    """dot.notation access to dictionary attributes"""

    def __init__(self):
      self._cvxpy_problem = None

    @property
    def cvxpy_problem(self):
      if self._cvxpy_problem is None:
        self._cvxpy_problem = self._generate_cvxpy_problem()
      return self._cvxpy_problem


    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        u = np.copy(self.u)
        u[u == np.inf] = 1e9
        l = np.copy(self.l)
        l[l == -np.inf] = -1e9
        x_var = cvxpy.Variable(self.n)
        objective = self.q * x_var + .5 * cvxpy.quad_form(x_var, self.P)
        constraints = [self.A * x_var <= u, self.A * x_var >= l]
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        return problem

class RandomProbRunner(object):
    '''
    Examples runner
    '''
    def __init__(self,
                 m, n,
                 seed,
                 num_probs,
                 solvers,
                 settings,
                 output_folder,
                 infeasible,
                 unbounded):
        self.solvers = solvers
        self.settings = settings
        self.output_folder = output_folder
        self.infeasible = infeasible
        self.unbounded = unbounded
        self.seed = seed
        self.num_probs = num_probs
        self.shape = (m,n)

    def problems(self):
      for prob in range(self.num_probs):
        seed = self.seed + prob
        np.random.seed(seed)
        name = f'prob={prob}:seed={seed}:shape={self.shape}'
        if self.infeasible:
          yield 'INFEASIBLE:' + name, gen_infeasible(*self.shape)
        elif self.unbounded:
          yield 'UNBOUNDED:' + name, gen_unbounded(*self.shape)
        else:
          yield 'FEASIBLE:' + name, gen_feasible(*self.shape)

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

        print("Solving random QP problems")
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
                if parallel:
                    results = pool.starmap(self.solve_single_example,
                                           zip(self.problems,
                                               repeat(solver),
                                               repeat(settings)))
                else:
                    results = []
                    for name, problem in self.problems():
                        results.append(self.solve_single_example(name,
                                                                 problem,
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
                             prob_name,
                             problem,
                             solver, settings):
        '''
        Solve NETLIB 'problem' with 'solver'

        Args:
            dimension: problem leading dimension
            instance_number: number of the instance
            solver: solver name
            settings: settings dictionary for the solver

        '''
        print(" - Solving %s with solver %s" % (prob_name, solver))

        # Create example instance
        instance = _Problem()
        instance.qp_problem = {}
        instance.qp_problem['P'] = scipy.sparse.csc_matrix(problem[0])
        instance.qp_problem['A'] = scipy.sparse.csc_matrix(problem[1])
        instance.qp_problem['q'] = problem[3]
        instance.qp_problem['u'] = problem[2]
        instance.qp_problem['l'] = -np.inf * np.ones(self.shape[0])
        instance.qp_problem['m'] = self.shape[0]
        instance.qp_problem['n'] = self.shape[1]

        instance.name = prob_name
        instance.P = instance.qp_problem['P']
        instance.A = instance.qp_problem['A']
        instance.q = instance.qp_problem['q']
        instance.l = instance.qp_problem['l']
        instance.u = instance.qp_problem['u']
        instance.m = instance.qp_problem['m']
        instance.n = instance.qp_problem['n']

        # Solve problem
        s = SOLVER_MAP[solver](settings)
        results = s.solve(instance)

        # Create solution as pandas table
        P = instance.qp_problem['P']
        A = instance.qp_problem['A']
        N = P.nnz + A.nnz

        # Add constant part to objective value
        obj = results.obj_val
        solution_dict = {'name': [prob_name],
                         'solver': [solver],
                         'status': [results.status],
                         'run_time': [results.run_time],
                         'iter': [results.niter],
                         'obj_val': [obj],
                         'n': [self.shape[1]],
                         'm': [self.shape[0]],
                         'N': [N]}

        # Add status polish if OSQP
        if solver[:4] == 'OSQP':
            solution_dict['status_polish'] = results.status_polish
            solution_dict['setup_time'] = results.setup_time
            solution_dict['solve_time'] = results.solve_time
            solution_dict['update_time'] = results.update_time
            solution_dict['rho_updates'] = results.rho_updates

        print(" - Solved %s with solver %s" % (prob_name, solver))

        # Return solution
        return pd.DataFrame(solution_dict)
