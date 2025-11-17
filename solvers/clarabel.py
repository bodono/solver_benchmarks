import sys
sys.path.append("..") # Adds higher directory to python modules path.

import clarabel 
import scipy.sparse
from . import statuses
from .results import Results
from utils.general import is_qp_solution_optimal
from utils.general import is_cone_solution_optimal
import time
import numpy as np
import os
import pandas as pd

class ClarabelSolver(object):

    STATUS_MAP = {'Solved': statuses.OPTIMAL,
                  'PrimalInfeasible': statuses.PRIMAL_INFEASIBLE,
                  'DualInfeasible': statuses.DUAL_INFEASIBLE,
                  'Failed': statuses.MAX_ITER_REACHED, # unbounded inaccurate
                  }

    def __init__(self, settings={}):
        '''
        Initialize solver object by setting require settings
        '''
        self._settings = settings

    @property
    def settings(self):
        """Solver settings"""
        return self._settings

    def solve(self, example):
        '''
        Solve problem

        Args:
            problem: problem structure with QP matrices

        Returns:
            Results structure
        '''

        settings = self._settings.copy()
        high_accuracy = settings.pop('high_accuracy', None)
        settings.pop('solver')
        
        if hasattr(example, 'qp_problem'):
          problem = example.qp_problem
          A = problem['A']
          (m, n) = problem['A'].shape
          # Hack out the equality constraints
          idxs = (problem['u'] - problem['l'] < 1e-6)
          idxs &= (problem['u'] < 1e20)
          idxs &= (problem['l'] > -1e20)
          o_idxs = np.array(range(A.shape[0])) # no need +1 for box cone
          o_idxs = np.hstack((o_idxs[idxs], o_idxs[~idxs]))
          inv_perm = np.argsort(o_idxs)
    
          if np.all(idxs): # no box cone
            A_scs = A.copy()
            b_scs = problem['u'].copy()
          else:
            bad_u_idxs = (problem['u'] > 1e7)
            bad_l_idxs = (problem['l'] < -1e7)
            A_scs = scipy.sparse.vstack((A[idxs, :], A[~idxs & ~bad_u_idxs, :], -A[~idxs & ~bad_l_idxs, :]))
            u = problem['u']
            l = problem['l']
            b_scs = np.hstack((u[idxs], u[~idxs & ~bad_u_idxs], -l[~idxs & ~bad_l_idxs]))
            
          p=scipy.sparse.csc_matrix(problem['P'])
          c=problem['q']
          a=scipy.sparse.csc_matrix(A_scs)
          b=b_scs
          z=int(np.sum(idxs))

        else:
          raise ValueError('Unrecognized problem type')

        m = a.shape[0]
        cones = [clarabel.ZeroConeT(z), clarabel.NonnegativeConeT(m - z)]
        settings = clarabel.DefaultSettings()
        settings.equilibrate_enable = True 
        settings.presolve_enable = False
        #settings.iterative_refinement_abstol = 1e-13
        #settings.iterative_refinement_reltol = 1e-13
        # settings.iterative_refinement_max_iter = 10
        settings.dynamic_regularization_enable = False
        settings.max_iter = 100
        start = time.time()
        solver = clarabel.DefaultSolver(p, c, a, b, cones, settings)
        try:
          solution = solver.solve()
          x = np.array(solution.x)
          s = np.array(solution.s)
          y = np.array(solution.z)
          status = solution.status
        except Exception as e:
          print('Failed')
          print(e)
          status = 'failed'
        end = time.time()
        
        #run_time = 1e-3 * (results['info']['solve_time']
        #                  + results['info']['setup_time'])
        status = self.STATUS_MAP.get(str(solution.status), statuses.SOLVER_ERROR)
        if str(solution.status) == "Solved" or str(solution.status) == "AlmostSolved":
          print(f'{c @ x + 0.5 * x @ p @ x=:.3e}')
          print(f'{-b @ y - 0.5 * x @ p @ x=:.3e}')
          print(f'{np.min(y[z:], initial=0)=:.3e}')
          print(f'{np.min(s[z:], initial=0)=:.3e}')
          print(f'{np.linalg.norm(a @ x + s - b, np.inf)=:.3e}')
          print(f'{np.linalg.norm(p @ x + a.T @ y + c, np.inf)=:.3e}')
          print(f'{c @ x + x @ p @ x + b @ y=:.3e}')
          print(f'{s @ y=:.3e}')

        if str(solution.status) == "PrimalInfeasible":
          print(f'{-b @ y=:.3e}')
          print(f'{np.min(y[z:], initial=0)=:.3e}')
          print(f'{np.linalg.norm(a.T @ y, np.inf)=:.3e}')

        if str(solution.status) == "DualInfeasible":
          print(f'{c @ x=:.3e}')
          print(f'{np.min(s[z:], initial=0)=:.3e}')
          print(f'{np.linalg.norm(a @ x + s, np.inf)=:.3e}')
          print(f'{np.linalg.norm(p @ x, np.inf)=:.3e}')

        run_time = end - start
        print(f"{run_time=}")
        pcost = solution.obj_val 
        it = solution.iterations
        return_results = Results(status,
                                 pcost,
                                 x,
                                 y,
                                 run_time,
                                 it)
        
        return_results.info = {'pres': solution.r_prim, 'dres': solution.r_dual}
        return return_results
