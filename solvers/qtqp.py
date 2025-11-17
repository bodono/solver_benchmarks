import sys
sys.path.append("..") # Adds higher directory to python modules path.

import qtqp 
import scipy.sparse
from . import statuses
from .results import Results
from utils.general import is_qp_solution_optimal
from utils.general import is_cone_solution_optimal
import time
import numpy as np
import os
import pandas as pd

class QTQPSolver(object):

    STATUS_MAP = {'solved': statuses.OPTIMAL,
                  'infeasible': statuses.PRIMAL_INFEASIBLE,
                  'unbounded': statuses.DUAL_INFEASIBLE,
                  'failed': statuses.MAX_ITER_REACHED, # unbounded inaccurate
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
          print(f"{np.max(np.abs(b))=}, {np.max(np.abs(a.data))=}, {np.max(np.abs(c))=}")
          if p.data.size > 0:
            print(f"{np.max(np.abs(p.data))}")  

        else:
          raise ValueError('Unrecognized problem type')

        start = time.time()
        solver = qtqp.QTQP(a=a, b=b, c=c, z=z, p=p)
        try:
          solution = solver.solve(**settings)
          x, y, s, results, status = solution.x, solution.y, solution.s, solution.stats, solution.status.value
        except Exception as e:
          print('Failed')
          print(e)
          status = 'failed'
          results = None
        end = time.time()
        if status != 'failed':
          results_df = pd.DataFrame(results)
        else:
          x = y = s = np.nan
          results_df = None

        if status == "solved":
          print(f'{c @ x + 0.5 * x @ p @ x=:.3e}')
          print(f'{-b @ y - 0.5 * x @ p @ x=:.3e}')
          print(f'{np.min(y[z:], initial=0)=:.3e}')
          print(f'{np.min(s[z:], initial=0)=:.3e}')
          print(f'{np.linalg.norm(a @ x + s - b, np.inf)=:.3e}')
          print(f'{np.linalg.norm(p @ x + a.T @ y + c, np.inf)=:.3e}')
          print(f'{c @ x + x @ p @ x + b @ y=:.3e}')
          print(f'{s @ y=:.3e}')
          print(f'{np.max(s * y)=:.3e}')

        if status == "infeasible":
          print(f'{-b @ y=:.3e}')
          print(f'{np.min(y[z:], initial=0)=:.3e}')
          print(f'{np.linalg.norm(a.T @ y, np.inf)=:.3e}')

        if status == "unbounded":
          print(f'{c @ x=:.3e}')
          print(f'{np.min(s[z:], initial=0)=:.3e}')
          print(f'{np.linalg.norm(a @ x + s, np.inf)=:.3e}')
          print(f'{np.linalg.norm(p @ x, np.inf)=:.3e}')



        bstatus = self.STATUS_MAP.get(status, statuses.SOLVER_ERROR)
        #run_time = 1e-3 * (results['info']['solve_time']
        #                  + results['info']['setup_time'])
        run_time = end - start
        pcost = results_df.tail(1).pcost.values[0] if results_df is not None else np.nan
        it = results_df.tail(1).iter.values[0] if results_df is not None else np.nan
        return_results = Results(bstatus,
                                 pcost,
                                 x,
                                 y,
                                 run_time,
                                 it)
        if results is not None and not np.isnan(it):
          return_results.info = results[it]
        else:
          return_results.info = {}


        return return_results
