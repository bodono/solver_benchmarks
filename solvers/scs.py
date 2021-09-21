import scs
import scipy.sparse
from . import statuses
from .results import Results
from utils.general import is_qp_solution_optimal
from utils.general import is_cone_solution_optimal
import time
import numpy as np
import os

class SCSSolver(object):

    STATUS_MAP = {1: statuses.OPTIMAL,
                  2: statuses.MAX_ITER_REACHED,
                  -2: statuses.PRIMAL_INFEASIBLE,
                  -1: statuses.DUAL_INFEASIBLE,
                  -6: statuses.MAX_ITER_REACHED, # unbounded inaccurate
                  -7: statuses.MAX_ITER_REACHED, # infeasible inaccurate
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
        
        if 'log_csv_filename' in settings:
            if settings['log_csv_filename'] is not None:
                settings['log_csv_filename'] = os.path.join(
                    settings['log_csv_filename'], example.prob_name)
        
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
            A_scs = scipy.sparse.vstack((A[idxs, :], np.zeros((1, n)), -A[~idxs, :]))
            b_scs = np.hstack((problem['u'][idxs], 1, np.zeros(m - np.sum(idxs))))

          data = dict(P=scipy.sparse.csc_matrix(problem['P']), c=problem['q'],
                      A=scipy.sparse.csc_matrix(A_scs), b=b_scs)
          cone = dict(z=np.int(np.sum(idxs)), bl=problem['l'][~idxs].tolist(),
                      bu=problem['u'][~idxs].tolist())

        elif hasattr(example, 'sdp_problem'):
          problem = example.sdp_problem
          cone = problem['cone']
          A = problem['A']
          (m, n) = problem['A'].shape
          data = dict(P=scipy.sparse.csc_matrix((n,n)),
                      A=problem['A'], b=problem['b'], c=problem['q'])
        else:
          raise ValueError('Unrecognized problem type')

        start = time.time()
        results = scs.solve(data, cone, **settings)
        end = time.time()

        status = self.STATUS_MAP.get(results['info']['status_val'], statuses.SOLVER_ERROR)
        if hasattr(example, 'qp_problem'):
          def _inv(y):
            if len(y) == cone['z']:
                return y
            y = y.copy()
            y[cone['z']:] *= -1.
            y = np.delete(y, cone['z']) # remove perspective var from y
            y = y[inv_perm]
            return y

          y = _inv(results['y'])
          s = _inv(results['s']) # just for debugging
          if status in statuses.SOLUTION_PRESENT:
            qp_optimal = is_qp_solution_optimal(problem,
                                          results['x'],
                                          y,
                                          high_accuracy=high_accuracy)
            cone_optimal = is_cone_solution_optimal(data, cone, results['x'],
                                                    results['y'], results['s'],
                                                    high_accuracy=high_accuracy)
            if (not qp_optimal and not cone_optimal):
              status = statuses.SOLVER_ERROR

        # Verify solver time
        if settings.get('time_limit') is not None:
            pass

        #run_time = 1e-3 * (results['info']['solve_time']
        #                  + results['info']['setup_time'])
        run_time = end - start
        return_results = Results(status,
                                 results['info']['pobj'],
                                 results['x'],
                                 -results['y'][1:],
                                 run_time,
                                 results['info']['iter'])

        return_results.setup_time = results['info']['setup_time']
        return_results.solve_time = results['info']['solve_time']

        return_results.info = results['info']

        return return_results
