import scs
import scipy.sparse
from . import statuses
from .results import Results
from utils.general import is_qp_solution_optimal
from utils.general import is_cone_solution_optimal
import time
import numpy as np

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
        if hasattr(example, 'qp_problem'):
          problem = example.qp_problem
          A = problem['A']
          (m, n) = problem['A'].shape
          # Hack out the equality constraints
          idxs = (problem['u'] - problem['l'] < 1e-6)
          o_idxs = np.array(range(A.shape[0])) # no need +1 for box cone
          o_idxs = np.hstack((o_idxs[idxs], o_idxs[~idxs]))
          inv_perm = np.argsort(o_idxs)

          A_scs = scipy.sparse.vstack((A[idxs, :], np.zeros((1, n)), -A[~idxs, :]))
          b_scs = np.hstack((problem['u'][idxs], 1, np.zeros(m - np.sum(idxs))))

          data = dict(P=scipy.sparse.csc_matrix(problem['P']), c=problem['q'],
                      A=scipy.sparse.csc_matrix(A_scs), b=b_scs)
          cone = dict(f=np.int(np.sum(idxs)), bl=problem['l'][~idxs].tolist(),
                      bu=problem['u'][~idxs].tolist())

        elif hasattr(example, 'sdp_problem'):
          problem = example.sdp_problem
          cone = problem['cone']
          data = dict(A=problem['A'], b=problem['b'], c=problem['q'])
        else:
          raise ValueError('Unrecognized problem type')

        start = time.time()
        results = scs.solve(data, cone, **settings)
        end = time.time()

        status = self.STATUS_MAP.get(results['info']['statusVal'], statuses.SOLVER_ERROR)
        if hasattr(example, 'qp_problem'):
          def _inv(y):
            y = y.copy()
            y[cone['f']:] *= -1.
            y = np.delete(y, cone['f']) # remove perspective var from y
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
            if results.info.run_time > settings.get('time_limit'):
                status = statuses.TIME_LIMIT

        #run_time = 1e-3 * (results['info']['solveTime']
        #                  + results['info']['setupTime'])
        run_time = end - start
        return_results = Results(status,
                                 results['info']['pobj'],
                                 results['x'],
                                 -results['y'][1:],
                                 run_time,
                                 results['info']['iter'])

        return_results.setup_time = results['info']['setupTime']
        return_results.solve_time = results['info']['solveTime']
        # TODO XXX add this to SCS
        #return_results.update_time = results['info']['coneTime']
        #return_results.update_time = results['info']['linSysTime']
        #return_results.update_time = results['info']['accelTime']
        #return_results.rho_updates = results['info']['scale_updates']

        return return_results
