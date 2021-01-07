import scs
import scipy.sparse
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal
import time
import numpy as np

class SCSSolver(object):

    STATUS_MAP = {1: s.OPTIMAL,
                  2: s.MAX_ITER_REACHED,
                  -2: s.PRIMAL_INFEASIBLE,
                  -1: s.DUAL_INFEASIBLE,
                  -6: s.DUAL_INFEASIBLE,
                  -7: s.PRIMAL_INFEASIBLE}

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

          # Solve
          (m,n) = problem['A'].shape

          # Hack out the equality constraints
          idxs = (problem['u'] - problem['l'] < 1e-5)
          A_scs = scipy.sparse.vstack((problem['A'][idxs, :],
                                      np.zeros((1, n)),
                                      -problem['A'][~idxs, :]))

          b_scs = np.hstack((problem['u'][idxs],
                            1,
                            np.zeros(m - np.sum(idxs))))

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

        # XXX TODO invert HACK above (on s and y)

        status = self.STATUS_MAP.get(results['info']['statusVal'], s.SOLVER_ERROR)

        #if status in s.SOLUTION_PRESENT:
        #    if not is_qp_solution_optimal(problem,
        #                                  results['x'],
        #                                  -results['y'][1:],
        #                                  high_accuracy=high_accuracy):
        #        status = s.SOLVER_ERROR

        # Verify solver time
        if settings.get('time_limit') is not None:
            if results.info.run_time > settings.get('time_limit'):
                status = s.TIME_LIMIT

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
        #return_results.rho_updates = results['info']['rho_updates']

        return return_results
