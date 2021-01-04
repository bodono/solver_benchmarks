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
        problem = example.qp_problem
        settings = self._settings.copy()
        high_accuracy = settings.pop('high_accuracy', None)

        # Solve
        (m,n) = problem["A"].shape
        b = np.zeros((m,))
        b_scs = np.hstack((1, b))
        A_scs = scipy.sparse.vstack((np.zeros((1, n)), -problem["A"]))

        data = dict(P=scipy.sparse.csc_matrix(problem['P']), c=problem['q'],
                    A=scipy.sparse.csc_matrix(A_scs), b=b_scs)
        cone = dict(bl=problem['l'].tolist(), bu=problem['u'].tolist())
        settings["verbose"]=True
        start = time.time()
        results = scs.solve(data, cone, **settings)
        end = time.time()
        status = self.STATUS_MAP.get(results['info']['statusVal'], s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            if not is_qp_solution_optimal(problem,
                                          results['x'],
                                          -results['y'][1:],
                                          high_accuracy=high_accuracy):
                status = s.SOLVER_ERROR

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

        #return_results.status_polish = results.info.status_polish
        #return_results.setup_time = results.info.setup_time
        #return_results.solve_time = results.info.solve_time
        #return_results.update_time = results.info.update_time
        #return_results.rho_updates = results.info.rho_updates

        return return_results
