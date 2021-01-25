import scs
import scipy.sparse
from . import statuses
from .results import Results
from utils.general import is_qp_solution_optimal
from utils.general import is_cone_solution_optimal
import time
import numpy as np
import qpalm as qp

class QPALMSolver(object):

    STATUS_MAP = {1: statuses.OPTIMAL,
                  -2: statuses.MAX_ITER_REACHED,
                  -3: statuses.PRIMAL_INFEASIBLE,
                  -4: statuses.DUAL_INFEASIBLE,
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
        problem = example.qp_problem
        high_accuracy = settings.pop('high_accuracy', None)

        (m, n) = problem['A'].shape

        start = time.time()
        solver = qp.Qpalm()
        solver._settings.contents.eps_abs = settings['eps_abs']
        solver._settings.contents.eps_rel = settings['eps_rel']
        solver._settings.contents.eps_prim_inf = settings['eps_prim_inf']
        solver._settings.contents.eps_dual_inf = settings['eps_dual_inf']
        solver.set_data(Q=problem['P'], A=problem['A'], q=problem['q'], bmin=problem['l'], bmax=problem['u'])
        solver._solve()
        end = time.time()
        status = solver._work.contents.info.contents.status_val
        status = self.STATUS_MAP.get(status, statuses.SOLVER_ERROR)

        x = solver._work.contents.solution.contents.x
        x = np.ctypeslib.as_array(x, shape=(n,))

        y = solver._work.contents.solution.contents.y
        y = np.ctypeslib.as_array(y, shape=(m,))

        iters = solver._work.contents.info.contents.iter
        pobj = solver._work.contents.info.contents.objective
        if status in statuses.SOLUTION_PRESENT:
          qp_optimal = is_qp_solution_optimal(problem,
                                            x,
                                            y,
                                            high_accuracy=high_accuracy)
          if not qp_optimal:
            status = statuses.SOLVER_ERROR

        # Verify solver time
        #if settings.get('time_limit') is not None:
        #    if results.info.run_time > settings.get('time_limit'):
        #        status = statuses.TIME_LIMIT

        #run_time = 1e-3 * (results['info']['solveTime']
        #                  + results['info']['setupTime'])
        run_time = end - start
        return_results = Results(status,
                                 pobj,
                                 x,
                                 y,
                                 run_time,
                                 iters)


        return return_results
