import scipy.sparse
from . import statuses as s
from .results import Results
from utils.general import is_qp_solution_optimal
import time
import numpy as np

def scs_2_cosmo(A, b, cone):

  A_cosmo = scipy.sparse.dok_matrix(A.shape)
  b_cosmo = np.zeros_like(b, dtype=np.float64)

  # need to permute the rows in A and b
  seen_rows = 0
  if 'f' in cone:
    seen_rows = int(cone['f'])
  if 'l' in cone:
    seen_rows += int(cone['l'])
  if 'q' in cone:
    seen_rows += int(np.sum(cone['q']))

  if seen_rows > 0:
    A_cosmo[:seen_rows, :] = A[:seen_rows, :]
    b_cosmo[:seen_rows] = b[:seen_rows]

  for s in cone['s']:
    ll = int(s * (s + 1) // 2)
    scs_cols, scs_rows = np.triu_indices(s)
    cosmo_rows, cosmo_cols = np.tril_indices(s)
    scs_mat = np.zeros((s,s), dtype=np.int64)
    scs_mat[scs_rows, scs_cols] = range(ll)
    mapping = scs_mat[cosmo_rows, cosmo_cols]
    A_cosmo[seen_rows:seen_rows + ll, :] = A[seen_rows + mapping, :]
    b_cosmo[seen_rows:seen_rows + ll] = b[seen_rows + mapping]
    seen_rows += ll

  return A_cosmo.tocsc(), b_cosmo

class COSMOSolver(object):
    STATUS_MAP = {'Solved': s.OPTIMAL,
                  'Max_iter_reached' : s.MAX_ITER_REACHED,
                  'Primal_infeasible': s.PRIMAL_INFEASIBLE,
                  'Dual_infeasible': s.DUAL_INFEASIBLE}

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

          A = -problem['A']
          b = np.zeros(m)
          P = problem['P']
          q = problem['q']
          l = problem['l']
          u = problem['u']
          cone = dict(b=l.shape[0])

        elif hasattr(example, 'sdp_problem'):
          problem = example.sdp_problem
          scs_cone = problem['cone']
          data = dict(A=problem['A'], b=problem['b'], c=problem['q'])
          P = None
          u = None
          l = None
          q = problem['q']
          A, b = scs_2_cosmo(problem['A'], problem['b'], scs_cone)
          cone = scs_cone.copy()
          cone['s'] = [int(p*(p+1) / 2) for p in cone['s']]
        else:
          raise ValueError('Unrecognized problem type')

        try:
          model = cosmo.Model()
        except:
          # julia:
          from julia.api import Julia
          jl = Julia(compiled_modules=False)
          import cosmopy as cosmo
          model = cosmo.Model()

        start = time.time()
        model.setup(P=P, q=q, A=A, b=b, u=u, l=l, cone=cone, **settings)
        model.optimize()
        end = time.time()
        status = self.STATUS_MAP.get(model.get_status(), s.SOLVER_ERROR)
        if hasattr(example, 'qp_problem'):
          if status in s.SOLUTION_PRESENT:
            if not is_qp_solution_optimal(problem,
                                          model.get_x(),
                                          -model.get_y(),
                                          high_accuracy=high_accuracy):
              status = s.SOLVER_ERROR


        run_time = end - start # this is poor due to python/julia overhead
        # will have to trust cosmo itself unforunately
        run_time = model.get_times()['solver_time']
        return_results = Results(status,
                                 model.get_objective_value(),
                                 model.get_x(),
                                 model.get_y(),
                                 run_time,
                                 model.get_iter())

        return return_results
