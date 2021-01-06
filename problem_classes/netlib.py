import cvxpy
from pysmps import smps_loader as mps
import numpy as np
import scipy.sparse

class NETLIB(object):
    '''
    NETLIB
    '''
    def __init__(self, file_name):
        '''
        Generate Maros problem in QP format and CVXPY format

        NB. By default, the CVXPY problem is not created
        '''
        # Load problem from file
        self._load_netlib_problem(file_name)

        self.qp_problem = self._generate_qp_problem()
        self._cvxpy_problem = None


    @property
    def cvxpy_problem(self):
      if self._cvxpy_problem is None:
        self._cvxpy_problem = self._generate_cvxpy_problem()
      return self._cvxpy_problem

    # min  q'x
    # s.t. l <= Ax <= u
    def _load_netlib_problem(self, filename, verbose=False):
      data = mps.load_mps(filename)
      if len(data["rhs_names"]) > 1:
        raise ValueError("more than one rhs")
      if len(data["bnd_names"]) > 1:
        raise ValueError("more than one bnd")

      A_mps = data["A"]
      c = data["c"]
      (m, n) = A_mps.shape
      types = np.array(data["types"])
      if not data["rhs"]: # if RHS totally missing, assume zeros
        b_mps = np.zeros(m)
      else:
        b_mps = data["rhs"][data["rhs_names"][0]]
      if not data["bnd_names"]: # if BOUNDS totally missing don't set them
        bounds = None
      else:
        bounds = data["bnd"][data["bnd_names"][0]]

      A_l = A_mps[types == "G",:]
      A_u = A_mps[types == "L",:]

      # check if A_l and A_u are equal
      if A_l.shape == A_u.shape and len((A_l != A_u).data) == 0:
        A_box = -A_u
        l = b_mps[types == "G"]
        u = b_mps[types == "L"]
      else:
        A_box = scipy.sparse.vstack((A_l, A_u))
        l = np.hstack((b_mps[types == "G"], -np.inf*np.ones(sum(types == "L"))))
        u = np.hstack((np.inf*np.ones(sum(types == "G")), b_mps[types == "L"]))

      # variable bounds vl <= x <= vu
      if bounds:
        vl = bounds['LO']
        vu = bounds['UP']

        assert np.squeeze(vl).shape[0] == n
        assert np.squeeze(vu).shape[0] == n

      else:
        #idxs = []
        vl = np.zeros(n)
        vu = np.inf * np.ones(n)

      l = np.hstack((vl, l))
      u = np.hstack((vu, u))

      A = scipy.sparse.vstack((scipy.sparse.eye(n, format='dok'), A_box))
      A = scipy.sparse.vstack((A_mps[types == "E", :],
                               A))
      # OSQP stack equality b on top
      l = np.hstack((b_mps[types == "E"], l))
      u = np.hstack((b_mps[types == "E"], u))

      # Assign final values to problem
      self.m, self.n = A.shape
      self.l = l
      self.u = u
      self.A = scipy.sparse.csc_matrix(A)
      self.P = scipy.sparse.csc_matrix((self.n, self.n))
      self.q = c
      self.r = 0.
      self.obj_type = 'min'
      #if self.obj_type == 'max':
      #    self.P *= -1
      #    self.q *= -1
      #    self.r *= -1

    @staticmethod
    def name():
        return 'NETLIB'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''
        problem = {}
        problem['P'] = self.P
        problem['q'] = self.q
        problem['r'] = self.r
        problem['A'] = self.A
        problem['l'] = self.l
        problem['u'] = self.u
        problem['n'] = self.n
        problem['m'] = self.m

        return problem


    # XXX this method might be wrong:
    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        u = np.copy(self.u)
        u[u == np.inf] = 1e9
        l = np.copy(self.l)
        l[l == -np.inf] = -1e9
        x_var = cvxpy.Variable(self.n)
        objective = self.q * x_var + self.r
        constraints = [self.A * x_var <= u, self.A * x_var >= l]
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        return problem

    # XXX this method might be wrong:
    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = variables[0].value

        # dual solution
        y = None
        if constraints[0].dual_value is not None:
          y = constraints[0].dual_value - constraints[1].dual_value

        return x, y
