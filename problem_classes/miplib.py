import cvxpy
import numpy as np
import scipy.sparse
import problem_classes.qpsreader


class MIPLIB(object):
    '''
    MIPLIB
    '''
    def __init__(self, file_name, prob_name):
        '''
        Generate Maros problem in QP format and CVXPY format

        NB. By default, the CVXPY problem is not created
        '''
        # Load problem from file
        self._load_miplib_problem(file_name)

        self.qp_problem = self._generate_qp_problem()
        self._cvxpy_problem = None
        self.prob_name = prob_name


    @property
    def cvxpy_problem(self):
      if self._cvxpy_problem is None:
        self._cvxpy_problem = self._generate_cvxpy_problem()
      return self._cvxpy_problem

    # min  q'x
    # s.t. l <= Ax <= u
    def _load_miplib_problem(self, filename, verbose=False):
      (A, c, l, u) = problem_classes.qpsreader.readMpsLp(filename)
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
        return 'MIPLIB'

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
