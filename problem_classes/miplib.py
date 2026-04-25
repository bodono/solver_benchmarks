import scipy.sparse
import problem_classes.qpsreader


class MIPLIB(object):
    '''
    MIPLIB
    '''
    def __init__(self, file_name, prob_name):
        '''
        Generate MIPLIB root-node LP relaxation in QP format.
        '''
        # Load problem from file
        self._load_miplib_problem(file_name)

        self.qp_problem = self._generate_qp_problem()
        self.prob_name = prob_name


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
