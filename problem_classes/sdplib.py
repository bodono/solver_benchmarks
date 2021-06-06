import cvxpy
import h5py
import numpy as np
import scipy.sparse
import scs


class SDPLIB(object):
    '''
    SDPLIB
    '''
    def __init__(self, file_name, prob_name):
        '''
        Generate SDPlib problem using cvxpy
        '''
        # Load problem from file
        self._load_sdplib_problem(file_name)

        self.sdp_problem = self._generate_sdp_problem()
        self._cvxpy_problem = None
        self.prob_name = prob_name


    def _parse_out_f(self, f, i):
      ref = f[f['F'][i]]
      value = np.array(ref).item()
      m = value[0]
      n = value[1]
      colptr = np.array(f[value[2]]) - 1 # julia is 1 indexed
      rowptr = np.array(f[value[3]]) - 1 # julia is 1 indexed
      data = np.array(f[value[4]])
      return scipy.sparse.csc_matrix((data, rowptr, colptr), shape=(m, n))


    def _load_sdplib_problem(self, filename, verbose=False):
      Fs = []
      with h5py.File(filename, "r") as f:
        print('opt objective val:', np.array(f['optVal']))
        m = np.array(f['m']).item()
        n = np.array(f['n']).item()
        c = np.array(f['c'])
        for i in range(m + 1):
          Fs.append(self._parse_out_f(f, i))

      print('cvxpy forming problem')
      x_var = cvxpy.Variable(m)
      objective = c.T @ x_var
      # F0 is not multiplied by var
      constraints = [cvxpy.sum([Fs[i+1] * x_var[i] for i in range(m)]) >> Fs[0]]
      problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
      print('cvxpy parsing scs data')
      prob_data = problem.get_problem_data(solver=cvxpy.SCS)
      print('cvxpy done')

      data = prob_data[0]
      dims = data['dims']
      self.cone = dict(f=dims.zero, s=dims.psd)
      self.A = data['A']
      self.P = None
      self.r = 0.
      self.n, self.m = self.A.shape
      self.b = data['b']
      self.q = data['c']
      self.cvxpy_problem = problem
      self.obj_type = 'min'

    @staticmethod
    def name():
        return 'SDPLIB'

    def _generate_sdp_problem(self):
        '''
        Generate QP problem
        '''
        problem = {}
        problem['P'] = self.P
        problem['r'] = self.r
        problem['q'] = self.q
        problem['A'] = self.A
        problem['b'] = self.b
        problem['n'] = self.n
        problem['m'] = self.m
        problem['cone'] = self.cone

        return problem

