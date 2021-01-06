import cvxpy
import h5py
import numpy as np
import scipy.sparse

class SDPLIB(object):
    '''
    SDPLIB
    '''
    def __init__(self, file_name):
        '''
        Generate Maros problem in QP format and CVXPY format

        NB. By default, the CVXPY problem is not created
        '''
        # Load problem from file
        self._load_sdplib_problem(file_name)

        self.sdp_problem = self._generate_sdp_problem()
        self._cvxpy_problem = None


    @property
    def cvxpy_problem(self):
      if self._cvxpy_problem is None:
        self._cvxpy_problem = self._generate_cvxpy_problem()
      return self._cvxpy_problem


    def _parse_out_f(self, f, i):
      #import pbd;pdb.set_trace
      ref = f[f['F'][i]]
      value = np.array(ref).item()
      m = value[0]
      n = value[1]
      colptr = np.array(f[value[2]]) - 1 # julia is 1 indexed
      rowptr = np.array(f[value[3]]) - 1 # julia is 1 indexed
      data = np.array(f[value[4]])
      return scipy.sparse.csc_matrix((data, rowptr, colptr), shape=(m, n))


    def _load_sdplib_problem(self, filename, verbose=False):
      f = h5py.File(filename, "r")
      m = np.array(f['m']).item()
      n = np.array(f['n']).item()
      c = np.array(f['c'])
      Fs = [] # F0 is not multiplied by var
      for i in range(m + 1):
        Fs.append(self._parse_out_f(f, i))

      print('opt objective val:', np.array(f['optVal']))
      x_var = cvxpy.Variable(m)
      objective = c.T * x_var
      constraints = [cvxpy.sum([Fs[i+1] * x_var[i] for i in range(m)]) >> Fs[0]]
      problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
      # problem.get_problem_data(solver=cvxpy.SCS)
      problem.solve(verbose=True, solver=cvxpy.SCS, acceleration_interval=10,
                    acceleration_lookback=20,
                    eps_abs=1e-5, eps_rel=1e-5, max_iters=int(1e6),
                    eps_infeas=1e-9)
                    #adaptive_scaling=False, scale=0.1)


    @staticmethod
    def name():
        return 'SDPLIB'

    def _generate_sdplib_problem(self):
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
