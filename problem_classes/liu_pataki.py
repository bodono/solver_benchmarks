import cvxpy
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.io
import scs


class LIU_PATAKI(object):
    def __init__(self, file_name):
        '''
        Generate problem using cvxpy
        '''
        # Load problem from file
        self._load_liu_pataki_problem(file_name)

        self.sdp_problem = self._generate_sdp_problem()
        self._cvxpy_problem = None


    @property
    def cvxpy_problem(self):
      if self._cvxpy_problem is None:
        self._cvxpy_problem = self._generate_cvxpy_problem()
      return self._cvxpy_problem


    def _load_liu_pataki_problem(self, filename, verbose=False):
      d = scipy.io.loadmat(filename)
      c = -d['b'].astype(np.float)
      b = d['c'].astype(np.float)
      K = d['K']

      if type(c) is not np.ndarray:
        c = np.squeeze(c.toarray())
      else:
        c = np.squeeze(c)

      if type(b) is not np.ndarray:
        b = np.squeeze(b.toarray())
      else:
        b = np.squeeze(b)

      try:
        _A = d['A'].T
      except:
        _A = d['At']
      #_A.indptr = _A.indices.astype(np.int)
      #_A.indices = _A.indices.astype(np.int)
      _A.data = _A.data.astype(np.float)

      if len(b.shape) == 0:
        b = np.zeros(_A.shape[0])
      if len(c.shape) == 0:
        c = np.zeros(_A.shape[1])

      K_names = K.dtype.names
      if 'r' in K_names:
        if np.sum(K['r'][0][0]) > 0:
          raise ValueError('cannot handle rotated Lorentz cones')

      curr_idx = 0
      to_drop = []
      new_free_idxs = []
      cone = dict()
      if 'f' in K_names:
        cone['f'] = np.sum(K['f'][0][0])
        curr_idx += int(cone['f'])
      if 'q' in K_names:
        cone['q'] = np.squeeze(K['q'][0][0].astype(np.int)).tolist()
        curr_idx += int(np.sum(cone['q']))
        if type(cone['q']) is int:
          cone['q'] = [cone['q']]
      if 'l' in K_names:
        cone['l'] = int(np.sum(K['l'][0][0]))
        curr_idx += cone['l']
      if 's' in K_names:
        cone['s'] = np.squeeze(K['s'][0][0].astype(np.int)).tolist()
        if type(cone['s']) is int:
          cone['s'] = [cone['s']]
        for s in cone['s']:
          #import pdb;pdb.set_trace()
          M = np.arange(s * s).reshape(s, s).T
          M_diags = np.diag(M)
          idxs = np.triu_indices(s, 1)
          add_from_idxs = curr_idx + M[idxs]
          add_to_idxs = curr_idx + M[idxs[1], idxs[0]]
          diag_idxs = curr_idx + M_diags
          #_A[add_to_idxs, :] += _A[add_from_idxs, :]
          #_A[add_to_idxs, :] /= 2.
          _A[diag_idxs, :] /= np.sqrt(2.)
          #b[add_to_idxs] += b[add_from_idxs]
          #b[add_to_idxs] /= 2.
          b[diag_idxs] /= np.sqrt(2.)
          to_drop += add_from_idxs.tolist()
          new_free_idxs += add_to_idxs.tolist()
          curr_idx += s * s

      A_free = _A[new_free_idxs, :] - _A[to_drop, :]
      b_free = b[new_free_idxs] - b[to_drop]

      if np.linalg.norm(b_free) > 0 or scipy.sparse.linalg.norm(A_free) > 0:
        raise ValueError('handle this case')

      #if 'f' in cone:
      #  cone['f'] += len(b_free)
      #else:
      #  cone['f'] = len(b_free)

      rows_to_keep = [r for r in range(_A.shape[0]) if r not in to_drop]
      #A = scipy.sparse.csc_matrix(np.vstack((A_free, _A[rows_to_keep, :])))
      #b = np.vstack((b_free, b[rows_to_keep]))
      A = scipy.sparse.csc_matrix(_A[rows_to_keep, :])
      b = b[rows_to_keep]

      A = A.sorted_indices()

      data = dict(A=A, b=b, c=c)

      #scs.solve(data, cone, verbose=True, eps_abs=1e-7, eps_rel=1e-7,
      #    max_iters=int(1e6))

      self.cone = cone
      self.A = A
      self.P = None
      self.r = 0.
      self.n, self.m = self.A.shape
      self.b = b
      self.q = c
      self.obj_type = 'min'

    @staticmethod
    def name():
        return 'LIU_PATAKI'

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

