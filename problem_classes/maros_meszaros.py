import numpy as np
import scipy.io as spio

# The archived .mat files use +-1e20 as their infinity sentinel, in a few
# files stored with ULPs of representation error (e.g. -9.99...e+19).
# Anything at or beyond this magnitude on the correct side is normalized
# to a true infinity (same 1e19 cut as transforms/cones.py).
_INF_SENTINEL = 1.0e19


class MarosMeszaros:
    '''
    Maros Meszaros
    '''
    def __init__(self, file_name, prob_name):
        '''
        Generate Maros problem in QP format.
        '''
        # Load problem from file
        self.P, self.q, self.r, self.A, self.l, self.u, self.n, self.m = \
            self._load_maros_meszaros_problem(file_name)

        self.qp_problem = self._generate_qp_problem()
        self.prob_name = prob_name

    @staticmethod
    def _load_maros_meszaros_problem(f):
        # Load file
        m = spio.loadmat(f)

        # Convert matrices
        P = m['P'].astype(float).tocsc()
        q = m['q'].T.flatten().astype(float)
        r = m['r'].T.flatten().astype(float)[0]
        A = m['A'].astype(float).tocsc()
        l = m['l'].T.flatten().astype(float)
        u = m['u'].T.flatten().astype(float)
        l = np.where(l <= -_INF_SENTINEL, -np.inf, l)
        u = np.where(u >= _INF_SENTINEL, np.inf, u)
        n = m['n'].T.flatten().astype(int)[0]
        m = m['m'].T.flatten().astype(int)[0]

        return P, q, r, A, l, u, n, m

    @staticmethod
    def name():
        return 'Maros Meszaros'

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
