import scipy.sparse as spa
import scipy.io as spio


class MarosMeszaros(object):
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
