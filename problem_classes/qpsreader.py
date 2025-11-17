import numpy as np
from scipy import sparse
from datetime import datetime
# from cylp.py.QP import QPSReader
import highspy


def _readMpsLp(filename):
    print('Starting from file QPSREADER: ', datetime.now())
    (P, c, A, b, G, c_low, c_up, x_low, x_up, n, _, _, _) = QPSReader.readQPS(filename)
    print('Finished from file QPSREADER: ', datetime.now())
    assert not P

    l = np.array([])
    u = np.array([])


    mats = []
    if isinstance(b, np.ndarray):
        l = b.copy()
        u = b.copy()
        mats.append(A)

    if isinstance(c_low, np.ndarray):
        idxs = (c_low > -1e20) | (c_up < 1e20)
        l = np.hstack((l, c_low[idxs]))
        u = np.hstack((u, c_up[idxs]))
        mats.append(G[idxs, :])

    if isinstance(x_low, np.ndarray):
        idxs = (x_low > -1e20) | (x_up < 1e20)
        l = np.hstack((l, x_low[idxs]))
        u = np.hstack((u, x_up[idxs]))
        mats.append(sparse.eye(n, format='csr')[idxs, :])

    mat = sparse.vstack(mats)
    print('Finished parsing QPSREADER: ', datetime.now())
    return (mat, c, l, u)

def readMpsLp(filename):
    filename = filename.rstrip('.gz')
    h = highspy.Highs()
    print('Starting from file HiGHS: ', datetime.now())
    h.readModel(filename)
    lp = h.getLp()
    A = lp.a_matrix_
    c = np.array(lp.col_cost_)
    n = len(c)
    x_low = np.array(lp.col_lower_)
    x_upper = np.array(lp.col_upper_)
    c_low = np.array(lp.row_lower_)
    c_upper = np.array(lp.row_upper_)
    print('Finished from file HiGHS: ', datetime.now())

    A = sparse.csc_matrix((A.value_, A.index_, A.start_), shape=(A.num_row_, A.num_col_))

    mats = [A]

    idxs = (x_low > -1e20) | (x_upper < 1e20)
    l = np.hstack((c_low, x_low[idxs]))
    u = np.hstack((c_upper, x_upper[idxs]))
    mats.append(sparse.eye(n, format='csr')[idxs, :])

    mat = sparse.vstack(mats)
    print('Finished parsing HiGHS: ', datetime.now())
    return (mat, c, l, u)

