import numpy as np
from scipy import sparse
from cylp.py.QP import QPSReader

def readMpsLp(filename):
    (P, c, A, b, G, c_low, c_up, x_low, x_up, n, _, _, _) = QPSReader.readQPS(filename)

    assert not P

    mat = sparse.dok_matrix((0, n))
    l = np.array([])
    u = np.array([])

    if isinstance(b, np.ndarray):
        l = b.copy()
        u = b.copy()
        mat = sparse.vstack((mat, A))

    if isinstance(c_low, np.ndarray):
        idxs = (c_low > -1e20) | (c_up < 1e20)
        l = np.hstack((l, c_low[idxs]))
        u = np.hstack((u, c_up[idxs]))
        mat = sparse.vstack((mat, G[idxs, :]))

    if isinstance(x_low, np.ndarray):
        idxs = (x_low > -1e20) | (x_up < 1e20)
        l = np.hstack((l, x_low[idxs]))
        u = np.hstack((u, x_up[idxs]))
        mat = sparse.vstack((mat, sparse.eye(n, format='dok')[idxs, :]))

    return (mat, c, l, u)

