import numpy as np
from scipy import sparse
from cylp.py.QP import QPSReader


def readMpsLp(filename):
    (P, c, A, b, G, c_low, c_up, x_low, x_up, n, _, _, _) = QPSReader.readQPS(filename)
    assert not P

    mat = sparse.coo_matrix((0, n))
    l = np.array([])
    u = np.array([])

    if isinstance(b, np.ndarray):
        l = b.copy()
        u = b.copy()
        mat = sparse.vstack((mat, A))

    if isinstance(c_low, np.ndarray):
        l = np.hstack((l, c_low))
        u = np.hstack((u, c_up))
        mat = sparse.vstack((mat, G))

    if isinstance(x_low, np.ndarray):
        l = np.hstack((l, x_low))
        u = np.hstack((u, x_up))
        mat = sparse.vstack((mat, sparse.eye(n, format='dok')))
    return (mat, c, l, u)

