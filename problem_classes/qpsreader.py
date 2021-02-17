import numpy as np
from scipy import sparse
from cylp.py.QP import QPSReader 


def readMpsLp(filename):
    (P, c, A, b, G, c_low, c_up, x_low, x_up, _, _, _, _) = QPSReader.readQPS(filename)
    assert not P
    (m, n) = A.shape
    l = b.copy()
    u = b.copy()
    if isinstance(c_low, np.ndarray):
        l = np.hstack((l, c_low))
        u = np.hstack((u, c_up))
        A = sparse.vstack((A, G))

    if isinstance(x_low, np.ndarray):
        l = np.hstack((l, x_low))
        u = np.hstack((u, x_up))
        A = sparse.vstack((A, sparse.eye(n, format='dok')))
    return (A, c, l, u)

