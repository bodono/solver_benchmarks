import numpy as np
from scipy import sparse
from datetime import datetime
from cylp.py.QP import QPSReader

def readMpsLp(filename):
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

