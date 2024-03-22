#!/usr/bin/env python
import time
import numpy as np
import numba as nb

def mydot(a, v):
    w = np.zeros(v.shape)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            w[i] += a[i][j] * v[j]
    return w


@nb.njit("float64[:](float64[:,:], float64[:])", parallel=True)
def mydot2(a, v):
    w = np.zeros(v.shape)
    for i in nb.prange(a.shape[0]):
        for j in range(a.shape[1]):
            w[i] += a[i][j] * v[j]
    return w

if __name__ == "__main__":
    N = 5000
    v = np.random.rand(N).astype(np.float64)
    A = np.random.rand(N, N).astype(np.float64)

    t0 = time.time()
    w1 = mydot(A, v)
    print("Python dot: %f s" % (time.time() - t0))

    t0 = time.time()
    w2 = A.dot(v)
    print("Numpy dot: %f s" % (time.time() - t0))

    if not np.allclose(w1, w2):
        raise ValueError("python dot and numpy dot do not match.")

    t0 = time.time()
    w3 = mydot2(A, v)
    print("Numba dot: %f s" % (time.time() - t0))

    if not np.allclose(w1, w3):
        raise ValueError("python dot and numba dot do not match.")
