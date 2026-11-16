# Accurate and memory-lean, but centers by subtracting in place, corrupting
# the caller's arrays.
import numpy as np


def pairwise_sq_dists(A, B):
    mu = (A.sum(axis=0) + B.sum(axis=0)) / (len(A) + len(B))
    A -= mu
    B -= mu
    out = A @ B.T
    out *= -2.0
    out += np.einsum("ij,ij->i", A, A)[:, None]
    out += np.einsum("ij,ij->i", B, B)[None, :]
    np.maximum(out, 0.0, out=out)
    return out
