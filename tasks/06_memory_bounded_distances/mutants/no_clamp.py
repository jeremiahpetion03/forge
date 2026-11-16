# Centered and memory-efficient, but forgets that rounding can produce tiny
# negative values (which become NaN after a sqrt downstream).
import numpy as np


def pairwise_sq_dists(A, B):
    mu = (A.sum(axis=0) + B.sum(axis=0)) / (len(A) + len(B))
    A = A - mu
    B = B - mu
    out = A @ B.T
    out *= -2.0
    out += np.einsum("ij,ij->i", A, A)[:, None]
    out += np.einsum("ij,ij->i", B, B)[None, :]
    return out
