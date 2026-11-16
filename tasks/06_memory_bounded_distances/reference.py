import numpy as np


def pairwise_sq_dists(A, B):
    # Center on a shared mean first: distances are translation-invariant, and
    # it removes the large ||a||^2 + ||b||^2 terms whose cancellation would
    # otherwise destroy precision when points are far from the origin.
    mu = (A.sum(axis=0) + B.sum(axis=0)) / (len(A) + len(B))
    A = A - mu
    B = B - mu
    out = A @ B.T                          # the only N x M allocation
    out *= -2.0                            # every later step works in place
    out += np.einsum("ij,ij->i", A, A)[:, None]
    out += np.einsum("ij,ij->i", B, B)[None, :]
    np.maximum(out, 0.0, out=out)          # rounding can still dip just below 0
    return out
