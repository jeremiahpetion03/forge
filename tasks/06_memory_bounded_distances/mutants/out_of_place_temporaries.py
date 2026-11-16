# Right formula, centered and clamped, but written as one expression: NumPy
# materializes several N x M temporaries at once and blows the budget.
import numpy as np


def pairwise_sq_dists(A, B):
    mu = (A.sum(axis=0) + B.sum(axis=0)) / (len(A) + len(B))
    A = A - mu
    B = B - mu
    a2 = (A * A).sum(axis=1)[:, None] + np.zeros((1, len(B)))
    b2 = (B * B).sum(axis=1)[None, :] + np.zeros((len(A), 1))
    return np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)
