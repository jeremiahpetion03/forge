# Memory-efficient expansion with clamping, but no centering: never negative,
# yet badly inaccurate far from the origin due to catastrophic cancellation.
import numpy as np


def pairwise_sq_dists(A, B):
    out = A @ B.T
    out *= -2.0
    out += np.einsum("ij,ij->i", A, A)[:, None]
    out += np.einsum("ij,ij->i", B, B)[None, :]
    np.maximum(out, 0.0, out=out)
    return out
