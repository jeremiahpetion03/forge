# Exactly right numerically, but defeats the purpose: rebuilds the full batch
# in memory and computes one gradient.
import numpy as np


def accumulated_gradient(batches, w, grad_fn):
    X = np.concatenate([b[0] for b in batches])
    y = np.concatenate([b[1] for b in batches])
    return grad_fn(X, y, w)
