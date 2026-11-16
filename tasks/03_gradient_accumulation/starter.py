import numpy as np


def accumulated_gradient(batches, w, grad_fn):
    grads = [grad_fn(X, y, w) for X, y in batches]
    return np.mean(grads, axis=0)
