import numpy as np


def log_softmax(x, axis=-1):
    e = np.exp(x)
    return np.log(e / e.sum(axis=axis, keepdims=True))
