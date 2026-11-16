import numpy as np


def log_softmax(x, axis=-1):
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)  # finite: every slice has a finite entry
    z = x - m  # largest entry becomes 0, so exp never overflows
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))
