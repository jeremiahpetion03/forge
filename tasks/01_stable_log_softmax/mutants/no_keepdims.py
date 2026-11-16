# Classic broadcasting slip: reductions without keepdims only line up by
# accident (square inputs, axis=0), otherwise misalign or crash.
import numpy as np


def log_softmax(x, axis=-1):
    m = np.max(x, axis=axis)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis))
