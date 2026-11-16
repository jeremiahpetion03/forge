# Numerically fine, but silently converts float32 to float64, doubling memory
# for every downstream tensor.
import numpy as np


def log_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    z = x - np.max(x, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))
