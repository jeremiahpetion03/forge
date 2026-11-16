# Correct and stable, but hard-codes the last axis.
import numpy as np


def log_softmax(x, axis=-1):
    z = x - np.max(x, axis=-1, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))
