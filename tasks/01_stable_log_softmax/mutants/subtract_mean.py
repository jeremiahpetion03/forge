# Plausible wrong fix: centers on the mean instead of the max, so the largest
# logit can still be far above zero and overflow exp.
import numpy as np


def log_softmax(x, axis=-1):
    z = x - np.mean(x, axis=axis, keepdims=True)
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))
