# Correct caching, but softmax without max-subtraction overflows on large scores.
import numpy as np


class CachedAttention:
    def __init__(self, Wq, Wk, Wv):
        self.Wq, self.Wk, self.Wv = Wq, Wk, Wv
        self.K, self.V = [], []

    def step(self, x):
        self.K.append(x @ self.Wk)
        self.V.append(x @ self.Wv)
        K, V = np.array(self.K), np.array(self.V)
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.exp(K @ (x @ self.Wq) / np.sqrt(len(x)))
            return (p / p.sum()) @ V
