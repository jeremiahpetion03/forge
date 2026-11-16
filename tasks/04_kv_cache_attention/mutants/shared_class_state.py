# Classic Python bug: the cache lives on the class, so every instance shares it.
# Correct for a single generation, silently wrong for two at once.
import numpy as np


class CachedAttention:
    K = []
    V = []

    def __init__(self, Wq, Wk, Wv):
        self.Wq, self.Wk, self.Wv = Wq, Wk, Wv
        type(self).K.clear()
        type(self).V.clear()

    def step(self, x):
        self.K.append(x @ self.Wk)
        self.V.append(x @ self.Wv)
        K, V = np.array(self.K), np.array(self.V)
        s = K @ (x @ self.Wq) / np.sqrt(len(x))
        p = np.exp(s - s.max())
        return (p / p.sum()) @ V
