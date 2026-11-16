# Off-by-one: attends to the cache BEFORE appending the new token, so each
# token never sees itself.
import numpy as np


class CachedAttention:
    def __init__(self, Wq, Wk, Wv):
        self.Wq, self.Wk, self.Wv = Wq, Wk, Wv
        self.K, self.V = [], []

    def step(self, x):
        q = x @ self.Wq
        if self.K:
            K, V = np.array(self.K), np.array(self.V)
            s = K @ q / np.sqrt(len(q))
            p = np.exp(s - s.max())
            out = (p / p.sum()) @ V
        else:
            out = np.zeros(self.Wv.shape[1])
        self.K.append(x @ self.Wk)
        self.V.append(x @ self.Wv)
        return out
