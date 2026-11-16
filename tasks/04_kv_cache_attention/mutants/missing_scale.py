# Fast and stable, but forgets the 1/sqrt(d) scaling, so attention is too sharp.
import numpy as np


class CachedAttention:
    def __init__(self, Wq, Wk, Wv):
        self.Wq, self.Wk, self.Wv = Wq, Wk, Wv
        self.K, self.V = [], []

    def step(self, x):
        self.K.append(x @ self.Wk)
        self.V.append(x @ self.Wv)
        K, V = np.array(self.K), np.array(self.V)
        s = K @ (x @ self.Wq)
        p = np.exp(s - s.max())
        return (p / p.sum()) @ V
