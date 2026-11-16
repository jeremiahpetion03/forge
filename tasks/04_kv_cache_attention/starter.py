import numpy as np


class CachedAttention:
    def __init__(self, Wq, Wk, Wv):
        self.Wq, self.Wk, self.Wv = Wq, Wk, Wv
        self.xs = []

    def step(self, x):
        self.xs.append(x)
        X = np.stack(self.xs)          # every token so far
        K = X @ self.Wk                # re-projected on every step
        V = X @ self.Wv
        q = x @ self.Wq
        s = K @ q / np.sqrt(len(q))
        p = np.exp(s - s.max())
        return (p / p.sum()) @ V
