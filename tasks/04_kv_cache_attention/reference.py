import numpy as np


class CachedAttention:
    def __init__(self, Wq, Wk, Wv):
        self.Wq, self.Wk, self.Wv = Wq, Wk, Wv
        d = Wk.shape[1]
        self.scale = 1.0 / np.sqrt(Wq.shape[1])
        self.K = np.empty((16, d), dtype=Wk.dtype)  # grows by doubling: amortized O(1) append
        self.V = np.empty((16, Wv.shape[1]), dtype=Wv.dtype)
        self.n = 0

    def _append(self, k, v):
        if self.n == len(self.K):
            self.K = np.concatenate([self.K, np.empty_like(self.K)])
            self.V = np.concatenate([self.V, np.empty_like(self.V)])
        self.K[self.n] = k
        self.V[self.n] = v
        self.n += 1

    def step(self, x):
        self._append(x @ self.Wk, x @ self.Wv)  # project only the new token
        K, V = self.K[: self.n], self.V[: self.n]
        s = (K @ (x @ self.Wq)) * self.scale
        p = np.exp(s - s.max())
        return (p / p.sum()) @ V
