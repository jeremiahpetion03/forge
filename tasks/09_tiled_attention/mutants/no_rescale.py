# Tiles correctly but never rescales the running sum and accumulator when a
# later tile raises the max: old tiles are weighted with a stale max.
import numpy as np


def tiled_attention(Q, K, V, key_mask, block):
    n, d = Q.shape
    run_max = np.full((n, 1), -np.inf)
    run_sum = np.zeros((n, 1))
    acc = np.zeros((n, V.shape[1]))
    for s in range(0, K.shape[0], block):
        valid = key_mask[s:s + block]
        if not valid.any():
            continue
        S = Q @ K[s:s + block][valid].T / np.sqrt(d)
        run_max = np.maximum(run_max, S.max(axis=1, keepdims=True))
        P = np.exp(S - run_max)
        run_sum = run_sum + P.sum(axis=1, keepdims=True)
        acc = acc + P @ V[s:s + block][valid]
    return np.divide(acc, run_sum, out=np.zeros_like(acc), where=run_sum > 0)
