# Loops over m // block full tiles only, so when block does not divide m the
# trailing keys are silently ignored.
import numpy as np


def tiled_attention(Q, K, V, key_mask, block):
    n, d = Q.shape
    run_max = np.full((n, 1), -np.inf)
    run_sum = np.zeros((n, 1))
    acc = np.zeros((n, V.shape[1]))
    for t in range(K.shape[0] // block):
        s = t * block
        valid = key_mask[s:s + block]
        if not valid.any():
            continue
        S = Q @ K[s:s + block][valid].T / np.sqrt(d)
        new_max = np.maximum(run_max, S.max(axis=1, keepdims=True))
        alpha = np.exp(run_max - new_max)
        P = np.exp(S - new_max)
        run_sum = run_sum * alpha + P.sum(axis=1, keepdims=True)
        acc = acc * alpha + P @ V[s:s + block][valid]
        run_max = new_max
    return np.divide(acc, run_sum, out=np.zeros_like(acc), where=run_sum > 0)
