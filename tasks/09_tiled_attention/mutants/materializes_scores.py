# Correct and stable, and it even loops over tiles, but it first writes every
# tile's scores into a full n x m buffer: exactly the memory traffic tiling is
# meant to avoid.
import numpy as np


def tiled_attention(Q, K, V, key_mask, block):
    n, d = Q.shape
    m = K.shape[0]
    S = np.empty((n, m))
    for s in range(0, m, block):
        S[:, s:s + block] = Q @ K[s:s + block].T / np.sqrt(d)
    S[:, ~key_mask] = -np.inf
    if not key_mask.any():
        return np.zeros((n, V.shape[1]))
    S -= S.max(axis=1, keepdims=True)
    np.exp(S, out=S)
    return (S @ V) / S.sum(axis=1, keepdims=True)
