import numpy as np


def tiled_attention(Q, K, V, key_mask, block):
    # Correct, but ignores `block` and builds the full n x m score matrix.
    S = Q @ K.T / np.sqrt(Q.shape[1])
    S = np.where(key_mask[None, :], S, -np.inf)
    mx = S.max(axis=1, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    P = np.exp(S - mx)
    total = P.sum(axis=1, keepdims=True)
    out = P @ V
    return np.divide(out, total, out=np.zeros_like(out), where=total > 0)
