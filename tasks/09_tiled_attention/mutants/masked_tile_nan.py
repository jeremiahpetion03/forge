# Applies the mask as -inf inside each tile but never skips a fully masked
# tile. Then the tile max is -inf and exp(-inf - (-inf)) = NaN poisons the
# running statistics: the same failure class as fully masked padding rows.
import numpy as np


def tiled_attention(Q, K, V, key_mask, block):
    n, d = Q.shape
    run_max = np.full((n, 1), -np.inf)
    run_sum = np.zeros((n, 1))
    acc = np.zeros((n, V.shape[1]))
    with np.errstate(invalid="ignore"):
        for s in range(0, K.shape[0], block):
            S = Q @ K[s:s + block].T / np.sqrt(d)
            S = np.where(key_mask[s:s + block][None, :], S, -np.inf)
            new_max = np.maximum(run_max, S.max(axis=1, keepdims=True))
            alpha = np.exp(run_max - new_max)
            P = np.exp(S - new_max)
            run_sum = run_sum * alpha + P.sum(axis=1, keepdims=True)
            acc = acc * alpha + P @ V[s:s + block]
            run_max = new_max
        return np.divide(acc, run_sum, out=np.zeros_like(acc), where=run_sum > 0)
