import numpy as np


def tiled_attention(Q, K, V, key_mask, block):
    n, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    run_max = np.full((n, 1), -np.inf)     # running max of scores per query
    run_sum = np.zeros((n, 1))             # running sum of exp(score - run_max)
    acc = np.zeros((n, V.shape[1]))        # running sum of exp(score - run_max) * v

    for start in range(0, K.shape[0], block):
        stop = start + block
        valid = key_mask[start:stop]
        if not valid.any():
            continue                       # nothing to add; avoids -inf - (-inf)
        Kb, Vb = K[start:stop][valid], V[start:stop][valid]
        S = Q @ Kb.T
        S *= scale                         # in place: one (n, block) temporary
        new_max = np.maximum(run_max, S.max(axis=1, keepdims=True))  # always finite here
        # Rescale what was accumulated under the old max. On the first valid
        # tile run_max is -inf, so the factor is exp(-inf) = 0, not NaN.
        alpha = np.exp(run_max - new_max)
        S -= new_max
        np.exp(S, out=S)                   # S now holds this tile's probabilities (unnormalized)
        run_sum = run_sum * alpha + S.sum(axis=1, keepdims=True)
        acc *= alpha
        acc += S @ Vb
        run_max = new_max

    # All keys masked: run_sum is 0, return zeros instead of 0/0.
    return np.divide(acc, run_sum, out=np.zeros_like(acc), where=run_sum > 0)
