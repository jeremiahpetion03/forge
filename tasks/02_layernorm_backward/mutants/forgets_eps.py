# Drops eps from the normalizer: indistinguishable from correct on ordinary
# data, wrong when the variance is small.
import numpy as np


def layernorm_backward(dy, x, gamma, eps=1e-5):
    D = x.shape[-1]
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var)
    xhat = (x - mu) * rstd
    dgamma = (dy * xhat).reshape(-1, D).sum(axis=0)
    dbeta = dy.reshape(-1, D).sum(axis=0)
    dxhat = dy * gamma
    dx = rstd * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    return dx, dgamma, dbeta
