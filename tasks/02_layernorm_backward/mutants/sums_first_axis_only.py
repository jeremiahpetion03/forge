# Reduces dgamma/dbeta over axis 0 only: right for 2-D input, wrong shape for 3-D.
import numpy as np


def layernorm_backward(dy, x, gamma, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * rstd
    dgamma = (dy * xhat).sum(axis=0)
    dbeta = dy.sum(axis=0)
    dxhat = dy * gamma
    dx = rstd * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
    return dx, dgamma, dbeta
