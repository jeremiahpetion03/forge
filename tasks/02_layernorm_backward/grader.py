import numpy as np

from forge.checks import check

EPS = 1e-5


def forward(x, gamma, beta, eps=EPS):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def numeric_grads(dy, x, gamma, beta, h=1e-6):
    """Central differences of L = sum(dy * y) w.r.t. x, gamma, beta (ground truth)."""
    def loss(x_, g_, b_):
        return float((dy * forward(x_, g_, b_)).sum())

    out = []
    for arr in (x, gamma, beta):
        g = np.zeros_like(arr)
        for i in np.ndindex(arr.shape):
            old = arr[i]
            arr[i] = old + h
            plus = loss(x, gamma, beta)
            arr[i] = old - h
            minus = loss(x, gamma, beta)
            arr[i] = old
            g[i] = (plus - minus) / (2 * h)
        out.append(g)
    return out


def case(shape, seed, scale=1.0):
    r = np.random.default_rng(seed)
    x = r.standard_normal(shape) * scale + r.standard_normal(shape[-1:]) * scale
    gamma = r.uniform(0.5, 1.5, shape[-1])
    beta = r.standard_normal(shape[-1])
    dy = r.standard_normal(shape)
    return dy, x, gamma, beta


def compare(sol, shape, seed, scale=1.0, rtol=1e-6, atol=1e-8):
    dy, x, gamma, beta = case(shape, seed, scale)
    dx, dgamma, dbeta = sol.layernorm_backward(dy.copy(), x.copy(), gamma.copy(), EPS)
    nx, ng, nb = numeric_grads(dy, x, gamma, beta)
    np.testing.assert_allclose(dx, nx, rtol=rtol, atol=atol, err_msg="dx vs finite differences")
    np.testing.assert_allclose(dgamma, ng, rtol=rtol, atol=atol, err_msg="dgamma vs finite differences")
    np.testing.assert_allclose(dbeta, nb, rtol=rtol, atol=atol, err_msg="dbeta vs finite differences")


@check("output shapes for 2-D and 3-D input")
def shapes(sol):
    for shape in ((4, 6), (2, 3, 6)):
        dy, x, gamma, _ = case(shape, 0)
        dx, dgamma, dbeta = sol.layernorm_backward(dy, x, gamma, EPS)
        assert np.shape(dx) == shape, f"dx shape {np.shape(dx)} for input {shape}"
        assert np.shape(dgamma) == (6,), f"dgamma shape {np.shape(dgamma)} for input {shape}"
        assert np.shape(dbeta) == (6,), f"dbeta shape {np.shape(dbeta)} for input {shape}"


@check("all gradients match finite differences (2-D)", weight=2)
def fd_2d(sol):
    compare(sol, (5, 7), seed=1)


@check("all gradients match finite differences (3-D batch)", weight=2)
def fd_3d(sol):
    compare(sol, (2, 3, 7), seed=2)


@check("correct when variance is comparable to eps", kind="robustness")
def low_variance(sol):
    compare(sol, (4, 8), seed=3, scale=3e-3)


@check("dx sums to zero along the normalized axis (shift invariance)")
def shift_invariance(sol):
    # y is unchanged by x + c, so the gradient must be orthogonal to the ones vector.
    dy, x, gamma, _ = case((6, 10), 4)
    dx, _, _ = sol.layernorm_backward(dy, x, gamma, EPS)
    np.testing.assert_allclose(np.sum(dx, axis=-1), 0.0, atol=1e-10)


@check("does not modify its inputs", kind="robustness")
def no_mutation(sol):
    dy, x, gamma, _ = case((3, 5), 5)
    copies = [a.copy() for a in (dy, x, gamma)]
    sol.layernorm_backward(dy, x, gamma, EPS)
    for name, a, c in zip(("dy", "x", "gamma"), (dy, x, gamma), copies, strict=True):
        assert np.array_equal(a, c), f"{name} was modified in place"
