import time

import numpy as np

from forge.checks import check

D = 64


def weights(seed, scale=1.0):
    r = np.random.default_rng(seed)
    return [r.standard_normal((D, D)) * scale / np.sqrt(D) for _ in range(3)]


def full_attention(X, Wq, Wk, Wv):
    """Ground truth: full causal attention over the whole sequence at once."""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    S = Q @ K.T / np.sqrt(D)
    S = np.where(np.triu(np.ones((len(X), len(X)), dtype=bool), 1), -np.inf, S)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P / P.sum(axis=1, keepdims=True)) @ V


def run_steps(sol, X, W):
    att = sol.CachedAttention(*W)
    return np.stack([att.step(x) for x in X])


@check("first step returns that token's value vector")
def first_step(sol):
    W = weights(0)
    x = np.random.default_rng(1).standard_normal(D)
    np.testing.assert_allclose(sol.CachedAttention(*W).step(x), x @ W[2], rtol=1e-12, atol=1e-12)


@check("matches full causal attention at every step", weight=2)
def matches_full(sol):
    W = weights(2)
    X = np.random.default_rng(3).standard_normal((40, D))
    np.testing.assert_allclose(run_steps(sol, X, W), full_attention(X, *W), rtol=1e-10, atol=1e-12)


@check("stable when attention scores are large", kind="robustness")
def large_scores(sol):
    W = weights(4, scale=40.0)  # scores in the thousands: naive exp overflows
    X = np.random.default_rng(5).standard_normal((30, D))
    out = run_steps(sol, X, W)
    assert np.isfinite(out).all(), "non-finite outputs with large attention scores"
    np.testing.assert_allclose(out, full_attention(X, *W), rtol=1e-8, atol=1e-8)


@check("two caches used at the same time do not interfere", kind="robustness")
def independent(sol):
    # Compares the solution against ITSELF run alone, so this checks isolation
    # only; whether the outputs are correct is judged by the other checks.
    Wa, Wb = weights(6), weights(7)
    r = np.random.default_rng(8)
    Xa, Xb = r.standard_normal((12, D)), r.standard_normal((12, D))
    alone_a, alone_b = run_steps(sol, Xa, Wa), run_steps(sol, Xb, Wb)
    a, b = sol.CachedAttention(*Wa), sol.CachedAttention(*Wb)
    outs_a, outs_b = [], []
    for xa, xb in zip(Xa, Xb, strict=True):  # interleaved
        outs_a.append(a.step(xa))
        outs_b.append(b.step(xb))
    np.testing.assert_allclose(np.stack(outs_a), alone_a, rtol=1e-12, atol=1e-12,
                               err_msg="first cache changed when a second one was used alongside it")
    np.testing.assert_allclose(np.stack(outs_b), alone_b, rtol=1e-12, atol=1e-12,
                               err_msg="second cache changed when a first one was used alongside it")


class _Recompute:
    """Baseline that re-projects the whole history on every step (the starter's approach)."""

    def __init__(self, Wq, Wk, Wv):
        self.W = (Wq, Wk, Wv)
        self.xs = []

    def step(self, x):
        self.xs.append(x)
        X = np.stack(self.xs)
        K, V = X @ self.W[1], X @ self.W[2]
        s = K @ (x @ self.W[0]) / np.sqrt(D)
        p = np.exp(s - s.max())
        return (p / p.sum()) @ V


def _best_time(make, X, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        att = make()
        t0 = time.perf_counter()
        for x in X:
            att.step(x)
        best = min(best, time.perf_counter() - t0)
    return best


@check("at least 3x faster than re-projecting the history (800 steps)", weight=2, kind="performance")
def speed(sol):
    # Measured against a baseline timed in the same process, so the check
    # holds on slow and fast machines alike. Best of 3 suppresses noise.
    W = weights(9)
    X = np.random.default_rng(10).standard_normal((800, D))
    base = _best_time(lambda: _Recompute(*W), X)
    mine = _best_time(lambda: sol.CachedAttention(*W), X)
    assert mine * 3 <= base, f"{base / mine:.1f}x faster than baseline; need >= 3x"
