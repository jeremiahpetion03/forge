import tracemalloc

import numpy as np

from forge.checks import check


def full_attention(Q, K, V, key_mask):
    """Ground truth: the whole score matrix at once (small inputs only)."""
    S = Q @ K.T / np.sqrt(Q.shape[1])
    S = np.where(key_mask[None, :], S, -np.inf)
    if not key_mask.any():
        return np.zeros((Q.shape[0], V.shape[1]))
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P / P.sum(axis=1, keepdims=True)) @ V


def qkv(seed, n, m, d=16, dv=8, scale=1.0):
    r = np.random.default_rng(seed)
    return (r.standard_normal((n, d)) * scale, r.standard_normal((m, d)) * scale,
            r.standard_normal((m, dv)))


def run(sol, Q, K, V, mask, block):
    return sol.tiled_attention(Q.copy(), K.copy(), V.copy(), mask.copy(), block)


@check("matches full attention across tiles", weight=2)
def matches(sol):
    Q, K, V = qkv(1, 30, 70)
    mask = np.random.default_rng(2).random(70) < 0.8
    np.testing.assert_allclose(run(sol, Q, K, V, mask, 16), full_attention(Q, K, V, mask),
                               rtol=1e-10, atol=1e-12)


@check("any block size: 1, not dividing m, larger than m")
def block_sizes(sol):
    Q, K, V = qkv(3, 12, 37)
    mask = np.ones(37, dtype=bool)
    exp = full_attention(Q, K, V, mask)
    for b in (1, 5, 37, 100):
        np.testing.assert_allclose(run(sol, Q, K, V, mask, b), exp, rtol=1e-10, atol=1e-12,
                                   err_msg=f"block={b}")


@check("stable when scores are in the thousands", kind="robustness")
def large_scores(sol):
    Q, K, V = qkv(4, 20, 50, scale=12.0)  # scores ~ thousands
    mask = np.ones(50, dtype=bool)
    out = run(sol, Q, K, V, mask, 8)
    assert np.isfinite(out).all(), "non-finite output with large scores"
    np.testing.assert_allclose(out, full_attention(Q, K, V, mask), rtol=1e-8, atol=1e-10)


@check("a tile whose keys are all masked does not produce NaN", kind="robustness")
def masked_tile(sol):
    Q, K, V = qkv(5, 10, 40)
    mask = np.ones(40, dtype=bool)
    mask[:16] = False  # the first two tiles of 8 are fully masked
    out = run(sol, Q, K, V, mask, 8)
    assert np.isfinite(out).all(), "NaN after a fully masked tile (-inf - -inf?)"
    np.testing.assert_allclose(out, full_attention(Q, K, V, mask), rtol=1e-10, atol=1e-12)


@check("all keys masked returns zeros, not NaN", kind="robustness")
def all_masked(sol):
    Q, K, V = qkv(6, 5, 12)
    out = run(sol, Q, K, V, np.zeros(12, dtype=bool), 4)
    assert np.shape(out) == (5, 8), f"shape {np.shape(out)}"
    assert np.isfinite(out).all() and (out == 0).all(), "expected exactly zeros"


@check("peak memory at most 50% of the full score matrix (n=m=2000)", weight=2, kind="performance")
def memory(sol):
    Q, K, V = qkv(7, 2000, 2000, d=32, dv=32)
    mask = np.ones(2000, dtype=bool)
    full_bytes = 2000 * 2000 * 8
    tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    sol.tiled_attention(Q, K, V, mask, 128)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    used = peak - base
    assert used <= 0.5 * full_bytes, (f"peak {used / 2**20:.1f} MB; budget "
                                       f"{0.5 * full_bytes / 2**20:.1f} MB (full matrix is "
                                       f"{full_bytes / 2**20:.1f} MB)")
