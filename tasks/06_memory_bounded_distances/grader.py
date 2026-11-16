import tracemalloc

import numpy as np

from forge.checks import check


def exact(A, B):
    """Ground truth, row by row (small inputs only)."""
    return np.stack([((a - B) ** 2).sum(axis=1) for a in A])


def points(n, d=32, seed=0, offset=0.0):
    return np.random.default_rng(seed).standard_normal((n, d)) + offset


@check("matches exact distances on ordinary data", weight=2)
def ordinary(sol):
    A, B = points(60, seed=1), points(45, seed=2)
    out = sol.pairwise_sq_dists(A, B)
    assert np.shape(out) == (60, 45), f"shape {np.shape(out)}"
    np.testing.assert_allclose(out, exact(A, B), rtol=1e-9, atol=1e-9)


@check("peak extra memory under 3x the output size", weight=2, kind="performance")
def memory(sol):
    A, B = points(2000, seed=3), points(500, seed=4)
    out_bytes = 2000 * 500 * 8
    tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    sol.pairwise_sq_dists(A, B)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    used = peak - base
    assert used <= 3 * out_bytes, f"peak {used / 2**20:.1f} MB; budget {3 * out_bytes / 2**20:.1f} MB"


@check("no negative distances when points are far from the origin", kind="robustness")
def nonnegative_far(sol):
    A = points(80, seed=5, offset=1e4)
    out = sol.pairwise_sq_dists(A, A)
    assert (out >= 0).all(), f"most negative distance {out.min():.3e}"


@check("self-distance ~0 and values accurate far from the origin", kind="robustness")
def accurate_far(sol):
    # Typical squared distance here is ~64; ||a||^2 is ~3e11. Without centering,
    # cancellation leaves errors around 1e-4; a centered solution stays near
    # 1e-14. The 1e-8 threshold sits orders of magnitude from both.
    A = points(80, seed=6, offset=1e5)
    out = sol.pairwise_sq_dists(A, A)
    assert np.abs(np.diag(out)).max() < 1e-8, f"self-distance up to {np.abs(np.diag(out)).max():.3e}"
    np.testing.assert_allclose(out, exact(A, A), rtol=1e-8, atol=1e-8)


@check("does not modify its inputs", kind="robustness")
def no_mutation(sol):
    A, B = points(20, seed=7), points(15, seed=8)
    a, b = A.copy(), B.copy()
    sol.pairwise_sq_dists(A, B)
    assert np.array_equal(A, a) and np.array_equal(B, b), "inputs were modified"
