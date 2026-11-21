import math
import time

import numpy as np

from forge.checks import check


def ndcg_loop(scores, relevance, k):
    """Independent ground truth: one query at a time, with an explicit sort key
    (score descending, then lower index). Shares no code with a vectorized
    solution."""
    out = []
    for s, r in zip(scores.tolist(), relevance.tolist()):
        n = len(s)
        kk = min(k, n)
        order = sorted(range(n), key=lambda i: (-s[i], i))[:kk]
        dcg = sum((2.0 ** r[i] - 1.0) / math.log2(pos + 2) for pos, i in enumerate(order))
        ideal = sorted(r, reverse=True)[:kk]
        idcg = sum((2.0 ** g - 1.0) / math.log2(pos + 2) for pos, g in enumerate(ideal))
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(out)


def data(seed, q, n):
    """Mostly irrelevant documents with a few graded hits (labels 1-3)."""
    r = np.random.default_rng(seed)
    scores = r.standard_normal((q, n))
    rel = np.where(r.random((q, n)) < 0.7, 0, r.integers(1, 4, (q, n))).astype(np.float64)
    return scores, rel


@check("matches an independent reference on graded relevance", weight=2)
def matches_reference(sol):
    scores, rel = data(1, 50, 30)
    out = sol.ndcg_at_k(scores.copy(), rel.copy(), 10)
    assert np.shape(out) == (50,), f"shape {np.shape(out)}; expected (50,)"
    np.testing.assert_allclose(out, ndcg_loop(scores, rel, 10), rtol=1e-12, atol=1e-12)


@check("a perfect ranking scores 1.0")
def perfect(sol):
    _, rel = data(2, 20, 15)
    rel[:, 0] = 3.0  # every query has at least one relevant document
    out = sol.ndcg_at_k(rel.copy(), rel.copy(), 5)  # score = label: the ideal order
    np.testing.assert_allclose(out, 1.0, rtol=0, atol=1e-12)


@check("tied scores rank the lower document index first")
def ties(sol):
    r = np.random.default_rng(3)
    scores = r.integers(0, 3, (40, 12)).astype(np.float64)  # heavy ties
    _, rel = data(4, 40, 12)
    out = sol.ndcg_at_k(scores.copy(), rel.copy(), 4)
    np.testing.assert_allclose(out, ndcg_loop(scores, rel, 4), rtol=1e-12, atol=1e-12)


@check("a query with no relevant documents scores 0.0, not NaN", kind="robustness")
def no_relevant(sol):
    scores, rel = data(5, 10, 20)
    rel[3] = 0.0
    rel[0, 0] = 2.0  # keep the other rows non-trivial
    out = sol.ndcg_at_k(scores.copy(), rel.copy(), 10)
    assert np.isfinite(out).all(), "non-finite nDCG (0/0 for a query with no relevant documents?)"
    assert out[3] == 0.0, f"query with no relevant documents scored {out[3]!r}, expected 0.0"
    np.testing.assert_allclose(out, ndcg_loop(scores, rel, 10), rtol=1e-12, atol=1e-12)


@check("k larger than the number of documents", kind="robustness")
def k_exceeds_n(sol):
    scores, rel = data(6, 8, 5)
    out = sol.ndcg_at_k(scores.copy(), rel.copy(), 50)
    np.testing.assert_allclose(out, ndcg_loop(scores, rel, 50), rtol=1e-12, atol=1e-12)


@check("does not modify its inputs", kind="robustness")
def no_mutation(sol):
    scores, rel = data(7, 6, 9)
    s0, r0 = scores.copy(), rel.copy()
    sol.ndcg_at_k(scores, rel, 4)
    assert np.array_equal(scores, s0), "scores were modified in place"
    assert np.array_equal(rel, r0), "relevance was modified in place"


def _best_time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@check("at least 5x faster than a per-query Python loop (Q=2000, N=100)", weight=2, kind="performance")
def speed(sol):
    # Baseline timed in the same process, so the check holds on slow and fast
    # machines alike. Best of 3 suppresses noise.
    scores, rel = data(8, 2000, 100)
    base = _best_time(lambda: ndcg_loop(scores, rel, 10))
    mine = _best_time(lambda: sol.ndcg_at_k(scores, rel, 10))
    assert mine * 5 <= base, f"{base / mine:.1f}x faster than the loop; need >= 5x"
