# Off-by-one in the position discount: divides by log2(i + 2) instead of
# log2(i + 1). A perfect ranking still scores exactly 1.0, so a test suite that
# only checks the ideal case would miss it.
import numpy as np


def ndcg_at_k(scores, relevance, k):
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)
    q, n = scores.shape
    k = min(int(k), n)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    discount = 1.0 / np.log2(np.arange(3, k + 3))
    dcg = (2.0 ** np.take_along_axis(rel, top, axis=1) - 1.0) @ discount
    idcg = (2.0 ** -np.sort(-rel, axis=1)[:, :k] - 1.0) @ discount
    out = np.zeros(q)
    np.divide(dcg, idcg, out=out, where=idcg > 0)
    return out
