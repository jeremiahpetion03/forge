# Uses the raw label as the gain instead of 2**rel - 1. Same ranking, same
# discounts, but graded relevance is weighted wrongly (a 3 counts 3x a 1,
# not 7x), so scores are off whenever labels above 1 appear.
import numpy as np


def ndcg_at_k(scores, relevance, k):
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)
    q, n = scores.shape
    k = min(int(k), n)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.take_along_axis(rel, top, axis=1) @ discount
    idcg = -np.sort(-rel, axis=1)[:, :k] @ discount
    out = np.zeros(q)
    np.divide(dcg, idcg, out=out, where=idcg > 0)
    return out
