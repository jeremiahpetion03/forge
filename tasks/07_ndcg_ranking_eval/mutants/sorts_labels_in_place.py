# Correct numbers, but builds the ideal ranking by sorting the caller's
# relevance array IN PLACE (after the gains were read), silently corrupting
# the labels for anything that uses them next.
import numpy as np


def ndcg_at_k(scores, relevance, k):
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)  # same object for float64 input
    q, n = scores.shape
    k = min(int(k), n)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (2.0 ** np.take_along_axis(rel, top, axis=1) - 1.0) @ discount
    rel.sort(axis=1)
    idcg = (2.0 ** rel[:, ::-1][:, :k] - 1.0) @ discount
    out = np.zeros(q)
    np.divide(dcg, idcg, out=out, where=idcg > 0)
    return out
