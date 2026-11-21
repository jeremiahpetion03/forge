# Builds the ideal ranking from only the documents the model retrieved in its
# top k, not from every document. Relevant documents the model missed never
# enter IDCG, so scores are inflated: a common bug in hand-rolled evaluators.
import numpy as np


def ndcg_at_k(scores, relevance, k):
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)
    q, n = scores.shape
    k = min(int(k), n)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    got = np.take_along_axis(rel, top, axis=1)
    dcg = (2.0 ** got - 1.0) @ discount
    idcg = (2.0 ** -np.sort(-got, axis=1) - 1.0) @ discount
    out = np.zeros(q)
    np.divide(dcg, idcg, out=out, where=idcg > 0)
    return out
