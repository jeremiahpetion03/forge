# Divides without guarding IDCG = 0, so a query with no relevant documents
# becomes 0/0 = NaN, which then poisons any mean taken over queries.
import numpy as np


def ndcg_at_k(scores, relevance, k):
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)
    q, n = scores.shape
    k = min(int(k), n)
    top = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (2.0 ** np.take_along_axis(rel, top, axis=1) - 1.0) @ discount
    idcg = (2.0 ** -np.sort(-rel, axis=1)[:, :k] - 1.0) @ discount
    with np.errstate(invalid="ignore", divide="ignore"):
        return dcg / idcg
