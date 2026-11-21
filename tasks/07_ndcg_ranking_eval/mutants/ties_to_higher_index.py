# Sorts ascending and reverses. The ranking is descending, but reversing a
# stable sort sends tied documents to the HIGHER index first. Invisible on
# continuous random scores, wrong whenever scores tie.
import numpy as np


def ndcg_at_k(scores, relevance, k):
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)
    q, n = scores.shape
    k = min(int(k), n)
    top = np.argsort(scores, axis=1, kind="stable")[:, ::-1][:, :k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (2.0 ** np.take_along_axis(rel, top, axis=1) - 1.0) @ discount
    idcg = (2.0 ** -np.sort(-rel, axis=1)[:, :k] - 1.0) @ discount
    out = np.zeros(q)
    np.divide(dcg, idcg, out=out, where=idcg > 0)
    return out
