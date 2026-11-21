import numpy as np


def ndcg_at_k(scores, relevance, k):
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(relevance, dtype=np.float64)
    n_queries, n_docs = scores.shape
    k = min(int(k), n_docs)

    # Top-k selection in O(N) per query instead of a full stable sort:
    # 1. find each row's k-th largest score v (np.partition, a copy),
    # 2. keep every score > v, then fill the remaining slots with the
    #    LOWEST-index documents whose score == v (the tie-break rule).
    # Exactly k documents are selected per row; -inf scores work unchanged.
    v = -np.partition(-scores, k - 1, axis=1)[:, k - 1:k]
    above = scores > v
    at = scores == v
    need = k - above.sum(axis=1, keepdims=True)
    chosen = above | (at & (np.cumsum(at, axis=1) <= need))
    idx = np.nonzero(chosen)[1].reshape(n_queries, k)  # ascending index order

    # Order only the k chosen documents. A STABLE sort of the negated scores
    # ranks descending and keeps lower indices first among ties.
    order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1, kind="stable")
    top = np.take_along_axis(idx, order, axis=1)

    discount = 1.0 / np.log2(np.arange(2, k + 2))       # positions 1..k
    dcg = (2.0 ** np.take_along_axis(rel, top, axis=1) - 1.0) @ discount
    # The ideal ranking uses ALL documents' labels, not just the retrieved ones.
    ideal = -np.sort(np.partition(-rel, k - 1, axis=1)[:, :k], axis=1)
    idcg = (2.0 ** ideal - 1.0) @ discount

    out = np.zeros(n_queries)
    np.divide(dcg, idcg, out=out, where=idcg > 0)      # IDCG = 0 -> 0.0, not 0/0
    return out
