import math

import numpy as np


def ndcg_at_k(scores, relevance, k):
    # Correct, but one query at a time in pure Python: far too slow at scale.
    out = []
    for s, r in zip(scores, relevance):
        n = len(s)
        kk = min(k, n)
        order = sorted(range(n), key=lambda i: (-s[i], i))[:kk]
        dcg = sum((2.0 ** r[i] - 1.0) / math.log2(pos + 2) for pos, i in enumerate(order))
        ideal = sorted(r, reverse=True)[:kk]
        idcg = sum((2.0 ** g - 1.0) / math.log2(pos + 2) for pos, g in enumerate(ideal))
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(out)
