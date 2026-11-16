def accumulated_gradient(batches, w, grad_fn):
    total = None
    n = 0
    for X, y in batches:
        k = len(y)
        if k == 0:
            continue
        g = grad_fn(X, y, w) * k  # mean gradient -> sum of per-row gradients
        total = g if total is None else total + g
        n += k
    if n == 0:
        raise ValueError("no rows in any micro-batch")
    return total / n
