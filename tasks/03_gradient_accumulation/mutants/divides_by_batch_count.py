# Weights correctly by rows but divides by the number of micro-batches times
# the nominal batch size, which is wrong whenever the last batch is short.
def accumulated_gradient(batches, w, grad_fn):
    nominal = max(len(y) for _, y in batches)
    total = 0
    for X, y in batches:
        if len(y):
            total = total + grad_fn(X, y, w) * len(y)
    return total / (nominal * len(batches))
