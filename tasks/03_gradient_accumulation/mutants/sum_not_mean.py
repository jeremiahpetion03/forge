# Sums the mean gradients: equivalent to multiplying the learning rate by the
# number of micro-batches.
def accumulated_gradient(batches, w, grad_fn):
    total = 0
    for X, y in batches:
        if len(y):
            total = total + grad_fn(X, y, w)
    return total
