import numpy as np

from forge.checks import check

D = 5


def mse_grad(X, y, w):
    """Mean gradient of mean squared error for linear regression."""
    if len(y) == 0:
        raise ZeroDivisionError("grad_fn called on an empty micro-batch")
    return 2.0 * X.T @ (X @ w - y) / len(y)


class Recorder:
    """Wraps grad_fn and records the largest batch it was ever given."""

    def __init__(self):
        self.max_rows = 0
        self.calls = 0

    def __call__(self, X, y, w):
        self.max_rows = max(self.max_rows, len(y))
        self.calls += 1
        return mse_grad(X, y, w)


def make(sizes, seed=0):
    r = np.random.default_rng(seed)
    X = r.standard_normal((sum(sizes), D))
    y = X @ r.standard_normal(D) + 0.1 * r.standard_normal(sum(sizes))
    w = r.standard_normal(D)
    batches, start = [], 0
    for s in sizes:
        batches.append((X[start:start + s], y[start:start + s]))
        start += s
    return batches, X, y, w


def assert_exact(sol, sizes, seed=0):
    batches, X, y, w = make(sizes, seed)
    rec = Recorder()
    got = sol.accumulated_gradient(batches, w, rec)
    np.testing.assert_allclose(got, mse_grad(X, y, w), rtol=1e-12, atol=1e-12,
                               err_msg=f"micro-batch sizes {sizes}")
    return rec


@check("equal-sized micro-batches")
def equal(sol):
    assert_exact(sol, [8, 8, 8, 8])


@check("uneven micro-batches (dataset not a multiple of batch size)", weight=2)
def uneven(sol):
    assert_exact(sol, [16, 16, 16, 5], seed=1)
    assert_exact(sol, [1, 30, 7], seed=2)


@check("single micro-batch")
def single(sol):
    assert_exact(sol, [13], seed=3)


@check("empty micro-batches are skipped, not averaged in", kind="robustness")
def empty(sol):
    assert_exact(sol, [10, 0, 10, 4, 0], seed=4)


@check("never materializes more than one micro-batch", kind="performance")
def memory_bound(sol):
    # Tests only the memory property; correctness is judged by the other checks.
    sizes = [16, 16, 16, 16, 9]
    batches, _, _, w = make(sizes, seed=5)
    rec = Recorder()
    sol.accumulated_gradient(batches, w, rec)
    assert rec.max_rows <= max(sizes), (
        f"grad_fn received {rec.max_rows} rows at once; limit is the largest micro-batch ({max(sizes)})")
