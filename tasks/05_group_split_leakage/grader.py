import numpy as np

from forge.checks import check


def dataset(seed=0, n_users=300):
    """Users with very different activity levels (1 to 40 rows), rows interleaved."""
    r = np.random.default_rng(seed)
    counts = r.integers(1, 41, n_users)
    ids = r.choice(10**6, n_users, replace=False)
    groups = np.repeat(ids, counts)
    return groups[r.permutation(len(groups))]


def split(sol, groups, frac=0.2, seed=0):
    tr, va = sol.group_split(groups.copy(), frac, seed)
    return np.asarray(tr), np.asarray(va)


@check("train and validation partition the rows exactly")
def partition(sol):
    g = dataset(1)
    tr, va = split(sol, g)
    both = np.concatenate([tr, va])
    assert len(both) == len(g), f"{len(both)} indices returned for {len(g)} rows"
    assert np.array_equal(np.sort(both), np.arange(len(g))), "indices missing or duplicated"


@check("no user appears in both train and validation", weight=3)
def no_leakage(sol):
    for s in range(3):
        g = dataset(2 + s)
        tr, va = split(sol, g, seed=s)
        leaked = np.intersect1d(g[tr], g[va])
        assert len(leaked) == 0, f"{len(leaked)} users appear in both splits"


@check("validation size within 0.05 of the requested fraction of rows")
def size(sol):
    for frac in (0.1, 0.2, 0.3):
        g = dataset(5)
        _, va = split(sol, g, frac)
        got = len(va) / len(g)
        assert abs(got - frac) <= 0.05, f"requested {frac}, got {got:.3f}"


@check("same seed reproduces the split exactly", kind="robustness")
def reproducible(sol):
    g = dataset(6)
    a, b = split(sol, g, seed=11), split(sol, g, seed=11)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), "same seed gave different splits"


@check("different seeds give different splits", kind="robustness")
def seed_matters(sol):
    g = dataset(7)
    vals = {tuple(np.sort(split(sol, g, seed=s)[1])) for s in range(5)}
    assert len(vals) == 5, f"only {len(vals)} distinct splits across 5 seeds"


@check("user selection does not depend on id order", kind="robustness")
def not_by_id(sol):
    # Across seeds, the validation users should cover the whole id range, not
    # just the smallest or largest ids.
    g = dataset(8)
    ranks = []
    users = np.unique(g)
    for s in range(10):
        _, va = split(sol, g, seed=s)
        ranks.extend(np.searchsorted(users, np.unique(g[va])) / len(users))
    ranks = np.asarray(ranks)
    assert 0.35 < ranks.mean() < 0.65, f"mean id-rank of validation users is {ranks.mean():.2f}"
