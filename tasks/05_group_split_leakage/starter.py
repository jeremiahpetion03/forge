import numpy as np


def group_split(groups, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(groups))
    n_val = int(round(val_fraction * len(groups)))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])
