# Leak-free and seeded, but picks a fraction of USERS rather than ROWS. Usually
# close; off whenever heavy users land on one side.
import numpy as np


def group_split(groups, val_fraction, seed):
    users = np.unique(groups)
    rng = np.random.default_rng(seed)
    k = int(round(val_fraction * len(users)))
    # pick users weighted toward heavy ones (a "stratified" attempt gone wrong)
    counts = np.array([np.sum(groups == u) for u in users], dtype=float)
    val_users = rng.choice(users, k, replace=False, p=counts / counts.sum())
    in_val = np.isin(groups, val_users)
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)
