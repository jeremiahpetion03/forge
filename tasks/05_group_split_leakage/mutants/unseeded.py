# Leak-free, but uses fresh OS entropy instead of the seed: every run differs,
# so no experiment can be reproduced.
import numpy as np


def group_split(groups, val_fraction, seed):
    users = np.unique(groups)
    rng = np.random.default_rng()
    val_users = rng.choice(users, int(round(val_fraction * len(users))), replace=False)
    in_val = np.isin(groups, val_users)
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)
