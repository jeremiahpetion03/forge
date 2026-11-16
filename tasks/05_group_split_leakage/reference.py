import numpy as np


def group_split(groups, val_fraction, seed):
    groups = np.asarray(groups)
    users, counts = np.unique(groups, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(users))
    target = val_fraction * len(groups)
    val_users, rows = [], 0
    for i in order:  # add whole users in random order until the row target is met
        if rows >= target:
            break
        val_users.append(users[i])
        rows += counts[i]
    in_val = np.isin(groups, val_users)
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)
