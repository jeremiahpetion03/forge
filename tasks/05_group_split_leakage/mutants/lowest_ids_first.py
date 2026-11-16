# Leak-free and the right size, but always puts the lowest user ids in
# validation (ignores the seed). If ids correlate with sign-up date, this
# silently evaluates on the oldest users only.
import numpy as np


def group_split(groups, val_fraction, seed):
    users, counts = np.unique(groups, return_counts=True)
    cum = np.cumsum(counts)
    val_users = users[cum <= val_fraction * len(groups)]
    in_val = np.isin(groups, val_users)
    return np.flatnonzero(~in_val), np.flatnonzero(in_val)
