# Selects validation users correctly but returns ALL rows as train, so the
# validation rows are also trained on.
import numpy as np


def group_split(groups, val_fraction, seed):
    users = np.unique(groups)
    rng = np.random.default_rng(seed)
    val_users = rng.choice(users, int(round(val_fraction * len(users))), replace=False)
    return np.arange(len(groups)), np.flatnonzero(np.isin(groups, val_users))
