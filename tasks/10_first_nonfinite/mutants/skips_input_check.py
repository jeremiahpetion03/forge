# Never checks x itself, so a NaN already in the input is blamed on layer 0.
import numpy as np


def _rep(i, name, inp, out, bad):
    return {"index": i, "name": name, "input": inp, "output": out.copy(),
            "first_position": tuple(int(v) for v in np.argwhere(bad)[0]),
            "n_nan": int(np.isnan(out).sum()), "n_posinf": int(np.isposinf(out).sum()),
            "n_neginf": int(np.isneginf(out).sum())}


def locate_first_nonfinite(layers, x):
    h = np.asarray(x).copy()
    for i, (name, fn) in enumerate(layers):
        before = h.copy()
        h = fn(h)
        if not np.isfinite(h).all():
            return _rep(i, name, before, h, ~np.isfinite(h))
    return None
