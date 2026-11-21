# Saves a reference to the layer's input instead of a copy. An in-place layer
# then overwrites it, so the "input" in the report is really the output.
import numpy as np


def _rep(i, name, inp, out, bad):
    return {"index": i, "name": name, "input": inp, "output": out.copy(),
            "first_position": tuple(int(v) for v in np.argwhere(bad)[0]),
            "n_nan": int(np.isnan(out).sum()), "n_posinf": int(np.isposinf(out).sum()),
            "n_neginf": int(np.isneginf(out).sum())}


def locate_first_nonfinite(layers, x):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        return _rep(-1, "<input>", x.copy(), x, ~np.isfinite(x))
    h = x.copy()
    for i, (name, fn) in enumerate(layers):
        before = h
        h = fn(h)
        if not np.isfinite(h).all():
            return _rep(i, name, before, h, ~np.isfinite(h))
    return None
