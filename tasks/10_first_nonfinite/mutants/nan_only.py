# Looks only for NaN, so it walks past the layer that produced inf and blames
# the later layer where inf - inf finally became NaN.
import numpy as np


def _rep(i, name, inp, out, bad):
    return {"index": i, "name": name, "input": inp, "output": out.copy(),
            "first_position": tuple(int(v) for v in np.argwhere(bad)[0]),
            "n_nan": int(np.isnan(out).sum()), "n_posinf": int(np.isposinf(out).sum()),
            "n_neginf": int(np.isneginf(out).sum())}


def locate_first_nonfinite(layers, x):
    x = np.asarray(x)
    if np.isnan(x).any():
        return _rep(-1, "<input>", x.copy(), x, np.isnan(x))
    h = x.copy()
    for i, (name, fn) in enumerate(layers):
        before = h.copy()
        h = fn(h)
        if np.isnan(h).any():
            return _rep(i, name, before, h, np.isnan(h))
    return None
