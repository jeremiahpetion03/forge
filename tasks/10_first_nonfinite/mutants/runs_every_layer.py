# Correct report, but runs the whole forward pass before scanning, so it keeps
# executing layers after the failure (wasted work, and side effects).
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
    h, trace = x.copy(), []
    for name, fn in layers:
        before = h.copy()
        h = fn(h)
        trace.append((name, before, h.copy()))
    for i, (name, before, out) in enumerate(trace):
        if not np.isfinite(out).all():
            return _rep(i, name, before, out, ~np.isfinite(out))
    return None
