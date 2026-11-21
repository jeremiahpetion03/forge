import numpy as np


def _report(index, name, inp, out):
    bad = ~np.isfinite(out)                       # catches NaN, +inf and -inf
    return {
        "index": index,
        "name": name,
        "input": inp,
        "output": out.copy(),
        "first_position": tuple(int(v) for v in np.argwhere(bad)[0]),  # argwhere is C order
        "n_nan": int(np.isnan(out).sum()),
        "n_posinf": int(np.isposinf(out).sum()),
        "n_neginf": int(np.isneginf(out).sum()),
    }


def locate_first_nonfinite(layers, x):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        return _report(-1, "<input>", x.copy(), x)
    h = x.copy()                                  # an in-place first layer must not touch x
    for i, (name, fn) in enumerate(layers):
        before = h.copy()                         # snapshot BEFORE the layer can mutate it
        h = fn(h)
        if not np.isfinite(h).all():
            return _report(i, name, before, h)    # stop: later layers never run
    return None
