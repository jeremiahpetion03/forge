# Reports the LAST non-finite element instead of the first.
import numpy as np


def locate_first_nonfinite(layers, x):
    x = np.asarray(x)
    def rep(i, name, inp, out):
        bad = ~np.isfinite(out)
        return {"index": i, "name": name, "input": inp, "output": out.copy(),
                "first_position": tuple(int(v) for v in np.argwhere(bad)[-1]),
                "n_nan": int(np.isnan(out).sum()), "n_posinf": int(np.isposinf(out).sum()),
                "n_neginf": int(np.isneginf(out).sum())}
    if not np.isfinite(x).all():
        return rep(-1, "<input>", x.copy(), x)
    h = x.copy()
    for i, (name, fn) in enumerate(layers):
        before = h.copy()
        h = fn(h)
        if not np.isfinite(h).all():
            return rep(i, name, before, h)
    return None
