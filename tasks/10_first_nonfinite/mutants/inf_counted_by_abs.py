# Counts infinities with abs(x) == inf and puts them all under +inf, so -inf
# is reported as +inf.
import numpy as np


def locate_first_nonfinite(layers, x):
    x = np.asarray(x)
    def rep(i, name, inp, out):
        bad = ~np.isfinite(out)
        return {"index": i, "name": name, "input": inp, "output": out.copy(),
                "first_position": tuple(int(v) for v in np.argwhere(bad)[0]),
                "n_nan": int(np.isnan(out).sum()), "n_posinf": int((np.abs(out) == np.inf).sum()),
                "n_neginf": 0}
    if not np.isfinite(x).all():
        return rep(-1, "<input>", x.copy(), x)
    h = x.copy()
    for i, (name, fn) in enumerate(layers):
        before = h.copy()
        h = fn(h)
        if not np.isfinite(h).all():
            return rep(i, name, before, h)
    return None
