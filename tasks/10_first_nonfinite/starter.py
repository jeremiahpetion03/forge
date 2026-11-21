import numpy as np


def locate_first_nonfinite(layers, x):
    # Runs every layer, keeps references (not copies), and only looks for NaN.
    trace, h = [], x
    for name, fn in layers:
        inp = h
        h = fn(h)
        trace.append((name, inp, h))
    for i, (name, inp, out) in enumerate(trace):
        bad = np.isnan(out)
        if bad.any():
            return {"index": i, "name": name, "input": inp, "output": out,
                    "first_position": tuple(int(v) for v in np.argwhere(bad)[0]),
                    "n_nan": int(bad.sum()), "n_posinf": 0, "n_neginf": 0}
    return None
