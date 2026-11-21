import numpy as np

from forge.checks import check


class Net:
    """Builds layer lists and records which layers ran."""

    def __init__(self):
        self.calls = []

    def layer(self, name, fn):
        def wrapped(h):
            self.calls.append(name)
            return fn(h)
        return name, wrapped


def scale(c):
    return lambda h: h * c


def relu_inplace(h):
    np.maximum(h, 0, out=h)      # modifies its input, like torch's relu_
    return h


def add_inplace(c):
    def f(h):
        h += c
        return h
    return f


def overflow_layer(h):
    out = h.copy()
    out[1, 2] = np.inf           # a single +inf at row 1, col 2
    out[2, 0] = -np.inf
    return out


def inf_minus_inf(h):
    return h - h                 # inf - inf = NaN; finite - finite = 0


def make_x(seed=0):
    return np.random.default_rng(seed).standard_normal((4, 5))


def run(sol, layers, x):
    return sol.locate_first_nonfinite(layers, x)


def expected_report(index, name, inp, out):
    """Independent ground truth with explicit loops."""
    first = None
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            if not np.isfinite(out[i, j]) and first is None:
                first = (i, j)
    flat = out.ravel().tolist()
    return {"index": index, "name": name, "first_position": first,
            "n_nan": sum(v != v for v in flat),
            "n_posinf": sum(v == float("inf") for v in flat),
            "n_neginf": sum(v == float("-inf") for v in flat)}


def assert_fields(got, exp):
    assert got is not None, "returned None, but a non-finite value occurs"
    for k, v in exp.items():
        assert got[k] == v, f"{k}: got {got[k]!r}, expected {v!r}"


@check("all finite returns None")
def all_finite(sol):
    net = Net()
    layers = [net.layer("a", scale(2.0)), net.layer("b", relu_inplace)]
    assert run(sol, layers, make_x()) is None


@check("finds the layer that produced inf, not the later one where NaN appears", weight=2)
def inf_before_nan(sol):
    net = Net()
    layers = [net.layer("proj", scale(3.0)), net.layer("overflow", overflow_layer),
              net.layer("residual", inf_minus_inf), net.layer("out", scale(0.5))]
    got = run(sol, layers, make_x(1))
    out = overflow_layer(scale(3.0)(make_x(1)))
    assert_fields(got, expected_report(1, "overflow", None, out))


@check("stops immediately: later layers never run")
def stops(sol):
    net = Net()
    layers = [net.layer("a", scale(1.0)), net.layer("overflow", overflow_layer),
              net.layer("later", scale(2.0)), net.layer("last", scale(2.0))]
    run(sol, layers, make_x(2))
    assert net.calls == ["a", "overflow"], f"layers that ran: {net.calls}"


@check("input snapshot is the value before an in-place layer ran", weight=2)
def snapshot_before_inplace(sol):
    def inplace_overflow(h):
        h *= 1e308               # in place; large values overflow to +/-inf
        return h
    net = Net()
    x = make_x(3)
    layers = [net.layer("a", scale(1.0)), net.layer("blowup", inplace_overflow)]
    got = run(sol, layers, x)
    assert got is not None and got["index"] == 1, f"got {got and got['index']}"
    np.testing.assert_array_equal(got["input"], x, err_msg="snapshot shows the mutated input")
    assert np.isfinite(got["input"]).all(), "input snapshot contains non-finite values"


@check("counts and first position for NaN, +inf and -inf", weight=2)
def counts_and_position(sol):
    def mixed(h):
        out = h.copy()
        out[0, 3] = np.nan
        out[0, 4] = -np.inf
        out[2, 1] = np.inf
        out[3, 3] = np.nan
        return out
    net = Net()
    x = make_x(4)
    got = run(sol, [net.layer("mixed", mixed)], x)
    assert_fields(got, expected_report(0, "mixed", None, mixed(x)))


@check("returned arrays are copies: later edits cannot change the report", kind="robustness")
def copies(sol):
    net = Net()
    holder = {}

    def keep_and_overflow(h):
        out = overflow_layer(h)
        holder["out"] = out
        return out
    got = run(sol, [net.layer("a", relu_inplace), net.layer("k", keep_and_overflow)], make_x(5))
    snapshot = got["output"].copy()
    holder["out"][:] = 7.0       # the model keeps using its buffer
    np.testing.assert_array_equal(got["output"], snapshot, err_msg="output is a view, not a copy")


@check("a non-finite input is reported before any layer runs", kind="robustness")
def bad_input(sol):
    net = Net()
    x = make_x(6)
    x[3, 1] = np.nan
    got = run(sol, [net.layer("a", scale(2.0))], x)
    assert net.calls == [], f"layers ran on a non-finite input: {net.calls}"
    assert_fields(got, expected_report(-1, "<input>", None, x))


@check("does not modify the caller's input", kind="robustness")
def no_mutation(sol):
    net = Net()
    x = make_x(7)
    x0 = x.copy()
    run(sol, [net.layer("relu", relu_inplace), net.layer("add", add_inplace(1.0))], x)
    np.testing.assert_array_equal(x, x0, err_msg="the caller's x was modified by an in-place layer")
