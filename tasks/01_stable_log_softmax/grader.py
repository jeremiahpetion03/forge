import numpy as np

from forge.checks import check


def expected(x, axis=-1):
    """Independent reference via logaddexp.reduce (pairwise, overflow-safe)."""
    x64 = np.asarray(x, dtype=np.float64)
    return x64 - np.logaddexp.reduce(x64, axis=axis, keepdims=True)


def rng():
    return np.random.default_rng(2024)


@check("matches reference on ordinary logits")
def ordinary(sol):
    x = rng().standard_normal((6, 9))
    np.testing.assert_allclose(sol.log_softmax(x), expected(x), rtol=1e-12, atol=1e-12)


@check("finite and exact for huge logits (1e4)", kind="robustness")
def huge(sol):
    x = rng().uniform(0, 50, (4, 10)) + 1e4
    out = sol.log_softmax(x)
    assert np.isfinite(out).all(), "non-finite output for large logits"
    np.testing.assert_allclose(out, expected(x), rtol=1e-9, atol=1e-9)


@check("finite and exact for very negative logits (-1e4)", kind="robustness")
def very_negative(sol):
    x = rng().uniform(0, 50, (4, 10)) - 1e4
    out = sol.log_softmax(x)
    assert np.isfinite(out).all(), "non-finite output for very negative logits (underflow)"
    np.testing.assert_allclose(out, expected(x), rtol=1e-9, atol=1e-9)


@check("float32 stays float32 and survives logits that overflow exp", kind="robustness")
def float32(sol):
    x = rng().uniform(0, 200, (5, 8)).astype(np.float32)  # float32 exp overflows above ~88.7
    out = sol.log_softmax(x)
    assert out.dtype == np.float32, f"dtype changed to {out.dtype}"
    assert np.isfinite(out).all(), "non-finite float32 output"
    np.testing.assert_allclose(out, expected(x), rtol=1e-5, atol=1e-4)


@check("masked (-inf) entries give exactly -inf, never NaN", kind="robustness")
def masked(sol):
    x = rng().standard_normal((3, 6)) * 5
    x[0, 1:] = -np.inf  # only one live entry in row 0
    x[1, ::2] = -np.inf
    out = sol.log_softmax(x)
    assert not np.isnan(out).any(), "NaN produced by masked entries"
    assert (out[np.isneginf(x)] == -np.inf).all(), "masked entries must be exactly -inf"
    live = ~np.isneginf(x)
    np.testing.assert_allclose(out[live], expected(x)[live], rtol=1e-12, atol=1e-12)
    assert out[0, 0] == 0.0, "a single live entry must have log-probability exactly 0"


@check("respects the axis argument on 3-D input")
def axes(sol):
    x = rng().standard_normal((3, 4, 5)) * 3
    for axis in (0, 1, -1):
        out = sol.log_softmax(x, axis=axis)
        assert out.shape == x.shape, f"shape {out.shape} for axis={axis}"
        np.testing.assert_allclose(out, expected(x, axis), rtol=1e-12, atol=1e-12,
                                   err_msg=f"axis={axis}")
