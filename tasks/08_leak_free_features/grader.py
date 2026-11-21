import numpy as np

from forge.checks import check

CITIES = np.array(["austin", "boston", "chicago", "denver", "el_paso"])
PLANS = np.array(["free", "pro", "team"])


def data(seed, n, unseen=False):
    r = np.random.default_rng(seed)
    num = np.column_stack([r.normal(50, 10, n), r.exponential(3.0, n), r.normal(0, 1, n) * 1e3])
    num[r.random(num.shape) < 0.15] = np.nan
    cities = CITIES if not unseen else np.append(CITIES, ["fresno", "gary"])
    cat = np.column_stack([r.choice(cities[:4] if not unseen else cities, n), r.choice(PLANS, n)])
    return num, cat


def expected(train_num, train_cat, num, cat):
    """Independent ground truth, column by column with explicit loops."""
    out_cols = []
    for j in range(train_num.shape[1]):
        obs = train_num[:, j][~np.isnan(train_num[:, j])]
        med, mean, std = float(np.median(obs)), float(obs.mean()), float(obs.std())
        std = std if std > 0 else 1.0
        col = num[:, j].copy()
        col[np.isnan(col)] = med
        out_cols.append((col - mean) / std)
    for j in range(train_cat.shape[1]):
        for c in sorted(set(train_cat[:, j].tolist())):
            out_cols.append(np.array([1.0 if v == c else 0.0 for v in cat[:, j]]))
    return np.column_stack(out_cols)


def fitted(sol, seed=0, n=400):
    num, cat = data(seed, n)
    return sol.FeaturePipeline().fit(num.copy(), cat.copy()), num, cat


@check("matches an independent reference on held-out data", weight=2)
def matches(sol):
    pipe, tn, tc = fitted(sol, 1)
    vn, vc = data(2, 150)
    out = pipe.transform(vn.copy(), vc.copy())
    exp = expected(tn, tc, vn, vc)
    assert np.shape(out) == exp.shape, f"shape {np.shape(out)}; expected {exp.shape}"
    np.testing.assert_allclose(out, exp, rtol=1e-12, atol=1e-12)


@check("a row transformed alone equals the same row in a batch (no batch statistics)", weight=2)
def no_batch_stats(sol):
    pipe, _, _ = fitted(sol, 3)
    vn, vc = data(4, 60)
    batch = pipe.transform(vn.copy(), vc.copy())
    for i in (0, 7, 31):
        alone = pipe.transform(vn[i:i + 1].copy(), vc[i:i + 1].copy())
        np.testing.assert_allclose(alone[0], batch[i], rtol=1e-12, atol=1e-12,
                                   err_msg=f"row {i} differs when transformed alone")


@check("statistics ignore missing values before imputation")
def stats_before_impute(sol):
    r = np.random.default_rng(5)
    num = r.normal(10, 4, (300, 3))
    num[:150, 0] = np.nan  # half missing: imputing first would shrink std a lot
    cat = np.column_stack([r.choice(CITIES[:4], 300), r.choice(PLANS, 300)])
    pipe = sol.FeaturePipeline().fit(num.copy(), cat.copy())
    out = pipe.transform(num.copy(), cat.copy())
    np.testing.assert_allclose(out, expected(num, cat, num, cat), rtol=1e-12, atol=1e-12)


@check("unseen categories become all zeros, never an error", kind="robustness")
def unseen(sol):
    pipe, tn, tc = fitted(sol, 6)
    vn, vc = data(7, 80, unseen=True)
    out = pipe.transform(vn.copy(), vc.copy())
    np.testing.assert_allclose(out, expected(tn, tc, vn, vc), rtol=1e-12, atol=1e-12)


@check("a constant column gives finite features", kind="robustness")
def constant(sol):
    num, cat = data(8, 100)
    num[:, 1] = 2.5
    out = sol.FeaturePipeline().fit(num.copy(), cat.copy()).transform(num.copy(), cat.copy())
    assert np.isfinite(out).all(), "non-finite features (std 0 for a constant column?)"
    np.testing.assert_allclose(out, expected(num, cat, num, cat), rtol=1e-12, atol=1e-12)


@check("does not modify its inputs", kind="robustness")
def no_mutation(sol):
    num, cat = data(9, 50)
    n0, c0 = num.copy(), cat.copy()
    pipe = sol.FeaturePipeline().fit(num, cat)
    pipe.transform(num, cat)
    assert np.array_equal(num, n0, equal_nan=True), "numeric input was modified"
    assert np.array_equal(cat, c0), "categorical input was modified"
