import numpy as np
import pandas as pd

from causalaudit.estimate import aipw, difference
from causalaudit.falsify import gcm_lgbm_test, holm, partial_corr_test


def confounded(n=4000, ate=2.0, seed=0):
    r = np.random.default_rng(seed)
    z = r.normal(size=n)
    t = (r.random(n) < 1 / (1 + np.exp(-1.5 * z))).astype(float)
    y = ate * t + 3 * z + r.normal(size=n)
    return pd.DataFrame({"Z": z, "T": t, "Y": y})


def test_aipw_recovers_effect_where_naive_fails():
    df = confounded()
    naive = difference(df, "T", "Y")
    est = aipw(df, "T", "Y", ["Z"], learner="linear")
    assert not (naive["lo"] <= 2.0 <= naive["hi"])
    assert est["lo"] <= 2.0 <= est["hi"]


def test_holm_exact_values():
    # sorted p: 0.01 (x3) = 0.03, 0.03 (x2) = 0.06, 0.04 (x1) = 0.04 -> monotone 0.06
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert holm([0.5, float("nan")]) == [1.0, 1.0]


def test_partial_corr_detects_linear_dependence():
    df = confounded()
    assert partial_corr_test(df, "Z", "Y", ("T",)) < 1e-6


def test_gcm_detects_nonlinear_dependence_and_accepts_independence():
    r = np.random.default_rng(1)
    s = r.normal(size=2000)
    a = np.sin(2 * s) + 0.3 * r.normal(size=2000)
    b_dep = a ** 2 + 0.3 * r.normal(size=2000) - (a ** 2).mean()
    b_ind = np.cos(2 * s) + 0.3 * r.normal(size=2000)          # depends on s only
    df = pd.DataFrame({"S": s, "A": a, "Bd": b_dep, "Bi": b_ind})
    assert gcm_lgbm_test(df, "A", "Bi", ("S",)) > 0.01
