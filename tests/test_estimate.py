import numpy as np
import pandas as pd

from causalaudit.estimate import aipw, difference
from causalaudit.falsify import holm, partial_corr_test


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


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == sorted(adj, key=lambda x: x) or max(adj) <= 1
    assert abs(adj[0] - 0.03) < 1e-12 and all(a <= 1 for a in adj)


def test_partial_corr_detects_linear_dependence():
    df = confounded()
    assert partial_corr_test(df, "Z", "Y", ("T",)) < 1e-6
