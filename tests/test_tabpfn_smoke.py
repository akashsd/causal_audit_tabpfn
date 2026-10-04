"""Smoke test of the TabPFN-3.5 paths (local weights). Skipped without a GPU/licensed key."""
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _available():
    try:
        from causalaudit.backend import load_token
        load_token()
        return torch.cuda.is_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="needs CUDA + TABPFN_API_KEY (licensed local weights)")


def test_crt_rejects_dependence_and_aipw_runs():
    from causalaudit.estimate import aipw
    from causalaudit.falsify import crt_test

    r = np.random.default_rng(0)
    n = 400
    z = r.normal(size=n)
    t = (r.random(n) < 1 / (1 + np.exp(-z))).astype(float)
    y = 1.0 * t + np.sin(2 * z) + 0.3 * r.normal(size=n)
    df = pd.DataFrame({"Z": z, "T": t, "Y": y})
    assert crt_test(df, "Z", "Y", ("T",)) < 0.01          # Y depends on Z beyond T
    est = aipw(df, "T", "Y", ["Z"], learner="tabpfn_fast")
    assert est["lo"] < 1.0 < est["hi"]
