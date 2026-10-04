import numpy as np
import pandas as pd

from causalaudit.backend import _hash


def test_cache_key_depends_on_training_rows_and_test_rows():
    X = pd.DataFrame({"a": np.arange(10.0)})
    y = pd.Series(np.arange(10.0) % 2, name="y")
    k = _hash("tabpfn", "classification", "v3.5", None, "predict_proba", X.iloc[:5], y.iloc[:5], X.iloc[5:])
    k_other_fold = _hash("tabpfn", "classification", "v3.5", None, "predict_proba", X.iloc[5:], y.iloc[5:], X.iloc[:5])
    k_other_test = _hash("tabpfn", "classification", "v3.5", None, "predict_proba", X.iloc[:5], y.iloc[:5], X.iloc[6:])
    assert len({k, k_other_fold, k_other_test}) == 3
