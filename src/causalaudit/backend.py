"""Paths, Prior Labs credentials, and a cached TabPFN-3.5 API wrapper."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_token() -> str:
    """Read the Prior Labs key from .env (TABPFN_API_KEY) and export it as TABPFN_TOKEN, which both
    the API client and the local `tabpfn` package (license-gated weight download) use."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    tok = os.environ.get("TABPFN_API_KEY") or os.environ.get("TABPFN_TOKEN")
    if not tok:
        raise RuntimeError("Set TABPFN_API_KEY in .env (see .env.example)")
    os.environ["TABPFN_TOKEN"] = tok
    return tok


def _hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (pd.DataFrame, pd.Series)):
            h.update(pd.util.hash_pandas_object(p, index=False).to_numpy().tobytes())
            cols = p.columns if isinstance(p, pd.DataFrame) else [p.name]
            dtypes = p.dtypes if isinstance(p, pd.DataFrame) else [p.dtype]
            h.update(repr(list(cols)).encode())
            h.update(repr([str(t) for t in dtypes]).encode())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:24]


class TabPFNAPI:
    """TabPFN-3.5 via the Prior Labs API (Plus by default; Thinking with `thinking=`).

    fit() only stores the training data; the API call happens at predict time, and the result is
    cached on disk keyed by (config, training rows, test rows) — re-runs cost zero credits, and
    different cross-fitting folds can never share a cache entry.
    """

    def __init__(self, task: str, version: str = "v3.5", thinking: str | None = None):
        self.task, self.version, self.thinking = task, version, thinking
        self.cache_hit = None

    def fit(self, X, y):
        self.X = pd.DataFrame(X).reset_index(drop=True)
        self.y = pd.Series(np.asarray(y), name="y")
        return self

    def _run(self, X, method: str):
        Xt = pd.DataFrame(X).reset_index(drop=True)
        key = _hash("tabpfn", self.task, self.version, self.thinking, method, self.X, self.y, Xt)
        path = RESULTS / "cache" / f"{key}.npy"
        if path.exists():
            self.cache_hit = True
            return np.load(path)
        import tabpfn_client as tc

        self.cache_hit = False
        tc.set_access_token(load_token())
        cls = tc.TabPFNClassifier if self.task == "classification" else tc.TabPFNRegressor
        kw = {"thinking_mode": True, "thinking_effort": self.thinking} if self.thinking else {}
        m = cls.create_default_for_version(self.version, **kw).fit(self.X, self.y)
        out = np.asarray(getattr(m, method)(Xt))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
        return out

    def predict_proba(self, X):
        return self._run(X, "predict_proba")

    def predict(self, X):
        return self._run(X, "predict")
