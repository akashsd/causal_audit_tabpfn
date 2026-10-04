"""Effect estimation for a binary treatment.

- aipw:        cross-fitted doubly-robust (AIPW) estimator with influence-function CIs.
               Nuisances (propensity, per-arm outcome models) come from a pluggable learner.
- s_learner:   "just hand it to the model": one model on [treatment + all covariates],
               ATE = mean f(x, T=1) - f(x, T=0). No CI; shrinks effects toward zero.
- difference:  naive difference in means (no adjustment), Welch CI.
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

Z95 = 1.959963984540054


# ----------------------------------------------------------------------------- learners
def _tabpfn(kind: str, version: str):
    def make():
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        from .backend import load_token as _load_token

        _load_token()
        v = {"fast": ModelVersion.V3_5_FAST, "base": ModelVersion.V3_5}[version]
        cls = TabPFNClassifier if kind == "clf" else TabPFNRegressor
        return cls.create_default_for_version(v)
    return make


def _tabpfn_api(kind: str, thinking: bool = False):
    def make():
        from .backend import TabPFNAPI

        m = TabPFNAPI("classification" if kind == "clf" else "regression", "v3.5",
                      thinking="medium" if thinking else None)
        if kind == "clf":  # sklearn-style classes_ for predict_proba column lookup
            m.classes_ = np.array([0, 1])
        return m
    return make


def _lgbm(kind: str):
    def make():
        import lightgbm as lgb
        return (lgb.LGBMClassifier if kind == "clf" else lgb.LGBMRegressor)(verbose=-1, random_state=0)
    return make


def _lgbm_cv(kind: str):
    """LightGBM tuned by 3-fold randomized search (12 configurations) on the training fold."""
    def make():
        import lightgbm as lgb
        from sklearn.model_selection import RandomizedSearchCV
        base = (lgb.LGBMClassifier if kind == "clf" else lgb.LGBMRegressor)(verbose=-1, random_state=0)
        grid = {"n_estimators": [100, 200, 400], "learning_rate": [0.02, 0.05, 0.1],
                "num_leaves": [4, 8, 16, 31], "min_child_samples": [10, 20, 50, 100],
                "reg_lambda": [0.0, 1.0, 10.0], "subsample": [0.7, 1.0], "subsample_freq": [1]}
        return RandomizedSearchCV(base, grid, n_iter=12, cv=3, random_state=0, n_jobs=-1,
                                  scoring="neg_log_loss" if kind == "clf" else "neg_mean_squared_error")
    return make


def _linear(kind: str):
    def make():
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        est = LogisticRegression(max_iter=2000) if kind == "clf" else LinearRegression()
        return make_pipeline(StandardScaler(), est)
    return make


LEARNERS: dict[str, dict[str, Callable]] = {
    "tabpfn_fast": {"clf": _tabpfn("clf", "fast"), "reg": _tabpfn("reg", "fast")},
    "tabpfn": {"clf": _tabpfn("clf", "base"), "reg": _tabpfn("reg", "base")},
    "tabpfn_plus_api": {"clf": _tabpfn_api("clf"), "reg": _tabpfn_api("reg")},
    "tabpfn_thinking_api": {"clf": _tabpfn_api("clf", True), "reg": _tabpfn_api("reg", True)},
    "lgbm": {"clf": _lgbm("clf"), "reg": _lgbm("reg")},
    "lgbm_cv": {"clf": _lgbm_cv("clf"), "reg": _lgbm_cv("reg")},
    "linear": {"clf": _linear("clf"), "reg": _linear("reg")},
}


def _is_binary(v: np.ndarray) -> bool:
    u = np.unique(v[~np.isnan(v)])
    return len(u) <= 2 and set(u) <= {0.0, 1.0}


def _fit_predict(learner: str, X_tr, y_tr, X_te, binary: bool) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if binary:
            if len(np.unique(y_tr)) < 2:  # degenerate fold: every training label identical
                return np.full(len(X_te), float(np.unique(y_tr)[0]))
            m = LEARNERS[learner]["clf"]().fit(X_tr, y_tr.astype(int))
            p = np.asarray(m.predict_proba(X_te))
            classes = list(getattr(m, "classes_", [0, 1]))
            return p[:, classes.index(1)] if 1 in classes else np.zeros(len(X_te))
        return np.asarray(LEARNERS[learner]["reg"]().fit(X_tr, y_tr).predict(X_te), dtype=float)


# ----------------------------------------------------------------------------- estimators
def aipw(df: pd.DataFrame, treatment: str, outcome: str, adjust: list[str], learner: str = "tabpfn_fast",
         folds: int = 3, seed: int = 0, clip: float = 0.01) -> dict:
    t = df[treatment].to_numpy(dtype=float).round().astype(int)
    y = df[outcome].to_numpy(dtype=float)
    X = df[list(adjust)]
    n, ybin = len(df), _is_binary(y)
    e, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(n), t):
        if adjust:
            e[te] = _fit_predict(learner, X.iloc[tr], t[tr], X.iloc[te], binary=True)
        else:
            e[te] = t[tr].mean()
        for arm, mu in ((1, mu1), (0, mu0)):
            idx = tr[t[tr] == arm]
            mu[te] = (_fit_predict(learner, X.iloc[idx], y[idx], X.iloc[te], ybin) if adjust
                      else y[idx].mean())
    n_clipped = int(((e < clip) | (e > 1 - clip)).sum())
    ec = np.clip(e, clip, 1 - clip)
    psi = mu1 - mu0 + t * (y - mu1) / ec - (1 - t) * (y - mu0) / (1 - ec)
    ate, se = float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(n))
    return {"estimator": "aipw", "learner": learner, "ate": ate, "se": se,
            "lo": ate - Z95 * se, "hi": ate + Z95 * se, "n": n, "adjust": list(adjust),
            "propensity_min": float(e.min()), "propensity_max": float(e.max()), "n_clipped": n_clipped}


def s_learner(df: pd.DataFrame, treatment: str, outcome: str, covariates: list[str],
              learner: str = "tabpfn_fast") -> dict:
    t = df[treatment].to_numpy(dtype=float).round()
    y = df[outcome].to_numpy(dtype=float)
    X = df[[treatment, *covariates]].astype(float)
    X1, X0 = X.copy(), X.copy()
    X1[treatment], X0[treatment] = 1.0, 0.0
    ybin = _is_binary(y)
    both = pd.concat([X1, X0], ignore_index=True)
    pred = _fit_predict(learner, X, y, both, ybin)
    ate = float(pred[: len(df)].mean() - pred[len(df):].mean())
    return {"estimator": "s_learner", "learner": learner, "ate": ate, "se": np.nan, "lo": np.nan,
            "hi": np.nan, "n": len(df), "adjust": list(covariates)}


def difference(df: pd.DataFrame, treatment: str, outcome: str) -> dict:
    t = df[treatment].to_numpy(dtype=float).round().astype(int)
    y = df[outcome].to_numpy(dtype=float)
    y1, y0 = y[t == 1], y[t == 0]
    ate = float(y1.mean() - y0.mean())
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    return {"estimator": "difference", "learner": "-", "ate": ate, "se": se,
            "lo": ate - Z95 * se, "hi": ate + Z95 * se, "n": len(df), "adjust": []}
