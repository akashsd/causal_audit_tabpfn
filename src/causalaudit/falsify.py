"""Falsification: test every conditional independence a claim implies.

Engine: TabPFN-CRT (tabpfn-extensions) — a conditional randomization test that uses
TabPFN-3.5's full predictive distributions both to model the target and to *simulate*
the tested variable from p(A | S). This is what makes nonlinear, mixed-type CI testing
work out of the box on small data. Baseline: linear partial correlation (Fisher-style).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .claim import Claim


def _patch_crt():
    """tabpfn-extensions 0.6.3 builds the target tensor in the logits' dtype (float64 with
    TabPFN-3.5) while the bar-distribution borders are float32 -> RuntimeError on
    continuous targets. Compute the log-density in the criterion's dtype instead."""
    import torch
    import tabpfn_extensions.pval_crt.crt as crt

    if getattr(crt, "_causalaudit_patched", False):
        return

    def logp_from_full_output(full_out, y_np):
        crit, logits = full_out["criterion"], full_out["logits"]
        dt = crit.borders.dtype
        y = torch.as_tensor(np.asarray(y_np), device=logits.device, dtype=dt).view(*logits.shape[:-1])
        return (-crit(logits.to(dt), y)).detach().cpu().numpy().reshape(-1)

    crt.logp_from_full_output = logp_from_full_output
    crt._causalaudit_patched = True


@dataclass
class CITest:
    a: str
    b: str
    given: tuple[str, ...]
    p_value: float
    method: str
    seconds: float = 0.0
    p_adjusted: float | None = None

    @property
    def label(self) -> str:
        return f"{self.a} ⟂ {self.b} given {{{', '.join(self.given)}}}"


def crt_test(df: pd.DataFrame, a: str, b: str, given: tuple[str, ...], fast: bool = True,
             B: int = 100, seed: int = 0) -> float:
    """p-value for H0: a ⟂ b | given, with b as the CRT target and a resampled from p(a | given)."""
    from tabpfn.constants import ModelVersion
    from tabpfn_extensions.pval_crt import tabpfn_crt

    from .backend import load_token as _load_token

    _patch_crt()
    _load_token()  # local weights download is license-gated by the Prior Labs key
    df = df[[a, b, *given]].dropna()
    X = df[[a, *given]].copy()
    if not given:
        # Marginal independence: the CRT still needs a model p(a | rest). An independent noise
        # column keeps the test valid (p(a | noise) = p(a)) and gives TabPFN a feature to fit on.
        X["_independent_noise"] = np.random.default_rng(seed).normal(size=len(X))
    y = df[b].to_numpy()
    if not np.issubdtype(y.dtype, np.number):
        y = pd.factorize(y)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = tabpfn_crt(X, y.astype(np.float32), j=0, B=B, seed=seed,
                         model_version=ModelVersion.V3_5_FAST if fast else ModelVersion.V3_5)
    # The rank p-value (1 + #{T_null >= T_obs}) / (B + 1) is floored at 1/(B+1): with B=100 and Holm
    # over m tests nothing below m/101 is reachable, so claims with >= 6 tests could never be rejected.
    # Use a normal approximation to the CRT null distribution instead (calibration is checked
    # empirically by the false-alarm rate on true claims in experiment E2).
    null = np.asarray(res["T_null"], dtype=float)
    sd = null.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float(res["p_value"])
    return float(stats.norm.sf((float(res["T_obs"]) - null.mean()) / sd))


def partial_corr_test(df: pd.DataFrame, a: str, b: str, given: tuple[str, ...]) -> float:
    """Linear baseline: correlation of OLS residuals of a and b on `given`."""
    df = df[[a, b, *given]].dropna()
    Z = np.column_stack([np.ones(len(df))] + [pd.to_numeric(df[g], errors="coerce").to_numpy() for g in given])
    def resid(v):
        v = pd.to_numeric(df[v], errors="coerce").to_numpy(dtype=float)
        return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return float(stats.pearsonr(resid(a), resid(b)).pvalue)


def holm(ps: list[float]) -> list[float]:
    m, order = len(ps), np.argsort(ps)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


def falsify(claim: Claim, df: pd.DataFrame, method: str = "tabpfn_crt", alpha: float = 0.05,
            seed: int = 0, max_tests: int | None = None) -> dict:
    """Test all implied independencies of `claim`; the claim is REJECTED if any Holm-adjusted
    p-value < alpha. Returns {"rejected", "tests": [CITest], "n_tests"}."""
    import time

    implied = claim.implied_independencies()
    if max_tests:
        implied = implied[:max_tests]
    tests = []
    for a, b, given in implied:
        t0 = time.perf_counter()
        if method == "tabpfn_crt":
            p = crt_test(df, a, b, given, fast=True, seed=seed)
        elif method == "tabpfn_crt_base":
            p = crt_test(df, a, b, given, fast=False, seed=seed)
        elif method == "partial_corr":
            p = partial_corr_test(df, a, b, given)
        else:
            raise ValueError(method)
        tests.append(CITest(a, b, given, p, method, round(time.perf_counter() - t0, 2)))
    for t, pa in zip(tests, holm([t.p_value for t in tests]) if tests else []):
        t.p_adjusted = pa
    return {"rejected": any(t.p_adjusted < alpha for t in tests), "n_tests": len(tests),
            "testable": bool(tests), "tests": tests}
