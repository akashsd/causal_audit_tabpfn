"""Summarize E1/E2 logs into the headline tables (results/causal/summary.md)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from . import causalds as C
from .bench import OUT


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _read(name: str) -> pd.DataFrame:
    p = OUT / name
    return pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]) if p.exists() else pd.DataFrame()


def trap_scenes() -> set[str]:
    """Scenes where adjusting for all observed covariates is NOT a valid adjustment."""
    out = set()
    for s in C.benchmark_scenes()["identifiable"]:
        o = C.oracle_claim(s)
        covs = set(o.observed) - {o.treatment, o.outcome}
        if not o.is_valid_adjustment(covs):
            out.add(s)
    return out


def e1_table() -> pd.DataFrame:
    d = _read("e1_estimation.jsonl")
    if d.empty:
        return d
    sd = {s: C.ground_truth(s)["outcome_stats"]["std"] for s in d.scene.unique()}
    traps = trap_scenes()
    d["sd"] = d.scene.map(sd)
    d["err_sd"] = (d.ate - d.truth) / d.sd
    d["width_sd"] = (d.hi - d.lo) / d.sd
    d["scenario"] = np.where(d.scene.isin(traps), "trap", "no trap")
    rows = []
    for (scen, arm, learner), g in d.groupby(["scenario", "arm", "learner"]):
        est = g[g.estimator != "abstain"]
        cov = est.covered.dropna()
        k, n = int(cov.sum()), len(cov)
        lo, hi = wilson(k, n)
        rows.append({"scenario": scen, "arm": arm, "learner": learner, "n_estimates": len(est),
                     "abstained": int((g.estimator == "abstain").sum()),
                     "mean |error| (SD)": est.err_sd.abs().mean(), "RMSE (SD)": np.sqrt((est.err_sd ** 2).mean()),
                     "coverage": k / n if n else np.nan, "coverage 95% CI": f"[{lo:.2f}, {hi:.2f}]" if n else "-",
                     "CI width (SD)": est.width_sd.mean() if n else np.nan})
    return pd.DataFrame(rows)


def e2_table() -> pd.DataFrame:
    d = _read("e2_falsification.jsonl")
    if d.empty:
        return d
    def kind(c):
        if c.startswith("oracle"):
            return "true claim"
        if c.startswith("claude"):
            return "Claude blind claim"
        return "wrong: hidden confounding ignored" if "hidden" in c else "wrong: reversed edge at treatment"
    d["claim_kind"] = d.claim.map(kind)
    rows = []
    for (k, m), g in d.groupby(["claim_kind", "method"]):
        t = g[g.testable]
        r, n = int(t.rejected.sum()), len(t)
        lo, hi = wilson(r, n)
        rows.append({"claim": k, "method": m, "runs": len(g), "testable runs": n,
                     "rejected": r, "rejection rate (testable)": r / n if n else np.nan,
                     "95% CI": f"[{lo:.2f}, {hi:.2f}]" if n else "-",
                     "rejection rate (all)": r / len(g)})
    return pd.DataFrame(rows)


def main():
    parts = ["# CausalDS results", ""]
    e1 = e1_table()
    if not e1.empty:
        parts += ["## E1 — estimation (bias/RMSE/CI width in outcome-SD units; coverage of the true ATE)", "",
                  e1.sort_values(["scenario", "RMSE (SD)"]).to_markdown(index=False, floatfmt=".3f"), ""]
    e2 = e2_table()
    if not e2.empty:
        parts += ["## E2 — falsification (rejection = at least one implied independency contradicted, Holm α=0.05)", "",
                  e2.sort_values(["claim", "method"]).to_markdown(index=False, floatfmt=".2f"), ""]
    text = "\n".join(parts)
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
