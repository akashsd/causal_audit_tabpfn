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


def cluster_bootstrap(values: pd.Series, clusters: pd.Series, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% CI for a mean when observations are clustered (replicates within a scene are correlated):
    resample whole scenes with replacement."""
    df = pd.DataFrame({"v": values.astype(float).to_numpy(), "c": clusters.to_numpy()}).dropna()
    if df.empty:
        return (np.nan, np.nan)
    groups = [g.v.to_numpy() for _, g in df.groupby("c")]
    r = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        pick = r.integers(0, len(groups), len(groups))
        means.append(np.concatenate([groups[i] for i in pick]).mean())
    return tuple(np.percentile(means, [2.5, 97.5]))


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
    # the audit's overlap check: more than 2% of propensities clipped -> verdict "FRAGILE"
    d["flagged"] = d.n_clipped.fillna(0) > 0.02 * d.n
    rows = []
    for (scen, arm, learner), g in d.groupby(["scenario", "arm", "learner"]):
        est = g[g.estimator != "abstain"]
        cov = est.covered.dropna()
        k, n = int(cov.sum()), len(cov)
        lo, hi = cluster_bootstrap(est.covered.dropna().astype(float), est.loc[est.covered.notna(), "scene"])
        ok = est[~est.flagged].covered.dropna()
        rows.append({"scenario": scen, "arm": arm, "learner": learner, "n_estimates": len(est),
                     "abstained": int((g.estimator == "abstain").sum()),
                     "mean |error| (SD)": est.err_sd.abs().mean(), "RMSE (SD)": np.sqrt((est.err_sd ** 2).mean()),
                     "coverage": k / n if n else np.nan, "coverage 95% CI (scene-clustered)": f"[{lo:.2f}, {hi:.2f}]" if n else "-",
                     "CI width (SD)": est.width_sd.mean() if n else np.nan,
                     "% propensities clipped": 100 * (est.n_clipped / est.n).mean() if n else np.nan,
                     "flagged FRAGILE": int(est.flagged.sum()),
                     "coverage when not flagged": ok.mean() if len(ok) else np.nan})
    return pd.DataFrame(rows)


def claim_kinds(d: pd.DataFrame) -> pd.Series:
    """Label each E2 run: true claim, Claude's blind claim, or a wrong claim split into
    'detectable' (some tested implication is false in the true graph) vs 'equivalent' (none is)."""
    from .bench import MAX_TESTS, detectable, wrong_claims

    det = {}
    for scene in d.scene.unique():
        o = C.oracle_claim(scene)
        for w in wrong_claims(scene):
            det[(scene, w.author)] = detectable(w, o, MAX_TESTS)

    def kind(r):
        if r.claim.startswith("oracle"):
            return "true claim"
        if r.claim.startswith("claude"):
            return "Claude blind claim"
        base = "wrong: ignores hidden confounding" if r.claim == "wrong: assumes no hidden confounding"             else "wrong: reversed edge at treatment"
        return base + (" (detectable)" if det.get((r.scene, r.claim)) else " (equivalent — untestable)")
    return d.apply(kind, axis=1)


def e2_table() -> pd.DataFrame:
    d = _read("e2_falsification.jsonl")
    if d.empty:
        return d
    d["claim_kind"] = claim_kinds(d)
    rows = []
    for (k, m, n_rows), g in d.groupby(["claim_kind", "method", "n"]):
        t = g[g.testable]
        r, n = int(t.rejected.sum()), len(t)
        lo, hi = cluster_bootstrap(t.rejected.astype(float), t.scene) if n else (np.nan, np.nan)
        rows.append({"claim": k, "method": m, "n": n_rows, "scenes": g.scene.nunique(), "runs": len(g),
                     "testable runs": n, "rejected": r, "rejection rate": r / n if n else np.nan,
                     "95% CI (scene-clustered)": f"[{lo:.2f}, {hi:.2f}]" if n else "-"})
    return pd.DataFrame(rows)


def matched_power(target_fa: float = 0.05) -> pd.DataFrame:
    """Detection rate at an EQUAL false-alarm rate: per method (and n), pick the threshold on the
    claim-level score (smallest Holm-adjusted p) so that at most `target_fa` of TRUE claims are
    rejected, then report how many detectable wrong claims fall below it."""
    d = _read("e2_falsification.jsonl")
    if d.empty:
        return d
    d["claim_kind"] = claim_kinds(d)
    d = d[d.testable].copy()
    d["score"] = d.tests.map(lambda ts: min(t["p_holm"] for t in ts))
    rows = []
    for (m, n), g in d.groupby(["method", "n"]):
        true = np.sort(g[g.claim_kind == "true claim"].score.to_numpy())
        if not len(true):
            continue
        k = int(np.floor(target_fa * len(true)))      # number of true claims we allow to be rejected
        thr = true[k] if k < len(true) else 1.0        # reject iff score < thr
        for kind in sorted(g.claim_kind.unique()):
            if "detectable" not in kind:
                continue
            w = g[g.claim_kind == kind]
            rows.append({"method": m, "n": n, "claim": kind, "threshold": thr,
                         "false alarms on true claims": float((true < thr).mean()),
                         "detection at matched false-alarm rate": float((w.score < thr).mean()),
                         "runs": len(w)})
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
