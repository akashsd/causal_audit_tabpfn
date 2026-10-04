"""End-to-end audit of one causal question on one dataset.

  claim (explicit assumptions)  ->  identification  ->  falsification (TabPFN-CRT)
  ->  overlap check  ->  estimation (AIPW with TabPFN-3.5 nuisances)  ->  contrasts  ->  verdict
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .claim import Claim
from .estimate import aipw, difference, s_learner
from .falsify import falsify


def e_value(ate: float, lo: float, hi: float, outcome_sd: float) -> dict:
    """VanderWeele & Ding E-value via the standardized-difference approximation
    RR ≈ exp(0.91·d): how strong (risk-ratio scale) an unmeasured confounder would need to be,
    with both treatment and outcome, to explain the estimate (and the CI bound) away."""
    def ev(d):
        rr = math.exp(0.91 * abs(d))
        return rr + math.sqrt(rr * (rr - 1))
    d = ate / outcome_sd
    bound = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi)) / outcome_sd
    return {"estimate": round(ev(d), 2), "ci_bound": round(ev(bound), 2) if bound else 1.0}


def run(claim: Claim, df: pd.DataFrame, learner: str = "tabpfn_fast", ci_method: str = "tabpfn_crt",
        alpha: float = 0.05, max_tests: int | None = 12, seed: int = 0) -> dict:
    T, Y = claim.treatment, claim.outcome
    missing = [c for c in claim.observed if c not in df.columns]
    if missing:
        raise KeyError(f"claim mentions columns not in data: {missing}")
    df = df[claim.observed].dropna(subset=[T, Y]).reset_index(drop=True)
    covariates = [c for c in claim.observed if c not in (T, Y)]

    out: dict = {"question": claim.question or f"Effect of {T} on {Y}", "treatment": T, "outcome": Y,
                 "n": len(df), "claim_author": claim.author, "learner": learner}

    # 1. identification (pure logic on the claim)
    adj = claim.adjustment_set()
    out["identification"] = {"identifiable_by_adjustment": adj is not None,
                             "adjustment_set": sorted(adj) if adj is not None else None,
                             "forbidden": claim.forbidden_controls()}

    # 2. falsification: does the data contradict the claim's testable implications?
    fz = falsify(claim, df, method=ci_method, alpha=alpha, seed=seed, max_tests=max_tests)
    out["falsification"] = {"method": ci_method, "rejected": fz["rejected"], "n_tests": fz["n_tests"],
                            "tests": [{"test": t.label, "p": round(t.p_value, 4),
                                       "p_holm": round(t.p_adjusted, 4)} for t in fz["tests"]]}

    # 3-4. overlap + estimate. If only the hypothesised latent confounders block identification,
    # also report the estimate *conditional on* their absence, with an E-value saying how strong
    # they would have to be to explain the effect away.
    est_adj, conditional, working = adj, False, claim
    if adj is None and claim.latent:
        working = Claim(T, Y, claim.observed,
                        [e for e in claim.edges if e.src not in claim.latent and e.dst not in claim.latent],
                        background=claim.background, author=claim.author)
        est_adj, conditional = working.adjustment_set(), True
    # Which contradicted implications actually matter for THIS effect? A failure is consequential if
    # some local repair (edge either way, or a hidden common cause) would invalidate the adjustment set.
    contradicted = [t for t in fz["tests"] if t.p_adjusted < alpha]
    for t, row in zip(fz["tests"], out["falsification"]["tests"]):
        if t.p_adjusted < alpha:
            row["consequential"] = (est_adj is None) or working.failure_is_consequential(t.a, t.b, est_adj)
    out["falsification"]["consequential_failures"] = sum(
        bool(r.get("consequential")) for r in out["falsification"]["tests"])
    out["conditional_on_no_latent_confounding"] = conditional and est_adj is not None
    if est_adj is not None:
        adj = est_adj
        est = aipw(df, T, Y, sorted(adj), learner, seed=seed)
        out["estimate"] = est
        sd = float(df[Y].std())
        out["sensitivity"] = {"e_value": e_value(est["ate"], est["lo"], est["hi"], sd)}
        out["overlap"] = {"propensity_range": [round(est["propensity_min"], 3), round(est["propensity_max"], 3)],
                          "n_clipped": est["n_clipped"],
                          "ok": est["n_clipped"] <= 0.02 * len(df)}
    # 5. contrasts: what you would have concluded without the harness
    out["contrasts"] = {
        "naive_difference": difference(df, T, Y),
        "tabpfn_all_features_s_learner": s_learner(df, T, Y, covariates, learner),
    }
    if est_adj is not None and set(covariates) != set(adj):
        out["contrasts"]["aipw_all_features"] = aipw(df, T, Y, covariates, learner, seed=seed)

    # 6. verdict
    if adj is None:
        verdict = "NOT IDENTIFIABLE by adjustment under the stated assumptions — no effect reported."
    elif fz["rejected"] and out["falsification"]["consequential_failures"]:
        verdict = "ASSUMPTIONS REJECTED by the data — revise the causal claim before trusting any estimate."
    elif out["conditional_on_no_latent_confounding"]:
        ev = out["sensitivity"]["e_value"]
        verdict = (f"NOT IDENTIFIED WITHOUT AN EXTRA ASSUMPTION — the claim names unmeasured confounding "
                   f"({', '.join(claim.latent)}). If it were absent, ATE = {est['ate']:+.3f} "
                   f"(95% CI {est['lo']:+.3f} to {est['hi']:+.3f}); to explain this away it would need a "
                   f"risk-ratio association of ≥ {ev['estimate']} with both treatment and outcome "
                   f"(≥ {ev['ci_bound']} to move the CI to include zero).")
    elif not out["overlap"]["ok"]:
        verdict = "FRAGILE — poor overlap between treated and untreated; estimate relies on extrapolation."
    elif est["lo"] <= 0 <= est["hi"]:
        verdict = "INCONCLUSIVE — assumptions survive testing, but the confidence interval includes zero."
    else:
        verdict = "SUPPORTED — assumptions survive testing and the effect is distinguishable from zero."
    if contradicted and not out["falsification"]["consequential_failures"] and est_adj is not None:
        verdict = (f"PARTLY REJECTED, INCONSEQUENTIAL — {len(contradicted)} implied independenc"
                   f"{'y' if len(contradicted) == 1 else 'ies'} contradicted, but no local repair of them changes "
                   f"which variables must be adjusted for, so the estimate stands. " + verdict)
    out["verdict"] = verdict
    return out


def to_markdown(r: dict) -> str:
    L = [f"# Causal audit: {r['question']}", "", f"**Verdict: {r['verdict']}**", "",
         f"- data: {r['n']:,} rows · treatment `{r['treatment']}` · outcome `{r['outcome']}`",
         f"- assumptions by: {r['claim_author']} · nuisance learner: `{r['learner']}`", ""]
    idn = r["identification"]
    L += ["## 1. Identification (from the stated causal graph)", "",
          ("- adjust for: " + (", ".join(f"`{v}`" for v in idn["adjustment_set"]) if idn["adjustment_set"] else "(nothing)") if idn["identifiable_by_adjustment"] else "- **not identifiable by adjustment** (the claim includes unmeasured confounding)")]
    for v, why in idn["forbidden"].items():
        L.append(f"- do **not** adjust for `{v}`: {why}")
    fz = r["falsification"]
    L += ["", f"## 2. Falsification ({fz['method']}, {fz['n_tests']} implied independencies, Holm-corrected)", ""]
    if fz["tests"]:
        L += ["| implied by the claim | p | p (Holm) | |", "|---|---|---|---|"]
        def status(t):
            if t["p_holm"] >= 0.05:
                return "✓ consistent"
            return "❌ contradicted — " + ("**changes the adjustment set**" if t.get("consequential")
                                          else "does not affect the adjustment set")
        L += [f"| {t['test']} | {t['p']:.3f} | {t['p_holm']:.3f} | {status(t)} |" for t in fz["tests"]]
    else:
        L.append("_The claim implies no testable independencies among observed variables (untestable assumptions — see sensitivity)._")
    if "estimate" in r:
        e = r["estimate"]
        L += ["", "## 3. Estimate (cross-fitted AIPW, TabPFN-3.5 nuisance models)" + (" — *conditional on no unmeasured confounding*" if r.get("conditional_on_no_latent_confounding") else ""), "",
              f"- adjusted for: {', '.join(e['adjust']) or '(nothing)'}", f"- **ATE = {e['ate']:+.3f}** (95% CI {e['lo']:+.3f} to {e['hi']:+.3f})",
              f"- overlap: propensity range {r['overlap']['propensity_range']}, clipped {r['overlap']['n_clipped']}",
              f"- E-value: {r['sensitivity']['e_value']['estimate']} (CI bound {r['sensitivity']['e_value']['ci_bound']}) "
              "— the risk-ratio strength an unmeasured confounder would need to explain the effect away"]
    L += ["", "## 4. What you would have concluded without the audit", "", "| approach | ATE | 95% CI |", "|---|---|---|"]
    for k, c in r["contrasts"].items():
        ci = f"{c['lo']:+.3f} to {c['hi']:+.3f}" if np.isfinite(c.get("lo", np.nan)) else "—"
        L.append(f"| {k.replace('_', ' ')} | {c['ate']:+.3f} | {ci} |")
    return "\n".join(L) + "\n"


def save(r: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(r, indent=1, default=float, ensure_ascii=False), encoding="utf-8")
    (out_dir / "audit.md").write_text(to_markdown(r), encoding="utf-8")
    return out_dir / "audit.md"
